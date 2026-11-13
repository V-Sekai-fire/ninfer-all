// See docs/maintainer/chip-tower.md.

#include "artifact/schema.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "models/qwen3_5/chip/tower.h"
#include "models/qwen3_5/load.h"
#include "runtime/engine/model_instance.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

using namespace ninfer;
namespace qwen = ninfer::models::qwen3_5;

std::string read_text(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

template <typename T>
std::vector<T> read_raw(const std::filesystem::path& path, std::size_t count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    std::vector<T> out(count);
    file.read(reinterpret_cast<char*>(out.data()),
              static_cast<std::streamsize>(count * sizeof(T)));
    if (static_cast<std::size_t>(file.gcount()) != count * sizeof(T) || file.peek() != EOF) {
        throw std::runtime_error("unexpected size of " + path.string());
    }
    return out;
}

struct Golden {
    std::size_t positions = 0;
    std::size_t inject_at = 0;
    std::vector<std::int32_t> dims;
    std::vector<std::int32_t> ids;
    std::vector<std::uint32_t> kept;
};

Golden load_golden(const std::filesystem::path& dir) {
    const artifact::Json meta = artifact::parse_json(read_text(dir / "golden.json"), "golden.json");
    Golden golden;
    golden.positions = meta.at("positions").get<std::size_t>();
    golden.inject_at = meta.at("inject_at").get<std::size_t>();
    for (const artifact::Json& dim : meta.at("dims")) { golden.dims.push_back(dim.get<std::int32_t>()); }
    golden.ids  = read_raw<std::int32_t>(dir / meta.at("ids").get<std::string>(), golden.positions);
    golden.kept = read_raw<std::uint32_t>(dir / meta.at("kept").get<std::string>(),
                                          golden.positions * golden.dims.size());
    return golden;
}

struct Outcome {
    std::size_t positions      = 0;
    std::size_t inexact        = 0;
    std::size_t inexact_prompt = 0;
    std::optional<std::size_t> first_bad;
    std::int32_t first_bad_dim = -1;
    std::uint32_t got_bits     = 0;
    std::uint32_t want_bits    = 0;
    std::size_t owned        = 0;
    std::size_t wrong_tokens = 0;
    std::size_t decode_errors = 0;
    std::optional<std::size_t> first_wrong;
    double seconds = 0.0;

    [[nodiscard]] bool ok() const { return inexact == 0 && wrong_tokens == 0 && decode_errors == 0; }
};

std::string json_number(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, 4);
    char text[32];
    std::snprintf(text, sizeof(text), std::isfinite(value) ? "%.9g" : "\"%g\"",
                  static_cast<double>(value));
    return text;
}

class FlippedWeight {
public:
    explicit FlippedWeight(const qwen::Model& model) {
        const std::vector<qwen::BlockWeights>& layers = model.weights().chip->layers;
        const qwen::DenseWeights& dense = std::get<qwen::DenseWeights>(layers[layers.size() / 2].ffn);
        const Weight weight = native_weight(model.weight(dense.down).view);
        const std::size_t count = static_cast<std::size_t>(weight.n) * weight.k;
        std::vector<std::uint16_t> host(count);
        CUDA_CHECK(cudaMemcpy(host.data(), weight.qdata, count * 2, cudaMemcpyDeviceToHost));
        std::size_t index = 0;
        while (index < count && (host[index] & 0x7fffU) == 0) { ++index; }
        if (index == count) { throw std::runtime_error("control: zero down projection"); }
        word_ = static_cast<std::uint16_t*>(const_cast<void*>(weight.qdata)) + index;
        bits_ = host[index];
        const std::uint16_t spoiled = bits_ ^ 0x8000U;
        CUDA_CHECK(cudaMemcpy(word_, &spoiled, 2, cudaMemcpyHostToDevice));
    }
    ~FlippedWeight() { (void)cudaMemcpy(word_, &bits_, 2, cudaMemcpyHostToDevice); }
    FlippedWeight(const FlippedWeight&)            = delete;
    FlippedWeight& operator=(const FlippedWeight&) = delete;

private:
    std::uint16_t* word_ = nullptr;
    std::uint16_t bits_  = 0;
};

Outcome run(const qwen::Model& model, cudaStream_t stream, const Golden& golden,
            std::string_view control, std::size_t limit) {
    std::optional<FlippedWeight> flipped;
    if (control == "weight") { flipped.emplace(model); }
    qwen::chip::Tower tower(model, stream, {.fp16_state = control == "fp16"});
    Outcome outcome;
    outcome.positions = std::min(limit, golden.positions);
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    tower.reset();
    for (std::size_t t = 0; t < outcome.positions; ++t) {
        if (t == golden.inject_at) { tower.load_firmware(); }
        tower.step(golden.ids[t]);
        const std::span<const std::uint16_t> residual = tower.residual();
        bool bad = false;
        for (std::size_t c = 0; c < golden.dims.size(); ++c) {
            const std::uint32_t got =
                static_cast<std::uint32_t>(residual[static_cast<std::size_t>(golden.dims[c])])
                << 16;
            const std::uint32_t want = golden.kept[t * golden.dims.size() + c];
            if (got == want) { continue; }
            bad = true;
            if (!outcome.first_bad) {
                outcome.first_bad     = t;
                outcome.first_bad_dim = golden.dims[c];
                outcome.got_bits      = got;
                outcome.want_bits     = want;
            }
        }
        outcome.inexact += bad;
        outcome.inexact_prompt += bad && t < golden.inject_at;
        if (t < golden.inject_at || t + 1 >= golden.positions) { continue; }
        try {
            const std::optional<TokenId> owner = tower.owner();
            if (!owner) { continue; }
            ++outcome.owned;
            if (*owner != golden.ids[t + 1]) {
                ++outcome.wrong_tokens;
                if (!outcome.first_wrong) { outcome.first_wrong = t; }
            }
        } catch (const std::exception&) {
            ++outcome.decode_errors;
            if (!outcome.first_wrong) { outcome.first_wrong = t; }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    outcome.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return outcome;
}

void report(const Golden& golden, std::string_view control, std::string_view profile,
            const Outcome& o) {
    std::printf("{\"control\": \"%.*s\", \"device_profile\": \"%.*s\", \"positions\": %zu, "
                "\"inject_at\": %zu, \"inexact_positions\": %zu, \"inexact_before_inject\": %zu, ",
                static_cast<int>(control.size()), control.data(), static_cast<int>(profile.size()),
                profile.data(), o.positions, golden.inject_at, o.inexact, o.inexact_prompt);
    if (!o.first_bad) {
        std::printf("\"first_bad_position\": null, ");
    } else {
        std::printf("\"first_bad_position\": %zu, \"first_bad_dim\": %d, \"got\": %s, "
                    "\"want\": %s, \"got_bits\": \"0x%08x\", \"want_bits\": \"0x%08x\", ",
                    *o.first_bad, o.first_bad_dim, json_number(o.got_bits).c_str(),
                    json_number(o.want_bits).c_str(), o.got_bits, o.want_bits);
    }
    std::printf("\"owned_positions\": %zu, \"wrong_tokens\": %zu, \"decode_errors\": %zu, ",
                o.owned, o.wrong_tokens, o.decode_errors);
    if (o.first_wrong) { std::printf("\"first_wrong_position\": %zu, ", *o.first_wrong); }
    std::printf("\"seconds\": %.1f, \"ok\": %s}\n", o.seconds, o.ok() ? "true" : "false");
    std::fflush(stdout);
}

int run_main(int argc, char** argv) {
    std::vector<std::string> paths;
    std::string control  = "all";
    std::string firmware = "calc-bridge";
    std::string profile  = "auto";
    std::size_t limit    = static_cast<std::size_t>(-1);
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const std::function<std::string()> next = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument("missing value"); }
            return argv[++i];
        };
        if (arg == "--control") {
            control = next();
        } else if (arg == "--firmware") {
            firmware = next();
        } else if (arg == "--device-profile") {
            profile = next();
        } else if (arg == "--limit") {
            limit = static_cast<std::size_t>(std::atoll(next().c_str()));
        } else {
            paths.emplace_back(arg);
        }
    }
    if (paths.empty()) {
        const char* artifact = std::getenv("NINFER_CHIP_ARTIFACT");
        const char* golden   = std::getenv("NINFER_CHIP_TOWER_GOLDEN");
        if (artifact == nullptr || golden == nullptr) {
            std::cout << "SKIP: supply ARTIFACT GOLDEN.raw\n";
            return 77;
        }
        paths = {artifact, golden};
    }
    if (paths.size() != 2) { throw std::invalid_argument("usage: ARTIFACT GOLDEN.raw [options]"); }

    const Golden golden = load_golden(paths[1]);
    DeviceContext device;
    EngineOptions engine;
    engine.device_profile = profile;
    runtime::install_device_route_profile_for(engine, device);
    models::LoadOptions options;
    options.chip_firmware = firmware;
    const std::unique_ptr<qwen::Model> model = qwen::load_model(paths[0], options, device);
    device.synchronize();

    if (control != "all") {
        const Outcome outcome = run(*model, device.stream, golden, control, limit);
        report(golden, control, profile, outcome);
        return outcome.ok() ? 0 : 1;
    }
    const Outcome plain = run(*model, device.stream, golden, "none", limit);
    report(golden, "none", profile, plain);
    const Outcome fp16 = run(*model, device.stream, golden, "fp16", limit);
    report(golden, "fp16", profile, fp16);
    const Outcome weight = run(*model, device.stream, golden, "weight", limit);
    report(golden, "weight", profile, weight);
    return plain.ok() && fp16.inexact != 0 && weight.inexact != 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "FATAL: " << error.what() << std::endl;
        return 1;
    }
}
