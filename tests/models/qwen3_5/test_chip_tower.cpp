// Runs an exported chip tower one token at a time through the public Ops and compares the kept
// residual dims with the reference bundle bit for bit. See docs/maintainer/chip-tower.md.

#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"

#include "core/arena.h"
#include "core/weight.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using ninfer::DeviceBuffer;
using ninfer::DType;
using ninfer::Tensor;
using ninfer::Weight;
namespace ops = ninfer::ops;

struct Json {
    enum class Kind { Null, Bool, Number, String, Array, Object };
    Kind kind      = Kind::Null;
    double number  = 0.0;
    bool boolean   = false;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> fields;

    const Json* find(std::string_view key) const {
        for (const auto& field : fields) {
            if (field.first == key) { return &field.second; }
        }
        return nullptr;
    }
    const Json& at(std::string_view key) const {
        const Json* value = find(key);
        if (value == nullptr) { throw std::runtime_error("json: missing key " + std::string(key)); }
        return *value;
    }
    std::int64_t integer() const {
        if (kind != Kind::Number) { throw std::runtime_error("json: expected a number"); }
        return static_cast<std::int64_t>(number);
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : s_(text) {}

    Json parse() {
        Json value = parse_value();
        skip_space();
        if (i_ != s_.size()) { fail("trailing text"); }
        return value;
    }

private:
    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error("json: " + std::string(what) + " at byte " + std::to_string(i_));
    }
    void skip_space() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\t' ||
                                  s_[i_] == '\r')) {
            ++i_;
        }
    }
    char peek() {
        skip_space();
        if (i_ >= s_.size()) { fail("unexpected end"); }
        return s_[i_];
    }
    void expect(char c) {
        if (peek() != c) { fail("unexpected character"); }
        ++i_;
    }
    bool literal(std::string_view word) {
        if (s_.substr(i_, word.size()) != word) { return false; }
        i_ += word.size();
        return true;
    }
    std::string parse_string() {
        expect('"');
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            char c = s_[i_++];
            if (c == '\\') {
                if (i_ >= s_.size()) { fail("bad escape"); }
                const char e = s_[i_++];
                if (e == 'u') {
                    if (i_ + 4 > s_.size()) { fail("bad escape"); }
                    const unsigned long code =
                        std::strtoul(std::string(s_.substr(i_, 4)).c_str(), nullptr, 16);
                    i_ += 4;
                    c = code < 0x80 ? static_cast<char>(code) : '?';
                } else {
                    c = e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e;
                }
            }
            out.push_back(c);
        }
        if (i_ >= s_.size()) { fail("unterminated string"); }
        ++i_;
        return out;
    }
    Json parse_value() {
        Json value;
        const char c = peek();
        if (c == '{') {
            value.kind = Json::Kind::Object;
            ++i_;
            if (peek() == '}') {
                ++i_;
                return value;
            }
            for (;;) {
                std::string key = parse_string();
                expect(':');
                value.fields.emplace_back(std::move(key), parse_value());
                if (peek() == ',') {
                    ++i_;
                    continue;
                }
                expect('}');
                return value;
            }
        }
        if (c == '[') {
            value.kind = Json::Kind::Array;
            ++i_;
            if (peek() == ']') {
                ++i_;
                return value;
            }
            for (;;) {
                value.items.push_back(parse_value());
                if (peek() == ',') {
                    ++i_;
                    continue;
                }
                expect(']');
                return value;
            }
        }
        if (c == '"') {
            value.kind = Json::Kind::String;
            value.text = parse_string();
            return value;
        }
        if (literal("true") || literal("false")) {
            value.kind    = Json::Kind::Bool;
            value.boolean = s_[i_ - 2] == 'u';
            return value;
        }
        if (literal("null")) { return value; }
        const std::string rest(s_.substr(i_, 64));
        char* end    = nullptr;
        value.number = std::strtod(rest.c_str(), &end);
        if (end == rest.c_str()) { fail("unexpected token"); }
        value.kind = Json::Kind::Number;
        i_ += static_cast<std::size_t>(end - rest.c_str());
        return value;
    }

    std::string_view s_;
    std::size_t i_ = 0;
};

std::string read_text(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path); }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

template <typename T>
std::vector<T> read_raw(const std::string& path, std::size_t count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path); }
    std::vector<T> out(count);
    file.read(reinterpret_cast<char*>(out.data()),
              static_cast<std::streamsize>(count * sizeof(T)));
    if (static_cast<std::size_t>(file.gcount()) != count * sizeof(T) || file.peek() != EOF) {
        throw std::runtime_error("unexpected size of " + path);
    }
    return out;
}

class SafeTensors {
public:
    struct Entry {
        std::string dtype;
        std::vector<std::int64_t> shape;
        std::uint64_t begin = 0;
        std::uint64_t end   = 0;
    };

    explicit SafeTensors(const std::string& path) : file_(path, std::ios::binary) {
        if (!file_) { throw std::runtime_error("cannot open " + path); }
        unsigned char size_bytes[8];
        file_.read(reinterpret_cast<char*>(size_bytes), 8);
        std::uint64_t header_bytes = 0;
        for (int i = 7; i >= 0; --i) { header_bytes = (header_bytes << 8) | size_bytes[i]; }
        std::string header(header_bytes, '\0');
        file_.read(header.data(), static_cast<std::streamsize>(header_bytes));
        if (!file_) { throw std::runtime_error("safetensors: truncated header"); }
        base_ = 8 + header_bytes;
        for (const auto& [name, value] : JsonParser(header).parse().fields) {
            if (name == "__metadata__") { continue; }
            Entry entry;
            entry.dtype = value.at("dtype").text;
            for (const Json& extent : value.at("shape").items) {
                entry.shape.push_back(extent.integer());
            }
            entry.begin = static_cast<std::uint64_t>(value.at("data_offsets").items.at(0).number);
            entry.end   = static_cast<std::uint64_t>(value.at("data_offsets").items.at(1).number);
            entries_.emplace(name, std::move(entry));
        }
    }

    const Entry& entry(const std::string& name, std::string_view dtype) const {
        const auto it = entries_.find(name);
        if (it == entries_.end()) { throw std::runtime_error("safetensors: no tensor " + name); }
        if (it->second.dtype != dtype) {
            throw std::runtime_error("safetensors: " + name + " is " + it->second.dtype);
        }
        return it->second;
    }

    template <typename T>
    std::vector<T> read(const std::string& name, std::string_view dtype) {
        const Entry& e = entry(name, dtype);
        std::vector<T> out((e.end - e.begin) / sizeof(T));
        file_.seekg(static_cast<std::streamoff>(base_ + e.begin));
        file_.read(reinterpret_cast<char*>(out.data()),
                   static_cast<std::streamsize>(e.end - e.begin));
        if (!file_) { throw std::runtime_error("safetensors: truncated tensor " + name); }
        return out;
    }

    std::string find_suffix(std::string_view prefix, std::string_view suffix) const {
        for (const auto& [name, entry] : entries_) {
            if (name.starts_with(prefix) && name.ends_with(suffix)) { return name; }
        }
        throw std::runtime_error("safetensors: no tensor " + std::string(prefix) + "*" +
                                 std::string(suffix));
    }

private:
    std::ifstream file_;
    std::uint64_t base_ = 0;
    std::map<std::string, Entry> entries_;
};

struct Matrix {
    DeviceBuffer device;
    std::int32_t n = 0;
    std::int32_t k = 0;

    Weight weight() const {
        Weight w{};
        w.payload         = device.p;
        w.payload_bytes   = static_cast<std::size_t>(n) * k * sizeof(std::uint16_t);
        w.qtype           = ninfer::QType::BF16;
        w.shape[0]        = n;
        w.shape[1]        = k;
        w.padded_shape[0] = n;
        w.padded_shape[1] = k;
        w.ndim            = 2;
        w.qdata           = device.p;
        w.n               = n;
        w.k               = k;
        w.layout          = ninfer::QuantLayout::Contiguous;
        return w;
    }
};

template <typename T>
DeviceBuffer upload(const std::vector<T>& host) {
    DeviceBuffer device(host.size() * sizeof(T));
    device.copy_from_host(host.data(), device.bytes);
    return device;
}

Matrix load_matrix(SafeTensors& file, const std::string& name) {
    const auto& entry = file.entry(name, "BF16");
    if (entry.shape.size() != 2) { throw std::runtime_error(name + " is not a matrix"); }
    Matrix m;
    m.n      = static_cast<std::int32_t>(entry.shape[0]);
    m.k      = static_cast<std::int32_t>(entry.shape[1]);
    m.device = upload(file.read<std::uint16_t>(name, "BF16"));
    return m;
}

DeviceBuffer load_fp32_vector(SafeTensors& file, const std::string& name) {
    const std::vector<std::uint16_t> bits = file.read<std::uint16_t>(name, "BF16");
    std::vector<float> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) { values[i] = ninfer::test::bf16_to_f32(bits[i]); }
    return upload(values);
}

struct Layer {
    Matrix qkv, z, a, b, out, gate, up, down;
    DeviceBuffer conv, norm1, norm2, gated_norm, a_log, dt_bias;
};

struct Tower {
    std::int32_t hidden      = 0;
    std::int32_t qk_heads    = 0;
    std::int32_t value_heads = 0;
    std::int32_t head_dim    = 0;
    std::int32_t channels    = 0;
    float eps                = 0.0F;
    Matrix embedding;
    std::vector<Layer> layers;
    std::vector<float> firmware;
};

Tower load_tower(const std::string& dir) {
    const Json config = JsonParser(read_text(dir + "/config.json")).parse();
    SafeTensors file(dir + "/model.safetensors");
    Tower tower;
    tower.qk_heads    = static_cast<std::int32_t>(config.at("linear_num_key_heads").integer());
    tower.value_heads = static_cast<std::int32_t>(config.at("linear_num_value_heads").integer());
    tower.head_dim    = static_cast<std::int32_t>(config.at("linear_key_head_dim").integer());
    tower.eps         = static_cast<float>(config.at("rms_norm_eps").number);
    tower.embedding   = load_matrix(file, "model.embed_tokens.weight");
    tower.hidden      = tower.embedding.k;
    tower.channels    = (2 * tower.qk_heads + tower.value_heads) * tower.head_dim;
    if (config.at("linear_conv_kernel_dim").integer() != 4 || tower.head_dim != 128 ||
        config.at("linear_value_head_dim").integer() != 128) {
        throw std::runtime_error("tower: unsupported GDN geometry");
    }
    const auto count = config.at("chip_config").at("chip_layers").integer();
    for (std::int64_t index = 0; index < count; ++index) {
        const std::string p = "model.layers." + std::to_string(index) + ".";
        Layer layer;
        layer.qkv  = load_matrix(file, p + "linear_attn.in_proj_qkv.weight");
        layer.z    = load_matrix(file, p + "linear_attn.in_proj_z.weight");
        layer.a    = load_matrix(file, p + "linear_attn.in_proj_a.weight");
        layer.b    = load_matrix(file, p + "linear_attn.in_proj_b.weight");
        layer.out  = load_matrix(file, p + "linear_attn.out_proj.weight");
        layer.gate = load_matrix(file, p + "mlp.gate_proj.weight");
        layer.up   = load_matrix(file, p + "mlp.up_proj.weight");
        layer.down = load_matrix(file, p + "mlp.down_proj.weight");
        const auto conv = file.read<std::uint16_t>(p + "linear_attn.conv1d.weight", "BF16");
        const std::size_t channels = static_cast<std::size_t>(tower.channels);
        if (conv.size() != channels * 4 || layer.qkv.n != tower.channels) {
            throw std::runtime_error("tower: unexpected conv shape");
        }
        std::vector<std::uint16_t> taps(conv.size());
        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t j = 0; j < 4; ++j) { taps[j * channels + c] = conv[c * 4 + j]; }
        }
        layer.conv       = upload(taps);
        layer.norm1      = upload(file.read<std::uint16_t>(p + "input_layernorm.weight", "BF16"));
        layer.norm2 =
            upload(file.read<std::uint16_t>(p + "post_attention_layernorm.weight", "BF16"));
        layer.gated_norm = upload(file.read<std::uint16_t>(p + "linear_attn.norm.weight", "BF16"));
        layer.a_log      = load_fp32_vector(file, p + "linear_attn.A_log");
        layer.dt_bias    = load_fp32_vector(file, p + "linear_attn.dt_bias");
        tower.layers.push_back(std::move(layer));
    }
    const std::string state = file.find_suffix("firmware.", ".state");
    const auto& shape       = file.entry(state, "F32").shape;
    if (shape != std::vector<std::int64_t>{count, tower.value_heads, 128, 128}) {
        throw std::runtime_error("tower: unexpected firmware state shape");
    }
    tower.firmware = file.read<float>(state, "F32");
    return tower;
}

struct Golden {
    std::int64_t positions = 0;
    std::int64_t inject_at = 0;
    std::vector<std::int32_t> dims;
    std::vector<std::int32_t> ids;
    std::vector<std::uint32_t> kept;
};

Golden load_golden(const std::string& dir) {
    const Json meta = JsonParser(read_text(dir + "/golden.json")).parse();
    Golden golden;
    golden.positions = meta.at("positions").integer();
    golden.inject_at = meta.at("inject_at").integer();
    for (const Json& dim : meta.at("dims").items) {
        golden.dims.push_back(static_cast<std::int32_t>(dim.integer()));
    }
    const auto n = static_cast<std::size_t>(golden.positions);
    golden.ids   = read_raw<std::int32_t>(dir + "/" + meta.at("ids").text, n);
    golden.kept  = read_raw<std::uint32_t>(dir + "/" + meta.at("kept").text, n * golden.dims.size());
    return golden;
}

std::uint16_t f32_to_f16(float value) {
    std::uint32_t u;
    std::memcpy(&u, &value, 4);
    const std::uint32_t sign = (u >> 16) & 0x8000U;
    const std::int32_t exp   = static_cast<std::int32_t>((u >> 23) & 0xffU) - 127 + 15;
    std::uint32_t frac       = u & 0x7fffffU;
    if (((u >> 23) & 0xffU) == 0xffU) {
        return static_cast<std::uint16_t>(sign | 0x7c00U | (frac ? 0x200U : 0U));
    }
    if (exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
    if (exp <= 0) {
        if (exp < -10) { return static_cast<std::uint16_t>(sign); }
        frac |= 0x800000U;
        const int shift          = 14 - exp;
        const std::uint32_t half = frac >> shift;
        const std::uint32_t rest = frac & ((1U << shift) - 1U);
        const std::uint32_t mid  = 1U << (shift - 1);
        return static_cast<std::uint16_t>(
            sign | (half + ((rest > mid || (rest == mid && (half & 1U))) ? 1U : 0U)));
    }
    const std::uint32_t half = (static_cast<std::uint32_t>(exp) << 10) | (frac >> 13);
    const std::uint32_t rest = frac & 0x1fffU;
    return static_cast<std::uint16_t>(
        sign | (half + ((rest > 0x1000U || (rest == 0x1000U && (half & 1U))) ? 1U : 0U)));
}

struct Options {
    std::string control       = "none";
    std::int64_t limit        = -1;
    std::int64_t dump_at      = -1;
    std::string dump_dir;
};

struct Outcome {
    std::int64_t positions      = 0;
    std::int64_t inexact        = 0;
    std::int64_t inexact_prompt = 0;
    std::int64_t first_bad      = -1;
    std::int32_t first_bad_dim  = -1;
    std::uint32_t got_bits      = 0;
    std::uint32_t want_bits     = 0;
    double seconds              = 0.0;
    bool ok() const { return inexact == 0; }
};

std::string json_number(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, 4);
    char text[32];
    std::snprintf(text, sizeof(text), std::isfinite(value) ? "%.9g" : "\"%g\"",
                  static_cast<double>(value));
    return text;
}

using Named = std::initializer_list<std::pair<const char*, const Tensor*>>;

void dump(const Options& options, std::size_t layer, const char* name, const Tensor& tensor) {
    std::vector<char> host(tensor.bytes());
    ninfer::test::cuda_check(
        cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost), "dump");
    char file[64];
    std::snprintf(file, sizeof(file), "/L%02zu.%s.bin", layer, name);
    std::ofstream(options.dump_dir + file, std::ios::binary)
        .write(host.data(), static_cast<std::streamsize>(host.size()));
}

void load_firmware(const Tower& tower, bool fp16, std::vector<DeviceBuffer>& states) {
    const std::size_t per_head  = 128 * 128;
    const std::size_t per_layer = per_head * static_cast<std::size_t>(tower.value_heads);
    std::vector<float> state(per_layer);
    for (std::size_t layer = 0; layer < tower.layers.size(); ++layer) {
        const float* source = tower.firmware.data() + layer * per_layer;
        for (std::size_t h = 0; h < static_cast<std::size_t>(tower.value_heads); ++h) {
            for (std::size_t key = 0; key < 128; ++key) {
                for (std::size_t value = 0; value < 128; ++value) {
                    state[h * per_head + value * 128 + key] =
                        source[h * per_head + key * 128 + value];
                }
            }
        }
        if (fp16) {
            std::vector<std::uint16_t> halves(per_layer);
            for (std::size_t i = 0; i < per_layer; ++i) { halves[i] = f32_to_f16(state[i]); }
            states[layer].copy_from_host(halves.data(), halves.size() * 2);
        } else {
            states[layer].copy_from_host(state.data(), state.size() * 4);
        }
    }
}

Outcome run(Tower& tower, const Golden& golden, const Options& options) {
    const bool fp16          = options.control == "fp16";
    const std::int32_t D     = tower.hidden;
    const std::int32_t C     = tower.channels;
    const std::int32_t Hq    = tower.qk_heads;
    const std::int32_t Hv    = tower.value_heads;
    const std::int32_t V     = Hv * 128;
    const std::int32_t inter = tower.layers.front().gate.n;
    const std::size_t layers = tower.layers.size();
    const float scale        = 1.0F / std::sqrt(128.0F);

    std::uint16_t flipped_bits   = 0;
    std::size_t flipped_offset   = 0;
    DeviceBuffer* flipped_buffer = nullptr;
    if (options.control == "weight") {
        Matrix& down = tower.layers[layers / 2].down;
        std::vector<std::uint16_t> host(static_cast<std::size_t>(down.n) * down.k);
        down.device.copy_to_host(host.data(), host.size() * 2);
        std::size_t index = 0;
        while (index < host.size() && (host[index] & 0x7fffU) == 0) { ++index; }
        if (index == host.size()) { throw std::runtime_error("control: zero down_proj"); }
        flipped_bits                = host[index];
        flipped_offset              = index * 2;
        flipped_buffer              = &down.device;
        const std::uint16_t spoiled = flipped_bits ^ 0x8000U;
        flipped_buffer->copy_from_host(&spoiled, 2, flipped_offset);
    }

    std::vector<DeviceBuffer> states, windows;
    for (std::size_t i = 0; i < layers; ++i) {
        states.emplace_back(static_cast<std::size_t>(Hv) * 128 * 128 * (fp16 ? 2 : 4));
        windows.emplace_back(static_cast<std::size_t>(C) * 3 * 2);
        states.back().fill(0);
        windows.back().fill(0);
    }
    DeviceBuffer b_id(4), b_h(D * 2), b_n(D * 2), b_qkv(C * 2), b_z(V * 2), b_a(Hv * 2),
        b_b(Hv * 2), b_conv(C * 2), b_g(Hv * 4), b_beta(Hv * 4), b_core(V * 2),
        b_gated(V * 2), b_y(D * 2), b_gate(inter * 2), b_up(inter * 2), b_act(inter * 2);
    ninfer::WorkspaceArena ws(std::max<std::size_t>(
        256, ops::gated_delta_net_workspace_capacity_bytes(Hq, Hv, true, 1, 1)));

    Tensor id(b_id.p, DType::I32, {1});
    Tensor h(b_h.p, DType::BF16, {D, 1}), n(b_n.p, DType::BF16, {D, 1});
    Tensor qkv(b_qkv.p, DType::BF16, {C, 1}), z(b_z.p, DType::BF16, {V, 1});
    Tensor a(b_a.p, DType::BF16, {Hv, 1}), b(b_b.p, DType::BF16, {Hv, 1});
    Tensor conv(b_conv.p, DType::BF16, {C, 1});
    auto* conv_bytes = static_cast<std::uint8_t*>(b_conv.p);
    Tensor q(conv_bytes, DType::BF16, {128, Hq, 1});
    Tensor k(conv_bytes + static_cast<std::size_t>(Hq) * 128 * 2, DType::BF16, {128, Hq, 1});
    Tensor v(conv_bytes + static_cast<std::size_t>(Hq) * 128 * 4, DType::BF16, {128, Hv, 1});
    Tensor g(b_g.p, DType::FP32, {Hv, 1}), beta(b_beta.p, DType::FP32, {Hv, 1});
    Tensor core(b_core.p, DType::BF16, {128, Hv, 1});
    Tensor z_heads(b_z.p, DType::BF16, {128, Hv, 1});
    Tensor gated(b_gated.p, DType::BF16, {128, Hv, 1});
    Tensor gated_flat(b_gated.p, DType::BF16, {V, 1});
    Tensor y(b_y.p, DType::BF16, {D, 1});
    Tensor gate(b_gate.p, DType::BF16, {inter, 1}), up(b_up.p, DType::BF16, {inter, 1});
    Tensor act(b_act.p, DType::BF16, {inter, 1});

    Outcome outcome;
    outcome.positions =
        options.limit >= 0 ? std::min(options.limit, golden.positions) : golden.positions;
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(D));
    const auto start = std::chrono::steady_clock::now();
    for (std::int64_t t = 0; t < outcome.positions; ++t) {
        if (t == golden.inject_at) { load_firmware(tower, fp16, states); }
        const bool dumping = t == options.dump_at;
        b_id.copy_from_host(&golden.ids[static_cast<std::size_t>(t)], 4);
        ops::embedding(id, tower.embedding.weight(), h, nullptr);
        for (std::size_t index = 0; index < layers; ++index) {
            Layer& layer = tower.layers[index];
            Tensor state(states[index].p, fp16 ? DType::FP16 : DType::FP32, {128, 128, Hv});
            Tensor window(windows[index].p, DType::BF16, {C, 3});
            Tensor norm1(layer.norm1.p, DType::BF16, {D}), norm2(layer.norm2.p, DType::BF16, {D});
            Tensor gated_weight(layer.gated_norm.p, DType::BF16, {128});
            Tensor conv_weight(layer.conv.p, DType::BF16, {C, 4});
            Tensor a_log(layer.a_log.p, DType::FP32, {Hv});
            Tensor dt_bias(layer.dt_bias.p, DType::FP32, {Hv});
            if (dumping) {
                dump(options, index, "resid_in", h);
                dump(options, index, "state_in", state);
                dump(options, index, "window_in", window);
            }
            ops::rmsnorm(h, norm1, tower.eps, true, n, nullptr);
            ops::linear(n, layer.qkv.weight(), qkv, nullptr);
            ops::linear(n, layer.z.weight(), z, nullptr);
            ops::linear(n, layer.a.weight(), a, nullptr);
            ops::linear(n, layer.b.weight(), b, nullptr);
            ops::causal_conv1d_silu(qkv, conv_weight, window, conv, nullptr);
            ops::gdn_gating(a, b, a_log, dt_bias, g, beta, nullptr);
            ops::gated_delta_net(q, k, v, g, beta, scale, true, ws, state, core, nullptr);
            ops::gated_rmsnorm(core, gated_weight, z_heads, ops::GateActivation::Silu, tower.eps,
                               gated, nullptr);
            ops::linear(gated_flat, layer.out.weight(), y, nullptr);
            if (dumping) {
                for (const auto& [name, tensor] : Named{
                         {"norm1", &n}, {"qkv", &qkv}, {"z", &z}, {"a", &a}, {"b", &b},
                         {"conv", &conv}, {"g", &g}, {"beta", &beta}, {"core", &core},
                         {"gated", &gated}, {"attn", &y}, {"state_out", &state}}) {
                    dump(options, index, name, *tensor);
                }
            }
            ops::residual_add(y, h, nullptr);
            if (dumping) { dump(options, index, "resid_mid", h); }
            ops::rmsnorm(h, norm2, tower.eps, true, n, nullptr);
            ops::linear(n, layer.gate.weight(), gate, nullptr);
            ops::linear(n, layer.up.weight(), up, nullptr);
            ops::silu_mul(gate, up, act, nullptr);
            ops::linear(act, layer.down.weight(), y, nullptr);
            if (dumping) {
                for (const auto& [name, tensor] : Named{
                         {"norm2", &n}, {"gate", &gate}, {"up", &up}, {"act", &act}, {"mlp", &y}}) {
                    dump(options, index, name, *tensor);
                }
            }
            ops::residual_add(y, h, nullptr);
            if (dumping) { dump(options, index, "resid_out", h); }
        }
        b_h.copy_to_host(residual.data(), residual.size() * 2);
        bool bad = false;
        for (std::size_t c = 0; c < golden.dims.size(); ++c) {
            const std::uint32_t got =
                static_cast<std::uint32_t>(residual[static_cast<std::size_t>(golden.dims[c])])
                << 16;
            const std::uint32_t want =
                golden.kept[static_cast<std::size_t>(t) * golden.dims.size() + c];
            if (got == want) { continue; }
            bad = true;
            if (outcome.first_bad < 0) {
                outcome.first_bad     = t;
                outcome.first_bad_dim = golden.dims[c];
                outcome.got_bits      = got;
                outcome.want_bits     = want;
            }
        }
        outcome.inexact += bad;
        outcome.inexact_prompt += bad && t < golden.inject_at;
    }
    ninfer::test::cuda_synchronize();
    outcome.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (flipped_buffer != nullptr) { flipped_buffer->copy_from_host(&flipped_bits, 2, flipped_offset); }
    return outcome;
}

void report(const Golden& golden, const Options& options, const Outcome& outcome) {
    std::printf("{\"control\": \"%s\", \"positions\": %lld, \"inject_at\": %lld, "
                "\"inexact_positions\": %lld, \"inexact_before_inject\": %lld, ",
                options.control.c_str(), static_cast<long long>(outcome.positions),
                static_cast<long long>(golden.inject_at), static_cast<long long>(outcome.inexact),
                static_cast<long long>(outcome.inexact_prompt));
    if (outcome.first_bad < 0) {
        std::printf("\"first_bad_position\": null, \"first_bad_dim\": null, \"got\": null, "
                    "\"want\": null, ");
    } else {
        std::printf("\"first_bad_position\": %lld, \"first_bad_dim\": %d, \"got\": %s, "
                    "\"want\": %s, \"got_bits\": \"0x%08x\", \"want_bits\": \"0x%08x\", ",
                    static_cast<long long>(outcome.first_bad), outcome.first_bad_dim,
                    json_number(outcome.got_bits).c_str(), json_number(outcome.want_bits).c_str(),
                    outcome.got_bits, outcome.want_bits);
    }
    std::printf("\"seconds\": %.1f, \"ok\": %s}\n", outcome.seconds,
                outcome.ok() ? "true" : "false");
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> paths;
    Options options;
    options.control = "all";
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto next            = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument("missing value"); }
            return argv[++i];
        };
        if (arg == "--control") {
            options.control = next();
        } else if (arg == "--limit") {
            options.limit = std::atoll(next().c_str());
        } else if (arg == "--dump") {
            options.dump_at  = std::atoll(next().c_str());
            options.dump_dir = next();
        } else {
            paths.emplace_back(arg);
        }
    }
    const char* tower_env  = std::getenv("NINFER_CHIP_TOWER");
    const char* golden_env = std::getenv("NINFER_CHIP_TOWER_GOLDEN");
    if (paths.empty() && tower_env != nullptr && golden_env != nullptr) {
        paths = {tower_env, golden_env};
    }
    if (paths.size() != 2) {
        std::cout << "SKIP: usage: ninfer_chip_tower_test TOWER_DIR GOLDEN_RAW_DIR "
                     "[--control all|none|fp16|weight] [--limit N] [--dump POSITION DIR]\n";
        return 77;
    }
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        Tower tower         = load_tower(paths[0]);
        const Golden golden = load_golden(paths[1]);
        const bool all      = options.control == "all";
        bool pass           = true;
        for (const char* control : {"none", "fp16", "weight"}) {
            if (!all && options.control != control) { continue; }
            Options one      = options;
            one.control      = control;
            const bool exact  = one.control == "none";
            const Outcome outcome = run(tower, golden, one);
            report(golden, one, outcome);
            pass = pass && outcome.ok() == exact;
        }
        return pass ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "chip tower test failed: " << error.what() << '\n';
        return 1;
    }
}
