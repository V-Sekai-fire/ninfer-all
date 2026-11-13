#include "models/qwen3_5/chip/tower.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "models/qwen3_5/chip/code.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::chip {
namespace {

constexpr std::int32_t kHeadDim = 128;
constexpr std::int32_t kWindow  = 3;

std::int32_t extent(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("chip tower dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
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

} // namespace

struct Tower::Impl {
    struct Layer {
        Tensor input_norm, post_attention_norm, convolution, norm, a_log, dt_bias;
        Weight query, key, value, z, a, b, output, gate, up, down;
        DeviceBuffer state, window;
    };

    const Model& model;
    cudaStream_t stream;
    bool fp16;
    Weight embedding{};
    std::vector<Layer> layers;
    DeviceBuffer firmware;
    std::size_t state_bytes = 0;
    WorkspaceArena workspace;
    DeviceBuffer b_id, b_h, b_n, b_qkv, b_z, b_a, b_b, b_conv, b_g, b_beta, b_core, b_gated, b_y,
        b_gate, b_up, b_act;
    std::int32_t id_host = 0;
    std::int32_t hidden = 0, channels = 0, key_heads = 0, value_heads = 0, key_width = 0,
                 value_width = 0, intermediate = 0;
    float eps = 0.0F;

    Impl(const Model& source, cudaStream_t stream_, bool fp16_, std::size_t workspace_bytes)
        : model(source), stream(stream_), fp16(fp16_), workspace(workspace_bytes) {}

    Weight matrix(WeightId id, std::int32_t rows, std::int32_t columns) const {
        Weight weight = ops::prepare_linear_weight(model.input(id)).weight;
        if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
            weight.n != rows || weight.k != columns) {
            throw std::invalid_argument(model.weight(id).name +
                                        ": the chip tower needs a dense BF16 matrix");
        }
        return weight;
    }

    Tensor vector(WeightId id, DType dtype, std::initializer_list<std::int32_t> shape) const {
        Tensor tensor = weight_tensor(model.weight(id).view, shape);
        if (tensor.dtype != dtype) {
            throw std::invalid_argument(model.weight(id).name +
                                        ": unexpected chip tower direct format");
        }
        return tensor;
    }
};

Tower::Tower(const Model& model, cudaStream_t stream, TowerOptions options) {
    if (!model.config().chip || !model.weights().chip) {
        throw std::invalid_argument("the model was loaded without a chip tower");
    }
    config_                = *model.config().chip;
    const ChipConfig& c    = config_;
    const ChipWeights& w   = *model.weights().chip;
    if (c.chip_vocab != chip_vocabulary() || c.family_bits != family_bits() ||
        c.payload_bits != 16) {
        throw std::invalid_argument("the chip tower's output code is not the rv1 layout's");
    }
    const std::int32_t key_heads   = extent(c.gdn.linear_num_key_heads);
    const std::int32_t value_heads = extent(c.gdn.linear_num_value_heads);
    impl_ = std::make_unique<Impl>(
        model, stream, options.fp16_state,
        std::max<std::size_t>(256, ops::gated_delta_net_workspace_capacity_bytes(
                                       key_heads, value_heads, true, 1, 1)));
    Impl& s        = *impl_;
    s.hidden       = extent(c.hidden_size);
    s.key_heads    = key_heads;
    s.value_heads  = value_heads;
    s.key_width    = extent(c.gdn.key_width());
    s.value_width  = extent(c.gdn.value_width());
    s.channels     = extent(c.gdn.conv_channels());
    s.intermediate = extent(c.intermediate_size);
    s.eps          = c.rms_norm_eps;

    s.embedding = native_weight(model.weight(w.token_embedding).view);
    if (s.embedding.qtype != QType::BF16 || s.embedding.layout != QuantLayout::Contiguous ||
        s.embedding.n != extent(c.vocab_size) || s.embedding.k != s.hidden) {
        throw std::invalid_argument("chip/token_embedding must be a dense BF16 table");
    }

    s.state_bytes = static_cast<std::size_t>(s.value_heads) * kHeadDim * kHeadDim *
                    (s.fp16 ? 2U : 4U);
    const std::size_t window_bytes = static_cast<std::size_t>(s.channels) * kWindow * 2U;
    s.layers.reserve(w.layers.size());
    for (const BlockWeights& block : w.layers) {
        const GdnWeights& g   = std::get<GdnWeights>(block.mixer);
        const DenseWeights& d = std::get<DenseWeights>(block.ffn);
        Impl::Layer layer;
        layer.input_norm          = s.vector(block.input_norm, DType::BF16, {s.hidden});
        layer.post_attention_norm = s.vector(block.post_attention_norm, DType::BF16, {s.hidden});
        layer.convolution = s.vector(g.convolution, DType::BF16, {s.channels, kWindow + 1});
        layer.norm        = s.vector(g.norm, DType::BF16, {kHeadDim});
        layer.a_log       = s.vector(g.a_log, DType::FP32, {s.value_heads});
        layer.dt_bias     = s.vector(g.dt_bias, DType::FP32, {s.value_heads});
        layer.query       = s.matrix(g.query, s.key_width, s.hidden);
        layer.key         = s.matrix(g.key, s.key_width, s.hidden);
        layer.value       = s.matrix(g.value, s.value_width, s.hidden);
        layer.z           = s.matrix(g.z, s.value_width, s.hidden);
        layer.a           = s.matrix(g.a_projection, s.value_heads, s.hidden);
        layer.b           = s.matrix(g.b_projection, s.value_heads, s.hidden);
        layer.output      = s.matrix(g.output, s.hidden, s.value_width);
        layer.gate        = s.matrix(d.gate, s.intermediate, s.hidden);
        layer.up          = s.matrix(d.up, s.intermediate, s.hidden);
        layer.down        = s.matrix(d.down, s.hidden, s.intermediate);
        layer.state       = DeviceBuffer(s.state_bytes);
        layer.window      = DeviceBuffer(window_bytes);
        s.layers.push_back(std::move(layer));
    }

    const Tensor stored = s.vector(
        w.firmware_state, DType::FP32,
        {kHeadDim, kHeadDim, s.value_heads, extent(s.layers.size())});
    const std::size_t per_head  = static_cast<std::size_t>(kHeadDim) * kHeadDim;
    const std::size_t per_layer = per_head * static_cast<std::size_t>(s.value_heads);
    std::vector<float> source(per_layer * s.layers.size());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(source.data(), stored.data, source.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<float> turned(source.size());
    for (std::size_t head = 0; head < source.size() / per_head; ++head) {
        const float* in = source.data() + head * per_head;
        float* out      = turned.data() + head * per_head;
        for (std::size_t key = 0; key < kHeadDim; ++key) {
            for (std::size_t value = 0; value < kHeadDim; ++value) {
                out[value * kHeadDim + key] = in[key * kHeadDim + value];
            }
        }
    }
    s.firmware = DeviceBuffer(s.state_bytes * s.layers.size());
    if (s.fp16) {
        std::vector<std::uint16_t> halves(turned.size());
        for (std::size_t i = 0; i < turned.size(); ++i) { halves[i] = f32_to_f16(turned[i]); }
        s.firmware.copy_from_host(halves.data(), halves.size() * 2);
    } else {
        s.firmware.copy_from_host(turned.data(), turned.size() * 4);
    }

    s.b_id    = DeviceBuffer(4);
    s.b_h     = DeviceBuffer(static_cast<std::size_t>(s.hidden) * 2U);
    s.b_n     = DeviceBuffer(static_cast<std::size_t>(s.hidden) * 2U);
    s.b_qkv   = DeviceBuffer(static_cast<std::size_t>(s.channels) * 2U);
    s.b_z     = DeviceBuffer(static_cast<std::size_t>(s.value_width) * 2U);
    s.b_a     = DeviceBuffer(static_cast<std::size_t>(s.value_heads) * 2U);
    s.b_b     = DeviceBuffer(static_cast<std::size_t>(s.value_heads) * 2U);
    s.b_conv  = DeviceBuffer(static_cast<std::size_t>(s.channels) * 2U);
    s.b_g     = DeviceBuffer(static_cast<std::size_t>(s.value_heads) * 4U);
    s.b_beta  = DeviceBuffer(static_cast<std::size_t>(s.value_heads) * 4U);
    s.b_core  = DeviceBuffer(static_cast<std::size_t>(s.value_width) * 2U);
    s.b_gated = DeviceBuffer(static_cast<std::size_t>(s.value_width) * 2U);
    s.b_y     = DeviceBuffer(static_cast<std::size_t>(s.hidden) * 2U);
    s.b_gate  = DeviceBuffer(static_cast<std::size_t>(s.intermediate) * 2U);
    s.b_up    = DeviceBuffer(static_cast<std::size_t>(s.intermediate) * 2U);
    s.b_act   = DeviceBuffer(static_cast<std::size_t>(s.intermediate) * 2U);
    residual_.assign(static_cast<std::size_t>(s.hidden), 0);
    reset();
}

Tower::~Tower() = default;

void Tower::reset() {
    Impl& s = *impl_;
    for (Impl::Layer& layer : s.layers) {
        CUDA_CHECK(cudaMemsetAsync(layer.state.p, 0, layer.state.bytes, s.stream));
        CUDA_CHECK(cudaMemsetAsync(layer.window.p, 0, layer.window.bytes, s.stream));
    }
    std::fill(residual_.begin(), residual_.end(), std::uint16_t{0});
}

void Tower::load_firmware() {
    Impl& s = *impl_;
    for (std::size_t index = 0; index < s.layers.size(); ++index) {
        CUDA_CHECK(cudaMemcpyAsync(s.layers[index].state.p,
                                   static_cast<const std::byte*>(s.firmware.p) +
                                       index * s.state_bytes,
                                   s.state_bytes, cudaMemcpyDeviceToDevice, s.stream));
    }
}

void Tower::step(TokenId token) {
    Impl& s = *impl_;
    if (token < 0 ||
        static_cast<std::uint64_t>(token) >=
            static_cast<std::uint64_t>(config_.chip_token_offset) + config_.chip_vocab) {
        throw std::invalid_argument("token " + std::to_string(token) +
                                    " is outside the chip tower's table");
    }
    const std::int32_t D  = s.hidden;
    const std::int32_t C  = s.channels;
    const std::int32_t Hq = s.key_heads;
    const std::int32_t Hv = s.value_heads;
    const std::int32_t K  = s.key_width;
    const std::int32_t V  = s.value_width;
    const std::int32_t M  = s.intermediate;
    const float scale     = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    cudaStream_t stream   = s.stream;

    Tensor id(s.b_id.p, DType::I32, {1});
    Tensor h(s.b_h.p, DType::BF16, {D, 1}), n(s.b_n.p, DType::BF16, {D, 1});
    std::uint8_t* qkv_bytes = static_cast<std::uint8_t*>(s.b_qkv.p);
    Tensor qkv(qkv_bytes, DType::BF16, {C, 1});
    Tensor q_raw(qkv_bytes, DType::BF16, {K, 1});
    Tensor k_raw(qkv_bytes + static_cast<std::size_t>(K) * 2, DType::BF16, {K, 1});
    Tensor v_raw(qkv_bytes + static_cast<std::size_t>(K) * 4, DType::BF16, {V, 1});
    Tensor z(s.b_z.p, DType::BF16, {V, 1});
    Tensor a(s.b_a.p, DType::BF16, {Hv, 1}), b(s.b_b.p, DType::BF16, {Hv, 1});
    std::uint8_t* conv_bytes = static_cast<std::uint8_t*>(s.b_conv.p);
    Tensor conv(conv_bytes, DType::BF16, {C, 1});
    Tensor q(conv_bytes, DType::BF16, {kHeadDim, Hq, 1});
    Tensor k(conv_bytes + static_cast<std::size_t>(K) * 2, DType::BF16, {kHeadDim, Hq, 1});
    Tensor v(conv_bytes + static_cast<std::size_t>(K) * 4, DType::BF16, {kHeadDim, Hv, 1});
    Tensor g(s.b_g.p, DType::FP32, {Hv, 1}), beta(s.b_beta.p, DType::FP32, {Hv, 1});
    Tensor core(s.b_core.p, DType::BF16, {kHeadDim, Hv, 1});
    Tensor z_heads(s.b_z.p, DType::BF16, {kHeadDim, Hv, 1});
    Tensor gated(s.b_gated.p, DType::BF16, {kHeadDim, Hv, 1});
    Tensor gated_flat(s.b_gated.p, DType::BF16, {V, 1});
    Tensor y(s.b_y.p, DType::BF16, {D, 1});
    Tensor gate(s.b_gate.p, DType::BF16, {M, 1}), up(s.b_up.p, DType::BF16, {M, 1});
    Tensor act(s.b_act.p, DType::BF16, {M, 1});

    s.id_host = token;
    CUDA_CHECK(cudaMemcpyAsync(s.b_id.p, &s.id_host, 4, cudaMemcpyHostToDevice, stream));
    ops::embedding(id, s.embedding, h, stream);
    for (Impl::Layer& layer : s.layers) {
        Tensor state(layer.state.p, s.fp16 ? DType::FP16 : DType::FP32, {kHeadDim, kHeadDim, Hv});
        Tensor window(layer.window.p, DType::BF16, {C, kWindow});
        ops::rmsnorm(h, layer.input_norm, s.eps, true, n, stream);
        ops::linear(n, layer.query, q_raw, stream);
        ops::linear(n, layer.key, k_raw, stream);
        ops::linear(n, layer.value, v_raw, stream);
        ops::linear(n, layer.z, z, stream);
        ops::linear(n, layer.a, a, stream);
        ops::linear(n, layer.b, b, stream);
        ops::causal_conv1d_silu(qkv, layer.convolution, window, conv, stream);
        ops::gdn_gating(a, b, layer.a_log, layer.dt_bias, g, beta, stream);
        ops::gated_delta_net(q, k, v, g, beta, scale, true, s.workspace, state, core, stream);
        ops::gated_rmsnorm(core, layer.norm, z_heads, ops::GateActivation::Silu, s.eps, gated,
                           stream);
        ops::linear(gated_flat, layer.output, y, stream);
        ops::residual_add(y, h, stream);
        ops::rmsnorm(h, layer.post_attention_norm, s.eps, true, n, stream);
        ops::linear(n, layer.gate, gate, stream);
        ops::linear(n, layer.up, up, stream);
        ops::silu_mul(gate, up, act, stream);
        ops::linear(act, layer.down, y, stream);
        ops::residual_add(y, h, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(residual_.data(), s.b_h.p, residual_.size() * 2,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ++steps_;
}

std::optional<TokenId> Tower::owner() const {
    const CodeLayout layout{.chip_token_offset = config_.chip_token_offset,
                            .family_bits       = config_.family_bits,
                            .payload_bits      = config_.payload_bits,
                            .code_dims         = config_.code_dims,
                            .printable         = config_.printable};
    const std::optional<std::uint32_t> token = owner_token(residual_, layout);
    if (!token) { return std::nullopt; }
    return static_cast<TokenId>(*token);
}

void Tower::begin(std::span<const TokenId> prompt) {
    reset();
    for (const TokenId token : prompt) { step(token); }
    load_firmware();
}

std::optional<TokenId> Tower::advance(std::span<const TokenId> tokens) {
    std::optional<TokenId> next;
    for (const TokenId token : tokens) {
        step(token);
        next = owner();
    }
    return next;
}

Burst Tower::burst(TokenId first, std::uint32_t max_chip_tokens) {
    Burst out;
    const TokenId offset        = static_cast<TokenId>(config_.chip_token_offset);
    std::optional<TokenId> next = first;
    while (next) {
        const TokenId token = *next;
        if (token >= offset) {
            if (out.chip_tokens == max_chip_tokens) {
                out.truncated = true;
                return out;
            }
            ++out.chip_tokens;
            out.stream.push_back(token);
            step(token);
            next = owner();
            if (!next) { throw std::runtime_error("the chip tower gave up a chip position"); }
        } else {
            out.echo.push_back(token);
            out.stream.push_back(token);
            step(token);
            next = owner();
        }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::chip
