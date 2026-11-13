#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

ChipWeights bind_chip(Bindings& b, const ChipConfig& config) {
    const TextConfig block = config.block_config();
    ChipWeights out;
    out.token_embedding =
        b.direct("chip/token_embedding", {config.vocab_size, config.hidden_size});
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        out.layers.push_back(bind_block(b, block, "chip/layers/" + std::to_string(i) + "/",
                                        MixerKind::LinearAttention));
    }
    out.firmware_state = b.direct("chip/firmware/" + config.firmware + "/state",
                                  {config.num_hidden_layers, config.gdn.linear_num_value_heads,
                                   config.gdn.linear_key_head_dim,
                                   config.gdn.linear_value_head_dim},
                                  QType::FP32);
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
