#pragma once

#include "models/qwen3_5/model.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::chip {

struct TowerOptions {
    bool fp16_state = false;
};

struct Burst {
    std::vector<TokenId> stream;
    std::vector<TokenId> echo;
    std::uint32_t chip_tokens = 0;
    bool truncated            = false;
};

// One conversation's chip tower state; see docs/maintainer/chip-tower.md.
class Tower {
public:
    Tower(const Model& model, cudaStream_t stream, TowerOptions options = {});
    ~Tower();

    Tower(const Tower&)            = delete;
    Tower& operator=(const Tower&) = delete;

    [[nodiscard]] const ChipConfig& config() const noexcept { return config_; }

    void reset();
    void load_firmware();
    void step(TokenId token);
    [[nodiscard]] std::span<const std::uint16_t> residual() const noexcept { return residual_; }
    [[nodiscard]] std::optional<TokenId> owner() const;

    void begin(std::span<const TokenId> prompt);
    [[nodiscard]] std::optional<TokenId> advance(std::span<const TokenId> tokens);
    [[nodiscard]] Burst burst(TokenId first, std::uint32_t max_chip_tokens);

    [[nodiscard]] std::uint64_t steps() const noexcept { return steps_; }

private:
    struct Impl;

    ChipConfig config_;
    std::unique_ptr<Impl> impl_;
    std::vector<std::uint16_t> residual_;
    std::uint64_t steps_ = 0;
};

} // namespace ninfer::models::qwen3_5::chip
