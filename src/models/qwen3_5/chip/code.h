#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::chip {

// The rv1-tower output code; see docs/maintainer/chip-tower.md.
inline constexpr std::array<std::uint32_t, 12> kFamilySizes{
    1U << 15, 1U << 15, 1U << 15, 4, 1U << 16, 1U << 16, 512, 8, 16, 16, 128, 256,
};
inline constexpr std::uint32_t kControls = 8;
inline constexpr std::uint32_t kEchoFamily =
    static_cast<std::uint32_t>(kFamilySizes.size()) + kControls;

[[nodiscard]] constexpr std::uint32_t control_base() noexcept {
    std::uint32_t base = 0;
    for (const std::uint32_t size : kFamilySizes) { base += size; }
    return base;
}

[[nodiscard]] constexpr std::uint32_t chip_vocabulary() noexcept {
    return control_base() + kControls;
}

[[nodiscard]] constexpr std::uint32_t family_bits() noexcept {
    std::uint32_t bits = 0;
    while ((kEchoFamily >> bits) != 0) { ++bits; }
    return bits;
}

struct CodeLayout {
    std::uint32_t chip_token_offset = 0;
    std::uint32_t family_bits       = 0;
    std::uint32_t payload_bits      = 0;
    std::span<const std::uint32_t> code_dims;
    std::span<const std::uint32_t> printable;
};

[[noreturn]] inline void no_token(std::uint32_t family, std::uint32_t payload) {
    throw std::runtime_error("the chip tower names no token (family " + std::to_string(family) +
                             ", payload " + std::to_string(payload) + ")");
}

[[nodiscard]] inline std::optional<std::uint32_t>
owner_token(std::span<const std::uint16_t> residual, const CodeLayout& layout) {
    bool owned            = false;
    std::uint32_t family  = 0;
    std::uint32_t payload = 0;
    for (std::size_t bit = 0; bit < layout.code_dims.size(); ++bit) {
        const std::uint16_t word = residual[layout.code_dims[bit]];
        const bool nonzero       = (word & 0x7fffU) != 0;
        const bool nan           = (word & 0x7f80U) == 0x7f80U && (word & 0x007fU) != 0;
        owned                    = owned || nonzero;
        if (!nonzero || nan || (word & 0x8000U) != 0) { continue; }
        if (bit < layout.family_bits) {
            family |= 1U << bit;
        } else {
            payload |= 1U << (bit - layout.family_bits);
        }
    }
    if (!owned) { return std::nullopt; }
    if (family == kEchoFamily) {
        if (payload >= layout.printable.size()) { no_token(family, payload); }
        return layout.printable[payload];
    }
    if (family > kEchoFamily) { no_token(family, payload); }
    if (family >= kFamilySizes.size()) {
        return layout.chip_token_offset + control_base() +
               (family - static_cast<std::uint32_t>(kFamilySizes.size()));
    }
    if (payload >= kFamilySizes[family]) { no_token(family, payload); }
    std::uint32_t base = 0;
    for (std::uint32_t index = 0; index < family; ++index) { base += kFamilySizes[index]; }
    return layout.chip_token_offset + base + payload;
}

} // namespace ninfer::models::qwen3_5::chip
