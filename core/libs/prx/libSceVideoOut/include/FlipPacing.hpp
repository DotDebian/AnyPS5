#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

inline std::uint64_t FlipTargetVblank(std::uint64_t lastFlipVblank, int flipRate) {
    if (flipRate < 0) throw std::invalid_argument("VideoOut: negative flip rate");
    const auto interval = static_cast<std::uint64_t>(flipRate) + 1;
    if (lastFlipVblank > std::numeric_limits<std::uint64_t>::max() - interval) throw std::overflow_error("VideoOut: flip interval overflow");
    return lastFlipVblank + interval;
}

inline std::uint64_t LostVblanks(std::uint64_t releasedVblank, std::uint64_t previousReleaseVblank, std::uint64_t readyVblank, int flipRate) {
    const auto earliest = std::max(FlipTargetVblank(previousReleaseVblank, flipRate), readyVblank);
    return releasedVblank > earliest ? releasedVblank - earliest : 0;
}

inline std::uint64_t HeldVblanks(std::uint64_t targetVblank, std::uint64_t previousReleaseVblank, std::uint64_t readyVblank, int flipRate) {
    const auto earliest = std::max(FlipTargetVblank(previousReleaseVblank, flipRate), readyVblank);
    return targetVblank > earliest ? targetVblank - earliest : 0;
}
