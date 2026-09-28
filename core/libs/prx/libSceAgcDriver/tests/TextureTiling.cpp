#include "GraphicsTests.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;

template<typename TAction>
void reject(TAction action, std::string_view reason) {
    try {
        action();
    } catch (const std::runtime_error& error) {
        Require(std::string_view(error.what()).find(reason) != std::string_view::npos, std::string("unexpected texture tiling test error: ") + error.what());
        return;
    }
    throw std::runtime_error(std::string("expected texture tiling rejection: ") + std::string(reason));
}

}

void RunTextureTilingTests() {
    {
        const auto mips = ComputeMipLayout(TextureTileMode::kLinear, 1, 4, 4, 2);
        Require(mips.size() == 2, "linear mip chain must contain the requested mip count");
        Require(mips[0].tiledOffset == 512 && mips[0].tiledSize == 1024, "linear mip 0 offset or size changed");
        Require(mips[0].width == 4 && mips[0].height == 4, "linear mip 0 dimensions changed");
        Require(mips[0].blocksPerRow == 256 && mips[0].pitchBytes == 256, "linear mip 0 row layout changed");
        Require(!mips[0].tail, "linear mips must never fall into a mip tail");
        Require(mips[1].tiledOffset == 0 && mips[1].tiledSize == 512, "linear mip 1 offset or size changed");
        Require(mips[1].width == 2 && mips[1].height == 2, "linear mip 1 dimensions changed");
        Require(mips[1].linearOffset == mips[1].tiledOffset && mips[1].linearSize == mips[1].tiledSize, "linear tiling must keep linear and tiled layout identical");
    }
    {
        const auto mips = ComputeMipLayout(TextureTileMode::kLinear, 169, 8, 8, 1);
        Require(mips.size() == 1, "compressed linear layout must contain one mip");
        Require(mips[0].width == 2 && mips[0].height == 2, "compressed linear mip block dimensions changed");
        Require(mips[0].blocksPerRow == 32 && mips[0].pitchBytes == 256, "compressed linear mip row layout changed");
        Require(mips[0].tiledSize == 512 && mips[0].linearSize == 512, "compressed linear mip size changed");
    }
    {
        const auto mips = ComputeMipLayout(TextureTileMode::kStandard256B, 1, 64, 64, 1);
        Require(mips.size() == 1, "standard 256B layout must contain one mip");
        Require(mips[0].tiledOffset == 0 && mips[0].tiledSize == 4096, "standard 256B mip 0 offset or size changed");
        Require(mips[0].width == 64 && mips[0].height == 64, "standard 256B mip 0 dimensions changed");
        Require(mips[0].blocksPerRow == 4 && mips[0].pitchBytes == 64, "standard 256B mip 0 row layout changed");
        Require(!mips[0].tail, "standard 256B textures must never use a mip tail");

        const auto surfaceSize = ComputeSurfaceSize(mips, 3);
        Require(surfaceSize == 4096ull * 3ull, "surface size must multiply the slice size by the array layer count");
    }
    {
        const auto mips = ComputeMipLayout(TextureTileMode::kStandard256B, 1, 32, 32, 6);
        Require(mips.size() == 6, "standard 256B mip chain must contain the requested mip count");
        for (const auto& mip : mips) Require(!mip.tail, "standard 256B tile mode must never produce a mip tail");
    }
    {
        const auto mips = ComputeMipLayout(TextureTileMode::kStandard64KB, 1, 1024, 1024, 11);
        Require(mips.size() == 11, "standard 64KB mip chain must contain the requested mip count");
        auto tailSeen = false;
        for (const auto& mip : mips) {
            Require(mip.width != 0 && mip.height != 0, "every standard 64KB mip must have nonzero dimensions");
            Require(mip.tiledSize != 0 && mip.linearSize != 0, "every standard 64KB mip must have a nonzero size");
            if (mip.tail) {
                tailSeen = true;
                Require(mip.blocksPerRow == 1, "mip tail levels must report a single block per row");
                Require(mip.tiledOffset == 0, "mip tail levels must share the tiled tail block offset");
            }
        }
        Require(tailSeen, "a deep standard 64KB mip chain must fall into the mip tail");
        Require(!mips.front().tail, "the base level of a deep mip chain must not be in the mip tail");
    }
    {
        const auto mips = ComputeMipLayout(TextureTileMode::kStandard4KB, 1, 512, 512, 10);
        Require(mips.size() == 10, "standard 4KB mip chain must contain the requested mip count");
        auto tailSeen = false;
        for (const auto& mip : mips) {
            if (mip.tail) tailSeen = true;
        }
        Require(tailSeen, "a deep standard 4KB mip chain must fall into the mip tail");
    }

    {
        const auto mips = ComputeMipLayout(TextureTileMode::RenderTarget64KB, 56, 257, 129, 1);
        Require(mips[0].blocksPerRow == 3 && mips[0].tiledSize == 393216, "render target surfaces must pad to complete 128 by 128 blocks for 32-bit pixels");
        // The detiler writes whole padded rows (TextureTiling.cpp: linearSize spans pitchBytes rows).
        Require(mips[0].pitchBytes == 1536 && mips[0].linearSize == 198144, "detiled render target rows must span the padded block width");
        Require(ComputeSurfaceSize(mips, 6) == 2359296, "render target cube faces must retain the padded guest slice stride");
    }
    for (const auto format : std::array<std::uint32_t, 5>{1, 7, 56, 71, 77}) {
        const auto mips = ComputeMipLayout(TextureTileMode::RenderTarget64KB, format, 1024, 513, 11);
        // The regular levels sit at their tiled offsets (smallest level first), the tail levels after
        // the chain: the ranges must not overlap in whatever order they lie.
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        for (const auto& mip : mips) ranges.emplace_back(mip.linearOffset, mip.linearOffset + mip.linearSize);
        std::sort(ranges.begin(), ranges.end());
        for (std::size_t i = 1; i < ranges.size(); ++i) Require(ranges[i].first >= ranges[i - 1].second, "detiled mip levels must occupy separate ranges");
        bool tailSeen = false;
        for (const auto& mip : mips) {
            Require(mip.linearOffset % 4 == 0, "detiled mip levels must be word-aligned");
            Require(mip.linearSize >= static_cast<std::uint64_t>(mip.pitchBytes) * mip.height, "detiled mip allocation must contain every row");
            Require(mip.linearSize % 4 == 0, "detiled mip sizes must preserve word alignment between array layers");
            if (mip.tail) {
                tailSeen = true;
                Require(mip.tiledOffset == 0 && mip.tiledSize == 65536, "render target mip tails must share one guest 64KB block");
            }
        }
        Require(tailSeen && !mips.front().tail, "render target mip chains must cover both regular blocks and mip tails");
    }
    // SW_64KB_R_X follows the generic 64 KiB thin-block rules for every format, block-compressed and
    // wide ones included.
    static_cast<void>(ComputeMipLayout(TextureTileMode::RenderTarget64KB, 169, 64, 64, 1));

    reject([] { ComputeMipLayout(TextureTileMode::kLinear, 1, 0, 4, 1); }, "zero-sized texture");
    reject([] { ComputeMipLayout(TextureTileMode::kLinear, 1, 4, 0, 1); }, "zero-sized texture");
    reject([] { ComputeMipLayout(TextureTileMode::kLinear, 1, 4, 4, 0); }, "mip count is out of range");
    reject([] { ComputeMipLayout(TextureTileMode::kLinear, 1, 4, 4, 17); }, "mip count is out of range");

    reject([] { ComputeSurfaceSize({}, 1); }, "empty mip chain");
    reject([] { ComputeSurfaceSize(ComputeMipLayout(TextureTileMode::kLinear, 1, 4, 4, 1), 0); }, "zero array layers");
}
