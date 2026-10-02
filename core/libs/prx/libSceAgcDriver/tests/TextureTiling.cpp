#include "GraphicsTests.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <bit>
#include <span>
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


struct ElementAddress {
    std::uint32_t mip;
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t z;
    std::uint64_t address;
};

std::uint64_t equationOffset(const TextureSwizzleEquation& equation, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    std::uint64_t offset = 0;
    for (std::uint32_t bit = 0; bit < 16; ++bit) {
        const auto mask = equation.bits[bit];
        const auto selected = (x & (mask & 0xfffu)) ^ ((y << 12) & (mask & 0xfff000u)) ^ ((z << 24) & (mask & 0xff000000u));
        offset |= static_cast<std::uint64_t>(std::popcount(selected) & 1u) << bit;
    }
    return offset;
}

std::uint64_t detiledAddress(const SurfaceGeometry& geometry, const TextureSwizzleEquation& equation, std::array<std::uint32_t, 2> blockExtent, std::uint32_t blockBytes, const ElementAddress& element) {
    const auto& mip = geometry.mips.at(element.mip);
    const auto base = geometry.GuestLayerOffset(element.z) + mip.tiledOffset;
    if (mip.tail) return base + equationOffset(equation, element.x + mip.tailX, element.y + mip.tailY, element.z);
    const auto blockIndex = static_cast<std::uint64_t>(element.y / blockExtent[1]) * mip.blocksPerRow + element.x / blockExtent[0];
    return base + blockIndex * blockBytes + equationOffset(equation, element.x, element.y, element.z);
}

GuestTextureResource volume(TextureTileMode tileMode, std::uint32_t format, std::uint32_t width, std::uint32_t height, std::uint32_t depth, std::uint32_t mipCount) {
    GuestTextureResource resource{};
    resource.baseAddress = 0x100000000ull;
    resource.width = width;
    resource.height = height;
    resource.depthOrLastArray = depth - 1u;
    resource.mipCount = mipCount;
    resource.lastLevel = mipCount - 1u;
    resource.tileMode = tileMode;
    resource.dimension = TextureDimension::k3D;
    resource.format = format;
    return resource;
}

void requireAddresses(const GuestTextureResource& resource, std::uint32_t swizzle, std::span<const ElementAddress> expected, std::uint64_t guestBytes, const char* what) {
    const auto geometry = DescribeSurface(resource);
    Require(geometry.guestBytes == guestBytes, std::string(what) + ": guest size differs from addrlib");
    Require(geometry.mips.size() == resource.mipCount, std::string(what) + ": mip count changed");
    const auto bytesPerElement = resource.format == 71 ? 8u : 4u;
    const auto* equation = FindTextureSwizzleEquation(swizzle, bytesPerElement);
    Require(equation != nullptr, std::string(what) + ": missing swizzle equation");
    std::array<std::uint32_t, 2> extent{};
    std::uint32_t blockBytes = 0;
    if (geometry.thick) {
        const auto thick = ThickBlockExtent(resource.tileMode, bytesPerElement);
        extent = {thick[0], thick[1]};
        blockBytes = resource.tileMode == TextureTileMode::kStandard4KB ? 4096u : 65536u;
    } else {
        const auto thin = ThinBlockLayout(resource.tileMode, bytesPerElement);
        extent = {thin[1], thin[2]};
        blockBytes = thin[0];
    }
    for (const auto& element : expected) {
        Require(element.z < geometry.LevelLayers(element.mip), std::string(what) + ": sample slice lies outside its level");
        const auto address = detiledAddress(geometry, *equation, extent, blockBytes, element);
        Require(address == element.address, std::string(what) + ": mip " + std::to_string(element.mip) + " element (" + std::to_string(element.x) + ", " + std::to_string(element.y) + ", " + std::to_string(element.z) + ") detiles from " + std::to_string(address) + " instead of addrlib's " + std::to_string(element.address));
    }
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
        Require(mips[0].pitchBytes == 1536 && mips[0].linearSize == 1536u * 129u, "detiled render target rows must span the padded block width");
        Require(ComputeSurfaceSize(mips, 6) == 2359296, "render target cube faces must retain the padded guest slice stride");
    }
    for (const auto format : std::array<std::uint32_t, 5>{1, 7, 56, 71, 77}) {
        const auto mips = ComputeMipLayout(TextureTileMode::RenderTarget64KB, format, 1024, 513, 11);
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        bool tailSeen = false;
        for (const auto& mip : mips) {
            Require(mip.linearOffset % 4 == 0, "detiled mip levels must start word-aligned");
            Require(mip.linearSize >= static_cast<std::uint64_t>(mip.pitchBytes) * mip.height, "detiled mip allocation must contain every row");
            Require(mip.linearSize % 4 == 0, "detiled mip sizes must preserve word alignment between array layers");
            ranges.emplace_back(mip.linearOffset, mip.linearOffset + mip.linearSize);
            if (mip.tail) {
                tailSeen = true;
                Require(mip.tiledOffset == 0 && mip.tiledSize == 65536, "render target mip tails must share one guest 64KB block");
            }
        }
        std::sort(ranges.begin(), ranges.end());
        for (std::size_t index = 1; index < ranges.size(); ++index) Require(ranges[index].first >= ranges[index - 1].second, "detiled mip levels must occupy separate ranges");
        Require(tailSeen && !mips.front().tail, "render target mip chains must cover both regular blocks and mip tails");
    }
    const auto compressed = ComputeMipLayout(TextureTileMode::RenderTarget64KB, 169, 64, 64, 1);
    Require(compressed.size() == 1 && compressed[0].tiledSize == 65536 && compressed[0].linearSize != 0, "block compressed render target layout is wrong");
    Require(ComputeMipLayout(TextureTileMode::RenderTarget64KB, 132, 64, 64, 1).size() == 1, "format 132 render target layout is missing");
    reject([] { ComputeMipLayout(TextureTileMode::RenderTarget64KB, 74, 64, 64, 1); }, "unsupported bytes per element");

    {
        constexpr ElementAddress thick64KB32[] = {{0, 0, 0, 0, 0x20000}, {0, 63, 63, 31, 0xbfffc}, {0, 32, 21, 16, 0x94108}, {0, 21, 63, 0, 0x4cb2c}, {0, 63, 0, 31, 0x9b6d4}, {1, 0, 0, 0, 0x10000}, {1, 31, 31, 15, 0x1fffc}, {1, 16, 10, 8, 0x1a820}, {1, 10, 31, 0, 0x15968}, {1, 31, 0, 15, 0x1b6d4}, {2, 0, 0, 0, 0x8000}, {2, 15, 15, 7, 0x9ffc}, {2, 8, 5, 4, 0x9508}, {2, 5, 15, 0, 0x8b2c}, {2, 15, 0, 7, 0x96d4}, {3, 0, 0, 0, 0x4000}, {3, 7, 7, 3, 0x43fc}, {3, 4, 2, 2, 0x42a0}, {3, 2, 7, 0, 0x4168}, {3, 7, 0, 3, 0x42d4}, {4, 0, 0, 0, 0x1000}, {4, 3, 3, 1, 0x107c}, {4, 2, 1, 1, 0x1058}, {4, 1, 3, 0, 0x102c}, {4, 3, 0, 1, 0x1054}, {5, 0, 0, 0, 0xa00}, {5, 1, 1, 0, 0xa0c}, {5, 1, 0, 0, 0xa04}, {5, 0, 1, 0, 0xa08}, {5, 1, 0, 0, 0xa04}, {6, 0, 0, 0, 0x900}, {6, 0, 0, 0, 0x900}, {6, 0, 0, 0, 0x900}, {6, 0, 0, 0, 0x900}, {6, 0, 0, 0, 0x900}};
        requireAddresses(volume(TextureTileMode::kStandard64KB, 22, 64, 64, 32, 7), 0x109u, thick64KB32, 786432, "SW_64KB_S 32-bit volume");
        constexpr ElementAddress thick64KB64[] = {{0, 0, 0, 0, 0x20000}, {0, 39, 23, 19, 0xb0bf8}, {0, 20, 8, 10, 0x2e280}, {0, 13, 23, 0, 0x41b28}, {0, 39, 0, 19, 0x902d8}, {1, 0, 0, 0, 0x10000}, {1, 19, 11, 9, 0x1e178}, {1, 10, 4, 5, 0x11c50}, {1, 6, 11, 0, 0x14360}, {1, 19, 0, 9, 0x1a058}, {2, 0, 0, 0, 0x8000}, {2, 9, 5, 4, 0x9c28}, {2, 5, 2, 2, 0x8388}, {2, 3, 5, 0, 0x8868}, {2, 9, 0, 4, 0x9408}, {3, 0, 0, 0, 0x4000}, {3, 4, 2, 1, 0x4310}, {3, 2, 1, 1, 0x4070}, {3, 1, 2, 0, 0x4108}, {3, 4, 0, 1, 0x4210}, {4, 0, 0, 0, 0x1000}, {4, 1, 0, 0, 0x1008}, {4, 1, 0, 0, 0x1008}, {4, 0, 0, 0, 0x1000}, {4, 1, 0, 0, 0x1008}, {5, 0, 0, 0, 0xa00}, {5, 0, 0, 0, 0xa00}, {5, 0, 0, 0, 0xa00}, {5, 0, 0, 0, 0xa00}, {5, 0, 0, 0, 0xa00}};
        requireAddresses(volume(TextureTileMode::kStandard64KB, 71, 40, 24, 20, 6), 0x109u, thick64KB64, 786432, "SW_64KB_S 64-bit volume");
        constexpr ElementAddress thick4KB32[] = {{0, 0, 0, 0, 0x3000}, {0, 31, 31, 7, 0xaffc}, {0, 16, 10, 4, 0x5c20}, {0, 10, 31, 0, 0x8968}, {0, 31, 0, 7, 0x66d4}, {1, 0, 0, 0, 0x1000}, {1, 15, 15, 3, 0x2bfc}, {1, 8, 5, 2, 0x2188}, {1, 5, 15, 0, 0x1b2c}, {1, 15, 0, 3, 0x22d4}, {2, 0, 0, 0, 0x800}, {2, 7, 7, 1, 0xb7c}, {2, 4, 2, 1, 0xa30}, {2, 2, 7, 0, 0x968}, {2, 7, 0, 1, 0xa54}, {3, 0, 0, 0, 0x300}, {3, 3, 3, 0, 0x36c}, {3, 2, 1, 0, 0x348}, {3, 1, 3, 0, 0x32c}, {3, 3, 0, 0, 0x344}, {4, 0, 0, 0, 0x200}, {4, 1, 1, 0, 0x20c}, {4, 1, 0, 0, 0x204}, {4, 0, 1, 0, 0x208}, {4, 1, 0, 0, 0x204}, {5, 0, 0, 0, 0x100}, {5, 0, 0, 0, 0x100}, {5, 0, 0, 0, 0x100}, {5, 0, 0, 0, 0x100}, {5, 0, 0, 0, 0x100}};
        requireAddresses(volume(TextureTileMode::kStandard4KB, 22, 32, 32, 8, 6), 0x105u, thick4KB32, 45056, "SW_4KB_S 32-bit volume");
        constexpr ElementAddress thinRX64[] = {{0, 0, 0, 0, 0x10000}, {0, 127, 63, 7, 0xff6f8}, {0, 64, 21, 4, 0x9d410}, {0, 42, 63, 0, 0x17eb0}, {0, 127, 0, 7, 0xfa168}, {1, 0, 0, 0, 0x8400}, {1, 63, 31, 3, 0x6f0f8}, {1, 32, 10, 2, 0x48980}, {1, 21, 31, 0, 0xd5d8}, {1, 63, 0, 3, 0x6a368}, {2, 0, 0, 0, 0x400}, {2, 31, 15, 1, 0x23ef8}, {2, 16, 5, 1, 0x21e10}, {2, 10, 15, 0, 0x34b0}, {2, 31, 0, 1, 0x22f68}, {3, 0, 0, 0, 0x800}, {3, 15, 7, 0, 0x39f8}, {3, 8, 2, 0, 0x2980}, {3, 5, 7, 0, 0x18d8}, {3, 15, 0, 0, 0x2968}};
        requireAddresses(volume(TextureTileMode::kR64KBX, 71, 128, 64, 8, 4), 27u, thinRX64, 1048576, "SW_64KB_R_X 64-bit volume");
        const auto geometry = DescribeSurface(volume(TextureTileMode::kStandard64KB, 22, 64, 64, 32, 7));
        Require(geometry.LevelLayers(0) == 32 && geometry.LevelLayers(3) == 4 && geometry.LevelLayers(6) == 1, "volume levels must hold their own depth in slices");
        Require(geometry.mips[1].tiledOffset == 65536 && geometry.mips[0].tiledOffset == 131072 && geometry.layerBytes == 393216, "thick mip chains must stack from the tail block out to level 0 inside each slab");
    }

    reject([] { ComputeMipLayout(TextureTileMode::kLinear, 1, 0, 4, 1); }, "zero-sized texture");
    reject([] { ComputeMipLayout(TextureTileMode::kLinear, 1, 4, 0, 1); }, "zero-sized texture");
    reject([] { ComputeMipLayout(TextureTileMode::kLinear, 1, 4, 4, 0); }, "mip count is out of range");
    reject([] { ComputeMipLayout(TextureTileMode::kLinear, 1, 4, 4, 17); }, "mip count is out of range");

    reject([] { ComputeSurfaceSize({}, 1); }, "empty mip chain");
    reject([] { ComputeSurfaceSize(ComputeMipLayout(TextureTileMode::kLinear, 1, 4, 4, 1), 0); }, "zero array layers");
}
