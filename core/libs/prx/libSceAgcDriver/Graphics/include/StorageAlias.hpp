#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_STORAGEALIAS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_STORAGEALIAS_HPP

#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include <cstdint>
#include <vector>

namespace AgcDriver::Graphics {

struct AliasSurface {
    TextureTileMode tileMode;
    std::uint32_t elementBytes;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t blocksPerRow;
    std::uint64_t blocksBegin;
    std::uint64_t blockCount;
    bool operator==(const AliasSurface&) const = default;
};

struct AliasCopy {
    std::uint32_t sourceX;
    std::uint32_t sourceY;
    std::uint32_t destinationX;
    std::uint32_t destinationY;
    std::uint32_t width;
    std::uint32_t height;
};

constexpr std::uint64_t AliasBlockBytes = 65536;
constexpr std::size_t AliasMaxCopiesPerBlock = 64;

bool AliasBlocksCompatible(const AliasSurface& source, const AliasSurface& destination);
std::uint32_t AliasBlockOffset(TextureTileMode tileMode, std::uint32_t elementBytes, std::uint32_t x, std::uint32_t y);
bool AliasBlockCopies(const AliasSurface& source, const AliasSurface& destination, std::uint64_t address, std::vector<AliasCopy>& copies);

}

#endif
