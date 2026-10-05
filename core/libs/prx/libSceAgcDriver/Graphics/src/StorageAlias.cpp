#include "prx/libSceAgcDriver/Graphics/include/StorageAlias.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <stdexcept>

namespace AgcDriver::Graphics {

namespace {

struct BlockShape {
    std::uint32_t width;
    std::uint32_t height;
    const TextureSwizzleEquation* equation;
};

std::optional<BlockShape> blockShape(TextureTileMode tileMode, std::uint32_t elementBytes) {
    const auto mode = XorSwizzleMode(tileMode);
    if (mode == 0 || !std::has_single_bit(elementBytes) || elementBytes > 16u) return std::nullopt;
    const auto* equation = FindTextureSwizzleEquation(mode, elementBytes);
    if (equation == nullptr) return std::nullopt;
    const auto layout = ThinBlockLayout(tileMode, elementBytes);
    if (layout[0] != AliasBlockBytes || !std::has_single_bit(layout[1]) || !std::has_single_bit(layout[2])) return std::nullopt;
    return BlockShape{layout[1], layout[2], equation};
}

std::uint32_t equationOffset(const TextureSwizzleEquation& equation, std::uint32_t x, std::uint32_t y) {
    std::uint32_t offset = 0;
    for (std::uint32_t bit = 0; bit < 16u; ++bit) {
        const auto mask = equation.bits[bit];
        const auto selected = (x & (mask & 0xfffu)) ^ ((y << 12u) & (mask & 0xfff000u));
        offset |= static_cast<std::uint32_t>(std::popcount(selected) & 1) << bit;
    }
    return offset;
}

bool locate(const AliasSurface& surface, const BlockShape& shape, std::uint64_t address, std::uint32_t& x, std::uint32_t& y) {
    if (address < surface.blocksBegin || surface.blocksPerRow == 0) return false;
    const auto relative = address - surface.blocksBegin;
    if (relative % AliasBlockBytes != 0) return false;
    const auto index = relative / AliasBlockBytes;
    if (index >= surface.blockCount) return false;
    const auto column = index % surface.blocksPerRow;
    const auto row = index / surface.blocksPerRow;
    if (column * shape.width >= surface.width || row * shape.height >= surface.height) return false;
    x = static_cast<std::uint32_t>(column) * shape.width;
    y = static_cast<std::uint32_t>(row) * shape.height;
    return true;
}

bool solveLocal(const BlockShape& shape, std::uint32_t offset, std::uint32_t& x, std::uint32_t& y) {
    struct Pivot {
        std::uint32_t value = 0;
        std::uint32_t unknowns = 0;
    };
    std::array<Pivot, 16> pivots{};
    const auto xBits = static_cast<std::uint32_t>(std::countr_zero(shape.width));
    const auto yBits = static_cast<std::uint32_t>(std::countr_zero(shape.height));
    for (std::uint32_t unknown = 0; unknown < xBits + yBits; ++unknown) {
        auto value = unknown < xBits ? equationOffset(*shape.equation, 1u << unknown, 0) : equationOffset(*shape.equation, 0, 1u << (unknown - xBits));
        auto unknowns = 1u << unknown;
        while (value != 0) {
            const auto top = static_cast<std::uint32_t>(std::bit_width(value) - 1);
            if (pivots[top].value == 0) {
                pivots[top] = {value, unknowns};
                break;
            }
            value ^= pivots[top].value;
            unknowns ^= pivots[top].unknowns;
        }
        if (value == 0) return false;
    }
    std::uint32_t unknowns = 0;
    while (offset != 0) {
        const auto top = static_cast<std::uint32_t>(std::bit_width(offset) - 1);
        if (pivots[top].value == 0) return false;
        offset ^= pivots[top].value;
        unknowns ^= pivots[top].unknowns;
    }
    x = unknowns & ((1u << xBits) - 1u);
    y = unknowns >> xBits;
    return true;
}

}

bool AliasBlocksCompatible(const AliasSurface& source, const AliasSurface& destination) {
    if (source.tileMode != destination.tileMode || source.elementBytes != destination.elementBytes) return false;
    if (source.width == 0 || source.height == 0 || destination.width == 0 || destination.height == 0) return false;
    return blockShape(source.tileMode, source.elementBytes).has_value();
}

std::uint32_t AliasBlockOffset(TextureTileMode tileMode, std::uint32_t elementBytes, std::uint32_t x, std::uint32_t y) {
    const auto shape = blockShape(tileMode, elementBytes);
    if (!shape) throw std::runtime_error("AGC graphics: storage alias block offsets need a 64 KiB XOR swizzle");
    return equationOffset(*shape->equation, x, y);
}

bool AliasBlockCopies(const AliasSurface& source, const AliasSurface& destination, std::uint64_t address, std::vector<AliasCopy>& copies) {
    if (!AliasBlocksCompatible(source, destination)) return false;
    const auto shape = *blockShape(source.tileMode, source.elementBytes);
    std::uint32_t sourceX = 0, sourceY = 0, destinationX = 0, destinationY = 0;
    if (!locate(source, shape, address, sourceX, sourceY) || !locate(destination, shape, address, destinationX, destinationY)) return false;
    const auto difference = equationOffset(*shape.equation, sourceX, sourceY) ^ equationOffset(*shape.equation, destinationX, destinationY);
    std::uint32_t flipX = 0, flipY = 0;
    if (!solveLocal(shape, difference, flipX, flipY)) return false;
    const auto stepX = flipX != 0 ? flipX & (~flipX + 1u) : shape.width;
    const auto stepY = flipY != 0 ? flipY & (~flipY + 1u) : shape.height;
    if (static_cast<std::size_t>(shape.width / stepX) * (shape.height / stepY) > AliasMaxCopiesPerBlock) return false;
    const auto sourceWidth = std::min(shape.width, source.width - sourceX);
    const auto sourceHeight = std::min(shape.height, source.height - sourceY);
    const auto destinationWidth = std::min(shape.width, destination.width - destinationX);
    const auto destinationHeight = std::min(shape.height, destination.height - destinationY);
    std::vector<AliasCopy> found;
    for (std::uint32_t tileY = 0; tileY < shape.height; tileY += stepY) {
        for (std::uint32_t tileX = 0; tileX < shape.width; tileX += stepX) {
            const auto fromX = tileX ^ flipX;
            const auto fromY = tileY ^ flipY;
            const auto heldX = std::min(tileX + stepX, destinationWidth);
            const auto heldY = std::min(tileY + stepY, destinationHeight);
            const auto readX = std::min(fromX + stepX, sourceWidth);
            const auto readY = std::min(fromY + stepY, sourceHeight);
            const bool held = heldX > tileX && heldY > tileY;
            const bool read = readX > fromX && readY > fromY;
            if (held != read) return false;
            if (!held) continue;
            if (heldX - tileX != readX - fromX || heldY - tileY != readY - fromY) return false;
            found.push_back({sourceX + fromX, sourceY + fromY, destinationX + tileX, destinationY + tileY, heldX - tileX, heldY - tileY});
        }
    }
    copies.insert(copies.end(), found.begin(), found.end());
    return true;
}

}
