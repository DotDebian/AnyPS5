#include "GraphicsTests.hpp"
#include "prx/libSceAgcDriver/Graphics/include/StorageAlias.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureSwizzleEquations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;

std::uint32_t referenceOffset(const TextureSwizzleEquation& equation, std::uint32_t x, std::uint32_t y) {
    std::uint32_t offset = 0;
    for (std::uint32_t bit = 0; bit < 16; ++bit) {
        const auto mask = equation.bits[bit];
        const auto selected = (x & (mask & 0xfffu)) ^ ((y << 12) & (mask & 0xfff000u));
        offset |= static_cast<std::uint32_t>(std::popcount(selected) & 1) << bit;
    }
    return offset;
}

AliasSurface makeSurface(TextureTileMode mode, std::uint32_t elementBytes, std::uint32_t width, std::uint32_t height, std::uint64_t base) {
    const auto mips = ComputeElementMipLayout(mode, elementBytes, width, height, 1);
    Require(mips.size() == 1 && !mips[0].tail && mips[0].tiledSize % AliasBlockBytes == 0, "storage alias test surface is not whole 64 KiB blocks");
    return {mode, elementBytes, width, height, mips[0].blocksPerRow, base + mips[0].tiledOffset, mips[0].tiledSize / AliasBlockBytes};
}

constexpr std::uint32_t None = ~0u;

struct BlockElements {
    std::vector<std::uint32_t> x;
    std::vector<std::uint32_t> y;
    std::size_t count = 0;
};

BlockElements blockElements(const AliasSurface& surface, std::uint64_t address) {
    const auto& equation = *FindTextureSwizzleEquation(XorSwizzleMode(surface.tileMode), surface.elementBytes);
    const auto block = ThinBlockLayout(surface.tileMode, surface.elementBytes);
    BlockElements elements;
    const auto slots = static_cast<std::size_t>(AliasBlockBytes / surface.elementBytes);
    elements.x.assign(slots, None);
    elements.y.assign(slots, None);
    if (address < surface.blocksBegin || address >= surface.blocksBegin + surface.blockCount * AliasBlockBytes) return elements;
    const auto index = (address - surface.blocksBegin) / AliasBlockBytes;
    const auto originX = static_cast<std::uint32_t>(index % surface.blocksPerRow) * block[1];
    const auto originY = static_cast<std::uint32_t>(index / surface.blocksPerRow) * block[2];
    for (std::uint32_t y = originY; y < std::min(originY + block[2], surface.height); ++y) {
        for (std::uint32_t x = originX; x < std::min(originX + block[1], surface.width); ++x) {
            const auto offset = referenceOffset(equation, x, y);
            Require(offset % surface.elementBytes == 0, "storage alias test element is not element aligned");
            const auto slot = offset / surface.elementBytes;
            Require(elements.x[slot] == None, "storage alias test equation maps two elements to one offset");
            elements.x[slot] = x;
            elements.y[slot] = y;
            ++elements.count;
        }
    }
    return elements;
}

struct PairResult {
    std::size_t shared = 0;
    std::size_t handed = 0;
};

PairResult checkPair(const AliasSurface& source, const AliasSurface& destination, const std::string& what) {
    PairResult result;
    const auto first = std::max(source.blocksBegin, destination.blocksBegin) - AliasBlockBytes;
    const auto last = std::min(source.blocksBegin + source.blockCount * AliasBlockBytes, destination.blocksBegin + destination.blockCount * AliasBlockBytes) + AliasBlockBytes;
    for (auto address = first; address < last; address += AliasBlockBytes) {
        const auto from = blockElements(source, address);
        const auto to = blockElements(destination, address);
        const bool shared = from.count != 0 && to.count != 0;
        bool same = shared;
        for (std::size_t slot = 0; same && slot < from.x.size(); ++slot) same = (from.x[slot] == None) == (to.x[slot] == None);
        std::vector<AliasCopy> copies{{1, 2, 3, 4, 5, 6}};
        const bool handed = AliasBlockCopies(source, destination, address, copies);
        const auto where = what + " block +" + std::to_string((address - destination.blocksBegin) / AliasBlockBytes);
        Require(handed == same, where + ": the hand-over decision differs from the element sets (" + (same ? "same" : "different") + ")");
        Require(copies.front().sourceX == 1 && copies.front().height == 6, where + ": the copies already listed were changed");
        if (shared) ++result.shared;
        if (!handed) {
            Require(copies.size() == 1, where + ": a refused block listed copies");
            continue;
        }
        ++result.handed;
        Require(copies.size() > 1 && copies.size() - 1 <= AliasMaxCopiesPerBlock, where + ": a handed block lists no copy or too many");
        std::vector<std::uint8_t> covered(from.x.size(), 0);
        const auto lookup = [&](const BlockElements& elements, std::uint32_t x, std::uint32_t y) {
            for (std::size_t slot = 0; slot < elements.x.size(); ++slot) {
                if (elements.x[slot] == x && elements.y[slot] == y) return static_cast<std::uint32_t>(slot);
            }
            return None;
        };
        std::size_t moved = 0;
        const auto* equation = FindTextureSwizzleEquation(XorSwizzleMode(destination.tileMode), destination.elementBytes);
        for (std::size_t c = 1; c < copies.size(); ++c) {
            const auto& copy = copies[c];
            Require(copy.width != 0 && copy.height != 0, where + ": an empty copy");
            const auto firstTo = lookup(to, copy.destinationX, copy.destinationY);
            const auto firstFrom = lookup(from, copy.sourceX, copy.sourceY);
            Require(firstTo != None && firstFrom != None, where + ": a copy starts outside the block's elements");
            for (std::uint32_t dy = 0; dy < copy.height; ++dy) {
                for (std::uint32_t dx = 0; dx < copy.width; ++dx) {
                    const auto toX = copy.destinationX + dx, toY = copy.destinationY + dy;
                    const auto fromX = copy.sourceX + dx, fromY = copy.sourceY + dy;
                    Require(toX < destination.width && toY < destination.height && fromX < source.width && fromY < source.height, where + ": a copy leaves an image");
                    const auto toOffset = referenceOffset(*equation, toX, toY) / destination.elementBytes;
                    const auto fromOffset = referenceOffset(*equation, fromX, fromY) / source.elementBytes;
                    Require(to.x[toOffset] == toX && to.y[toOffset] == toY, where + ": a copy writes an element of another block");
                    Require(from.x[fromOffset] == fromX && from.y[fromOffset] == fromY, where + ": a copy reads an element of another block");
                    Require(toOffset == fromOffset, where + ": a copy moves an element to other guest bytes");
                    Require(covered[toOffset] == 0, where + ": two copies write one element");
                    covered[toOffset] = 1;
                    ++moved;
                }
            }
        }
        Require(moved == to.count && moved == from.count, where + ": the copies do not move every element either image holds");
    }
    return result;
}

}

void RunStorageAliasTests() {
    constexpr std::uint64_t base = 0x53aa00000ull;
    constexpr auto renderTarget = TextureTileMode::kR64KBX;
    {
        const auto target = makeSurface(renderTarget, 8, 1920, 1080, base);
        const auto wide = makeSurface(renderTarget, 8, 2432, 1368, base);
        Require(target.blockCount == 255 && wide.blockCount == 418, "the title's transient surfaces have unexpected block counts");
        const auto forward = checkPair(target, wide, "1920x1080 -> 2432x1368 (8 bytes)");
        const auto back = checkPair(wide, target, "2432x1368 -> 1920x1080 (8 bytes)");
        Require(forward.shared == 255 && forward.handed == 240 && back.handed == 240, "the title's 8-byte hand-over does not take every whole block: " + std::to_string(forward.handed) + "/" + std::to_string(back.handed));
        std::vector<AliasCopy> copies;
        Require(AliasBlockCopies(target, wide, base + 15 * AliasBlockBytes, copies) && copies.size() == 4, "a block changing row parity is not handed over in four column groups");
        Require(copies[0].sourceX == 32 && copies[0].sourceY == 64 && copies[0].destinationX == 15 * 128 && copies[0].destinationY == 0 && copies[0].width == 32 && copies[0].height == 64, "the column groups of a row parity change are misplaced");
        copies.clear();
        Require(AliasBlockCopies(target, target, base + 17 * AliasBlockBytes, copies) && copies.size() == 1 && copies[0].sourceX == copies[0].destinationX && copies[0].width == 128 && copies[0].height == 64, "an identical surface is not handed over block for block");
        copies.clear();
        Require(!AliasBlockCopies(target, wide, base + 255 * AliasBlockBytes, copies) && copies.empty(), "a block outside the source was handed over");
        Require(!AliasBlockCopies(target, wide, base + 1, copies) && !AliasBlockCopies(target, wide, base - AliasBlockBytes, copies), "an unaligned or outside address was handed over");
        const auto shifted = makeSurface(renderTarget, 8, 1920, 1080, base + 0x300000);
        checkPair(wide, shifted, "2432x1368 -> 1920x1080 at +48 blocks (8 bytes)");
        checkPair(shifted, wide, "1920x1080 at +48 blocks -> 2432x1368 (8 bytes)");
        const auto half = makeSurface(renderTarget, 8, 960, 540, base);
        checkPair(wide, half, "2432x1368 -> 960x540 (8 bytes)");
        checkPair(half, target, "960x540 -> 1920x1080 (8 bytes)");
    }
    {
        const auto packed = makeSurface(renderTarget, 4, 1920, 1080, base);
        const auto wide = makeSurface(renderTarget, 4, 2432, 1368, base + 0x10000);
        checkPair(packed, wide, "1920x1080 -> 2432x1368 at +1 block (4 bytes)");
        checkPair(wide, packed, "2432x1368 at +1 block -> 1920x1080 (4 bytes)");
        const auto eight = makeSurface(renderTarget, 8, 1920, 1080, base);
        std::vector<AliasCopy> copies;
        Require(!AliasBlocksCompatible(eight, packed) && !AliasBlockCopies(eight, packed, base, copies) && copies.empty(), "surfaces of other element sizes were handed over by copies");
    }
    for (const auto mode : {TextureTileMode::kZ64KBX, TextureTileMode::kS64KBX, TextureTileMode::kD64KBX, TextureTileMode::kR64KBX}) {
        for (const std::uint32_t elementBytes : {1u, 2u, 4u, 8u, 16u}) {
            const auto name = "mode " + std::to_string(static_cast<int>(mode)) + " at " + std::to_string(elementBytes) + " bytes";
            const auto a = makeSurface(mode, elementBytes, 700, 300, base);
            const auto b = makeSurface(mode, elementBytes, 1100, 260, base);
            const auto c = makeSurface(mode, elementBytes, 520, 410, base + 2 * AliasBlockBytes);
            checkPair(a, b, name + " a -> b");
            checkPair(b, a, name + " b -> a");
            checkPair(c, b, name + " c -> b");
            checkPair(a, c, name + " a -> c");
            const auto same = checkPair(a, a, name + " a -> a");
            Require(same.handed == same.shared && same.shared == a.blockCount, name + ": an identical surface is not handed over whole");
        }
    }
    {
        const auto standard = AliasSurface{TextureTileMode::kStandard64KB, 8, 256, 128, 2, base, 2};
        std::vector<AliasCopy> copies;
        Require(!AliasBlocksCompatible(standard, standard) && !AliasBlockCopies(standard, standard, base, copies), "a surface without an XOR equation was handed over");
        auto other = makeSurface(renderTarget, 8, 256, 128, base);
        auto moved = other;
        moved.tileMode = TextureTileMode::kD64KBX;
        Require(!AliasBlocksCompatible(other, moved), "surfaces of other tile modes were handed over");
    }
}
