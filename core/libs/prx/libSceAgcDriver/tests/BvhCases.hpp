#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BVHCASES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BVHCASES_HPP

#include "BvhReference.hpp"
#include <limits>
#include <string>
#include <vector>

namespace BvhCases {

using BvhReference::Bits;
using BvhReference::Descriptor;
using BvhReference::Ray;
using BvhReference::Vec3;
using BvhReference::Words;

constexpr std::uint32_t SceneNodes = 2048;
constexpr std::uint32_t CrossingNode = 1023;
constexpr std::uint32_t Flags = 0x0e040906u;
constexpr std::array<std::uint32_t, 4> Boxes{0x1001u, 0x2002u, 0x3003u, 0x4004u};
constexpr std::array<std::uint32_t, 4> Grazing{0x5005u, 0x6006u, 0x7007u, 0x8008u};
constexpr std::uint32_t None = BvhReference::NoChild;
constexpr std::uint64_t TopBase = 0xffffffffff00ull;
constexpr std::uint64_t LargestNode = (std::uint64_t{1} << 42u) - 1u;

enum class Base { Scene, Zero, Top };

struct Case {
    std::string name;
    Descriptor descriptor;
    Ray ray;
    Words expected;
    Base base = Base::Scene;
    bool absolute = false;
    bool wideOnly = false;
};

inline std::uint32_t F(float value) { return Bits(value); }

inline void Triangles(std::uint32_t* node, const std::array<Vec3, 5>& vertices) {
    for (std::uint32_t vertex = 0; vertex < 5u; ++vertex) {
        for (std::uint32_t axis = 0; axis < 3u; ++axis) node[vertex * 3u + axis] = Bits(vertices[vertex][axis]);
    }
    node[15] = Flags;
}

inline void Box32(std::uint32_t* node, const std::array<std::uint32_t, 4>& children, const std::array<std::array<float, 6>, 4>& bounds) {
    for (std::uint32_t child = 0; child < 4u; ++child) {
        node[child] = children[child];
        for (std::uint32_t component = 0; component < 6u; ++component) node[4u + child * 6u + component] = Bits(bounds[child][component]);
    }
}

inline std::array<std::array<float, 6>, 4> ZBoxes() {
    return {{{-1, -1, 5, 1, 1, 6}, {-1, -1, 1, 1, 1, 2}, {-1, -1, 3, 1, 1, 4}, {2, -1, 7, 3, 1, 8}}};
}

inline std::vector<std::uint32_t> Scene() {
    std::vector<std::uint32_t> words(SceneNodes * 16u, 0u);
    const auto node = [&](std::uint32_t index) { return words.data() + index * 16u; };
    Triangles(node(0), {{{0, 0, 3}, {4, 0, 3}, {0, 4, 3}, {4, 4, 3}, {4, 8, 3}}});
    Triangles(node(1), {{{3, 0, 0}, {3, 4, 0}, {3, 0, 4}, {3, 4, 4}, {3, 4, 8}}});
    Triangles(node(2), {{{0, 3, 0}, {0, 3, 4}, {4, 3, 0}, {4, 3, 4}, {8, 3, 4}}});
    constexpr std::array<std::array<std::uint32_t, 3>, 4> halves{{{0xbc00bc00u, 0x3c004500u, 0x46003c00u}, {0xbc00bc00u, 0x3c003c00u, 0x40003c00u}, {0xbc00bc00u, 0x3c004200u, 0x44003c00u}, {0xbc004000u, 0x42004700u, 0x48003c00u}}};
    for (std::uint32_t child = 0; child < 4u; ++child) {
        node(3)[child] = Boxes[child];
        for (std::uint32_t word = 0; word < 3u; ++word) node(3)[4u + child * 3u + word] = halves[child][word];
    }
    Box32(node(4), Boxes, ZBoxes());
    const float above = std::nextafter(1.0f, 2.0f);
    Box32(node(6), Grazing, {{{0, 0, above, 1, 1, 2}, {-1, -1, -1, -0.0f, 1, 1}, {-1, -1, 1, 0, 1, 2}, {2, 2, 2, 3, 3, 3}}});
    Box32(node(CrossingNode), Boxes, ZBoxes());
    return words;
}

inline Ray Make(std::uint64_t node, Vec3 origin, Vec3 direction, Vec3 inverse, float extent) {
    Ray ray;
    ray.node = node;
    ray.origin = origin;
    ray.direction = direction;
    ray.inverse = inverse;
    ray.extent = extent;
    return ray;
}

inline std::vector<Case> All() {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto config = [](bool barycentrics, bool sort, std::uint32_t grow, std::uint64_t lastNode = SceneNodes - 1u) {
        Descriptor descriptor;
        descriptor.barycentrics = barycentrics;
        descriptor.sort = sort;
        descriptor.grow = grow;
        descriptor.lastNode = lastNode;
        return descriptor;
    };
    const auto plain = config(false, false, 0);
    const auto bary = config(true, false, 0);
    const auto sorted = config(false, true, 0);
    const Vec3 up{0, 0, 1};
    const Vec3 upInverse{inf, inf, 1};
    const Vec3 down{0, 0, -1};
    const Vec3 downInverse{inf, inf, -1};
    const auto z = [&](std::uint64_t node, Vec3 origin, float extent = 100.0f) { return Make(node, origin, up, upInverse, extent); };
    const auto diagonal = [&](float extent) { return Make((6u << 3u) | 5u, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, extent); };
    constexpr std::uint64_t box16 = (3u << 3u) | 4u;
    constexpr std::uint64_t box32 = (4u << 3u) | 5u;
    constexpr std::uint64_t crossing = (CrossingNode << 3u) | 5u;
    const Words miss{BvhReference::Infinity, BvhReference::One, None, 0u};
    const Words noBoxes{None, None, None, None};
    const Words forward{Boxes[0], Boxes[1], Boxes[2], None};
    const Words nearFirst{Boxes[1], Boxes[2], Boxes[0], None};
    std::vector<Case> cases{
        {"type 0 hit", plain, z(0, {1, 0.5f, 0}), {F(-48), F(-16), Flags, 1}},
        {"type 0 barycentrics", bary, z(0, {1, 0.5f, 0}), {F(-48), F(-16), F(-2), F(-4)}},
        {"type 1 hit", plain, z(1, {3.5f, 2, 0}), {F(-48), F(-16), Flags + 1u, 1}},
        {"type 1 barycentrics", bary, z(1, {3.5f, 2, 0}), {F(-48), F(-16), F(-6), F(-2)}},
        {"type 2 hit", plain, z(2, {3, 5, 0}), {F(-48), F(-16), Flags + 2u, 1}},
        {"type 2 barycentrics", bary, z(2, {3, 5, 0}), {F(-48), F(-16), F(-4), F(-8)}},
        {"type 3 hit", plain, z(3, {1, 3, 0}), {F(48), F(16), Flags + 3u, 1}},
        {"type 3 barycentrics", bary, z(3, {1, 3, 0}), {F(48), F(16), F(8), F(4)}},
        {"type 0 outside", plain, z(0, {5, 5, 0}), {BvhReference::Infinity, BvhReference::One, Flags, 0}},
        {"type 0 behind the origin", plain, z(0, {1, 0.5f, 5}), {BvhReference::Infinity, BvhReference::One, Flags, 0}},
        {"type 0 behind, barycentrics kept", bary, z(0, {1, 0.5f, 5}), {BvhReference::Infinity, BvhReference::One, F(-2), F(-4)}},
        {"negative z direction", plain, Make(0, {1, 0.5f, 6}, down, downInverse, 100), {F(48), F(16), Flags, 1}},
        {"negative z direction barycentrics", bary, Make(0, {1, 0.5f, 6}, down, downInverse, 100), {F(48), F(16), F(2), F(4)}},
        {"x-dominant direction", bary, Make(8, {0, 1, 0.5f}, {2, 0, 0}, {0.5f, inf, inf}, 100), {F(-192), F(-128), F(-16), F(-32)}},
        {"y-dominant direction", bary, Make(16, {0.5f, 0, 1}, {0, 2, 0}, {inf, 0.5f, inf}, 100), {F(-192), F(-128), F(-16), F(-32)}},
        {"shared diagonal edge, type 0 excluded", plain, z(0, {2, 2, 0}), {BvhReference::Infinity, BvhReference::One, Flags, 0}},
        {"shared diagonal edge, type 1 included", plain, z(1, {2, 2, 0}), {F(-48), F(-16), Flags + 1u, 1}},
        {"shared horizontal edge, type 1 included", plain, z(1, {2, 4, 0}), {F(-48), F(-16), Flags + 1u, 1}},
        {"shared horizontal edge, type 2 excluded", plain, z(2, {2, 4, 0}), {BvhReference::Infinity, BvhReference::One, Flags + 2u, 0}},
        {"bottom edge with the triangle above excluded", plain, z(0, {2, 0, 0}), {BvhReference::Infinity, BvhReference::One, Flags, 0}},
        {"box16 in node order", plain, z(box16, {0, 0, 0}, 10), forward},
        {"box32 in node order", plain, z(box32, {0, 0, 0}, 10), forward},
        {"box32 across two ranges in node order", plain, z(crossing, {0, 0, 0}, 10), forward},
        {"box16 sorted", sorted, z(box16, {0, 0, 0}, 10), nearFirst},
        {"box32 sorted", sorted, z(box32, {0, 0, 0}, 10), nearFirst},
        {"box32 across two ranges sorted", sorted, z(crossing, {0, 0, 0}, 10), nearFirst},
        {"box16 face at the extent excluded", plain, z(box16, {0, 0, 0}, 3), {None, Boxes[1], None, None}},
        {"box32 origin inside a child", sorted, z(box32, {0, 0, 1.5f}, 10), nearFirst},
        {"box16 negative direction sorted", sorted, Make(box16, {0, 0, 9}, down, downInverse, 10), {Boxes[0], Boxes[2], Boxes[1], None}},
        {"box32 behind the origin", plain, z(box32, {0, 0, 20}, 10), noBoxes},
        {"grazing box without growth", plain, diagonal(10), {None, Grazing[1], None, Grazing[3]}},
        {"grazing box grown one ulp, sorted", config(false, true, 1), diagonal(10), {Grazing[1], Grazing[0], Grazing[3], None}},
        {"grazing box grown 255 ulps", config(false, false, 255), diagonal(10), {Grazing[0], Grazing[1], None, Grazing[3]}},
        {"box entry equal to the extent excluded", plain, diagonal(2), {None, Grazing[1], None, None}},
        {"descriptor type not 8, triangle", [&] { auto d = plain; d.kind = 0; return d; }(), z(0, {1, 0.5f, 0}), miss},
        {"descriptor type not 8, box", [&] { auto d = plain; d.kind = 0; return d; }(), z(box16, {0, 0, 0}, 10), noBoxes},
        {"node type 6", plain, z((3u << 3u) | 6u, {0, 0, 0}, 10), noBoxes},
        {"node type 7", plain, z((3u << 3u) | 7u, {0, 0, 0}, 10), noBoxes},
        {"NaN origin, triangle", bary, z(0, {nan, 0.5f, 0}), miss},
        {"infinite origin, box", plain, z(box16, {-inf, 0, 0}, 10), noBoxes},
        {"NaN extent, triangle", plain, z(0, {1, 0.5f, 0}, nan), miss},
        {"NaN inverse direction, triangle", plain, Make(0, {1, 0.5f, 0}, up, {nan, inf, 1}, 100), miss},
        {"NaN direction, box", plain, Make(box16, {0, 0, 0}, {0, nan, 1}, upInverse, 10), noBoxes},
        {"triangle node at the last node", config(false, false, 0, 0), z(0, {1, 0.5f, 0}), {F(-48), F(-16), Flags, 1}},
        {"triangle node past the last node", config(false, false, 0, 0), z(8, {1, 0.5f, 0}), miss},
        {"box32 second half past the last node", config(false, true, 0, 4), z(box32, {0, 0, 0}, 10), noBoxes},
        {"box32 ending at the last node", config(false, true, 0, 5), z(box32, {0, 0, 0}, 10), nearFirst},
        {"node at 2^48", config(false, false, 0, LargestNode), z(4u << 3u, {1, 0.5f, 0}), miss, Base::Top},
        {"box32 second half at 2^48", config(false, false, 0, LargestNode), z((3u << 3u) | 5u, {0, 0, 0}, 10), noBoxes, Base::Top},
        {"absolute 64-bit triangle node", config(false, false, 0, LargestNode), z(0, {1, 0.5f, 0}), {F(-48), F(-16), Flags, 1}, Base::Zero, true, true},
        {"absolute 64-bit box32 across two ranges", config(false, true, 0, LargestNode), z(crossing, {0, 0, 0}, 10), nearFirst, Base::Zero, true, true},
        {"64-bit node far past the last node", plain, z(std::uint64_t{1} << 60u, {1, 0.5f, 0}), miss, Base::Scene, false, true},
    };
    return cases;
}

inline Words EncodeDescriptor(const Case& item, std::uint64_t scene) {
    auto descriptor = item.descriptor;
    descriptor.base = item.base == Base::Scene ? scene : item.base == Base::Top ? TopBase : 0u;
    return descriptor.Encode();
}

inline std::uint64_t NodePointer(const Case& item, std::uint64_t scene) {
    return item.absolute ? item.ray.node + (scene >> 3u) : item.ray.node;
}

}

#endif
