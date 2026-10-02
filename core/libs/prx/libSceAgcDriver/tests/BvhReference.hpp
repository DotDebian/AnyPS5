#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BVHREFERENCE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BVHREFERENCE_HPP

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>

namespace BvhReference {

using Words = std::array<std::uint32_t, 4>;
using Vec3 = std::array<float, 3>;
using BlockReader = std::function<const std::uint32_t*(std::uint64_t address)>;

constexpr std::uint32_t Infinity = 0x7f800000u;
constexpr std::uint32_t One = 0x3f800000u;
constexpr std::uint32_t NoChild = 0xffffffffu;

struct Ray {
    std::uint64_t node = 0;
    float extent = 0.0f;
    Vec3 origin{};
    Vec3 direction{};
    Vec3 inverse{};
};

struct Descriptor {
    std::uint64_t base = 0;
    std::uint64_t lastNode = 0;
    std::uint32_t grow = 0;
    bool sort = false;
    bool barycentrics = false;
    std::uint32_t kind = 8;

    [[nodiscard]] Words Encode() const {
        const auto units = base >> 8u;
        return {static_cast<std::uint32_t>(units), static_cast<std::uint32_t>((units >> 32u) & 0xffu) | ((grow & 0xffu) << 23u) | (sort ? 0x80000000u : 0u),
                static_cast<std::uint32_t>(lastNode), static_cast<std::uint32_t>((lastNode >> 32u) & 0x3ffu) | (barycentrics ? 0x01000000u : 0u) | ((kind & 0xfu) << 28u)};
    }
};

inline float Bits(std::uint32_t value) { return std::bit_cast<float>(value); }
inline std::uint32_t Bits(float value) { return std::bit_cast<std::uint32_t>(value); }

inline float Half(std::uint32_t half) {
    const std::uint32_t sign = (half & 0x8000u) << 16u;
    const std::uint32_t exponent = (half >> 10u) & 0x1fu;
    std::uint32_t mantissa = half & 0x3ffu;
    if (exponent == 0x1fu) return Bits(sign | 0x7f800000u | (mantissa << 13u));
    if (exponent != 0u) return Bits(sign | ((exponent + 112u) << 23u) | (mantissa << 13u));
    if (mantissa == 0u) return Bits(sign);
    std::uint32_t shift = 0;
    while ((mantissa & 0x400u) == 0u) {
        mantissa <<= 1u;
        ++shift;
    }
    return Bits(sign | ((113u - shift) << 23u) | ((mantissa & 0x3ffu) << 13u));
}

inline float NumberMax(float a, float b) {
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    return a < b ? b : a;
}

inline float NumberMin(float a, float b) {
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    return b < a ? b : a;
}

inline Words Triangle(const std::uint32_t* node, std::uint32_t type, const Ray& ray, bool barycentrics) {
    static constexpr std::array<std::array<std::uint32_t, 3>, 4> corners{{{0, 1, 2}, {1, 3, 2}, {2, 3, 4}, {2, 4, 0}}};
    const auto& d = ray.direction;
    const bool yMajor = std::fabs(d[0]) < std::fabs(d[1]);
    const bool zMajor = (yMajor ? std::fabs(d[1]) : std::fabs(d[0])) < std::fabs(d[2]);
    const auto rotate = [&](const Vec3& v) -> Vec3 {
        if (zMajor) return v;
        if (yMajor) return {v[2], v[0], v[1]};
        return {v[1], v[2], v[0]};
    };
    const auto origin = rotate(ray.origin);
    const auto direction = rotate(ray.direction);
    std::array<Vec3, 3> s{};
    for (std::uint32_t corner = 0; corner < 3u; ++corner) {
        const auto* p = node + corners[type][corner] * 3u;
        const auto position = rotate({Bits(p[0]), Bits(p[1]), Bits(p[2])});
        const Vec3 r{position[0] - origin[0], position[1] - origin[1], position[2] - origin[2]};
        const float x0 = r[0] * direction[2];
        const float x1 = direction[0] * r[2];
        const float y0 = r[1] * direction[2];
        const float y1 = direction[1] * r[2];
        s[corner] = {x0 - x1, y0 - y1, r[2]};
    }
    Vec3 edge{};
    Vec3 weight{};
    for (std::uint32_t i = 0; i < 3u; ++i) {
        const auto& a = s[(i + 1u) % 3u];
        const auto& b = s[(i + 2u) % 3u];
        const float left = b[0] * a[1];
        const float right = b[1] * a[0];
        edge[i] = left - right;
        weight[i] = edge[i] * direction[2];
    }
    const float n0 = edge[0] * s[0][2];
    const float n1 = edge[1] * s[1][2];
    const float n2 = edge[2] * s[2][2];
    const float n01 = n0 + n1;
    const float numerator = n01 + n2;
    const float w01 = weight[0] + weight[1];
    const float denominator = w01 + weight[2];
    const bool front = denominator > 0.0f;
    bool negative = false;
    bool positive = false;
    for (const float e : edge) {
        negative = negative || e < 0.0f;
        positive = positive || e > 0.0f;
    }
    bool missed = (negative && positive) || denominator == 0.0f || std::isnan(numerator) || (front ? numerator : -numerator) < 0.0f;
    for (std::uint32_t i = 0; i < 3u; ++i) {
        const float startY = s[(i + 1u) % 3u][1];
        const float endY = s[(i + 2u) % 3u][1];
        const bool horizontal = startY == 0.0f && endY == 0.0f;
        const bool below = startY < 0.0f || (startY == 0.0f && endY > 0.0f);
        const bool excluded = horizontal ? s[i][1] > 0.0f : below != front;
        missed = missed || (edge[i] == 0.0f && excluded);
    }
    const std::uint32_t flags = node[15];
    const std::uint32_t remap = flags >> (type * 8u);
    const auto barycentric = [&](std::uint32_t shift) {
        const auto source = (remap >> shift) & 3u;
        return Bits(source == 1u ? weight[1] : source == 2u ? weight[2] : weight[0]);
    };
    return {missed ? Infinity : Bits(numerator), missed ? One : Bits(denominator), barycentrics ? barycentric(0u) : flags + type, barycentrics ? barycentric(2u) : (missed ? 0u : 1u)};
}

inline Words Boxes(const std::uint32_t* first, const std::uint32_t* second, bool half, const Descriptor& descriptor, const Ray& ray) {
    const auto word = [&](std::uint32_t index) { return index < 16u ? first[index] : second[index - 16u]; };
    std::array<std::uint32_t, 4> children{};
    std::array<std::uint32_t, 4> keys{};
    for (std::uint32_t child = 0; child < 4u; ++child) {
        std::array<float, 6> bounds{};
        for (std::uint32_t component = 0; component < 6u; ++component) {
            if (half) {
                const auto packed = word(4u + child * 3u + component / 2u);
                bounds[component] = Half(component % 2u == 0u ? packed & 0xffffu : packed >> 16u);
            } else {
                bounds[component] = Bits(word(4u + child * 6u + component));
            }
        }
        float entry = 0.0f;
        float exit = 0.0f;
        for (std::uint32_t axis = 0; axis < 3u; ++axis) {
            const float lowOffset = bounds[axis] - ray.origin[axis];
            const float highOffset = bounds[axis + 3u] - ray.origin[axis];
            const float low = lowOffset * ray.inverse[axis];
            const float high = highOffset * ray.inverse[axis];
            const bool forward = ray.inverse[axis] >= 0.0f;
            const float near = forward ? low : high;
            const float far = forward ? high : low;
            entry = axis == 0u ? near : NumberMax(entry, near);
            exit = axis == 0u ? far : NumberMin(exit, far);
        }
        const std::uint32_t exitBits = exit == 0.0f ? 0u : Bits(exit);
        const float grown = Bits(std::min(exitBits + descriptor.grow, Infinity));
        const bool hit = exit >= 0.0f && entry < ray.extent && entry <= grown;
        children[child] = hit ? first[child] : NoChild;
        keys[child] = hit ? Bits(entry > 0.0f ? entry : 0.0f) : Infinity;
    }
    if (descriptor.sort) {
        static constexpr std::array<std::array<std::uint32_t, 2>, 5> network{{{0, 1}, {2, 3}, {0, 2}, {1, 3}, {1, 2}}};
        for (const auto& [a, b] : network) {
            if (keys[a] <= keys[b]) continue;
            std::swap(keys[a], keys[b]);
            std::swap(children[a], children[b]);
        }
    }
    return children;
}

inline Descriptor Decode(const Words& words) {
    Descriptor descriptor;
    descriptor.base = ((std::uint64_t{words[1] & 0xffu} << 32u) | words[0]) << 8u;
    descriptor.grow = (words[1] >> 23u) & 0xffu;
    descriptor.sort = (words[1] & 0x80000000u) != 0u;
    descriptor.lastNode = (std::uint64_t{words[3] & 0x3ffu} << 32u) | words[2];
    descriptor.barycentrics = (words[3] & 0x01000000u) != 0u;
    descriptor.kind = words[3] >> 28u;
    return descriptor;
}

inline Words Intersect(const Words& words, const Ray& ray, const BlockReader& read) {
    const auto descriptor = Decode(words);
    const auto type = static_cast<std::uint32_t>(ray.node & 7u);
    const auto index = ray.node >> 3u;
    const bool box32 = type == 5u;
    const auto lastIndex = index + (box32 ? 1u : 0u);
    const auto address = descriptor.base + (index << 6u);
    const auto lastBlock = descriptor.base + (lastIndex << 6u);
    const bool triangle = type < 4u;
    const Words invalid = triangle ? Words{Infinity, One, NoChild, 0u} : Words{NoChild, NoChild, NoChild, NoChild};
    bool valid = descriptor.kind == 8u && type <= 5u && lastIndex <= descriptor.lastNode && lastBlock < (std::uint64_t{1} << 48u) && !std::isnan(ray.extent);
    for (std::uint32_t axis = 0; axis < 3u; ++axis) {
        valid = valid && !std::isnan(ray.origin[axis]) && !std::isnan(ray.direction[axis]) && !std::isnan(ray.inverse[axis]) && !std::isinf(ray.origin[axis]);
    }
    if (!valid) return invalid;
    const auto* first = read(address);
    const auto* second = box32 ? read(address + 64u) : nullptr;
    if (first == nullptr || (box32 && second == nullptr)) return invalid;
    if (triangle) return Triangle(first, type, ray, descriptor.barycentrics);
    return Boxes(first, second, type == 4u, descriptor, ray);
}

}

#endif
