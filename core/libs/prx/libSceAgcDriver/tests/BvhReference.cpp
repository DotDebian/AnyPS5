#include "BvhCases.hpp"
#include <cinttypes>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace {

constexpr std::uint64_t SceneBase = 0x4000123400ull;

std::string Hex(const BvhReference::Words& words) {
    char text[64];
    std::snprintf(text, sizeof(text), "%08x %08x %08x %08x", words[0], words[1], words[2], words[3]);
    return text;
}

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void HandCases() {
    const auto scene = BvhCases::Scene();
    const BvhReference::BlockReader read = [&](std::uint64_t address) -> const std::uint32_t* {
        if (address < SceneBase || address - SceneBase >= scene.size() * 4u || address % 64u != 0u) return nullptr;
        return scene.data() + (address - SceneBase) / 4u;
    };
    for (const auto& item : BvhCases::All()) {
        auto ray = item.ray;
        ray.node = BvhCases::NodePointer(item, SceneBase);
        const auto actual = BvhReference::Intersect(BvhCases::EncodeDescriptor(item, SceneBase), ray, read);
        Require(actual == item.expected, item.name + ": " + Hex(actual) + ", expected " + Hex(item.expected));
    }
}

std::int64_t Cross(const std::array<std::int64_t, 2>& a, const std::array<std::int64_t, 2>& b, const std::array<std::int64_t, 2>& c) {
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

void SharedEdges() {
    std::mt19937 random(1234u);
    std::uniform_int_distribution<int> coordinate(-6, 6);
    std::array<std::uint32_t, 16> node{};
    const BvhReference::BlockReader read = [&](std::uint64_t address) -> const std::uint32_t* { return address == SceneBase ? node.data() : nullptr; };
    BvhReference::Descriptor descriptor;
    descriptor.base = SceneBase;
    const auto words = descriptor.Encode();
    std::uint64_t checked = 0;
    std::uint64_t onDiagonal = 0;
    for (std::uint32_t quad = 0; quad < 4000u; ++quad) {
        std::array<std::array<std::int64_t, 2>, 4> v{};
        for (auto& vertex : v) vertex = {coordinate(random), coordinate(random)};
        const std::array<std::array<std::int64_t, 2>, 4> ring{v[0], v[1], v[3], v[2]};
        std::int64_t sign = 0;
        bool convex = true;
        for (std::uint32_t i = 0; i < 4u; ++i) {
            const auto turn = Cross(ring[i], ring[(i + 1u) % 4u], ring[(i + 2u) % 4u]);
            if (turn == 0 || (sign != 0 && (turn > 0) != (sign > 0))) convex = false;
            sign = turn;
        }
        if (!convex) continue;
        const float depth = static_cast<float>(coordinate(random)) + 7.5f;
        for (std::uint32_t vertex = 0; vertex < 4u; ++vertex) {
            node[vertex * 3u] = BvhReference::Bits(static_cast<float>(v[vertex][0]));
            node[vertex * 3u + 1u] = BvhReference::Bits(static_cast<float>(v[vertex][1]));
            node[vertex * 3u + 2u] = BvhReference::Bits(depth);
        }
        for (std::int64_t x = -6; x <= 6; ++x) {
            for (std::int64_t y = -6; y <= 6; ++y) {
                const std::array<std::int64_t, 2> point{x, y};
                bool inside = true;
                for (std::uint32_t i = 0; i < 4u; ++i) inside = inside && (Cross(ring[i], ring[(i + 1u) % 4u], point) > 0) == (sign > 0) && Cross(ring[i], ring[(i + 1u) % 4u], point) != 0;
                if (!inside) continue;
                const float direction = (x + y) % 2 == 0 ? 1.0f : -1.0f;
                std::uint32_t hits = 0;
                for (std::uint32_t type = 0; type < 2u; ++type) {
                    BvhReference::Ray ray;
                    ray.node = type;
                    ray.extent = 1000.0f;
                    ray.origin = {static_cast<float>(x), static_cast<float>(y), direction > 0.0f ? -20.0f : 20.0f};
                    ray.direction = {0.0f, 0.0f, direction};
                    ray.inverse = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), direction};
                    const auto result = BvhReference::Intersect(words, ray, read);
                    if (result[3] == 1u) {
                        ++hits;
                        Require(BvhReference::Bits(result[0]) / BvhReference::Bits(result[1]) == std::fabs(depth - ray.origin[2]), "a shared-edge hit reported the wrong distance");
                    }
                }
                Require(hits == 1u, "a ray through the inside of a quad hit " + std::to_string(hits) + " of its two triangles");
                ++checked;
                if (Cross(v[1], v[2], point) == 0) ++onDiagonal;
            }
        }
    }
    Require(checked > 10000u && onDiagonal > 500u, "the shared-edge sweep checked too few rays");
    std::printf("shared edges: %" PRIu64 " rays, %" PRIu64 " on the shared edge\n", checked, onDiagonal);
}

}

int main() {
    try {
        HandCases();
        SharedEdges();
        std::puts("bvh reference tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
