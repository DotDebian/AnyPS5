#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/tests/BvhCases.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using BvhReference::Words;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t LaneDwords = 16;
constexpr std::uint32_t MemoryNodes = 4096;
constexpr std::size_t MemoryBytes = MemoryNodes * 64u;
constexpr std::size_t SplitBytes = 65536;
constexpr std::uint32_t Sentinel = 0xa5a50000u;
constexpr std::uint64_t PartialExec = 0x0f0f0f0f5555aaaaull;

alignas(256) std::array<std::uint32_t, Lanes * LaneDwords> Input{};
alignas(256) std::array<std::uint32_t, Lanes * 4u> Output{};

struct Variant {
    std::string name;
    std::uint32_t waveSize = 32;
    bool wide = false;
    bool a16 = false;
    bool nsa = false;
    bool partial = false;

    [[nodiscard]] std::uint32_t Components() const { return (wide ? 2u : 1u) + (a16 ? 7u : 10u); }
    [[nodiscard]] std::uint32_t Slot(std::uint32_t component) const { return nsa && component != 0u ? Components() - component : component; }
    [[nodiscard]] bool Active(std::uint32_t lane) const { return !partial || ((PartialExec >> (waveSize == 32u ? lane % 32u : lane)) & 1u) != 0u; }
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t records) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), records, 0x01016facu};
}

void Mubuf(std::vector<std::uint32_t>& code, std::uint32_t opcode, std::uint32_t data, std::uint32_t resource, std::uint32_t offset) {
    code.push_back((0x38u << 26u) | (opcode << 18u) | (1u << 13u) | offset);
    code.push_back(0u | (data << 8u) | ((resource / 4u) << 16u) | (0x80u << 24u));
}

void MoveScalar(std::vector<std::uint32_t>& code, std::uint32_t destination, std::uint32_t literal) {
    code.push_back(0xbe800000u | (destination << 16u) | (0x03u << 8u) | 0xffu);
    code.push_back(literal);
}

std::vector<std::uint32_t> Program(const Variant& variant) {
    std::vector<std::uint32_t> code;
    for (std::uint32_t reg = 20; reg < 24u; ++reg) {
        code.push_back(0x7e000000u | (reg << 17u) | (1u << 9u) | 0xffu);
        code.push_back(Sentinel | reg);
    }
    for (std::uint32_t quad = 0; quad < 3u; ++quad) Mubuf(code, 0x0eu, 2u + quad * 4u, 0u, quad * 16u);
    code.push_back(0xbf8c3f70u);
    if (variant.partial) {
        MoveScalar(code, 126u, static_cast<std::uint32_t>(PartialExec));
        MoveScalar(code, 127u, static_cast<std::uint32_t>(PartialExec >> 32u));
    }
    const std::uint32_t opcode = variant.wide ? 0xe7u : 0xe6u;
    const std::uint32_t extra = variant.Components() - 1u;
    const std::uint32_t nsaDwords = variant.nsa ? (extra + 3u) / 4u : 0u;
    code.push_back((0x3cu << 26u) | ((opcode & 0x7fu) << 18u) | (1u << 15u) | (1u << 12u) | (0xfu << 8u) | (nsaDwords << 1u) | (opcode >> 7u));
    code.push_back(2u | (20u << 8u) | (2u << 16u) | (variant.a16 ? 1u << 30u : 0u));
    for (std::uint32_t word = 0; word < nsaDwords; ++word) {
        std::uint32_t packed = 0;
        for (std::uint32_t byte = 0; byte < 4u; ++byte) {
            const auto component = 1u + word * 4u + byte;
            if (component <= extra) packed |= (2u + variant.Slot(component)) << (byte * 8u);
        }
        code.push_back(packed);
    }
    code.push_back(0xbf8c3f70u);
    if (variant.partial) {
        code.push_back(0xbefe03c1u);
        code.push_back(0xbeff03c1u);
    }
    Mubuf(code, 0x1eu, 20u, 4u, 0u);
    code.push_back(0xbf810000u);
    return code;
}

std::uint32_t HalfBits(float value) {
    const auto bits = BvhReference::Bits(value);
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23u) & 0xffu;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) return sign | 0x7c00u | (mantissa != 0u ? 0x200u : 0u);
    if (value == 0.0f) return sign;
    Require(exponent >= 113u && exponent <= 142u && (mantissa & 0x1fffu) == 0u, "an A16 test value is not an exact normal half");
    return sign | ((exponent - 112u) << 10u) | (mantissa >> 13u);
}

std::vector<std::uint32_t> Operand(const Variant& variant, const BvhReference::Ray& ray) {
    std::vector<std::uint32_t> components;
    components.push_back(static_cast<std::uint32_t>(ray.node));
    if (variant.wide) components.push_back(static_cast<std::uint32_t>(ray.node >> 32u));
    components.push_back(BvhReference::Bits(ray.extent));
    for (const float value : ray.origin) components.push_back(BvhReference::Bits(value));
    if (variant.a16) {
        const std::array<float, 6> halves{ray.direction[0], ray.direction[1], ray.direction[2], ray.inverse[0], ray.inverse[1], ray.inverse[2]};
        for (std::uint32_t i = 0; i < 6u; i += 2u) components.push_back(HalfBits(halves[i]) | (HalfBits(halves[i + 1u]) << 16u));
    } else {
        for (const float value : ray.direction) components.push_back(BvhReference::Bits(value));
        for (const float value : ray.inverse) components.push_back(BvhReference::Bits(value));
    }
    std::vector<std::uint32_t> slots(LaneDwords, 0u);
    for (std::uint32_t component = 0; component < components.size(); ++component) slots[variant.Slot(component)] = components[component];
    return slots;
}

std::vector<Words> Dispatch(AgcDriver::VulkanDevice& device, const Variant& variant, const Words& descriptor, const std::vector<BvhReference::Ray>& rays) {
    Require(rays.size() <= Lanes, "too many rays for one dispatch");
    Input.fill(0u);
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        const auto slots = Operand(variant, rays[lane < rays.size() ? lane : 0u]);
        std::copy(slots.begin(), slots.end(), Input.begin() + lane * LaneDwords);
    }
    Output.fill(0xdeadbeefu);
    std::vector<std::uint32_t> userData(16, 0u);
    const auto input = BufferDescriptor(Input.data(), LaneDwords * 4u, Lanes);
    const auto output = BufferDescriptor(Output.data(), 16u, Lanes);
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    std::copy(descriptor.begin(), descriptor.end(), userData.begin() + 8);
    const auto program = Program(variant);
    const std::span<const std::uint32_t> code(program);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {variant.waveSize, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    std::vector<Words> words(rays.size());
    for (std::uint32_t lane = 0; lane < rays.size(); ++lane) std::copy_n(Output.begin() + lane * 4u, 4u, words[lane].begin());
    return words;
}

std::string Hex(const Words& words) {
    char text[64];
    std::snprintf(text, sizeof(text), "%08x %08x %08x %08x", words[0], words[1], words[2], words[3]);
    return text;
}

Words Inactive() { return {Sentinel | 20u, Sentinel | 21u, Sentinel | 22u, Sentinel | 23u}; }

std::uint32_t HandCases(AgcDriver::VulkanDevice& device, const Variant& variant, std::uint64_t scene) {
    const auto cases = BvhCases::All();
    std::map<Words, std::vector<std::size_t>> groups;
    for (std::size_t index = 0; index < cases.size(); ++index) {
        if (cases[index].wideOnly && !variant.wide) continue;
        groups[BvhCases::EncodeDescriptor(cases[index], scene)].push_back(index);
    }
    std::uint32_t checked = 0;
    for (const auto& [descriptor, members] : groups) {
        std::vector<BvhReference::Ray> rays;
        for (std::size_t lane = 0; lane < Lanes; ++lane) {
            const auto& item = cases[members[lane % members.size()]];
            auto ray = item.ray;
            ray.node = BvhCases::NodePointer(item, scene);
            rays.push_back(ray);
        }
        const auto results = Dispatch(device, variant, descriptor, rays);
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto& item = cases[members[lane % members.size()]];
            const auto expected = variant.Active(lane) ? item.expected : Inactive();
            Require(results[lane] == expected, variant.name + ": " + item.name + " (lane " + std::to_string(lane) + "): " + Hex(results[lane]) + ", expected " + Hex(expected));
            ++checked;
        }
    }
    return checked;
}

class Fuzz {
public:
    Fuzz(std::uint32_t* memory, std::uint64_t scene) : memory(memory), scene(scene), random(0x5eedu) {}

    void Fill() {
        for (std::uint32_t node = BvhCases::SceneNodes; node < MemoryNodes; ++node) {
            auto* words = memory + node * 16u;
            switch (node + 1u < MemoryNodes ? node % 4u : 0u) {
            case 0:
            case 1:
                for (std::uint32_t i = 0; i < 15u; ++i) words[i] = BvhReference::Bits(coordinate());
                words[15] = static_cast<std::uint32_t>(random());
                break;
            case 2:
                for (std::uint32_t child = 0; child < 4u; ++child) {
                    words[child] = chance(10) ? BvhReference::NoChild : static_cast<std::uint32_t>(random());
                    std::array<float, 6> bounds = box();
                    for (std::uint32_t word = 0; word < 3u; ++word) words[4u + child * 3u + word] = half(bounds[word * 2u]) | (half(bounds[word * 2u + 1u]) << 16u);
                }
                break;
            default:
                for (std::uint32_t child = 0; child < 4u; ++child) {
                    words[child] = chance(10) ? BvhReference::NoChild : static_cast<std::uint32_t>(random());
                    const auto bounds = box();
                    for (std::uint32_t component = 0; component < 6u; ++component) words[4u + child * 6u + component] = BvhReference::Bits(bounds[component]);
                }
                break;
            }
        }
    }

    BvhReference::Descriptor Descriptor() {
        BvhReference::Descriptor descriptor;
        descriptor.base = scene;
        descriptor.lastNode = chance(80) ? MemoryNodes - 1u : BvhCases::SceneNodes + random() % (MemoryNodes - BvhCases::SceneNodes);
        descriptor.grow = chance(50) ? 0u : chance(50) ? random() % 4u : random() % 256u;
        descriptor.sort = chance(50);
        descriptor.barycentrics = chance(50);
        descriptor.kind = chance(95) ? 8u : random() % 16u;
        return descriptor;
    }

    BvhReference::Ray Ray(bool a16) {
        BvhReference::Ray ray;
        const auto node = BvhCases::SceneNodes + static_cast<std::uint32_t>(random() % (MemoryNodes - BvhCases::SceneNodes));
        const auto data = node % 4u;
        std::uint32_t type = data < 2u ? static_cast<std::uint32_t>(random() % 4u) : data == 2u ? 4u : 5u;
        if (chance(8)) type = static_cast<std::uint32_t>(random() % 8u);
        ray.node = (std::uint64_t{node} << 3u) | type;
        const auto* words = memory + node * 16u;
        std::array<float, 3> target{};
        if (data < 2u) {
            static constexpr std::array<std::array<std::uint32_t, 3>, 4> corners{{{0, 1, 2}, {1, 3, 2}, {2, 3, 4}, {2, 4, 0}}};
            std::array<float, 3> weights{uniform(0, 1), uniform(0, 1), 0};
            if (chance(30)) weights[random() % 2u] = 0.0f;
            if (weights[0] + weights[1] > 1.0f) weights = {1.0f - weights[0], 1.0f - weights[1], 0};
            weights[2] = 1.0f - weights[0] - weights[1];
            for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                for (std::uint32_t corner = 0; corner < 3u; ++corner) target[axis] += weights[corner] * BvhReference::Bits(words[corners[type % 4u][corner] * 3u + axis]);
            }
        } else {
            for (auto& value : target) value = uniform(-40, 40);
        }
        for (std::uint32_t axis = 0; axis < 3u; ++axis) ray.direction[axis] = a16 ? halfValue() : chance(15) ? 0.0f : uniform(-2, 2);
        if (chance(10)) ray.direction = {0, 0, 0};
        const float distance = chance(20) ? uniform(-5, 0) : uniform(0, 30);
        for (std::uint32_t axis = 0; axis < 3u; ++axis) {
            ray.origin[axis] = target[axis] - ray.direction[axis] * distance;
            if (chance(3)) ray.origin[axis] = special();
            ray.inverse[axis] = a16 ? halfValue() : chance(90) ? 1.0f / ray.direction[axis] : coordinate();
        }
        ray.extent = chance(85) ? uniform(0, 100) : special();
        return ray;
    }

private:
    bool chance(std::uint32_t percent) { return random() % 100u < percent; }
    float uniform(float low, float high) { return std::uniform_real_distribution<float>(low, high)(random); }

    float special() {
        static constexpr std::array<std::uint32_t, 8> values{0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x3f800000u, 0xc2c80000u, 0x7f7fffffu};
        return BvhReference::Bits(values[random() % values.size()]);
    }

    float coordinate() {
        if (chance(2)) return special();
        if (chance(40)) return static_cast<float>(static_cast<int>(random() % 17u) - 8);
        return uniform(-50, 50);
    }

    float halfValue() {
        static constexpr std::array<float, 10> values{0.0f, -0.0f, 1.0f, -1.0f, 2.0f, 0.5f, -0.25f, 3.0f, 1.5f, -4.0f};
        if (chance(5)) return std::numeric_limits<float>::infinity();
        return values[random() % values.size()];
    }

    std::array<float, 6> box() {
        std::array<float, 6> bounds{};
        for (std::uint32_t axis = 0; axis < 3u; ++axis) {
            const float a = static_cast<float>(static_cast<int>(random() % 81u) - 40) * 0.5f;
            const float b = a + static_cast<float>(random() % 20u) * 0.5f;
            bounds[axis] = chance(5) ? b : a;
            bounds[axis + 3u] = chance(5) ? a : b;
            if (chance(2)) bounds[axis] = special();
        }
        return bounds;
    }

    static std::uint32_t half(float value) {
        const auto bits = BvhReference::Bits(value);
        const std::uint32_t sign = (bits >> 16u) & 0x8000u;
        const std::uint32_t exponent = (bits >> 23u) & 0xffu;
        if (exponent == 0xffu) return sign | 0x7c00u | ((bits & 0x7fffffu) != 0u ? 0x200u : 0u);
        if (value == 0.0f || exponent < 113u) return sign;
        if (exponent > 142u) return sign | 0x7c00u;
        return sign | ((exponent - 112u) << 10u) | ((bits & 0x7fffffu) >> 13u);
    }

    std::uint32_t* memory;
    std::uint64_t scene;
    std::mt19937_64 random;
};

bool SameWord(std::uint32_t actual, std::uint32_t expected, bool floating) {
    if (actual == expected) return true;
    return floating && std::isnan(BvhReference::Bits(actual)) && std::isnan(BvhReference::Bits(expected));
}

std::uint32_t FuzzCases(AgcDriver::VulkanDevice& device, const Variant& variant, Fuzz& fuzz, const std::uint32_t* memory, std::uint64_t scene, std::uint32_t dispatches) {
    const BvhReference::BlockReader read = [&](std::uint64_t address) -> const std::uint32_t* {
        if (address < scene || address - scene >= MemoryBytes) return nullptr;
        return memory + (address - scene) / 4u;
    };
    std::uint32_t hits = 0;
    for (std::uint32_t round = 0; round < dispatches; ++round) {
        const auto descriptor = fuzz.Descriptor();
        const auto words = descriptor.Encode();
        std::vector<BvhReference::Ray> rays;
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) rays.push_back(fuzz.Ray(variant.a16));
        const auto results = Dispatch(device, variant, words, rays);
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto expected = variant.Active(lane) ? BvhReference::Intersect(words, rays[lane], read) : Inactive();
            const bool triangle = variant.Active(lane) && (rays[lane].node & 7u) < 4u;
            bool same = true;
            for (std::uint32_t word = 0; word < 4u; ++word) same = same && SameWord(results[lane][word], expected[word], triangle && (word < 2u || descriptor.barycentrics));
            if (!same) {
                const auto& ray = rays[lane];
                char detail[512];
                std::snprintf(detail, sizeof(detail), "node %llx extent %08x origin %08x %08x %08x direction %08x %08x %08x inverse %08x %08x %08x descriptor %s",
                              static_cast<unsigned long long>(ray.node), BvhReference::Bits(ray.extent), BvhReference::Bits(ray.origin[0]), BvhReference::Bits(ray.origin[1]), BvhReference::Bits(ray.origin[2]),
                              BvhReference::Bits(ray.direction[0]), BvhReference::Bits(ray.direction[1]), BvhReference::Bits(ray.direction[2]), BvhReference::Bits(ray.inverse[0]), BvhReference::Bits(ray.inverse[1]), BvhReference::Bits(ray.inverse[2]), Hex(words).c_str());
                Require(false, variant.name + " fuzz round " + std::to_string(round) + " lane " + std::to_string(lane) + ": " + Hex(results[lane]) + ", reference " + Hex(expected) + "; " + detail);
            }
            if (variant.Active(lane) && (triangle ? expected[0] != BvhReference::Infinity : expected[0] != BvhReference::NoChild)) ++hits;
        }
    }
    return hits;
}

class Registration {
public:
    explicit Registration(std::uint8_t* memory) : memory(memory) {
        GuestAllocations::Mutation mutation;
        mutation.Add(memory, SplitBytes, true, false);
        mutation.Add(memory + SplitBytes, MemoryBytes - SplitBytes, true, false);
    }
    ~Registration() {
        GuestAllocations::Mutation mutation;
        mutation.Remove(memory);
        mutation.Remove(memory + SplitBytes);
    }
    Registration(const Registration&) = delete;
    Registration& operator=(const Registration&) = delete;

private:
    std::uint8_t* memory;
};

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        auto* block = static_cast<std::uint8_t*>(std::aligned_alloc(SplitBytes, MemoryBytes));
        Require(block != nullptr, "the BVH test memory was not allocated");
        std::unique_ptr<std::uint8_t, decltype(&std::free)> owner(block, &std::free);
        auto* words = reinterpret_cast<std::uint32_t*>(block);
        std::memset(block, 0, MemoryBytes);
        const auto sceneWords = BvhCases::Scene();
        std::copy(sceneWords.begin(), sceneWords.end(), words);
        const auto scene = reinterpret_cast<std::uint64_t>(block);
        Fuzz fuzz(words, scene);
        fuzz.Fill();
        Registration registration(block);
        const std::vector<Variant> variants{
            {"wave32", 32},
            {"wave64", 64},
            {"wave32 a16", 32, false, true},
            {"wave32 nsa", 32, false, false, true},
            {"wave64 a16 nsa, partial exec", 64, false, true, true, true},
            {"wave32 bvh64", 32, true},
            {"wave64 bvh64 a16 nsa", 64, true, true, true},
            {"wave32 bvh64 nsa, partial exec", 32, true, false, true, true},
        };
        std::uint32_t checked = 0;
        for (const auto& variant : variants) checked += HandCases(*device, variant, scene);
        std::uint32_t hits = 0;
        hits += FuzzCases(*device, variants[0], fuzz, words, scene, 60);
        hits += FuzzCases(*device, variants[6], fuzz, words, scene, 30);
        hits += FuzzCases(*device, variants[4], fuzz, words, scene, 30);
        std::printf("bvh intersection tests passed: %u hand-case lanes, %u fuzz lanes (%u hits)\n", checked, 120u * Lanes, hits);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
