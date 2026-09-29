// agc_driver_gds_tests: GDS instructions recompiled and run on the GPU through VulkanDevice with
// the device's GDS buffer (made unless APS5_NO_GDS=1), which the CP's DMA_DATA reads and writes as well (see
// Pm4::InstallGdsBacking). Run by hand.
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "NggProgram.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Sentinel = 0xdeadbeefu;
constexpr std::uint32_t Groups = 4;
constexpr std::uint32_t GroupThreads = 64;
alignas(256) std::array<std::uint32_t, Groups * GroupThreads> Output{};

std::uint32_t low(const void* pointer) { return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer)); }
std::uint32_t high(const void* pointer) { return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer) >> 32u); }

std::vector<std::uint32_t> packet(std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    std::vector<std::uint32_t> result{0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u)};
    result.insert(result.end(), payload);
    return result;
}

// The CP's side of the GDS: DMA_DATA of an immediate dword into it, and of a dword out of it.
void WriteGds(std::uint32_t offset, std::uint32_t value) {
    AgcDriver::QueueState queue;
    const auto write = packet(0x50, {0x40100000u, value, 0, offset, 0, 4});
    AgcDriver::Pm4::Validate(write, 0);
    AgcDriver::Pm4::Execute(write, queue);
}

std::uint32_t ReadGds(std::uint32_t offset) {
    AgcDriver::QueueState queue;
    alignas(8) std::uint32_t value = 0;
    const auto read = packet(0x50, {0x20000000u, offset, 0, low(&value), high(&value), 4});
    AgcDriver::Pm4::Validate(read, 0);
    AgcDriver::Pm4::Execute(read, queue);
    return value;
}

// A structured buffer of `count` dwords (stride 4, 32_FLOAT, DST_SEL XYZW).
std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), count, 0x01016facu};
}

// Runs `code` as Groups groups of GroupThreads threads: user data s0 = M0, s[4:7] = the output V#,
// the group id in s8.
void Run(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, std::uint32_t waveSize, std::uint32_t m0) {
    Output.fill(Sentinel);
    std::vector<std::uint32_t> userData(8, 0u);
    userData[0] = m0;
    const auto descriptor = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size()));
    std::copy(descriptor.begin(), descriptor.end(), userData.begin() + 4);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{GroupThreads, 1, 1}, 0, {true, false, false}, false, 1};
    const ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {waveSize, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, Groups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
}

// The threads with (id & 3) != 0 each got one distinct value of [first, first + count) and the
// others kept the sentinel; `count` is the number of such threads.
void CheckDistinct(const char* what, std::uint32_t first) {
    std::vector<std::uint32_t> values;
    for (std::uint32_t thread = 0; thread < Output.size(); ++thread) {
        const bool active = (thread % GroupThreads & 3u) != 0;
        if (!active) {
            Require(Output[thread] == Sentinel, std::string(what) + ": inactive thread " + std::to_string(thread) + " stored " + std::to_string(Output[thread]));
            continue;
        }
        values.push_back(Output[thread]);
    }
    std::sort(values.begin(), values.end());
    for (std::uint32_t i = 0; i < values.size(); ++i) {
        Require(values[i] == first + i, std::string(what) + ": the threads' indices are not " + std::to_string(first) + " + [0, " + std::to_string(values.size()) + "): sorted value " + std::to_string(i) + " is " + std::to_string(values[i]));
    }
}

constexpr std::uint32_t ActiveThreads = Groups * GroupThreads * 3 / 4;

// s_mov_b32 m0, s0 / v_and_b32 v4, 3, v0 / v_cmpx_ne_u32 0, v4
// v_mbcnt_lo_u32_b32 v1, exec_lo, 0 / v_mbcnt_hi_u32_b32 v1, exec_hi, v1
// ds_append v2 offset:4 gds / s_waitcnt lgkmcnt(0) / v_add_nc_u32 v2, v2, v1
// s_lshl_b32 s9, s8, 6 / v_add_nc_u32 v3, s9, v0 / buffer_store_dword v2, v3, s[4:7], 0 idxen
// s_endpgm
alignas(256) constexpr std::array<std::uint32_t, 16> AppendCode{
    0xbefc0300, 0x36080083, 0x7daa0880, 0xd7650001, 0x0001007e, 0xd7660001, 0x0002027f, 0xd8fa0004,
    0x02000000, 0xbf8cc07f, 0x4a040302, 0x8f098608, 0x4a060009, 0xe0702000, 0x80010203, 0xbf810000,
};

// As AppendCode, with ds_consume v2 offset:8 gds and v2 = v2 - 1 - v1 (v_not_b32 v1, v1 /
// v_add_nc_u32 v2, v2, v1): lane i of a wave that consumed k from n takes n - 1 - i.
alignas(256) constexpr std::array<std::uint32_t, 17> ConsumeCode{
    0xbefc0300, 0x36080083, 0x7daa0880, 0xd7650001, 0x0001007e, 0xd7660001, 0x0002027f, 0xd8f60008,
    0x02000000, 0xbf8cc07f, 0x7e026f01, 0x4a040302, 0x8f098608, 0x4a060009, 0xe0702000, 0x80010203,
    0xbf810000,
};

// A linked-list walk as Astro Bot's light-list consumer (0x500597a00) runs it: wave64 over a
// 32x2 group (two lanes per invocation on a 32-wide subgroup), each lane loading its head, then
// following {data, next} nodes until next is 0, with the EXECZ exit taken only once every lane is
// done. s[0:3] heads, s[4:7] nodes (stride 8), s[8:11] the per-lane step counts.
//   v_lshl_add_u32 v2, v1, 5, v0 / s_lshl_b32 s14, s12, 6 / v_add_nc_u32 v2, s14, v2 (group id s12)
//   buffer_load_dword v7, v2, s[0:3], 0 idxen / v_mov_b32 v3, 0 / s_waitcnt vmcnt(0)
//   s_mov_b64 s[16:17], exec
// loop:
//   v_cmpx_lt_u32 0, v7 / s_cbranch_execz done / buffer_load_dwordx2 v[4:5], v7, s[4:7], 0 idxen
//   s_waitcnt vmcnt(0) / v_add_nc_u32 v3, 1, v3 / v_mov_b32 v7, v5 / s_branch loop
// done:
//   s_mov_b64 exec, s[16:17] / buffer_store_dword v3, v2, s[8:11], 0 idxen / s_endpgm
alignas(256) constexpr std::array<std::uint32_t, 21> ListWalkCode{
    0xd7460002, 0x04010b01, 0x8f0e860c, 0x4a04040e, 0xe0302000, 0x80000702, 0x7e060280, 0xbf8c3f70,
    0xbe90047e, 0x7da20e80, 0xbf880006, 0xe0342000, 0x80010407, 0xbf8c3f70, 0x4a060681, 0x7e0e0305,
    0xbf82fff8, 0xbefe0410, 0xe0702000, 0x80020302, 0xbf810000,
};

constexpr std::uint32_t WalkGroups = 64;
alignas(256) std::array<std::uint32_t, WalkGroups * 64> WalkHeads{};
alignas(256) std::array<std::uint32_t, 2 * (1 + WalkGroups * 64 * 24)> WalkNodes{};
alignas(256) std::array<std::uint32_t, WalkGroups * 64> WalkSteps{};

// Every lane walks a chain of its own length (0 to 23 nodes, varying within each wave and quad).
void RunListWalk(AgcDriver::VulkanDevice& device, std::uint32_t waveSize) {
    std::uint32_t next = 1;
    std::vector<std::uint32_t> expected(WalkHeads.size());
    for (std::uint32_t lane = 0; lane < WalkHeads.size(); ++lane) {
        const std::uint32_t length = (lane * 7u + lane / 64u * 5u) % 24u;
        std::uint32_t head = 0;
        for (std::uint32_t k = 0; k < length; ++k) {
            WalkNodes[next * 2u] = lane;
            WalkNodes[next * 2u + 1u] = head;
            head = next++;
        }
        WalkHeads[lane] = head;
        expected[lane] = length;
    }
    WalkSteps.fill(Sentinel);
    std::vector<std::uint32_t> userData(12, 0u);
    const auto heads = BufferDescriptor(WalkHeads.data(), static_cast<std::uint32_t>(WalkHeads.size()));
    auto nodes = BufferDescriptor(WalkNodes.data(), static_cast<std::uint32_t>(WalkNodes.size() / 2u));
    nodes[1] = (nodes[1] & 0xffffu) | (8u << 16u);
    const auto steps = BufferDescriptor(WalkSteps.data(), static_cast<std::uint32_t>(WalkSteps.size()));
    std::copy(heads.begin(), heads.end(), userData.begin());
    std::copy(nodes.begin(), nodes.end(), userData.begin() + 4);
    std::copy(steps.begin(), steps.end(), userData.begin() + 8);
    const std::span<const std::uint32_t> code(ListWalkCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{32, 2, 1}, 0, {true, false, false}, false, 2};
    const ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {waveSize, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, WalkGroups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    for (std::uint32_t lane = 0; lane < WalkHeads.size(); ++lane) {
        Require(WalkSteps[lane] == expected[lane], "list walk: lane " + std::to_string(lane) + " took " + std::to_string(WalkSteps[lane]) + " steps for a chain of " + std::to_string(expected[lane]));
    }
}

// The pixel test: triangles over a Width x Height target, each pixel shader invocation appending
// through the GDS and storing its index at its pixel's slot of PixelIndices.
constexpr std::uint32_t Width = 192;
constexpr std::uint32_t Height = 128;
alignas(256) std::array<std::byte, Width * Height * 4> Pixels{};
alignas(256) std::array<std::uint32_t, Width * Height> PixelIndices{};

// s_mov_b32 m0, s0 / v_cvt_u32_f32 v2, v0 / v_cvt_u32_f32 v3, v1 (POS_X, POS_Y)
// v_mbcnt_lo_u32_b32 v4, exec_lo, 0 / v_mbcnt_hi_u32_b32 v4, exec_hi, v4
// ds_append v5 offset:4 gds / s_waitcnt lgkmcnt(0) / v_add_nc_u32 v5, v5, v4
// v_mad_u32_u24 v6, v3, 192, v2 / buffer_store_dword v5, v6, s[4:7], 0 idxen
// v_mov_b32 v7, 1.0 / exp mrt0 v7, v7, v7, v7 done vm / s_endpgm
alignas(256) constexpr std::array<std::uint32_t, 20> AppendPixelCode{
    0xbefc0300, 0x7e040f00, 0x7e060f01, 0xd7650004, 0x0001007e, 0xd7660004, 0x0002087f, 0xd8fa0004,
    0x05000000, 0xbf8cc07f, 0x4a0a0905, 0xd5430006, 0x0409ff03, 0x000000c0, 0xe0702000, 0x80010506,
    0x7e0e02f2, 0xf800180f, 0x07070707, 0xbf810000,
};

// Triangles of assorted slopes and sizes, none overlapping another: their edges leave partly
// covered quads (helper invocations) all over the target.
std::vector<NggTest::Vertex> PixelTriangles() {
    std::vector<NggTest::Vertex> vertices;
    constexpr std::uint32_t columns = 6;
    constexpr std::uint32_t rows = 4;
    for (std::uint32_t cell = 0; cell < columns * rows; ++cell) {
        const float width = 2.0f / columns;
        const float height = 2.0f / rows;
        const float x = -1.0f + width * static_cast<float>(cell % columns);
        const float y = -1.0f + height * static_cast<float>(cell / columns);
        const float skew = 0.07f * static_cast<float>(cell % 5u);
        const std::array<float, 4> color{1, 1, 1, 1};
        vertices.push_back({{x + (0.05f + skew) * width, y + 0.05f * height, 0.5f, 1.0f}, color});
        vertices.push_back({{x + 0.93f * width, y + (0.11f + skew) * height, 0.5f, 1.0f}, color});
        vertices.push_back({{x + (0.37f - skew * 0.5f) * width, y + 0.91f * height, 0.5f, 1.0f}, color});
    }
    return vertices;
}

// Draws PixelTriangles with AppendPixelCode (M0 in user word 0, the PixelIndices V# in words 4..7).
void DrawAppendingPixels(AgcDriver::VulkanDevice& device, std::uint32_t m0) {
    static const auto triangles = PixelTriangles();
    Pixels.fill(std::byte{0});
    PixelIndices.fill(Sentinel);
    const auto target = device.Target();
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(triangles.size()), 0, 1, 0, false};
    const auto vertexBuffer = NggTest::VertexBufferDescriptor(triangles.data(), static_cast<std::uint32_t>(triangles.size()));
    std::vector<std::uint32_t> geometryUserData(12, 0u);
    const auto index = AgcDriver::Graphics::MeshIndexBufferDescriptor(draw, reinterpret_cast<std::uintptr_t>(NggTest::GeometryCode.data()));
    std::copy(index.begin(), index.end(), geometryUserData.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), geometryUserData.begin() + 8);
    const std::array<ShaderRecompiler::MemoryRegion, 1> geometryMemory{{{reinterpret_cast<std::uintptr_t>(NggTest::GeometryCode.data()), std::as_bytes(std::span(NggTest::GeometryCode))}}};
    ShaderRecompiler::RecompileRequest geometry{
        {ShaderStage::Mesh, reinterpret_cast<std::uintptr_t>(NggTest::GeometryCode.data()), NggTest::GeometryCode, 0, {}},
        {64, 0, geometryUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, geometryMemory},
        target,
        {0, 0, 0, ShaderRecompiler::MeshDrawPushOffsetBytes},
        ShaderRecompiler::GraphicsCompileContext{0, {}, NggTest::SmallSubgroup, std::nullopt, {draw.indexAddress, draw.indexCount, draw.indexSize, draw.instanceCount}}
    };
    const auto meshResult = ShaderRecompiler::Recompile(geometry);
    const auto meshPush = static_cast<std::uint32_t>(meshResult.pushConstants.size());

    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.interpolatorCount = 1;
    pixel.interpolatorSettings[0] = 0x400u;
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixel.posX = true;
    pixel.posY = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    std::vector<std::uint32_t> pixelUserData(8, 0u);
    pixelUserData[0] = m0;
    const auto indices = BufferDescriptor(PixelIndices.data(), static_cast<std::uint32_t>(PixelIndices.size()));
    std::copy(indices.begin(), indices.end(), pixelUserData.begin() + 4);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(AppendPixelCode.data()), std::as_bytes(std::span(AppendPixelCode))}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(AppendPixelCode.data()), AppendPixelCode, 0, {}},
        {64, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        {0, 0, meshPush, ShaderRecompiler::MeshDrawPushOffsetBytes - meshPush},
        std::nullopt
    };
    const auto pixelResult = ShaderRecompiler::Recompile(fragment);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Mesh, &meshResult, 0},
        {ShaderStage::Fragment, &pixelResult, meshPush}
    }};

    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Geometry, 0x20u, 64, 64, NggTest::SmallSubgroup, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, Pixels.size(), 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    device.Draw(state, draw, shaders);
    device.WaitIdle();
}

// Every covered pixel (the one color the shader exports) stored a distinct index of
// [first, first + covered), every other pixel kept the sentinel; returns the covered count.
std::uint32_t CheckPixelIndices(std::uint32_t first) {
    std::vector<std::uint32_t> values;
    for (std::uint32_t pixel = 0; pixel < PixelIndices.size(); ++pixel) {
        const bool covered = std::to_integer<std::uint8_t>(Pixels[pixel * 4u]) == 255u;
        if (!covered) {
            Require(PixelIndices[pixel] == Sentinel, "ds_append pixel: uncovered pixel " + std::to_string(pixel) + " stored " + std::to_string(PixelIndices[pixel]));
            continue;
        }
        values.push_back(PixelIndices[pixel]);
    }
    Require(values.size() > 1000, "ds_append pixel: the triangles covered only " + std::to_string(values.size()) + " pixels");
    std::sort(values.begin(), values.end());
    for (std::uint32_t i = 0; i < values.size(); ++i) {
        Require(values[i] == first + i, "ds_append pixel: the pixels' indices are not " + std::to_string(first) + " + [0, " + std::to_string(values.size()) + "): sorted value " + std::to_string(i) + " is " + std::to_string(values[i]));
    }
    return static_cast<std::uint32_t>(values.size());
}

}

int main() {
    try {
        // Tests do not fill the user's shader disk cache; the device makes its GDS buffer.
        setenv("APS5_NO_SHADER_DISK_CACHE", "1", 1);
        unsetenv("APS5_NO_GDS");
        AgcDriver::VulkanDevice device;

        // M0: base 0x100 (bits 31:16), size 0x20. The append counter is GDS dword 0x104.
        constexpr std::uint32_t m0 = (0x100u << 16u) | 0x20u;
        for (const auto waveSize : {64u, 32u}) {
            const auto name = std::string("ds_append wave") + std::to_string(waveSize);
            WriteGds(0x104, 1000);
            Run(device, AppendCode, waveSize, m0);
            CheckDistinct(name.c_str(), 1000);
            Require(ReadGds(0x104) == 1000 + ActiveThreads, name + ": the GDS counter is " + std::to_string(ReadGds(0x104)));

            const auto consume = std::string("ds_consume wave") + std::to_string(waveSize);
            WriteGds(0x108, 500);
            Run(device, ConsumeCode, waveSize, m0);
            CheckDistinct(consume.c_str(), 500 - ActiveThreads);
            Require(ReadGds(0x108) == 500 - ActiveThreads, consume + ": the GDS counter is " + std::to_string(ReadGds(0x108)));
        }

        // Lanes leaving a loop at different iterations (the wave takes the exit once all have).
        RunListWalk(device, 32);
        RunListWalk(device, 64);

        // A pixel shader's append: the invocations the GPU adds for partly covered quads (helper
        // invocations, whose atomics do nothing) must neither allocate nor be the lane that
        // performs the wave's atomic.
        for (std::uint32_t pass = 0; pass < 4; ++pass) {
            WriteGds(0x104, 77);
            DrawAppendingPixels(device, m0);
            const auto covered = CheckPixelIndices(77);
            Require(ReadGds(0x104) == 77 + covered, "ds_append pixel: the GDS counter is " + std::to_string(ReadGds(0x104)) + " for " + std::to_string(covered) + " covered pixels");
        }

        std::puts("GDS tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
