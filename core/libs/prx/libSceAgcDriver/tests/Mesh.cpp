// agc_driver_mesh_tests: an NGG geometry program (the merged ES/GS "primitive shader" form the
// geometry path runs, see State.cpp DecodeShaderStages) recompiled to a mesh shader and drawn on the
// GPU through VulkanDevice::Draw, with the pixels read back. Needs VK_EXT_mesh_shader.
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "Recompiler.hpp"
#include "NggProgram.hpp"
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
using namespace NggTest;

constexpr std::uint32_t Width = 192;
constexpr std::uint32_t Height = 128;
constexpr std::array<std::uint8_t, 4> Background{16, 24, 40, 255};
alignas(256) std::array<std::byte, Width * Height * 4> Pixels{};

// v_interp_mov_f32 v0..v3, p0, attr0.xyzw / exp mrt0 v0, v1, v2, v3 done vm / s_endpgm
alignas(256) constexpr std::array<std::uint32_t, 7> PixelCode{
    0xc8020002, 0xc8060102, 0xc80a0202, 0xc80e0302, 0xf800180f, 0x03020100, 0xbf810000,
};

constexpr std::uint32_t Columns = 8;
constexpr std::uint32_t Rows = 5;
constexpr std::uint32_t Triangles = Columns * Rows;

std::array<std::uint8_t, 4> TriangleColor(std::uint32_t triangle) {
    return {static_cast<std::uint8_t>(40u + 29u * triangle), static_cast<std::uint8_t>(200u + 53u * triangle), static_cast<std::uint8_t>(triangle % 2u == 0u ? 60u : 200u), 255};
}

// A triangle in cell `triangle` of a Columns x Rows grid over the viewport, all three vertices in its color.
std::array<Vertex, 3> TriangleVertices(std::uint32_t triangle) {
    const float cellWidth = 2.0f / Columns;
    const float cellHeight = 2.0f / Rows;
    const float x = -1.0f + cellWidth * static_cast<float>(triangle % Columns);
    const float y = -1.0f + cellHeight * static_cast<float>(triangle / Columns);
    const auto color = TriangleColor(triangle);
    const std::array<float, 4> rgba{color[0] / 255.0f, color[1] / 255.0f, color[2] / 255.0f, 1.0f};
    return {{
        {{x + 0.1f * cellWidth, y + 0.1f * cellHeight, 0.5f, 1.0f}, rgba},
        {{x + 0.9f * cellWidth, y + 0.1f * cellHeight, 0.5f, 1.0f}, rgba},
        {{x + 0.5f * cellWidth, y + 0.9f * cellHeight, 0.5f, 1.0f}, rgba},
    }};
}

// The pixel a point of cell `triangle` inside its triangle lands on (the viewport flips y).
std::size_t CellPixel(std::uint32_t triangle) {
    const auto column = triangle % Columns;
    const auto row = triangle / Columns;
    const auto x = (column * Width + Width / 2u) / Columns;
    const auto y = Height - 1u - (row * Height + (Height * 4u) / 10u) / Rows;
    return (static_cast<std::size_t>(y) * Width + x) * 4u;
}

// Vertices in scrambled order behind an index buffer, and the same triangles in draw order.
alignas(256) std::array<Vertex, 3 * Triangles> Scrambled{};
alignas(256) std::array<std::uint16_t, 3 * Triangles> Indices{};
alignas(256) std::array<Vertex, 3 * Triangles> Ordered{};
// A strip of two rows of quads (Columns cells wide, one color) over the lower half: 2 * Columns
// triangles, the odd ones with the hardware's vertex order swap.
alignas(256) std::array<Vertex, 2 * Columns + 2> Strip{};

void ClearPixels() {
    for (std::size_t i = 0; i < Pixels.size(); i += 4) {
        for (std::size_t c = 0; c < 4; ++c) Pixels[i + c] = std::byte{Background[c]};
    }
}

bool PixelIs(std::size_t offset, const std::array<std::uint8_t, 4>& color) {
    for (std::size_t c = 0; c < 4; ++c) {
        if (std::to_integer<std::uint8_t>(Pixels[offset + c]) != color[c]) return false;
    }
    return true;
}

std::string PixelText(std::size_t offset) {
    char text[64];
    std::snprintf(text, sizeof(text), "(%u, %u, %u, %u)", std::to_integer<unsigned>(Pixels[offset]), std::to_integer<unsigned>(Pixels[offset + 1]), std::to_integer<unsigned>(Pixels[offset + 2]), std::to_integer<unsigned>(Pixels[offset + 3]));
    return text;
}

struct MeshDraw {
    ShaderRecompiler::MeshConfiguration mesh;
    AgcDriver::Pm4::DrawParameters draw;
    std::array<std::uint32_t, 4> vertexBuffer;
};

// Recompiles the geometry and pixel programs for `setup` and draws into Pixels.
void DrawMesh(AgcDriver::VulkanDevice& device, const MeshDraw& setup) {
    const auto target = device.Target();
    // The hidden words of the merged program (the user data pointer, GS_TG_INFO and wave info the
    // prologue replaces, the index buffer V#), then the user SGPRs s[8:11]: the vertex buffer.
    std::vector<std::uint32_t> userData(12, 0u);
    const auto index = AgcDriver::Graphics::MeshIndexBufferDescriptor(setup.draw, reinterpret_cast<std::uintptr_t>(GeometryCode.data()));
    std::copy(index.begin(), index.end(), userData.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
    std::copy(setup.vertexBuffer.begin(), setup.vertexBuffer.end(), userData.begin() + 8);
    const std::array<ShaderRecompiler::MemoryRegion, 1> geometryMemory{{{reinterpret_cast<std::uintptr_t>(GeometryCode.data()), std::as_bytes(std::span(GeometryCode))}}};
    ShaderRecompiler::RecompileRequest geometry{
        {ShaderStage::Mesh, reinterpret_cast<std::uintptr_t>(GeometryCode.data()), GeometryCode, 0, {}},
        {64, 0, userData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, geometryMemory},
        target,
        {0, 0, 0, ShaderRecompiler::MeshDrawPushOffsetBytes},
        ShaderRecompiler::GraphicsCompileContext{0, {}, setup.mesh, std::nullopt, {setup.draw.indexAddress, setup.draw.indexCount, setup.draw.indexSize, setup.draw.instanceCount}}
    };
    const auto meshResult = ShaderRecompiler::Recompile(geometry);
    const auto meshPush = static_cast<std::uint32_t>(meshResult.pushConstants.size());

    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.interpolatorCount = 1;
    pixel.interpolatorSettings[0] = 0x400u;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(PixelCode.data()), std::as_bytes(std::span(PixelCode))}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(PixelCode.data()), PixelCode, 0, {}},
        {64, 0, {}, std::nullopt, pixel, std::nullopt, pixelMemory},
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
    state.stages = {AgcDriver::Graphics::ShaderPath::Geometry, 0x20u, 64, 64, setup.mesh, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, Pixels.size(), 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.renderExtent = {Width, Height};
    state.topology = setup.mesh.inputPrimitive == 6u ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    device.Draw(state, setup.draw, shaders);
    device.WaitIdle();
}

// Thirty-two triangles (96 ES and output vertices) per subgroup of two waves: a full and a partial
// workgroup, the second wave's ES vertices read by the first wave's GS threads across the barrier
// and its vertex threads exporting vertices 64 and up.
constexpr ShaderRecompiler::MeshConfiguration WideSubgroup{4u, 32u, 96u, 96u, 32u, 128u, 2048u, 0u, 4u};

void CheckTriangles(const char* what) {
    for (std::uint32_t triangle = 0; triangle < Triangles; ++triangle) {
        const auto offset = CellPixel(triangle);
        Require(PixelIs(offset, TriangleColor(triangle)), std::string(what) + ": triangle " + std::to_string(triangle) + " was not rendered in its color, pixel " + PixelText(offset));
    }
    Require(PixelIs(0, Background) && PixelIs(Pixels.size() - 4u, Background), std::string(what) + ": the corners changed");
}

}

int main() {
    try {
        // Tests do not fill the user's shader disk cache.
        setenv("APS5_NO_SHADER_DISK_CACHE", "1", 1);
        for (std::uint32_t triangle = 0; triangle < Triangles; ++triangle) {
            const auto vertices = TriangleVertices(triangle);
            for (std::uint32_t k = 0; k < 3; ++k) {
                Ordered[3 * triangle + k] = vertices[k];
                // Scrambled slot of vertex (triangle, k): reversed triangles, rotated corners.
                const auto slot = 3 * (Triangles - 1 - triangle) + (k + 1) % 3;
                Scrambled[slot] = vertices[k];
                Indices[3 * triangle + k] = static_cast<std::uint16_t>(slot);
            }
        }
        const std::array<float, 4> stripColor{128.0f / 255.0f, 1.0f, 64.0f / 255.0f, 1.0f};
        for (std::uint32_t i = 0; i < Strip.size(); ++i) {
            const float x = -1.0f + 2.0f * static_cast<float>(i / 2u) / Columns;
            const float y = i % 2u == 0u ? -1.0f : 0.0f;
            Strip[i] = {{x, y, 0.5f, 1.0f}, stripColor};
        }

        AgcDriver::VulkanDevice device;
        const auto target = device.Target();
        if (!target.mesh.has_value()) {
            std::puts("Mesh tests skipped: the device has no VK_EXT_mesh_shader");
            return 0;
        }

        for (const auto& [name, subgroup] : {std::pair{"one-wave subgroups", SmallSubgroup}, std::pair{"two-wave subgroups", WideSubgroup}}) {
            ClearPixels();
            DrawMesh(device, {subgroup, {reinterpret_cast<std::uintptr_t>(Indices.data()), static_cast<std::uint32_t>(Indices.size()), 2, 1, 0, true}, VertexBufferDescriptor(Scrambled.data(), static_cast<std::uint32_t>(Scrambled.size()))});
            CheckTriangles((std::string("indexed triangle list, ") + name).c_str());

            ClearPixels();
            DrawMesh(device, {subgroup, {0, static_cast<std::uint32_t>(Ordered.size()), 0, 1, 0, false}, VertexBufferDescriptor(Ordered.data(), static_cast<std::uint32_t>(Ordered.size()))});
            CheckTriangles((std::string("non-indexed triangle list, ") + name).c_str());
        }

        // A strip: three triangles (five ES vertices) per subgroup, so the sixteen take six
        // workgroups, every other one starting at an odd triangle (its first two vertices swapped).
        ClearPixels();
        const ShaderRecompiler::MeshConfiguration strip{6u, 3u, 5u, 9u, 3u, 64u, 1024u, 0u, 4u};
        DrawMesh(device, {strip, {0, static_cast<std::uint32_t>(Strip.size()), 0, 1, 0, false}, VertexBufferDescriptor(Strip.data(), static_cast<std::uint32_t>(Strip.size()))});
        const std::array<std::uint8_t, 4> stripPixel{128, 255, 64, 255};
        for (std::uint32_t y = Height / 2u + 2u; y < Height - 2u; y += 5u) {
            for (std::uint32_t x = 2u; x < Width - 2u; x += 5u) {
                const auto offset = (static_cast<std::size_t>(y) * Width + x) * 4u;
                Require(PixelIs(offset, stripPixel), "triangle strip: pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") is " + PixelText(offset));
            }
        }
        Require(PixelIs((static_cast<std::size_t>(Height / 4u) * Width + Width / 2u) * 4u, Background), "triangle strip: the upper half changed");

        std::puts("Mesh tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
