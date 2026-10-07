#include "Translation/NggPassthrough.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <string>

namespace {

constexpr std::array<std::uint32_t, 22> PerVertex{
    0x93ebff03, 0x00080008, 0x8f6a8c6b, 0x8700ff03, 0x000000ff, 0x887c6a00, 0xbf900009, 0x81ea6bc0, 0x90fe6ac1, 0xf8000941, 0x00000000,
    0x81ea00c0, 0xbf8cff0f, 0x90fe6ac1, 0x7e020d05, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020201, 0xf800020f, 0x01010101, 0xbf810000,
};

constexpr std::array<std::uint32_t, 23> CountFromShiftedWaveInfo{
    0x93eaff03, 0x00080008, 0x876bff03, 0x000000ff, 0x8f6a8c6a, 0x887c6a6b, 0xbf800000, 0xbf900009, 0x906a8803, 0x81ea6a80, 0x90fe6ac1, 0xf8000941,
    0x00000000, 0x81ea0380, 0x90fe6ac1, 0x7e020d05, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020201, 0xf800020f, 0x01010101, 0xbf810000,
};

constexpr std::array<std::uint32_t, 24> Culled{
    0x93ebff03, 0x00080008, 0x8f6a8c6b, 0x8700ff03, 0x000000ff, 0x887c6a00, 0xbf900009, 0x81ea6bc0, 0x90fe6ac1, 0x380000ff, 0x80000000, 0xf8000941,
    0x00000000, 0x81ea00c0, 0xbf8cff0f, 0x90fe6ac1, 0x7e020d05, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020201, 0xf800020f, 0x01010101, 0xbf810000,
};

constexpr std::array<std::uint32_t, 22> SubgroupRegister{
    0x93ebff03, 0x00080008, 0x8f6a8c6b, 0x8700ff03, 0x000000ff, 0x887c6a00, 0xbf900009, 0x81ea6bc0, 0x90fe6ac1, 0xf8000941, 0x00000000,
    0x81ea00c0, 0xbf8cff0f, 0x90fe6ac1, 0x7e020d05, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020201, 0xf800020f, 0x04040404, 0xbf810000,
};

constexpr std::array<std::uint32_t, 27> SharedMemory{
    0x93ebff03, 0x00080008, 0x8f6a8c6b, 0x8700ff03, 0x000000ff, 0x887c6a00, 0xbf900009, 0x81ea6bc0, 0x90fe6ac1, 0xf8000941, 0x00000000, 0x81ea00c0, 0xbf8cff0f, 0x90fe6ac1,
    0x7e020d05, 0x7e040280, 0x7e0602f2, 0xd8340000, 0x00000101, 0xd8d80000, 0x01000001, 0xbf8cc07f, 0xf80008cf, 0x03020201, 0xf800020f, 0x01010101, 0xbf810000,
};

constexpr std::array<std::uint32_t, 23> CountAsData{
    0x93ebff03, 0x00080008, 0x8f6a8c6b, 0x8700ff03, 0x000000ff, 0x887c6a00, 0xbf900009, 0x81ea6bc0, 0x90fe6ac1, 0xf8000941, 0x00000000, 0x81ea00c0,
    0xbf8cff0f, 0x90fe6ac1, 0x7e020d05, 0x7e040280, 0x7e0602f2, 0x7e080200, 0xf80008cf, 0x03020201, 0xf800020f, 0x04040404, 0xbf810000,
};

constexpr std::array<std::uint32_t, 24> LaneAsData{
    0x93ebff03, 0x00080008, 0x8f6a8c6b, 0x8700ff03, 0x000000ff, 0x887c6a00, 0xbf900009, 0x81ea6bc0, 0x90fe6ac1, 0xf8000941, 0x00000000, 0x81ea00c0,
    0xbf8cff0f, 0x90fe6ac1, 0x7e020d05, 0x7e040280, 0x7e0602f2, 0xd7650004, 0x000100c1, 0xf80008cf, 0x03020201, 0xf800020f, 0x04040404, 0xbf810000,
};

constexpr std::array<std::uint32_t, 28> GroupCountsAndLaneBranches{
    0x938cff02, 0x0009000c, 0x938dff02, 0x00090016, 0x8f0e8c0d, 0x887c0e0c, 0xbf900009, 0xd7650009,
    0x000100c1, 0xd7660009, 0x000212c1, 0x7da8120d, 0xbf880002, 0xf8000941, 0x00000000, 0xbefe04c1,
    0x7da8120c, 0xbf880009, 0xe0382000, 0x80021005, 0xe0382010, 0x80021405, 0xbf8c3f70, 0xf80008cf,
    0x13121110, 0xf800020f, 0x17161514, 0xbf810000,
};

constexpr std::array<std::uint32_t, 24> AllocationFromWaveId{
    0x93ebff03, 0x00080008, 0x9381ff03, 0x00080010, 0x8f6a8c01, 0x8700ff03, 0x000000ff, 0x887c6a00, 0xbf900009, 0x81ea6bc0, 0x90fe6ac1, 0xf8000941,
    0x00000000, 0x81ea00c0, 0xbf8cff0f, 0x90fe6ac1, 0x7e020d05, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020201, 0xf800020f, 0x01010101, 0xbf810000,
};

constexpr std::array<std::uint32_t, 21> NoAllocation{
    0x93ebff03, 0x00080008, 0x8f6a8c6b, 0x8700ff03, 0x000000ff, 0x887c6a00, 0x81ea6bc0, 0x90fe6ac1, 0xf8000941, 0x00000000, 0x81ea00c0, 0xbf8cff0f,
    0x90fe6ac1, 0x7e020d05, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020201, 0xf800020f, 0x01010101, 0xbf810000,
};

constexpr std::array<std::uint32_t, 22> AllocationOfVerticesOnly{
    0x93ebff03, 0x00080008, 0x8f6a8c6b, 0x8700ff03, 0x000000ff, 0xbefc0300, 0xbf900009, 0x81ea6bc0, 0x90fe6ac1, 0xf8000941, 0x00000000, 0x81ea00c0,
    0xbf8cff0f, 0x90fe6ac1, 0x7e020d05, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020201, 0xf800020f, 0x01010101, 0xbf810000,
};

constexpr std::array<std::uint32_t, 23> PrimitiveExportFromEveryLane{
    0x93ebff03, 0x00080008, 0x8f6a8c6b, 0x8700ff03, 0x000000ff, 0x887c6a00, 0xbf900009, 0x81ea6bc0, 0x90fe6ac1, 0xbefe04c1, 0xf8000941, 0x00000000,
    0x81ea00c0, 0xbf8cff0f, 0x90fe6ac1, 0x7e020d05, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020201, 0xf800020f, 0x01010101, 0xbf810000,
};

constexpr std::array<std::uint32_t, 26> OrderedIdBranch{
    0x93ebff03, 0x00080008, 0x8f6a8c6b, 0x8700ff03, 0x000000ff, 0x887c6a00, 0xbf900009, 0x81ea6bc0, 0x90fe6ac1, 0xf8000941, 0x00000000, 0x81ea00c0,
    0xbf8cff0f, 0x90fe6ac1, 0x8701ff02, 0x00000fff, 0xbf078001, 0xbf850007, 0x7e020d05, 0x7e040280, 0x7e0602f2, 0xf80008cf, 0x03020201, 0xf800020f,
    0x01010101, 0xbf810000,
};

int failures = 0;

void Expect(const char* name, std::span<const std::uint32_t> code, const ShaderRecompiler::NggSubgroupLimits& limits, bool perVertex) {
    const auto reason = ShaderRecompiler::NggPassthroughSubgroupDependence(code, 4u, limits);
    if (reason.has_value() == perVertex) {
        std::fprintf(stderr, "%s: expected %s, got %s\n", name, perVertex ? "the vertex path" : "the mesh path", reason ? reason->c_str() : "the vertex path");
        ++failures;
    }
}

}

int main() {
    const ShaderRecompiler::NggSubgroupLimits wave64{64u, 64u, 64u};
    Expect("vertex count gating from MERGED_WAVE_INFO", PerVertex, wave64, true);
    Expect("vertex count gating from shifted MERGED_WAVE_INFO", CountFromShiftedWaveInfo, wave64, true);
    Expect("wave32 subgroups", PerVertex, {32u, 32u, 32u}, true);
    Expect("two-wave subgroups that every wave allocates", PerVertex, {64u, 128u, 128u}, false);
    Expect("wave32 subgroups of two waves", PerVertex, {32u, 64u, 64u}, false);
    Expect("an allocation from the GS wave id", AllocationFromWaveId, wave64, false);
    Expect("no allocation request", NoAllocation, wave64, false);
    Expect("an allocation of vertices only", AllocationOfVerticesOnly, wave64, false);
    Expect("a primitive export from every lane", PrimitiveExportFromEveryLane, wave64, false);
    Expect("a branch on the ordered wave id", OrderedIdBranch, wave64, false);
    Expect("a culled primitive export", Culled, wave64, false);
    Expect("a GS input VGPR exported", SubgroupRegister, wave64, false);
    Expect("LDS traffic", SharedMemory, wave64, false);
    Expect("the vertex count exported as data", CountAsData, wave64, false);
    Expect("the lane index exported as data", LaneAsData, wave64, false);
    Expect("GS_TG_INFO counts with lane-dependent branches", GroupCountsAndLaneBranches, wave64, false);
    Expect("no subgroup limits", PerVertex, {64u, 0u, 0u}, false);
    if (failures != 0) return 1;
    std::puts("NGG passthrough routing tests passed");
    return 0;
}
