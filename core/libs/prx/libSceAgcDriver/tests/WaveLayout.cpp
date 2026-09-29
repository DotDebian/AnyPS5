// agc_driver_wave_tests: hand-assembled wave64 compute programs recompiled in both wave64 layouts
// of a 32-wide host (two guest lanes per invocation, and one lane per invocation with the wave's
// halves exchanging through workgroup memory, see SpirvWaveExchange.hpp) and run on the GPU; each
// layout's results must match a model of the guest wave, and each other. Run by hand.
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;
using ShaderRecompiler::WaveLayout;

constexpr std::uint32_t Results = 16;
constexpr std::uint32_t Groups = 2048;
constexpr std::uint32_t MaxThreads = 256;
alignas(256) std::array<std::uint32_t, Groups * MaxThreads> Input{};
alignas(256) std::array<std::uint32_t, Groups * MaxThreads * Results> Output{};

// Per thread tid = group * N + local (N threads per group, s8; the group id in s12; local = y * X + x
// from the thread's x and y ids in v0 and v1, the group's width X in s9), with
// in = Input[tid]; the program stores 16 results at Output[tid * 16 + j]:
//    0, 1  the wave's VCC after v_cmp_lt_u32 50, in (s_mov_b32 of vcc_lo / vcc_hi)
//    2     s_bcnt1_i32_b64 of it
//    3     v_readfirstlane_b32 of in under EXEC = that VCC (s_and_saveexec_b64; lane 0 if empty)
//    4, 5  v_readlane_b32 of in at lanes 37 and 5
//    6     v_readlane_b32 of in at lane (result 3 & 63)
//    7     v_mbcnt_lo/hi of the VCC words
//    8, 9  s_and_saveexec_b64 vcc / s_cbranch_execz over "s22 = 1, v9 = in + 1": s22, v9 (0 else)
//   10     after v_cmp_eq_u32 7, in / s_cbranch_vccz over "s23 = 1": s23
//   11     a loop that runs while any lane's (in & 15) counter is positive (v_cmpx_lt_u32 /
//          s_cbranch_execz out), counting its iterations in s26
//   12     v_writelane_b32 of result 5 into lane 40 of a zeroed VGPR
//   13     v_mov_b32_dpp row_shr:1 bound_ctrl:1 of in (0 in each row's lane 0)
//   14     ds_bpermute_b32 of in from lane ^ 1
//   15     LDS: each lane writes in at its thread's dword, then reads lane (lane + 33) & 63's
//          (a wave's lanes see each other's LDS writes without a barrier)
//
//   v_mad_u32_u24 v0, v1, s9, v0 / s_mul_i32 s13, s12, s8 / v_add_nc_u32 v1, s13, v0 / buffer_load_dword v2, v1, s[0:3], 0 idxen
//   v_lshlrev_b32 v3, 4, v1 / s_mov_b32 s22, 0 / s_mov_b32 s23, 0 / s_mov_b32 s26, 0
//   v_mov_b32 v9, 0 / v_mov_b32 v13, 0 / v_mov_b32 v14, 0x1234 / s_waitcnt vmcnt(0)
//   v_cmp_lt_u32 vcc, 50, v2 / s_mov_b32 s10, vcc_lo / s_mov_b32 s11, vcc_hi / v_mov_b32 v4, s10
//   v_mov_b32 v5, s11 / buffer_store_dword v4 (offset:0), v5 (offset:4)
//   s_bcnt1_i32_b64 s14, vcc / v_mov_b32 v6, s14 / buffer_store_dword v6 offset:8
//   s_and_saveexec_b64 s[16:17], vcc / v_readfirstlane_b32 s18, v2 / s_mov_b64 exec, s[16:17]
//   v_mov_b32 v7, s18 / buffer_store_dword v7 offset:12
//   v_readlane_b32 s19, v2, 37 / v_readlane_b32 s20, v2, 5 / (stored at offset:16, offset:20)
//   s_and_b32 s21, s18, 63 / v_readlane_b32 s21, v2, s21 / (offset:24)
//   v_mbcnt_lo_u32_b32 v8, s10, 0 / v_mbcnt_hi_u32_b32 v8, s11, v8 / (offset:28)
//   s_and_saveexec_b64 s[24:25], vcc / s_cbranch_execz skip1 / s_mov_b32 s22, 1
//   v_add_nc_u32 v9, 1, v2
// skip1:
//   s_or_b64 exec, exec, s[24:25] / (s22 at offset:32, v9 at offset:36)
//   v_cmp_eq_u32 vcc, 7, v2 / s_cbranch_vccz skip2 / s_mov_b32 s23, 1
// skip2:
//   (s23 at offset:40) / v_and_b32 v11, 15, v2 / s_mov_b64 s[28:29], exec
// loop:
//   v_cmpx_lt_u32 0, v11 / s_cbranch_execz done / v_add_nc_u32 v11, -1, v11 / s_add_u32 s26, s26, 1
//   s_branch loop
// done:
//   s_mov_b64 exec, s[28:29] / (s26 at offset:44)
//   v_writelane_b32 v13, s20, 40 / (offset:48)
//   v_mov_b32_dpp v14, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1 / (offset:52)
//   v_mbcnt_lo_u32_b32 v15, -1, 0 / v_mbcnt_hi_u32_b32 v15, -1, v15 / v_xor_b32 v16, 1, v15
//   v_lshlrev_b32 v16, 2, v16 / ds_bpermute_b32 v17, v16, v2 / s_waitcnt lgkmcnt(0) / (offset:56)
//   v_lshlrev_b32 v18, 2, v0 / ds_write_b32 v18, v2 / v_add_nc_u32 v19, 33, v0
//   v_and_b32 v19, 63, v19 / v_and_b32 v20, -64, v0 / v_or_b32 v19, v19, v20
//   v_lshlrev_b32 v19, 2, v19 / s_waitcnt lgkmcnt(0) / ds_read_b32 v21, v19
//   s_waitcnt lgkmcnt(0) / (offset:60) / s_endpgm
// (llvm-mc -triple=amdgcn -mcpu=gfx1030 -mattr=+wavefrontsize64; stores are
// buffer_store_dword vN, v3, s[4:7], 0 idxen offset:4j.)
alignas(256) constexpr std::array<std::uint32_t, 118> WaveOpsCode{
    0xd5430000, 0x04001301, 0x930d080c, 0x4a02000d, 0xe0302000, 0x80000201, 0x34060284, 0xbe960380,
    0xbe970380, 0xbe9a0380, 0x7e120280, 0x7e1a0280, 0x7e1c02ff, 0x00001234, 0xbf8c3f70, 0x7d8204b2,
    0xbe8a036a, 0xbe8b036b, 0x7e08020a, 0x7e0a020b, 0xe0702000, 0x80010403, 0xe0702004, 0x80010503,
    0xbe8e106a, 0x7e0c020e, 0xe0702008, 0x80010603, 0xbe90246a, 0x7e240502, 0xbefe0410, 0x7e0e0212,
    0xe070200c, 0x80010703, 0xd7600013, 0x00014b02, 0xd7600014, 0x00010b02, 0x7e080213, 0x7e0a0214,
    0xe0702010, 0x80010403, 0xe0702014, 0x80010503, 0x8715bf12, 0xd7600015, 0x00002b02, 0x7e080215,
    0xe0702018, 0x80010403, 0xd7650008, 0x0001000a, 0xd7660008, 0x0002100b, 0xe070201c, 0x80010803,
    0xbe98246a, 0xbf880002, 0xbe960381, 0x4a120481, 0x88fe187e, 0x7e080216, 0xe0702020, 0x80010403,
    0xe0702024, 0x80010903, 0x7d840487, 0xbf860001, 0xbe970381, 0x7e080217, 0xe0702028, 0x80010403,
    0x3616048f, 0xbe9c047e, 0x7da21680, 0xbf880003, 0x4a1616c1, 0x801a811a, 0xbf82fffb, 0xbefe041c,
    0x7e08021a, 0xe070202c, 0x80010403, 0xd761000d, 0x00015014, 0xe0702030, 0x80010d03, 0x7e1c02fa,
    0xff091102, 0xe0702034, 0x80010e03, 0xd765000f, 0x000100c1, 0xd766000f, 0x00021ec1, 0x3a201e81,
    0x34202082, 0xdacc0000, 0x11000210, 0xbf8cc07f, 0xe0702038, 0x80011103, 0x34240082, 0xd8340000,
    0x00000212, 0x4a2600a1, 0x362626bf, 0x362800ff, 0xffffffc0, 0x38262913, 0x34262682, 0xbf8cc07f,
    0xd8d80000, 0x15000013, 0xbf8cc07f, 0xe070203c, 0x80011503, 0xbf810000,
};

// The guest wave's results for thread `local` of group `group` (N threads per group); nullopt
// where the guest leaves them undefined (a lane the group does not have).
std::array<std::optional<std::uint32_t>, Results> Expected(std::uint32_t threads, std::uint32_t group, std::uint32_t local) {
    const std::uint32_t waveBase = local & ~63u;
    const auto exists = [&](std::uint32_t lane) { return waveBase + lane < threads; };
    const auto in = [&](std::uint32_t lane) { return Input[group * threads + waveBase + lane]; };
    const std::uint32_t lane = local & 63u;
    std::uint64_t vcc = 0;
    bool seven = false;
    std::uint32_t loops = 0;
    for (std::uint32_t l = 0; l < 64; ++l) {
        if (!exists(l)) continue;
        if (in(l) > 50u) vcc |= 1ull << l;
        seven |= in(l) == 7u;
        loops = std::max(loops, in(l) & 15u);
    }
    std::array<std::optional<std::uint32_t>, Results> result{};
    result[0] = static_cast<std::uint32_t>(vcc);
    result[1] = static_cast<std::uint32_t>(vcc >> 32u);
    result[2] = static_cast<std::uint32_t>(std::popcount(vcc));
    // With no lane active v_readfirstlane_b32 reads lane 0.
    const auto first = in(vcc != 0 ? static_cast<std::uint32_t>(std::countr_zero(vcc)) : 0u);
    result[3] = first;
    if (exists(first & 63u)) result[6] = in(first & 63u);
    if (exists(37)) result[4] = in(37);
    if (exists(5)) result[5] = in(5);
    result[7] = static_cast<std::uint32_t>(std::popcount(vcc & ((1ull << lane) - 1ull)));
    result[8] = vcc != 0 ? 1u : 0u;
    result[9] = in(lane) > 50u ? in(lane) + 1u : 0u;
    result[10] = seven ? 1u : 0u;
    result[11] = loops;
    if (exists(5)) result[12] = lane == 40u ? in(5) : 0u;
    result[13] = lane % 16u == 0u ? 0u : in(lane - 1u);
    if (exists(lane ^ 1u)) result[14] = in(lane ^ 1u);
    if (exists((lane + 33u) & 63u)) result[15] = in((lane + 33u) & 63u);
    return result;
}

// Wave patterns by the wave's index in the dispatch: 0 random, 1 no lane above 50 (the EXECZ
// branch skips, the loop still runs), 2 only lanes 32-63 above 50 (the first active lane is in
// the high half), 3 only lanes 0-31, with a 7 somewhere.
void FillInput(std::uint32_t threads) {
    std::uint64_t seed = 0x9e3779b97f4a7c15ull ^ threads;
    const auto next = [&] { seed ^= seed << 13u; seed ^= seed >> 7u; seed ^= seed << 17u; return static_cast<std::uint32_t>(seed); };
    std::uint32_t wave = 0;
    for (std::uint32_t group = 0; group < Groups; ++group) {
        for (std::uint32_t base = 0; base < threads; base += 64u, ++wave) {
            for (std::uint32_t lane = 0; lane < 64u && base + lane < threads; ++lane) {
                auto& value = Input[group * threads + base + lane];
                switch (wave % 4u) {
                case 0: value = next() % 100u; break;
                case 1: value = next() % 51u; break;
                case 2: value = lane >= 32u ? 51u + next() % 49u : next() % 51u; break;
                default: value = lane < 32u ? 51u + next() % 49u : (lane == 45u ? 7u : next() % 51u); break;
                }
            }
        }
    }
}

// A structured buffer of `count` dwords (stride 4, 32_FLOAT, DST_SEL XYZW).
std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (4u << 16u), count, 0x01016facu};
}

constexpr std::uint32_t Sentinel = 0xdeadbeefu;

std::vector<std::uint32_t> Run(AgcDriver::VulkanDevice& device, std::uint32_t width, std::uint32_t threads, WaveLayout layout) {
    std::fill(Output.begin(), Output.end(), Sentinel);
    std::vector<std::uint32_t> userData(12, 0u);
    const auto input = BufferDescriptor(Input.data(), Groups * threads);
    const auto output = BufferDescriptor(Output.data(), Groups * threads * Results);
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    userData[8] = threads;
    userData[9] = width;
    const std::span<const std::uint32_t> code(WaveOpsCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{width, threads / width, 1}, threads, {true, false, false}, false, 2};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.waveLayout = layout;
    Require(ShaderRecompiler::WaveLayoutFor(request) == layout, "wave layout: the request does not compile to the layout under test");
    const auto result = ShaderRecompiler::Recompile(request);
    // APS5_WAVE_TEST_SPV=<prefix>: writes each module to <prefix><threads>_<layout>.spv (for spirv-val).
    if (const char* prefix = std::getenv("APS5_WAVE_TEST_SPV")) {
        const auto path = std::string(prefix) + std::to_string(width) + "x" + std::to_string(threads / width) + "_" + (layout == WaveLayout::SingleLane ? "single" : "two") + ".spv";
        if (std::FILE* file = std::fopen(path.c_str(), "wb")) {
            std::fwrite(result.spirv.data(), sizeof(std::uint32_t), result.spirv.size(), file);
            std::fclose(file);
        }
    }
    device.Dispatch(result, Groups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
    device.WaitIdle();
    return {Output.begin(), Output.begin() + Groups * threads * Results};
}

const char* Name(WaveLayout layout) {
    return layout == WaveLayout::SingleLane ? "single-lane" : "two-lane";
}

void Check(std::uint32_t threads, WaveLayout layout, const std::vector<std::uint32_t>& output) {
    for (std::uint32_t group = 0; group < Groups; ++group) {
        for (std::uint32_t local = 0; local < threads; ++local) {
            const auto expected = Expected(threads, group, local);
            for (std::uint32_t j = 0; j < Results; ++j) {
                if (!expected[j]) continue;
                const auto actual = output[(group * threads + local) * Results + j];
                Require(actual == *expected[j], std::string("wave ops, ") + std::to_string(threads) + " threads, " + Name(layout) + ": group " + std::to_string(group) + " thread " + std::to_string(local) + " result " + std::to_string(j) + " is " + std::to_string(actual) + ", expected " + std::to_string(*expected[j]));
            }
        }
    }
}

}

int main() {
    try {
        setenv("APS5_NO_SHADER_DISK_CACHE", "1", 1);
        AgcDriver::VulkanDevice device;
        Require(device.Target().subgroupSize == 32u, "wave layout tests need a device with 32-wide subgroups");
        // One wave per group (the halves meet at workgroup barriers), several waves (they pair
        // through counters), a partial last wave (lanes 32-63 of the second wave missing), and
        // two-dimensional groups (a wave's lanes are consecutive invocation indices).
        for (const auto [width, threads] : {std::pair{64u, 64u}, std::pair{128u, 128u}, std::pair{96u, 96u}, std::pair{256u, 256u}, std::pair{160u, 160u}, std::pair{8u, 64u}, std::pair{16u, 256u}, std::pair{16u, 160u}}) {
            FillInput(threads);
            const auto two = Run(device, width, threads, WaveLayout::TwoLane);
            Check(threads, WaveLayout::TwoLane, two);
            const auto single = Run(device, width, threads, WaveLayout::SingleLane);
            Check(threads, WaveLayout::SingleLane, single);
            std::printf("wave ops, %ux%u threads: both layouts match the guest wave\n", width, threads / width);
        }
        std::puts("wave layout tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
