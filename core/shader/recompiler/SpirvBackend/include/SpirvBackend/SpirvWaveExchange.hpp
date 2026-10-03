// WaveLayout::SingleLane: a guest wave64 at one guest lane per invocation on 32-wide host subgroups.
//
// Guest lane L of wave W is the invocation with LocalInvocationIndex 64 W + L, so the wave's lanes
// 0-31 and 32-63 are two host subgroups (its halves; subgroups are formed from consecutive
// invocation indices, which the two-lane layout relies on too). Lane-local work needs nothing
// more, and an operation within 32 lanes of one half (DPP rows, permlane16, ds_swizzle,
// ds_bpermute) stays a subgroup operation of that half. What spans the wave is computed by each
// half over its own lanes, published to workgroup memory, and combined with the other half's
// value after a synchronization of the two halves:
//  - ballots (the EXEC, VCC and SGPR masks v_cmp and friends write) and the branches on them
//    (s_cbranch_execz / vccz / execnz / vccnz), which makes the wave's control flow uniform over
//    both halves, as the two-lane layout's branches are uniform over its subgroup;
//  - v_readfirstlane, v_readlane and the lane a ds_append / ds_consume performs its atomic in.
// Scalar values then follow from wave-uniform inputs in every lane of the wave.
//
// The halves synchronize with a workgroup barrier when the workgroup is one wave (its control flow
// is the wave's, uniform over the workgroup). With several waves the halves of one wave pair up
// through arrival counters in workgroup memory instead (the waves' control flows differ, so a
// workgroup barrier may not be reached by all): each half publishes its generation with a release
// store and waits for the other half's with acquire loads. The two halves of a wave run the same
// sequence of exchanges, so neither waits for a generation the other never reaches.
//
// Two generations of slots alternate: a half writes generation g + 2's slot only after the other
// half arrived at g + 1, so after it read generation g.
#ifndef CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVWAVEEXCHANGE_HPP
#define CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVWAVEEXCHANGE_HPP

#include "SpirvBackend/SpirvEmitterState.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler {

// The dwords one exchange carries from each half at most.
inline constexpr std::uint32_t WaveExchangeDwords = 2u;

// Sets up the SingleLane layout for a wave64 compute program on 32-wide subgroups whose workgroup
// has more than 32 threads (state.splitWave): its capabilities, builtins and variables. Before
// the module header.
void PrepareWaveExchange(SpirvEmitterState& state, const ShaderWorkgroupInputInfo& workgroup);

// The entry block's part: the invocation's wave, half and slots, and the counters' reset (with a
// workgroup barrier when the halves pair through counters). After the function's variables.
void EmitWaveExchangeEntry(SpirvEmitterState& state);

// Publishes up to WaveExchangeDwords values that are uniform over this half and returns, for each,
// {lanes 0-31's value, lanes 32-63's value}: this half's and the other's. A half the workgroup does
// not have (a partial last wave) reads as zero. Must be reached by both halves of the wave alike.
std::vector<std::array<std::uint32_t, 2>> EmitWaveExchange(SpirvEmitterState& state, std::span<const std::uint32_t> values);

// Orders the wave's workgroup memory accesses before it against those after it in both halves.
void EmitWaveSync(SpirvEmitterState& state);

// The guest lane (0-63) of the invocation, and whether it is in the wave's lanes 0-31.
std::uint32_t EmitSplitGuestLane(SpirvEmitterState& state);
std::uint32_t EmitSplitFirstHalf(SpirvEmitterState& state);

// This half's ballot of `predicate` placed at the half's word ({x, 0} or {0, x}, as uvec4), for
// testing lanes of this half; not the wave's ballot.
std::uint32_t EmitSplitHalfBallot(SpirvEmitterState& state, std::uint32_t predicate);

// This half's 32-bit ballot word of `predicate`.
std::uint32_t EmitSplitBallotWord(SpirvEmitterState& state, std::uint32_t predicate);

// The conditional branches of the program not on EXEC or VCC (SCC, scalar instruction and
// structurizer conditions) whose condition may differ between the lanes of a wave: a SingleLane
// wave's halves could take them apart, so such a program cannot take that layout.
std::vector<const BlockInfo*> LaneVaryingScalarBranches(const IrProgram& program);

}

#endif
