#ifndef CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVANALYSIS_HPP
#define CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVANALYSIS_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler {

struct SpirvRequirements {
    bool subgroupBallot = false;
    bool subgroupShuffle = false;
    bool subgroupLocalInvocationId = false;
    bool computeDerivatives = false;
    bool imageGatherExtended = false;
    bool functionLds = false;
    // With functionLds: the dwords the per-invocation LDS array needs (see FunctionLdsDwords).
    std::uint32_t functionLdsDwords = 0;
    // With functionLds: when not empty, every LDS access's byte address with the lane term taken
    // out (see FunctionLdsLaneAddresses); the array is indexed by these constant addresses.
    std::unordered_map<const IrValue*, std::uint32_t> functionLdsAddresses;
    bool ldsLock = false;
    bool functionScratch = false;
    bool pixelValidMask = false;
    bool bufferInt64Atomics = false;
    bool sharedInt64Atomics = false;
    bool float64 = false;
    bool coherentBuffers = false;
    std::vector<std::uint32_t> capabilities;
    std::vector<std::string> extensions;
};

[[nodiscard]] SpirvRequirements AnalyzeProgramRequirements(const IrProgram& program);
// The per-invocation LDS array of a stage without workgroup memory (vertex, fragment, ...) holds
// `FunctionLdsDwordLimit` dwords unless every LDS access's address has a provable upper bound
// (lane id, constants, adds, shifts, masks, minimums, selects and phis of those): then just the
// dwords those accesses can reach, rounded up to 64. No access reaches past the smaller array, so
// the bounds checks and results are as with the full one; the array stays out of NVIDIA local
// memory, where 32 KiB per invocation hung the GPU (Xid 31 / Xid 109).
inline constexpr std::uint32_t FunctionLdsDwordLimit = 8192u;
[[nodiscard]] std::uint32_t FunctionLdsDwords(const IrProgram& program);
// When every LDS access of such a stage has the address `lane * stride + constant` (lane id,
// constants, adds, multiplies and left shifts of those), with one stride that is a multiple of 4
// for all of them and no wrap past 2^32: each access's `constant` plus its instruction offset, by
// access. Otherwise (or with no lane term at all) empty. The lane id is the same for every access
// of an invocation, so taking `lane * stride` out of every address maps the addresses one to one
// and keeps which accesses alias; the per-invocation array then sees only constant indices.
[[nodiscard]] std::unordered_map<const IrValue*, std::uint32_t> FunctionLdsLaneAddresses(const IrProgram& program);
[[nodiscard]] std::unordered_set<const IrValue*> WaveUniformValues(const IrProgram& program);
[[nodiscard]] bool IsWaveMaskBranch(BranchCondition condition);

}

#endif
