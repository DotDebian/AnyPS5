#ifndef CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVANALYSIS_HPP
#define CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVANALYSIS_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include <cstdint>
#include <string>
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
    bool functionScratch = false;
    bool pixelValidMask = false;
    bool bufferInt64Atomics = false;
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
[[nodiscard]] std::unordered_set<const IrValue*> WaveUniformValues(const IrProgram& program);

}

#endif
