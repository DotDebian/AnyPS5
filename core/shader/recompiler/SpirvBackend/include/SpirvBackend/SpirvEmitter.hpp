#ifndef CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVEMITTER_HPP
#define CORE_SHADER_RECOMPILIER_SPIRVBACKEND_INCLUDE_SPIRVBACKEND_SPIRVEMITTER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/BindingAllocator.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include <cstdint>
#include <vector>
#include <span>
#include <stdexcept>
#include <string_view>

namespace ShaderRecompiler {

struct SpirvTargetOptions {
    std::uint32_t vulkanVersion;
    std::uint32_t spirvVersion;
    std::uint32_t subgroupSize;
    std::uint32_t bdaAbiVersion;
    std::span<const std::uint32_t> supportedCapabilities;
    std::span<const std::string_view> supportedExtensions;
    bool nonConstantImageOffsets = false;
    // The guest program's address, for the loop guard's report (APS5_LOOP_GUARD).
    std::uint64_t codeAddress = 0;
    std::uint32_t workgroupReserveBytes = 0;
};

// What Emit throws for a SingleLane program whose wave halves could take a branch apart (see
// SpirvWaveExchange.hpp): the caller compiles it two lanes per invocation instead.
class SingleLaneNotExact : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class SpirvEmitter {
public:
    [[nodiscard]] std::vector<std::uint32_t> Emit(const IrProgram& program, const BindingAllocationResult& bindings, const SpirvTargetOptions& target) const;
    [[nodiscard]] std::vector<std::uint32_t> Emit(const IrProgram& program, const ShaderStageInputInfo& inputInfo, const BindingAllocationResult& bindings, const SpirvTargetOptions& target) const;

};

}

#endif
