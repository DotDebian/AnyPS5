#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_SRTWALKER_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_SRTWALKER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

using SrtMemoryReader = bool (*)(void* userData, std::uint64_t address, std::uint32_t* value);
// Whether the dword at `address` is mapped readable guest memory (SrtRuntime::isReadable).
using SrtMemoryProbe = bool (*)(void* userData, std::uint64_t address);

// The guest addresses a walk dereferenced (Detail::Evaluator::EvaluateRawRead): the leaf read of
// each pure flat slot (IrResourcePlan::pureFlatSlots) as (flat offset, address), set by the
// flattened loop through `leaf`/`leafSlot` around that slot's evaluation, and every other raw read
// address.
struct SrtReadTrace {
    const IrValue* leaf = nullptr;
    std::uint32_t leafSlot = 0;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> leaves;
    std::vector<std::uint64_t> otherReads;
};

struct SrtRuntime {
    std::span<const std::uint32_t> userData;
    std::uint64_t shaderBase = 0;
    SrtMemoryReader readMemory = nullptr;
    void* userContext = nullptr;
    SrtMemoryReader readSpecializationMemory = nullptr;
    SrtReadTrace* readTrace = nullptr;
    // Set by a runtime over live guest memory. A flattened SRT slot (IrResourcePlan::srtReads) is a
    // scalar load the plan hoisted out of the program, wherever the program executes it: a
    // conditional one (SrtRead::conditional) is often under a branch taken only when its address
    // is valid (a BVH walk's instance pointer is null when the scene has no ray-traced instances).
    // Such a slot whose dword is unmapped reads as zero: the program cannot execute that load (it
    // would take a GPU page fault), so it cannot observe the value. An unconditional slot and the
    // descriptor sources are not covered; an unmapped read there still fails the walk.
    SrtMemoryProbe isReadable = nullptr;
};

enum class RuntimeValueType {
    Any,
    Integer
};

class SrtWalker {
public:
    void BuildPlan(IrProgram& program) const;
    [[nodiscard]] bool ValidateRuntimeValue(const IrResourcePlan& program, const IrValue* value, RuntimeValueType type = RuntimeValueType::Any) const;
    void EvaluateUniformValues(const IrResourcePlan& program, std::span<IrValue* const> values, const SrtRuntime& runtime, std::span<std::uint32_t> results) const;
    void EvaluateDescriptorSource(const IrResourcePlan& program, std::uint32_t source, const SrtRuntime& runtime, DescriptorValue& result) const;
    void EvaluateDescriptorSources(const IrResourcePlan& program, std::span<const std::uint32_t> sources, const SrtRuntime& runtime, std::vector<DescriptorValue>& results) const;
    void EvaluateRuntimeSources(const IrResourcePlan& program, std::span<const std::uint32_t> sources, const SrtRuntime& runtime, std::vector<DescriptorValue>& results, std::vector<std::uint32_t>& flat, std::span<const std::uint8_t> cleanFlatSlots, std::vector<std::uint8_t>& activeSources) const;
    void Walk(const IrResourcePlan& program, const SrtRuntime& runtime, std::vector<std::uint32_t>& flat) const;

};

}

#endif
