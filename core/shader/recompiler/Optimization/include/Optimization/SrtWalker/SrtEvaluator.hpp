#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTEVALUATOR_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTEVALUATOR_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/SrtWalker.hpp"

#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler::Detail {

class Evaluator {
public:
    Evaluator(const IrResourcePlan& program, const SrtRuntime& runtime, std::span<const std::uint8_t> cleanFlatSlots = {}, Evaluator* cleanEvaluator = nullptr, IrValue* activeMask = nullptr) : _program(program), _runtime(runtime), _cleanFlatSlots(cleanFlatSlots), _cleanEvaluator(cleanEvaluator), _activeMask(activeMask != nullptr ? activeMask->Resolve() : nullptr) {}

    bool Evaluate(IrValue* value, std::uint32_t& result);
    bool EvaluateWide(IrValue* raw, std::uint64_t& result);
    // From now on the raw read of a conditional flattened slot (SrtRead::conditional) whose dword
    // is unmapped (SrtRuntime::isReadable) evaluates to zero. For the flattened slots, which are
    // evaluated after every descriptor source.
    void ReadUnmappedAsZero() { _unmappedAsZero = true; }
    // The raw reads that found their dword unmapped and read zero.
    [[nodiscard]] std::uint32_t UnmappedReads() const { return _unmappedReads; }

private:
    static float Float32(std::uint64_t bits);
    static std::uint64_t Float32Bits(float value);

    bool Arg(IrValue& inst, std::size_t index, std::uint64_t& result);
    bool EvaluatePhi(IrValue& inst, std::uint64_t& result);
    bool EvaluateExtract(IrValue& inst, std::uint64_t& result);
    bool EvaluateRawRead(IrValue& inst, std::uint64_t& result);
    bool EvaluateInst(IrValue& inst, std::uint64_t& result);

    const IrResourcePlan& _program;
    const SrtRuntime& _runtime;
    std::span<const std::uint8_t> _cleanFlatSlots;
    Evaluator* _cleanEvaluator = nullptr;
    IrValue* _activeMask = nullptr;
    std::unordered_map<IrValue*, std::uint64_t> _cache;
    std::vector<IrValue*> _visiting;
    bool IsConditionalSlotRead(const IrValue& inst);

    bool _unmappedAsZero = false;
    bool _conditionalReadsBuilt = false;
    std::unordered_set<const IrValue*> _conditionalReads;
    std::uint32_t _unmappedReads = 0;
};

}

#endif
