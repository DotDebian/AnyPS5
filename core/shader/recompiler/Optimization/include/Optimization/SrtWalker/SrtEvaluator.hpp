#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTEVALUATOR_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_SRTWALKER_SRTEVALUATOR_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/SrtWalker.hpp"

#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler::Detail {

// The storage of a thread's evaluators, kept between walks: the value tables and the lists (the
// values being visited, the conditional reads of a walk). A capture builds two evaluators and one
// more for every ReadFirstLane it meets, and each grew its table from nothing, allocation after
// allocation (a fifth of a draw's preparation was the heap behind these tables). A table or a
// list taken here is given back by the evaluator's destructor; what it holds is the taker's to
// set (a table comes back with stale slots, a list comes back empty).
struct EvaluatedSlot {
    const IrValue* key = nullptr;
    std::uint64_t value = 0;
};
[[nodiscard]] std::vector<EvaluatedSlot> TakeEvaluatorTable();
void ReturnEvaluatorTable(std::vector<EvaluatedSlot>&& table) noexcept;
[[nodiscard]] std::vector<const IrValue*> TakeEvaluatorList();
void ReturnEvaluatorList(std::vector<const IrValue*>&& list) noexcept;

class EvaluatedValues {
public:
    EvaluatedValues() = default;
    EvaluatedValues(const EvaluatedValues&) = delete;
    EvaluatedValues& operator=(const EvaluatedValues&) = delete;
    ~EvaluatedValues() {
        if (_slots.capacity() != 0u) {
            ReturnEvaluatorTable(std::move(_slots));
        }
    }
    bool Find(const IrValue* key, std::uint64_t& value) const {
        if (_slots.empty()) {
            return false;
        }
        for (std::size_t index = Home(key);; index = (index + 1u) & (_slots.size() - 1u)) {
            const auto& slot = _slots[index];
            if (slot.key == key) {
                value = slot.value;
                return true;
            }
            if (slot.key == nullptr) {
                return false;
            }
        }
    }
    void Insert(const IrValue* key, std::uint64_t value) {
        if ((_count + 1u) * 2u > _slots.size()) {
            Grow();
        }
        for (std::size_t index = Home(key);; index = (index + 1u) & (_slots.size() - 1u)) {
            auto& slot = _slots[index];
            if (slot.key == key) {
                return;
            }
            if (slot.key == nullptr) {
                slot = {key, value};
                ++_count;
                return;
            }
        }
    }

private:
    using Slot = EvaluatedSlot;
    std::size_t Home(const IrValue* key) const {
        return static_cast<std::size_t>((reinterpret_cast<std::uintptr_t>(key) >> 4u) * 0x9e3779b97f4a7c15ull >> 32u) & (_slots.size() - 1u);
    }
    void Grow() {
        // The next table comes from the thread's storage (no allocation once it has held a table
        // of this size) and the previous one goes back to it.
        auto previous = TakeEvaluatorTable();
        previous.assign(_slots.empty() ? 64u : _slots.size() * 2u, Slot{});
        previous.swap(_slots);
        _count = 0;
        for (const auto& slot : previous) {
            if (slot.key != nullptr) {
                Insert(slot.key, slot.value);
            }
        }
        if (previous.capacity() != 0u) {
            ReturnEvaluatorTable(std::move(previous));
        }
    }
    std::vector<Slot> _slots;
    std::size_t _count = 0;
};

class Evaluator {
public:
    Evaluator(const IrResourcePlan& program, const SrtRuntime& runtime, std::span<const std::uint8_t> cleanFlatSlots = {}, Evaluator* cleanEvaluator = nullptr, IrValue* activeMask = nullptr) : _program(program), _runtime(runtime), _cleanFlatSlots(cleanFlatSlots), _cleanEvaluator(cleanEvaluator), _activeMask(activeMask != nullptr ? activeMask->Resolve() : nullptr) {}
    Evaluator(const Evaluator&) = delete;
    Evaluator& operator=(const Evaluator&) = delete;
    // Gives the lists back to the thread's storage (the table goes with _cache).
    ~Evaluator();

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
    EvaluatedValues _cache;
    // From the thread's storage, taken at the first push.
    std::vector<const IrValue*> _visiting;
    bool IsConditionalSlotRead(const IrValue& inst);

    bool _unmappedAsZero = false;
    bool _conditionalReadsBuilt = false;
    // The resolved values of the conditional slots, sorted (from the thread's storage).
    std::vector<const IrValue*> _conditionalReads;
    std::uint32_t _unmappedReads = 0;
};

}

#endif
