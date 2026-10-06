#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <bit>
#include <cstdlib>

namespace ShaderRecompiler {

bool BdaNoSearchExperiment();

namespace {

void ReturnBdaZeroIf(SpirvEmitterState& state, std::uint32_t condition) {
    const auto failed = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
    EmitLabel(state, failed);
    state.module.AddFunction(spv::OpReturnValue, BdaConstant(state, 0u));
    EmitLabel(state, next);
}

// (u64 address, u32 bytes, u32 instruction) -> u64 device address of the range holding all the bytes,
// or 0. With the cache (state.bdaCacheBegin, the default) the function first tries the last range a
// lookup of this invocation resolved: a guest program's accesses stay inside a few allocations (a
// frame's draws read ~6 MiB by address, all in one heap), while the search is ~11 dependent steps
// over the ~1200 ranges, 34 table loads in all, repeated at every access. A hit needs the access
// inside the cached range and the wanted permission, which is all the search would establish for
// it: the table's ranges are sorted and disjoint, so the range that holds the address is the one
// the search selects, and the driver never lets a range's device addresses wrap. Anything else
// (another range, a range without the permission, an unmapped or wrapping access) takes the
// search, with every fault it records today; a search that succeeds leaves its range in the
// cache. The variables are per host invocation and shared by its lanes (a wave64 program at two
// lanes, a mesh program's ES and GS threads): they only repeat what the table says, which no
// shader writes, so whichever lane filled them the next lookup gets the table's own answer.
// The cached form also leaves out what the driver guarantees of the table it builds
// (BdaResources): the header's version, entry size and reserved word and each entry's reserved
// word, four loads per search. The count against the buffer's length (a table cut short reads as
// InvalidTable, as today) and the entry's own sanity checks stay.
std::uint32_t DefineBdaLookup(SpirvEmitterState& state, const char* name, bool recordFaults, std::uint32_t permission = BdaAbi::Read) {
    const bool cached = state.bdaCacheBegin != 0;
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto load = [&](std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, value, pointer);
        return value;
    };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, name);
    state.module.AddFunction(spv::OpFunction, u64, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, u64, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto bytes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto low = state.module.AllocateId();
    const auto high = state.module.AllocateId();
    const auto pointer = TypePointer(state, spv::StorageClassFunction, u32);
    state.module.AddFunction(spv::OpVariable, pointer, low, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpVariable, pointer, high, spv::StorageClassFunction);
    const auto fail = [&](std::uint32_t condition, BdaAbi::FaultReason reason) {
        if (recordFaults) {
            ReturnBdaFailureIf(state, condition, address, bytes, instruction, reason);
        } else {
            ReturnBdaZeroIf(state, condition);
        }
    };
    const auto loadCache = [&](std::uint32_t type, std::uint32_t variable) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, variable);
        return value;
    };
    if (cached) {
        // An empty cache holds [0, 0), which no access lies in; a wrapping access (its end at or
        // below its address) and a zero-byte one miss too and fault in the search below.
        const auto accessEnd = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
        for (std::uint32_t way = 0; way < state.bdaCacheWays; way++) {
            const auto beginVariable = way == 0u ? state.bdaCacheBegin : state.bdaWayBegin[way - 1u];
            const auto endVariable = way == 0u ? state.bdaCacheEnd : state.bdaWayEnd[way - 1u];
            const auto baseVariable = way == 0u ? state.bdaCacheBase : state.bdaWayBase[way - 1u];
            const auto permissionVariable = way == 0u ? state.bdaCachePermissions : state.bdaWayPermissions[way - 1u];
            const auto cachedBegin = loadCache(u64, beginVariable);
            auto hit = binary(spv::OpUGreaterThanEqual, boolean, address, cachedBegin);
            hit = binary(spv::OpLogicalAnd, boolean, hit, binary(spv::OpULessThanEqual, boolean, accessEnd, loadCache(u64, endVariable)));
            hit = binary(spv::OpLogicalAnd, boolean, hit, binary(spv::OpUGreaterThan, boolean, accessEnd, address));
            hit = binary(spv::OpLogicalAnd, boolean, hit, binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, loadCache(u32, permissionVariable), constant(permission)), constant(0)));
            const auto found = state.module.AllocateId();
            const auto search = state.module.AllocateId();
            state.module.AddFunction(spv::OpSelectionMerge, search, spv::SelectionControlMaskNone);
            state.module.AddFunction(spv::OpBranchConditional, hit, found, search);
            EmitLabel(state, found);
            state.module.AddFunction(spv::OpReturnValue, binary(spv::OpIAdd, u64, loadCache(u64, baseVariable), binary(spv::OpISub, u64, address, cachedBegin)));
            EmitLabel(state, search);
        }
        if (BdaNoSearchExperiment()) {
            // Experiment only, wrong data: a miss reads the first cached range at the clamped offset
            // instead of searching the table (it stays inside that range, so the read is in bounds).
            const auto cachedBegin = loadCache(u64, state.bdaCacheBegin);
            const auto span = binary(spv::OpISub, u64, loadCache(u64, state.bdaCacheEnd), cachedBegin);
            const auto wide = Unary(state, spv::OpUConvert, u64, bytes);
            const auto usable = binary(spv::OpLogicalAnd, boolean, binary(spv::OpUGreaterThanEqual, boolean, span, wide), binary(spv::OpINotEqual, boolean, span, BdaConstant(state, 0u)));
            const auto found = state.module.AllocateId();
            const auto search = state.module.AllocateId();
            state.module.AddFunction(spv::OpSelectionMerge, search, spv::SelectionControlMaskNone);
            state.module.AddFunction(spv::OpBranchConditional, usable, found, search);
            EmitLabel(state, found);
            const auto offset = state.module.AllocateId();
            state.module.AddFunction(spv::OpExtInst, u64, offset, GlslStd450(state), 38u, binary(spv::OpISub, u64, address, cachedBegin), binary(spv::OpISub, u64, span, wide));
            state.module.AddFunction(spv::OpReturnValue, binary(spv::OpIAdd, u64, loadCache(u64, state.bdaCacheBase), offset));
            EmitLabel(state, search);
        }
    }
    const auto length = state.module.AllocateId();
    state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
    fail(binary(spv::OpULessThan, boolean, length, constant(4)), BdaAbi::FaultReason::InvalidTable);
    if (!cached) {
        fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(0)), constant(BdaAbi::Version)), BdaAbi::FaultReason::InvalidTable);
        fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(2)), constant(sizeof(BdaAbi::Range))), BdaAbi::FaultReason::InvalidTable);
        fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(3)), constant(0)), BdaAbi::FaultReason::InvalidTable);
    }
    const auto count = BdaLoadWord(state, constant(1));
    const auto available = binary(spv::OpISub, u32, length, constant(4));
    fail(binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, available, constant(7)), constant(0)), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpINotEqual, boolean, count, binary(spv::OpShiftRightLogical, u32, available, constant(3))), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpIEqual, boolean, bytes, constant(0)), BdaAbi::FaultReason::Overflow);
    const auto end = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
    fail(binary(spv::OpULessThanEqual, boolean, end, address), BdaAbi::FaultReason::Overflow);
    state.module.AddFunction(spv::OpStore, low, constant(0));
    state.module.AddFunction(spv::OpStore, high, count);
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto lower = load(low);
    const auto upper = load(high);
    const auto search = binary(spv::OpULessThan, boolean, lower, upper);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, search, body, merge);
    EmitLabel(state, body);
    const auto midpoint = binary(spv::OpIAdd, u32, lower, binary(spv::OpShiftRightLogical, u32, binary(spv::OpISub, u32, upper, lower), constant(1)));
    const auto entry = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, midpoint, constant(8)));
    const auto before = binary(spv::OpULessThan, boolean, address, BdaLoadAddress(state, entry));
    const auto newLow = state.module.AllocateId();
    const auto newHigh = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelect, u32, newLow, before, lower, binary(spv::OpIAdd, u32, midpoint, constant(1)));
    state.module.AddFunction(spv::OpSelect, u32, newHigh, before, midpoint, upper);
    state.module.AddFunction(spv::OpStore, low, newLow);
    state.module.AddFunction(spv::OpStore, high, newHigh);
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto index = load(low);
    fail(binary(spv::OpIEqual, boolean, index, constant(0)), BdaAbi::FaultReason::Unmapped);
    const auto selected = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, index, constant(1)), constant(8)));
    const auto at = [&](std::uint32_t offset) { return binary(spv::OpIAdd, u32, selected, constant(offset)); };
    const auto begin = BdaLoadAddress(state, selected);
    const auto finish = BdaLoadAddress(state, at(2));
    const auto base = BdaLoadAddress(state, at(4));
    const auto permissions = BdaLoadWord(state, at(6));
    if (!cached) fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, at(7)), constant(0)), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpUGreaterThanEqual, boolean, begin, finish), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpUGreaterThan, boolean, end, finish), BdaAbi::FaultReason::Unmapped);
    fail(binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, permissions, constant(permission)), constant(0)), BdaAbi::FaultReason::Permission);
    fail(binary(spv::OpIEqual, boolean, base, BdaConstant(state, 0)), BdaAbi::FaultReason::InvalidTable);
    const auto offset = binary(spv::OpISub, u64, address, begin);
    const auto result = binary(spv::OpIAdd, u64, base, offset);
    fail(binary(spv::OpULessThan, boolean, result, base), BdaAbi::FaultReason::Overflow);
    const auto deviceEnd = binary(spv::OpIAdd, u64, result, Unary(state, spv::OpUConvert, u64, bytes));
    fail(binary(spv::OpULessThanEqual, boolean, deviceEnd, result), BdaAbi::FaultReason::Overflow);
    if (cached && state.bdaCacheWays == 1u) {
        state.module.AddFunction(spv::OpStore, state.bdaCacheBegin, begin);
        state.module.AddFunction(spv::OpStore, state.bdaCacheEnd, finish);
        state.module.AddFunction(spv::OpStore, state.bdaCacheBase, base);
        state.module.AddFunction(spv::OpStore, state.bdaCachePermissions, permissions);
    } else if (cached) {
        const auto next = loadCache(u32, state.bdaCacheNext);
        for (std::uint32_t way = 0; way < state.bdaCacheWays; way++) {
            EmitIfCondition(state, binary(spv::OpIEqual, boolean, next, constant(way)), [&] {
                state.module.AddFunction(spv::OpStore, way == 0u ? state.bdaCacheBegin : state.bdaWayBegin[way - 1u], begin);
                state.module.AddFunction(spv::OpStore, way == 0u ? state.bdaCacheEnd : state.bdaWayEnd[way - 1u], finish);
                state.module.AddFunction(spv::OpStore, way == 0u ? state.bdaCacheBase : state.bdaWayBase[way - 1u], base);
                state.module.AddFunction(spv::OpStore, way == 0u ? state.bdaCachePermissions : state.bdaWayPermissions[way - 1u], permissions);
            });
        }
        const auto advanced = binary(spv::OpIAdd, u32, next, constant(1u));
        const auto wrapped = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelect, u32, wrapped, binary(spv::OpULessThan, boolean, advanced, constant(state.bdaCacheWays)), advanced, constant(0u));
        state.module.AddFunction(spv::OpStore, state.bdaCacheNext, wrapped);
    }
    state.module.AddFunction(spv::OpReturnValue, result);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

}

static std::uint32_t DefineBdaNoteWrite(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "note_bda_write");
    state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlMaskNone, state.module.Type(spv::OpTypeFunction, voidType, TypeScalarU64(state)));
    const auto address = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, TypeScalarU64(state), address);
    EmitLabel(state, state.module.AllocateId());
    const auto done = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, boolean), done, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpStore, done, ConstantBool(state, false));
    const auto page = binary(spv::OpIAdd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, TypeScalarU64(state), address, BdaConstant(state, BdaAbi::WrittenPageShift))), constant(1));
    const auto hash = binary(spv::OpShiftRightLogical, u32, binary(spv::OpIMul, u32, page, constant(0x9e3779b1u)), constant(32u - static_cast<std::uint32_t>(std::countr_zero(BdaAbi::WrittenPageSlots))));
    const auto scope = constant(spv::ScopeDevice);
    const auto relaxed = constant(spv::MemorySemanticsMaskNone);
    for (std::uint32_t probe = 0; probe < BdaAbi::WrittenPageProbes; probe++) {
        const auto pending = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, boolean, pending, done);
        EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, pending), [&] {
            const auto slot = binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenSlotsWord), binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, hash, constant(probe)), constant(BdaAbi::WrittenPageSlots - 1u)));
            const auto previous = state.module.AllocateId();
            state.module.AddFunction(spv::OpAtomicCompareExchange, u32, previous, BdaWord(state, state.faultBufferVariable, slot), scope, relaxed, relaxed, page, constant(0u));
            const auto taken = binary(spv::OpLogicalOr, boolean, binary(spv::OpIEqual, boolean, previous, constant(0u)), binary(spv::OpIEqual, boolean, previous, page));
            state.module.AddFunction(spv::OpStore, done, taken);
        });
    }
    const auto noted = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, noted, done);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, noted), [&] {
        state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenOverflowWord)), scope, relaxed, constant(1u));
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

// APS5_NO_BDA_LOOKUP_CACHE=1 emits the lookups as before the cache: the whole search and every
// table check at each access. An environment switch of the recompiler, so the shader disk cache
// keeps both forms apart.
bool BdaLookupCached() {
    static const bool disabled = std::getenv("APS5_NO_BDA_LOOKUP_CACHE") != nullptr;
    return !disabled;
}

// APS5_BDA_CACHE_WAYS=2..4 (not for upstream): the lookups keep that many ranges, filled in turn.
std::uint32_t BdaCacheWays() {
    static const std::uint32_t ways = [] {
        const char* text = std::getenv("APS5_BDA_CACHE_WAYS");
        const auto value = text != nullptr ? std::strtoul(text, nullptr, 10) : 1ul;
        return static_cast<std::uint32_t>(value < 1ul ? 1ul : value > 4ul ? 4ul : value);
    }();
    return ways;
}

// APS5_EXP_BDA_NO_SEARCH=1 (not for upstream, experiment, wrong image): a miss reads the cached
// range at a clamped offset instead of searching the table. Measures what the searches cost.
bool BdaNoSearchExperiment() {
    static const bool enabled = std::getenv("APS5_EXP_BDA_NO_SEARCH") != nullptr;
    return enabled;
}

void DefineGetBdaPointer(SpirvEmitterState& state) {
    if (!state.program.Info().usesDma) return;
    if (BdaLookupCached()) {
        // Initialized to the empty range [0, 0) with no permission.
        const auto zero = BdaConstant(state, 0u);
        const auto address = TypePointer(state, spv::StorageClassPrivate, TypeScalarU64(state));
        state.bdaCacheBegin = state.module.DefineInitializedGlobalVariable(address, spv::StorageClassPrivate, zero);
        state.bdaCacheEnd = state.module.DefineInitializedGlobalVariable(address, spv::StorageClassPrivate, zero);
        state.bdaCacheBase = state.module.DefineInitializedGlobalVariable(address, spv::StorageClassPrivate, zero);
        state.bdaCachePermissions = state.module.DefineInitializedGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeU32(state)), spv::StorageClassPrivate, ConstantU32(state, 0u));
        state.module.AddName(state.bdaCacheBegin, "bda_cache_begin");
        state.module.AddName(state.bdaCacheEnd, "bda_cache_end");
        state.module.AddName(state.bdaCacheBase, "bda_cache_base");
        state.module.AddName(state.bdaCachePermissions, "bda_cache_permissions");
        state.bdaCacheWays = BdaCacheWays();
        for (std::uint32_t way = 1; way < state.bdaCacheWays; way++) {
            state.bdaWayBegin[way - 1u] = state.module.DefineInitializedGlobalVariable(address, spv::StorageClassPrivate, zero);
            state.bdaWayEnd[way - 1u] = state.module.DefineInitializedGlobalVariable(address, spv::StorageClassPrivate, zero);
            state.bdaWayBase[way - 1u] = state.module.DefineInitializedGlobalVariable(address, spv::StorageClassPrivate, zero);
            state.bdaWayPermissions[way - 1u] = state.module.DefineInitializedGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeU32(state)), spv::StorageClassPrivate, ConstantU32(state, 0u));
        }
        if (state.bdaCacheWays > 1u) state.bdaCacheNext = state.module.DefineInitializedGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeU32(state)), spv::StorageClassPrivate, ConstantU32(state, 0u));
    }
    state.bdaPointerFunction = DefineBdaLookup(state, "get_bda_pointer", true);
    if (!BdaByteReadsForced()) state.bdaProbeFunction = DefineBdaLookup(state, "probe_bda_pointer", false);
    if (state.program.Info().bdaWrites) {
        state.bdaWritePointerFunction = DefineBdaLookup(state, "get_bda_write_pointer", true, BdaAbi::Write);
        state.bdaNoteWriteFunction = DefineBdaNoteWrite(state);
    }
}

}
