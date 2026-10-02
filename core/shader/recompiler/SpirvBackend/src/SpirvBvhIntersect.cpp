#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <spirv/unified1/GLSL.std.450.h>
#include <spirv/unified1/spirv.hpp>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace ShaderRecompiler {

namespace {

using Vec3 = std::array<std::uint32_t, 3>;

constexpr std::uint32_t PositiveInfinity = 0x7f800000u;
constexpr std::uint32_t One = 0x3f800000u;
constexpr std::uint32_t NoChild = 0xffffffffu;

class BvhEmitter {
public:
    explicit BvhEmitter(SpirvEmitterState& state)
        : state(state), u32(TypeU32(state)), u64(TypeScalarU64(state)), f32(TypeF32(state)), boolean(TypeBool(state)), result4(TypeU32Composite(state, 4u)) {}

    std::uint32_t Define() {
        const auto vec3 = TypeF32Vector(state, 3u);
        const auto function = state.module.AllocateId();
        state.module.AddName(function, "bvh_intersect");
        state.module.AddFunction(spv::OpFunction, result4, function, spv::FunctionControlMaskNone, state.module.Type(spv::OpTypeFunction, result4, result4, u64, f32, vec3, vec3, vec3, u32));
        const std::array<std::uint32_t, 7> types{result4, u64, f32, vec3, vec3, vec3, u32};
        std::array<std::uint32_t, 7> parameters{};
        for (std::uint32_t i = 0; i < parameters.size(); ++i) {
            parameters[i] = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionParameter, types[i], parameters[i]);
        }
        EmitLabel(state, state.module.AllocateId());
        for (std::uint32_t i = 0; i < 4u; ++i) words[i] = extract(u32, parameters[0], i);
        const auto node = parameters[1];
        extent = parameters[2];
        for (std::uint32_t axis = 0; axis < 3u; ++axis) {
            origin[axis] = extract(f32, parameters[3], axis);
            direction[axis] = extract(f32, parameters[4], axis);
            inverse[axis] = extract(f32, parameters[5], axis);
        }
        pc = parameters[6];
        state.module.AddFunction(spv::OpReturnValue, intersect(node));
        state.module.AddFunction(spv::OpFunctionEnd);
        return function;
    }

private:
    std::uint32_t uint(std::uint32_t value) { return ConstantU32(state, value); }
    std::uint32_t wide(std::uint64_t value) { return BdaConstant(state, value); }
    std::uint32_t real(std::uint32_t bits) { return ConstantF32(state, bits); }
    std::uint32_t op(spv::Op opcode, std::uint32_t type, std::uint32_t lhs, std::uint32_t rhs) { return Binary(state, opcode, type, lhs, rhs); }
    std::uint32_t unary(spv::Op opcode, std::uint32_t type, std::uint32_t value) { return Unary(state, opcode, type, value); }
    std::uint32_t both(std::uint32_t lhs, std::uint32_t rhs) { return op(spv::OpLogicalAnd, boolean, lhs, rhs); }
    std::uint32_t either(std::uint32_t lhs, std::uint32_t rhs) { return op(spv::OpLogicalOr, boolean, lhs, rhs); }
    std::uint32_t isNot(std::uint32_t value) { return unary(spv::OpLogicalNot, boolean, value); }
    std::uint32_t equal(std::uint32_t lhs, std::uint32_t value) { return op(spv::OpIEqual, boolean, lhs, uint(value)); }
    std::uint32_t bits(std::uint32_t value, std::uint32_t shift, std::uint32_t mask) { return op(spv::OpBitwiseAnd, u32, shift == 0u ? value : op(spv::OpShiftRightLogical, u32, value, uint(shift)), uint(mask)); }
    std::uint32_t pick(std::uint32_t type, std::uint32_t condition, std::uint32_t whenTrue, std::uint32_t whenFalse) { return Select(state, type, condition, whenTrue, whenFalse); }
    std::uint32_t asFloat(std::uint32_t value) { return unary(spv::OpBitcast, f32, value); }
    std::uint32_t asUint(std::uint32_t value) { return unary(spv::OpBitcast, u32, value); }
    std::uint32_t widen(std::uint32_t value) { return unary(spv::OpUConvert, u64, value); }
    std::uint32_t pair(std::uint32_t low, std::uint32_t high) { return op(spv::OpBitwiseOr, u64, widen(low), op(spv::OpShiftLeftLogical, u64, widen(high), wide(32u))); }

    std::uint32_t extract(std::uint32_t type, std::uint32_t composite, std::uint32_t index) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, type, value, composite, index);
        return value;
    }

    std::uint32_t compose(const std::array<std::uint32_t, 4>& values) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, result4, value, values[0], values[1], values[2], values[3]);
        return value;
    }

    std::uint32_t glsl(std::uint32_t instruction, std::uint32_t type, std::uint32_t lhs, std::uint32_t rhs = 0) {
        const auto value = state.module.AllocateId();
        if (rhs == 0) state.module.AddFunction(spv::OpExtInst, type, value, GlslStd450(state), instruction, lhs);
        else state.module.AddFunction(spv::OpExtInst, type, value, GlslStd450(state), instruction, lhs, rhs);
        return value;
    }

    std::uint32_t exact(spv::Op opcode, std::uint32_t lhs, std::uint32_t rhs) {
        const auto value = op(opcode, f32, lhs, rhs);
        state.module.AddAnnotation(spv::OpDecorate, value, spv::DecorationNoContraction);
        return value;
    }

    std::uint32_t less(std::uint32_t lhs, std::uint32_t rhs) { return op(spv::OpFOrdLessThan, boolean, lhs, rhs); }
    std::uint32_t greater(std::uint32_t lhs, std::uint32_t rhs) { return op(spv::OpFOrdGreaterThan, boolean, lhs, rhs); }
    std::uint32_t same(std::uint32_t lhs, std::uint32_t rhs) { return op(spv::OpFOrdEqual, boolean, lhs, rhs); }

    std::uint32_t locate(std::uint32_t address, std::uint32_t bytes) {
        const bool probing = state.bdaProbeFunction != 0;
        const auto physical = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, u64, physical, probing ? state.bdaProbeFunction : state.bdaPointerFunction, address, uint(bytes), pc);
        const auto mapped = op(spv::OpINotEqual, boolean, physical, wide(0u));
        const auto aligned = op(spv::OpIEqual, boolean, op(spv::OpBitwiseAnd, u64, physical, wide(3u)), wide(0u));
        if (probing) EmitIfCondition(state, isNot(mapped), [&] { RecordBdaFault(state, address, uint(bytes), pc, BdaAbi::FaultReason::Unmapped); });
        EmitIfCondition(state, both(mapped, isNot(aligned)), [&] { RecordBdaFault(state, address, uint(bytes), pc, BdaAbi::FaultReason::Unaligned); });
        return pick(u64, both(mapped, aligned), physical, wide(0u));
    }

    std::uint32_t load(std::uint32_t block, std::uint32_t byteOffset) {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, op(spv::OpIAdd, u64, block, byteOffset));
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, value, pointer, spv::MemoryAccessAlignedMask, 4u);
        return value;
    }

    std::uint32_t loadWord(std::uint32_t block, std::uint32_t word) { return load(block, wide(word * 4u)); }

    std::uint32_t intersect(std::uint32_t node) {
        const auto type = bits(unary(spv::OpUConvert, u32, node), 0u, 7u);
        const auto index = op(spv::OpShiftRightLogical, u64, node, wide(3u));
        const auto box32 = equal(type, 5u);
        const auto lastIndex = op(spv::OpIAdd, u64, index, pick(u64, box32, wide(1u), wide(0u)));
        const auto base = op(spv::OpShiftLeftLogical, u64, pair(words[0], bits(words[1], 0u, 0xffu)), wide(8u));
        const auto address = op(spv::OpIAdd, u64, base, op(spv::OpShiftLeftLogical, u64, index, wide(6u)));
        const auto lastBlock = op(spv::OpIAdd, u64, base, op(spv::OpShiftLeftLogical, u64, lastIndex, wide(6u)));
        const auto lastNode = pair(words[2], bits(words[3], 0u, 0x3ffu));
        auto valid = both(equal(bits(words[3], 28u, 0xfu), 8u), op(spv::OpULessThanEqual, boolean, type, uint(5u)));
        valid = both(valid, op(spv::OpULessThanEqual, boolean, lastIndex, lastNode));
        valid = both(valid, op(spv::OpULessThan, boolean, lastBlock, wide(std::uint64_t{1} << 48u)));
        valid = both(valid, isNot(unary(spv::OpIsNan, boolean, extent)));
        for (std::uint32_t axis = 0; axis < 3u; ++axis) {
            for (const auto value : {origin[axis], direction[axis], inverse[axis]}) valid = both(valid, isNot(unary(spv::OpIsNan, boolean, value)));
            valid = both(valid, isNot(unary(spv::OpIsInf, boolean, origin[axis])));
        }
        const auto triangle = op(spv::OpULessThan, boolean, type, uint(4u));
        const auto invalid = compose({pick(u32, triangle, uint(PositiveInfinity), uint(NoChild)), pick(u32, triangle, uint(One), uint(NoChild)), uint(NoChild), pick(u32, triangle, uint(0u), uint(NoChild))});
        return EmitValueOrDefaultIfCondition(state, valid, result4, invalid, [&] {
            const auto first = locate(address, 64u);
            const auto second = EmitValueOrDefaultIfCondition(state, box32, u64, wide(0u), [&] { return locate(op(spv::OpIAdd, u64, address, wide(64u)), 48u); });
            const auto fetched = both(op(spv::OpINotEqual, boolean, first, wide(0u)), either(isNot(box32), op(spv::OpINotEqual, boolean, second, wide(0u))));
            return EmitValueOrDefaultIfCondition(state, fetched, result4, invalid, [&] {
                return EmitValueIfElse(state, triangle, result4, [&] { return intersectTriangle(first, type); }, [&] {
                    return EmitValueIfElse(state, box32, result4, [&] { return intersectBoxes(first, second, false); }, [&] { return intersectBoxes(first, 0u, true); });
                });
            });
        });
    }

    std::uint32_t intersectTriangle(std::uint32_t block, std::uint32_t type) {
        const auto is = [&](std::uint32_t value) { return equal(type, value); };
        const std::array<std::uint32_t, 3> corners{
            glsl(GLSLstd450UMin, u32, type, uint(2u)),
            pick(u32, is(0u), uint(1u), pick(u32, is(3u), uint(4u), uint(3u))),
            pick(u32, is(2u), uint(4u), pick(u32, is(3u), uint(0u), uint(2u))),
        };
        const auto magnitude = [&](std::uint32_t value) { return glsl(GLSLstd450FAbs, f32, value); };
        const auto yMajor = less(magnitude(direction[0]), magnitude(direction[1]));
        const auto zMajor = less(pick(f32, yMajor, magnitude(direction[1]), magnitude(direction[0])), magnitude(direction[2]));
        const auto rotate = [&](const Vec3& value) {
            Vec3 rotated{};
            for (std::uint32_t i = 0; i < 3u; ++i) rotated[i] = pick(f32, zMajor, value[i], pick(f32, yMajor, value[(i + 2u) % 3u], value[(i + 1u) % 3u]));
            return rotated;
        };
        const auto from = rotate(origin);
        const auto toward = rotate(direction);
        std::array<Vec3, 3> sheared{};
        for (std::uint32_t corner = 0; corner < 3u; ++corner) {
            const auto first = op(spv::OpIMul, u32, corners[corner], uint(12u));
            Vec3 position{};
            for (std::uint32_t axis = 0; axis < 3u; ++axis) position[axis] = asFloat(load(block, widen(op(spv::OpIAdd, u32, first, uint(axis * 4u)))));
            position = rotate(position);
            Vec3 relative{};
            for (std::uint32_t axis = 0; axis < 3u; ++axis) relative[axis] = exact(spv::OpFSub, position[axis], from[axis]);
            sheared[corner] = {
                exact(spv::OpFSub, exact(spv::OpFMul, relative[0], toward[2]), exact(spv::OpFMul, toward[0], relative[2])),
                exact(spv::OpFSub, exact(spv::OpFMul, relative[1], toward[2]), exact(spv::OpFMul, toward[1], relative[2])),
                relative[2],
            };
        }
        Vec3 edge{};
        Vec3 weight{};
        for (std::uint32_t i = 0; i < 3u; ++i) {
            const auto& a = sheared[(i + 1u) % 3u];
            const auto& b = sheared[(i + 2u) % 3u];
            edge[i] = exact(spv::OpFSub, exact(spv::OpFMul, b[0], a[1]), exact(spv::OpFMul, b[1], a[0]));
            weight[i] = exact(spv::OpFMul, edge[i], toward[2]);
        }
        const auto numerator = exact(spv::OpFAdd, exact(spv::OpFAdd, exact(spv::OpFMul, edge[0], sheared[0][2]), exact(spv::OpFMul, edge[1], sheared[1][2])), exact(spv::OpFMul, edge[2], sheared[2][2]));
        const auto denominator = exact(spv::OpFAdd, exact(spv::OpFAdd, weight[0], weight[1]), weight[2]);
        const auto zero = real(0u);
        const auto frontFacing = greater(denominator, zero);
        auto negative = less(edge[0], zero);
        auto positive = greater(edge[0], zero);
        for (std::uint32_t i = 1; i < 3u; ++i) {
            negative = either(negative, less(edge[i], zero));
            positive = either(positive, greater(edge[i], zero));
        }
        auto missed = either(both(negative, positive), same(denominator, zero));
        missed = either(missed, unary(spv::OpIsNan, boolean, numerator));
        missed = either(missed, less(pick(f32, frontFacing, numerator, unary(spv::OpFNegate, f32, numerator)), zero));
        for (std::uint32_t i = 0; i < 3u; ++i) {
            const auto startY = sheared[(i + 1u) % 3u][1];
            const auto endY = sheared[(i + 2u) % 3u][1];
            const auto startOnAxis = same(startY, zero);
            const auto horizontal = both(startOnAxis, same(endY, zero));
            const auto below = either(less(startY, zero), both(startOnAxis, greater(endY, zero)));
            const auto excluded = pick(boolean, horizontal, greater(sheared[i][1], zero), op(spv::OpLogicalNotEqual, boolean, below, frontFacing));
            missed = either(missed, both(same(edge[i], zero), excluded));
        }
        const auto flags = loadWord(block, 15u);
        const auto remap = op(spv::OpShiftRightLogical, u32, flags, op(spv::OpShiftLeftLogical, u32, type, uint(3u)));
        const auto barycentric = [&](std::uint32_t shift) {
            const auto source = bits(remap, shift, 3u);
            return asUint(pick(f32, equal(source, 1u), weight[1], pick(f32, equal(source, 2u), weight[2], weight[0])));
        };
        const auto barycentrics = op(spv::OpINotEqual, boolean, bits(words[3], 24u, 1u), uint(0u));
        return compose({
            asUint(pick(f32, missed, real(PositiveInfinity), numerator)),
            asUint(pick(f32, missed, real(One), denominator)),
            pick(u32, barycentrics, barycentric(0u), op(spv::OpIAdd, u32, flags, type)),
            pick(u32, barycentrics, barycentric(2u), pick(u32, missed, uint(0u), uint(1u))),
        });
    }

    std::array<std::uint32_t, 6> bounds(std::uint32_t first, std::uint32_t second, bool half, std::uint32_t child) {
        std::array<std::uint32_t, 6> result{};
        if (half) {
            for (std::uint32_t word = 0; word < 3u; ++word) {
                const auto unpacked = glsl(GLSLstd450UnpackHalf2x16, TypeF32Vector(state, 2u), loadWord(first, 4u + child * 3u + word));
                result[word * 2u] = extract(f32, unpacked, 0u);
                result[word * 2u + 1u] = extract(f32, unpacked, 1u);
            }
            return result;
        }
        for (std::uint32_t component = 0; component < 6u; ++component) {
            const auto word = 4u + child * 6u + component;
            result[component] = asFloat(word < 16u ? loadWord(first, word) : loadWord(second, word - 16u));
        }
        return result;
    }

    std::uint32_t intersectBoxes(std::uint32_t first, std::uint32_t second, bool half) {
        const auto zero = real(0u);
        const auto grow = bits(words[1], 23u, 0xffu);
        const auto sort = op(spv::OpINotEqual, boolean, bits(words[1], 31u, 1u), uint(0u));
        Vec3 forward{};
        for (std::uint32_t axis = 0; axis < 3u; ++axis) forward[axis] = op(spv::OpFOrdGreaterThanEqual, boolean, inverse[axis], zero);
        std::array<std::uint32_t, 4> children{};
        std::array<std::uint32_t, 4> keys{};
        for (std::uint32_t child = 0; child < 4u; ++child) {
            const auto box = bounds(first, second, half, child);
            std::uint32_t entry = 0;
            std::uint32_t exit = 0;
            for (std::uint32_t axis = 0; axis < 3u; ++axis) {
                const auto low = exact(spv::OpFMul, exact(spv::OpFSub, box[axis], origin[axis]), inverse[axis]);
                const auto high = exact(spv::OpFMul, exact(spv::OpFSub, box[axis + 3u], origin[axis]), inverse[axis]);
                const auto near = pick(f32, forward[axis], low, high);
                const auto far = pick(f32, forward[axis], high, low);
                entry = axis == 0u ? near : glsl(GLSLstd450NMax, f32, entry, near);
                exit = axis == 0u ? far : glsl(GLSLstd450NMin, f32, exit, far);
            }
            const auto grown = asFloat(glsl(GLSLstd450UMin, u32, op(spv::OpIAdd, u32, asUint(pick(f32, same(exit, zero), zero, exit)), grow), uint(PositiveInfinity)));
            const auto hit = both(both(op(spv::OpFOrdGreaterThanEqual, boolean, exit, zero), less(entry, extent)), op(spv::OpFOrdLessThanEqual, boolean, entry, grown));
            children[child] = pick(u32, hit, loadWord(first, child), uint(NoChild));
            keys[child] = pick(u32, hit, asUint(pick(f32, greater(entry, zero), entry, zero)), uint(PositiveInfinity));
        }
        constexpr std::array<std::array<std::uint32_t, 2>, 5> network{{{0u, 1u}, {2u, 3u}, {0u, 2u}, {1u, 3u}, {1u, 2u}}};
        for (const auto& [a, b] : network) {
            const auto swap = both(sort, op(spv::OpUGreaterThan, boolean, keys[a], keys[b]));
            const std::array<std::uint32_t, 2> child{children[a], children[b]};
            const std::array<std::uint32_t, 2> key{keys[a], keys[b]};
            children[a] = pick(u32, swap, child[1], child[0]);
            children[b] = pick(u32, swap, child[0], child[1]);
            keys[a] = pick(u32, swap, key[1], key[0]);
            keys[b] = pick(u32, swap, key[0], key[1]);
        }
        return compose(children);
    }

    SpirvEmitterState& state;
    std::uint32_t u32;
    std::uint32_t u64;
    std::uint32_t f32;
    std::uint32_t boolean;
    std::uint32_t result4;
    std::array<std::uint32_t, 4> words{};
    std::uint32_t extent = 0;
    Vec3 origin{};
    Vec3 direction{};
    Vec3 inverse{};
    std::uint32_t pc = 0;
};

}

void DefineBvhIntersect(SpirvEmitterState& state) {
    if (state.bdaPointerFunction == 0) throw std::runtime_error("BVH intersection needs the BDA lookup");
    state.bvhIntersectFunction = BvhEmitter(state).Define();
}

}
