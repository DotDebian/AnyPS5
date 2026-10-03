#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawAhead.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/PreparedDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/Submission.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

void check(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

template<typename TAction>
void expectFailure(TAction action, const char* text) {
    try { action(); }
    catch (const std::runtime_error& error) {
        check(std::string(error.what()).find(text) != std::string::npos, error.what());
        return;
    }
    throw std::runtime_error("expected PM4 rejection");
}

std::vector<std::uint32_t> makePacket(std::uint32_t opcode, std::initializer_list<std::uint32_t> payload, std::uint32_t flags = 0) {
    std::vector<std::uint32_t> result{0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u) | flags};
    result.insert(result.end(), payload);
    return result;
}

std::uint32_t low(const void* pointer) { return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer)); }
std::uint32_t high(const void* pointer) { return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer) >> 32u); }

void execute(AgcDriver::QueueState& state, const std::vector<std::uint32_t>& packet) {
    AgcDriver::Pm4::Validate(packet, 0);
    AgcDriver::Pm4::Execute(packet, state);
}

void testCatalog() {
    std::set<std::uint32_t> values;
    for (const auto& opcode : AgcDriver::Pm4::Opcodes) {
        check(values.insert(opcode.value).second, "duplicate PM4 opcode");
        const auto packet = makePacket(opcode.value, {0});
        check(AgcDriver::Pm4::Name(packet[0]) == opcode.name, "opcode name mismatch");
        const auto reason = AgcDriver::Pm4::UnsupportedReason(packet[0]);
        if (!reason.empty()) expectFailure([&] { AgcDriver::Pm4::Validate(packet, 0); }, std::string(reason).c_str());
    }
    check(values.size() == 54, "reference opcode catalog is incomplete");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0xff, {0}), 0); }, "not known");
    const std::array<std::pair<std::uint32_t, const char*>, 11> custom{{
        {5, "DRAW_RESET"}, {6, "WAIT_FLIP_DONE"}, {9, "DISPATCH_RESET"}, {11, "PUSH_MARKER"},
        {12, "POP_MARKER"}, {20, "ACQUIRE_MEM_CUSTOM"}, {21, "WRITE_DATA_CUSTOM"}, {23, "FLIP"},
        {24, "RELEASE_MEM_CUSTOM"}, {25, "DMA_DATA_CUSTOM"}, {26, "CONTEXT_STATE"}
    }};
    for (const auto& [id, name] : custom) check(AgcDriver::Pm4::Name(makePacket(0x10, {0}, id << 2)[0]) == name, "custom opcode name mismatch");
}

void testRegisters() {
    AgcDriver::QueueState state;
    execute(state, makePacket(0x79, {0x242, 4}));
    check(state.userConfig.at(0x242) == 4, "primitive type register write was lost");
    execute(state, makePacket(0x79, {0x242, 6}));
    check(state.userConfig.at(0x242) == 6, "primitive type register update was lost");
    const std::array<std::uint32_t, 3> indirectOpcodes{0x9f, 0x63, 0x64};
    for (auto opcode : indirectOpcodes) {
        std::array<std::uint32_t, 6> pairs{0x10, 41, 0x11, 42, 0x10, 43};
        auto packet = makePacket(opcode, {low(pairs.data()), high(pairs.data()), 0x80000000, 3});
        execute(state, packet);
        const auto& registers = opcode == 0x9f ? state.context : opcode == 0x63 ? state.shader : state.userConfig;
        check(registers.at(0x10) == 43 && registers.at(0x11) == 42, "indirect register order or bank lost");
        pairs[0] = 0x12;
        pairs[4] = 0xffffffffu;
        expectFailure([&] { execute(state, packet); }, "sentinel");
        check(!registers.contains(0x12), "invalid indirect packet partially changed state");
        packet[1] = 0x1000;
        packet[2] = 0;
        expectFailure([&] { execute(state, packet); }, "guest");
        packet[3] = 0;
        expectFailure([&] { AgcDriver::Pm4::Validate(packet, 0); }, "control");
    }
    execute(state, makePacket(0x69, {0x11, 50, 51}));
    check(state.context.at(0x11) == 50 && state.context.at(0x12) == 51, "direct registers not sequential");
    execute(state, makePacket(0x7a, {0x10, 60}));
    check(state.userConfig.at(0x10) == 60, "uconfig index zero failed");
    execute(state, makePacket(0x7a, {0x20000243, 0x441}));
    check(state.indexType == 1 && state.userConfig.at(0x243) == 0x441, "indexed VGT_INDEX_TYPE write lost state");
    expectFailure([&] { execute(state, makePacket(0x7a, {0x10000010, 1})); }, "bank selection");
    expectFailure([&] { execute(state, makePacket(0x69, {0xffff, 1, 2})); }, "overflow");
    expectFailure([&] { AgcDriver::Pm4::Validate(makePacket(0x9f, {0, 0, 0x80000000, 0}), 0x20); }, "compute");
}

void testRegisterFile() {
    AgcDriver::Registers registers{{0x300, 3}, {0x10, 1}, {0x41, 2}};
    check(registers.size() == 3 && !registers.contains(0x11) && registers.at(0x41) == 2, "register file lookup");
    check(registers.find(0x12) == registers.end() && registers.find(0x10)->second == 1, "register file find");
    check(!registers.emplace(0x10, 9).second && registers.at(0x10) == 1, "register file emplace replaced a value");
    check(registers.insert_or_assign(0x10, 7).second == false && registers.at(0x10) == 7, "register file assignment");
    check(registers.lower_bound(0x11)->first == 0x41 && registers.upper_bound(0x41)->first == 0x300 && registers.lower_bound(0x301) == registers.end(), "register file bounds");
    std::vector<std::pair<std::uint32_t, std::uint32_t>> order;
    for (const auto& [offset, value] : registers) order.emplace_back(offset, value);
    check(order == std::vector<std::pair<std::uint32_t, std::uint32_t>>{{0x10, 7}, {0x41, 2}, {0x300, 3}}, "register file order");
    auto copy = registers;
    copy[0x7000] = 5;
    check(copy.size() == 4 && registers.size() == 3 && !registers.contains(0x7000) && !(copy == registers), "register file copy");
    check(copy.erase(0x7000) == 1 && copy.erase(0x7000) == 0 && copy == registers, "register file erase");
    bool threw = false;
    try {
        static_cast<void>(registers.at(0x42));
    } catch (const std::out_of_range&) {
        threw = true;
    }
    check(threw, "register file read an unset register");
}

void testContextAndBases() {
    AgcDriver::QueueState state;
    execute(state, makePacket(0x69, {0x10, 17}));
    execute(state, makePacket(0x76, {0x20c, 2}));
    execute(state, makePacket(0x10, {3, 0}, 0x68));
    check(state.context == AgcDriver::InitialContextRegisters() && state.shader.at(0x20c) == 2, "push-clear reset wrong state");
    expectFailure([&] { execute(state, makePacket(0x10, {1, 0}, 0x68)); }, "already pushed");
    execute(state, makePacket(0x69, {0x10, 19}));
    execute(state, makePacket(0x10, {2, 0}, 0x68));
    check(state.context.at(0x10) == 17, "pop did not restore context");
    expectFailure([&] { execute(state, makePacket(0x10, {2, 0}, 0x68)); }, "not been pushed");
    alignas(8) std::array<std::uint32_t, 4> arguments{7, 8, 9, 0};
    execute(state, makePacket(0x11, {1, low(arguments.data()), high(arguments.data())}, 2));
    auto packet = makePacket(0x16, {0, 0x8041});
    AgcDriver::Pm4::Validate(packet, 0);
    auto resolved = AgcDriver::Pm4::ResolveDispatch(packet, state);
    check(resolved == std::array<std::uint32_t, 5>{0xc0031500, 7, 8, 9, 0x8041}, "base-relative dispatch arguments changed");
    packet = makePacket(0x16, {low(arguments.data()), high(arguments.data()), 0x41});
    AgcDriver::Pm4::Validate(packet, 0x20);
    check(AgcDriver::Pm4::ResolveDispatch(packet, state)[3] == 9, "absolute indirect dispatch arguments changed");
    AgcDriver::Pm4::Validate(makePacket(0x15, {1, 1, 1, 0x2041}), 0);
    AgcDriver::Pm4::Validate(makePacket(0x16, {0, 0xa041}), 0);
    expectFailure([&] { AgcDriver::Pm4::Validate(makePacket(0x15, {1, 1, 1, 0x4041}), 0); }, "dispatch modifiers");
    execute(state, makePacket(0x13, {32}));
    execute(state, makePacket(0x26, {0x1000, 1}));
    execute(state, makePacket(0x2a, {1}));
    execute(state, makePacket(0x2f, {3}));
    check(state.indexBufferSize == 32 && state.indexBase == 0x100001000ull && state.indexType == 1 && state.instanceCount == 3, "draw setup state lost");
    execute(state, makePacket(0x10, {0x00636261}, 0x2c));
    check(state.markers.back() == "abc", "marker text lost");
    execute(state, makePacket(0x10, {0}, 0x30));
    expectFailure([&] { execute(state, makePacket(0x10, {0}, 0x30)); }, "underflow");
    execute(state, makePacket(0x10, {0}, 0x24));
    check(state.shader.empty() && state.context == AgcDriver::InitialContextRegisters() && state.dispatchIndirectBase == 0 && state.indexBase == 0 && !state.savedContext, "dispatch reset retained state");
}

void testAutoDraw() {
    check(AgcDriver::Pm4::AccessesMemory(0xc0012d00u), "auto draw must synchronize guest memory");
    AgcDriver::QueueState state;
    state.instanceCount = 4;
    state.indexBase = 1;
    state.indexType = 0xffffffffu;
    state.userConfig[0x24a] = 7;
    for (const auto flags : {2u, 0x22u}) {
        const auto draw = AgcDriver::Pm4::ResolveDraw(makePacket(0x2d, {3, flags}), state);
        check(!draw.indexed && draw.indexAddress == 0 && draw.indexSize == 0, "auto draw used the index buffer");
        check(draw.indexCount == 3 && draw.instanceCount == 4 && draw.firstVertex == 7 && draw.firstInstance == 0 && draw.flags == (flags & 0x20u), "auto draw parameters mismatch");
    }
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x2d, {3, 2}), 0x20); }, "compute queue");
    for (const auto flags : {0u, 1u, 3u, 0x20u, 0x42u}) {
        expectFailure([&] { AgcDriver::Pm4::Validate(makePacket(0x2d, {3, flags}), 0); }, "auto draw flags");
    }
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x2d, {3}), 0); }, "packet size");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x2d, {3, 2, 0}), 0); }, "packet size");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x2d, {3, 2}, 1), 0); }, "header flags");
    state.userConfig[0x24a] = std::numeric_limits<std::uint32_t>::max();
    check(AgcDriver::Pm4::ResolveDraw(makePacket(0x2d, {1, 2}), state).firstVertex == std::numeric_limits<std::uint32_t>::max(), "last vertex rejected");
    expectFailure([&] { AgcDriver::Pm4::ResolveDraw(makePacket(0x2d, {2, 2}), state); }, "vertex range overflow");
    check(AgcDriver::Pm4::ResolveDraw(makePacket(0x2d, {0, 2}), state).indexCount == 0, "empty auto draw rejected");
    state.userConfig.erase(0x24a);
    expectFailure([&] { AgcDriver::Pm4::ResolveDraw(makePacket(0x2d, {1, 2}), state); }, "GE_INDX_OFFSET");
}

void testIndexedDraw() {
    AgcDriver::QueueState state;
    alignas(4) std::array<std::uint32_t, 8> indices{};
    state.indexBase = reinterpret_cast<std::uintptr_t>(indices.data());
    state.instanceCount = 3;
    const auto packet = makePacket(0x35, {4, 2, 4, 0x20});
    for (std::uint32_t type = 0; type < 3; ++type) {
        state.indexType = type;
        const auto draw = AgcDriver::Pm4::ResolveDraw(packet, state);
        const auto size = type == 0 ? 2u : type == 1 ? 4u : 1u;
        check(draw.indexAddress == state.indexBase + 2 * size && draw.indexSize == size && draw.indexCount == 4 && draw.instanceCount == 3 && draw.flags == 0x20, "indexed draw state mismatch");
    }
    state.indexType = 0;
    check(AgcDriver::Pm4::ResolveDraw(packet, state).firstVertex == 0, "indexed draw invented a base vertex");
    state.userConfig[0x24a] = 0xd4d4;
    const auto offsetDraw = AgcDriver::Pm4::ResolveDraw(packet, state);
    check(offsetDraw.indexed && offsetDraw.firstVertex == 0xd4d4, "DRAW_INDEX_OFFSET_2 ignored GE_INDX_OFFSET");
    const auto address = reinterpret_cast<std::uintptr_t>(indices.data());
    const auto explicitDraw = AgcDriver::Pm4::ResolveDraw(makePacket(0x27, {4, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), 4, 0}), state);
    check(explicitDraw.indexed && explicitDraw.firstVertex == 0xd4d4 && explicitDraw.indexAddress == address, "DRAW_INDEX_2 ignored GE_INDX_OFFSET");
    state.userConfig.erase(0x24a);
    expectFailure([&] { AgcDriver::Pm4::ResolveDraw(packet, state); }, "GE_INDX_OFFSET");
    state.userConfig[0x24a] = 0;
    expectFailure([&] { AgcDriver::Pm4::Validate(packet, 0x20); }, "compute queue");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x35, {3, 0, 4, 0}), 0); }, "maximum index size");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x35, {4, 0, 4, 1}), 0); }, "draw flags");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x35, {4, 0, 4}), 0); }, "packet size");
    state.indexType = 3;
    expectFailure([&] { AgcDriver::Pm4::ResolveDraw(packet, state); }, "index type");
    state.indexType = 1;
    state.indexBase += 1;
    expectFailure([&] { AgcDriver::Pm4::ResolveDraw(packet, state); }, "misaligned index base");
    state.indexBase = std::numeric_limits<std::uint64_t>::max() - 3;
    expectFailure([&] { AgcDriver::Pm4::ResolveDraw(packet, state); }, "index address overflow");
    state.indexType = 2;
    state.indexBase = std::numeric_limits<std::uint64_t>::max() - 4;
    expectFailure([&] { AgcDriver::Pm4::ResolveDraw(packet, state); }, "address range overflow");
    state.indexBase = 0x1000;
    expectFailure([&] { AgcDriver::Pm4::ResolveDraw(packet, state); }, "guest");
}

void testIndirectDraw() {
    check(AgcDriver::Pm4::AccessesMemory(0xc0032400u) && AgcDriver::Pm4::AccessesMemory(0xc0083800u), "indirect draws must synchronize guest memory");
    check(AgcDriver::Pm4::UnsupportedReason(0xc0032400u).empty() && AgcDriver::Pm4::UnsupportedReason(0xc0082c00u).empty(), "indirect draws are rejected");
    AgcDriver::QueueState state;
    state.userConfig[0x24a] = 5;
    const auto packet = makePacket(0x24, {0, 0x280, 0x8e, 2});
    AgcDriver::Pm4::Validate(packet, 0);
    expectFailure([&] { AgcDriver::Pm4::Validate(packet, 0x20); }, "compute queue");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x24, {2, 0x280, 0x8e, 2}), 0); }, "misaligned indirect draw offset");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x24, {0, 0x10280, 0x8e, 2}), 0); }, "start-index location");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x24, {0, 0x200, 0x8e, 2}), 0); }, "register location");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x24, {0, 0x280, 0x8e, 0}), 0); }, "initiator");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x25, {0, 0x280, 0x8e, 2}), 0); }, "initiator");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x24, {0, 0x280, 0x8e}), 0); }, "packet size");
    AgcDriver::Pm4::Validate(makePacket(0x2c, {0, 0x8c, 0x280, 0x8d | (1u << 31u), 3, 0, 0, 16, 2}), 0);
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x2c, {0, 0x280, 0x8e, 0x280 | (1u << 27u), 3, 0, 0, 16, 2}), 0); }, "control bits");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x2c, {0, 0x280, 0x8e, 0x280, 3, 0, 0, 12, 2}), 0); }, "stride");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x38, {0, 0x280, 0x8e, 0x280, 3, 0, 0, 16, 0}), 0); }, "stride");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x2c, {0, 0x280, 0x8e, 0x280 | (1u << 30u), 3, 0, 0, 16, 2}), 0); }, "count address");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x2c, {0, 0x280, 0x8e, 0x280, 3, 0x1000, 0, 16, 2}), 0); }, "count address");
    expectFailure([&] { AgcDriver::Pm4::ResolveDraw(packet, state); }, "base has not been set");
    execute(state, makePacket(0x11, {1, 0x5d87fc40, 0x10}));
    auto draw = AgcDriver::Pm4::ResolveDraw(packet, state);
    check(draw.indirect.has_value() && draw.indirect->arguments == 0x105d87fc40ull && draw.indirect->opcode == 0x24 && draw.indirect->recordBytes == 16 && draw.indirect->stride == 16 && draw.indirect->count == 1 && !draw.indirect->countIndirect, "indirect draw arguments mismatch");
    check(draw.indirect->baseVertexLocation == 0x280 && draw.indirect->startInstanceLocation == 0x8e && draw.indirect->drawIndexLocation == 0x280 && !draw.indirect->drawIndexEnabled && draw.indirect->indxOffset == 5 && draw.firstVertex == 5 && !draw.indexed && draw.indexCount == 0 && draw.instanceCount == 0, "indirect draw locations mismatch");
    check(draw.indirect->RangeBytes() == 16 && draw.indirect->VertexDwordOffset() == 8 && draw.indirect->InstanceDwordOffset() == 12, "indirect draw record geometry mismatch");
    check(AgcDriver::Pm4::ResolveDraw(makePacket(0x24, {0x60, 0x280, 0x8e, 2}), state).indirect->arguments == 0x105d87fca0ull, "indirect draw offset not applied");
    const auto multi = AgcDriver::Pm4::ResolveDraw(makePacket(0x2c, {0x20, 0x8c, 0x280, 0x8d | (1u << 31u), 3, 0, 0, 32, 0x22}), state);
    check(multi.indirect->count == 3 && multi.indirect->stride == 32 && multi.indirect->drawIndexEnabled && multi.indirect->drawIndexLocation == 0x8d && !multi.indirect->countIndirect && multi.flags == 0x20 && multi.indirect->RangeBytes() == 80, "indirect multi-draw mismatch");
    expectFailure([&] { AgcDriver::Pm4::ResolveDraw(makePacket(0x25, {0, 0x8c, 0x280, 0}), state); }, "index base");
    alignas(4) std::array<std::uint16_t, 8> indices{};
    state.indexBase = reinterpret_cast<std::uintptr_t>(indices.data());
    state.indexType = 0;
    expectFailure([&] { AgcDriver::Pm4::ResolveDraw(makePacket(0x25, {0, 0x8c, 0x280, 0}), state); }, "INDEX_BUFFER_SIZE");
    state.indexBufferSize = 8;
    const auto indexed = AgcDriver::Pm4::ResolveDraw(makePacket(0x25, {0, 0x8c, 0x280, 0}), state);
    check(indexed.indexed && indexed.indexAddress == state.indexBase && indexed.indexCount == 8 && indexed.indexSize == 2 && indexed.indirect->recordBytes == 20 && indexed.indirect->stride == 20 && indexed.firstVertex == 0 && indexed.indirect->indxOffset == 0, "indexed indirect draw mismatch");
    check(indexed.indirect->VertexDwordOffset() == 12 && indexed.indirect->InstanceDwordOffset() == 16, "indexed record geometry mismatch");
    alignas(16) std::array<std::uint32_t, 10> records{3, 2, 7, 9, 4, 1, 5, 6, 0, 0};
    state.drawIndirectBase = reinterpret_cast<std::uintptr_t>(records.data());
    const auto local = AgcDriver::Pm4::ResolveDraw(makePacket(0x2c, {0, 0x280, 0x280, 0x280, 2, 0, 0, 16, 2}), state);
    const auto first = AgcDriver::Pm4::ReadDrawArguments(*local.indirect, 0);
    const auto second = AgcDriver::Pm4::ReadDrawArguments(*local.indirect, 1);
    check(first.count == 3 && first.instances == 2 && first.firstVertexOrIndex == 7 && first.vertexOffset == 0 && first.firstInstance == 9, "non-indexed record layout mismatch");
    check(second.count == 4 && second.instances == 1 && second.firstVertexOrIndex == 5 && second.firstInstance == 6, "second record mismatch");
    expectFailure([&] { AgcDriver::Pm4::ReadDrawArguments(*local.indirect, 2); }, "record index");
    records = {3, 2, 7, 9, 11, 0, 0, 0, 0, 0};
    const auto indexedRecord = AgcDriver::Pm4::ReadDrawArguments(*AgcDriver::Pm4::ResolveDraw(makePacket(0x25, {0, 0x8c, 0x280, 0}), state).indirect, 0);
    check(indexedRecord.count == 3 && indexedRecord.instances == 2 && indexedRecord.firstVertexOrIndex == 7 && indexedRecord.vertexOffset == 9 && indexedRecord.firstInstance == 11, "indexed record layout mismatch");
    alignas(4) std::uint32_t countValue = 2;
    const auto counted = AgcDriver::Pm4::ResolveDraw(makePacket(0x2c, {0, 0x280, 0x280, 0x280 | (1u << 30u), 5, low(&countValue), high(&countValue), 16, 2}), state);
    check(counted.indirect->countIndirect && counted.indirect->count == 5 && counted.indirect->countAddress == reinterpret_cast<std::uintptr_t>(&countValue) && AgcDriver::Pm4::ReadDrawCount(*counted.indirect) == 2, "indirect draw count mismatch");
    expectFailure([&] { AgcDriver::Pm4::ReadDrawCount(*local.indirect); }, "count address");
}

void testMemory() {
    AgcDriver::QueueState state;
    std::array<std::uint32_t, 4> data{0, 0, 0, 0};
    execute(state, makePacket(0x37, {0x100, low(data.data()), high(data.data()), 11, 12}));
    check(data[0] == 11 && data[1] == 12, "WRITE_DATA increment failed");
    execute(state, makePacket(0x37, {0x10100, low(data.data()), high(data.data()), 21, 22}));
    check(data[0] == 22 && data[1] == 12, "WRITE_DATA fixed destination failed");
    execute(state, makePacket(0x81, {4, 31, 32}));
    execute(state, makePacket(0x83, {4, 2, low(data.data()), high(data.data())}));
    check(data[0] == 31 && data[1] == 32, "constant RAM round trip failed");
    expectFailure([&] { execute(state, makePacket(0x81, {0xbffc, 1, 2})); }, "overflow");
    expectFailure([&] { execute(state, makePacket(0x40, {0x10105, 0, 0, low(data.data()), high(data.data())})); }, "64-bit immediate");
}

void testCopies() {
    AgcDriver::QueueState state;
    alignas(8) std::array<std::uint32_t, 4> source{11, 12, 13, 14};
    alignas(8) std::array<std::uint32_t, 4> destination{};
    execute(state, makePacket(0x40, {0x10101, low(source.data()), high(source.data()), low(destination.data()), high(destination.data())}));
    check(destination[0] == 11 && destination[1] == 12 && destination[2] == 0, "64-bit COPY_DATA failed");
    execute(state, makePacket(0x40, {0x105, 0x12345678, 0, low(destination.data()), high(destination.data())}));
    check(destination[0] == 0x12345678, "immediate COPY_DATA failed");
    execute(state, makePacket(0x50, {0x60000000, low(source.data()), high(source.data()), low(destination.data()), high(destination.data()), 16}));
    check(source == destination, "DMA_DATA copy failed");
    execute(state, makePacket(0x50, {0x40000000, 0x44332211, 0, low(destination.data()), high(destination.data()), 6}));
    check(destination[0] == 0x44332211 && destination[1] == 0x00002211, "DMA_DATA byte fill failed");
    constexpr std::uint32_t cachePolicies = (1u << 13u) | (2u << 25u);
    const auto toGds = makePacket(0x50, {0x60100000 | cachePolicies, low(source.data()), high(source.data()), 0x100, 0, 16});
    check(!AgcDriver::Pm4::ResolveStore(toGds, state, 64).has_value(), "DMA_DATA to GDS resolved as a memory store");
    execute(state, toGds);
    execute(state, makePacket(0x50, {0x20100000, 0x104, 0, 0xfff8, 0, 8}));
    destination = {};
    execute(state, makePacket(0x50, {0x20000000 | cachePolicies, 0xfff8, 0, low(destination.data()), high(destination.data()), 8}));
    check(destination[0] == 12 && destination[1] == 13 && destination[2] == 0, "DMA_DATA GDS to GDS round trip failed");
    const auto fromGds = makePacket(0x50, {0x20000000, 0x100, 0, low(destination.data()), high(destination.data()), 16});
    const auto store = AgcDriver::Pm4::ResolveStore(fromGds, state, 64);
    check(store.has_value() && store->Bytes().size() == 16 && std::memcmp(store->Bytes().data(), source.data(), 16) == 0, "DMA_DATA from GDS did not resolve its source bytes");
    destination = {};
    execute(state, fromGds);
    check(source == destination, "DMA_DATA GDS round trip failed");
    expectFailure([&] { execute(state, makePacket(0x50, {0x60100000, low(source.data()), high(source.data()), 0xfffc, 0, 8})); }, "exceeds the GDS");
    expectFailure([&] { execute(state, makePacket(0x50, {0x20000000, 0, 1, low(destination.data()), high(destination.data()), 4})); }, "exceeds the GDS");
    expectFailure([&] { execute(state, makePacket(0x50, {0x60200000, low(source.data()), high(source.data()), 0, 0, 4})); }, "destination is not implemented");
    expectFailure([&] { execute(state, makePacket(0x50, {0x60000000 | (1u << 15u), low(source.data()), high(source.data()), low(destination.data()), high(destination.data()), 4})); }, "reserved fields");
    expectFailure([&] { execute(state, makePacket(0x37, {0x100, 0x1000, 0, 1})); }, "guest");
#ifdef _WIN32
    auto* memory = VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    check(memory != nullptr, "VirtualAlloc failed");
    DWORD previous = 0;
    check(VirtualProtect(memory, 4096, PAGE_READONLY, &previous) != 0, "VirtualProtect failed");
    try {
        expectFailure([&] { execute(state, makePacket(0x37, {0x100, low(memory), high(memory), 1})); }, "write permission");
    } catch (...) { VirtualFree(memory, 0, MEM_RELEASE); throw; }
    check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
#endif
}

// DecodeMemoryCopy: only a memory-to-memory DMA_DATA with incrementing addresses and disjoint,
// non-empty ranges is a copy the driver may record on the GPU (VulkanDevice::CopyBuffer).
void testMemoryCopyDecode() {
    using AgcDriver::Pm4::DecodeMemoryCopy;
    constexpr std::uint64_t source = 0x1120000000ull, destination = 0x403d7b400ull;
    const auto packet = [&](std::uint32_t control, std::uint64_t from, std::uint64_t to, std::uint32_t command) {
        return makePacket(0x50, {control, static_cast<std::uint32_t>(from), static_cast<std::uint32_t>(from >> 32u), static_cast<std::uint32_t>(to), static_cast<std::uint32_t>(to >> 32u), command});
    };
    const auto copy = DecodeMemoryCopy(packet(0x60000000, source, destination, 0xbdd800));
    check(copy.has_value() && copy->source == source && copy->destination == destination && copy->bytes == 0xbdd800, "a memory-to-memory DMA_DATA did not decode as a copy");
    check(DecodeMemoryCopy(packet(0x00000000, source, destination, 64)).has_value(), "a DMA_DATA with memory selectors 0 did not decode as a copy");
    check(!DecodeMemoryCopy(packet(0x40000000, 0x44332211, destination, 64)).has_value(), "an immediate fill decoded as a copy");
    check(!DecodeMemoryCopy(packet(0x60100000, source, 0x100, 64)).has_value(), "a DMA_DATA to the GDS decoded as a copy");
    check(!DecodeMemoryCopy(packet(0x20000000, 0x100, destination, 64)).has_value(), "a DMA_DATA from the GDS decoded as a copy");
    check(!DecodeMemoryCopy(packet(0x60000000, source, destination, 64 | (1u << 28u))).has_value(), "a non-incrementing source decoded as a copy");
    check(!DecodeMemoryCopy(packet(0x60000000, source, destination, 64 | (1u << 29u))).has_value(), "a non-incrementing destination decoded as a copy");
    check(!DecodeMemoryCopy(packet(0x60000000, source, destination, 0)).has_value(), "an empty DMA_DATA decoded as a copy");
    check(!DecodeMemoryCopy(packet(0x60000000, source, source + 32, 64)).has_value(), "overlapping ranges decoded as a copy");
    check(DecodeMemoryCopy(packet(0x60000000, source, source + 64, 64)).has_value(), "adjacent ranges did not decode as a copy");
    check(!DecodeMemoryCopy(makePacket(0x40, {0x10101, 0, 0, 0, 0})).has_value(), "a COPY_DATA decoded as a DMA_DATA copy");
}

// The CP's GDS behind a device's backing (Pm4::InstallGdsBacking): the bytes move into the backing
// and back, DMA_DATA reads and writes the backing, and after a shader use a store from the GDS is
// not resolved ahead of the drain (its bytes may still be on the way), until the CP's next access.
std::array<std::byte, AgcDriver::Pm4::GdsBytes> gdsBacking{};
std::array<std::byte, AgcDriver::Pm4::GdsBytes> otherBacking{};

void testGdsBacking() {
    using namespace AgcDriver::Pm4;
    AgcDriver::QueueState state;
    alignas(8) std::array<std::uint32_t, 4> source{21, 22, 23, 24};
    alignas(8) std::array<std::uint32_t, 4> destination{};
    execute(state, makePacket(0x50, {0x60100000, low(source.data()), high(source.data()), 0x200, 0, 16}));
    const int owner = 0;
    const int other = 0;
    check(InstallGdsBacking(&owner, gdsBacking), "the GDS backing was refused");
    check(std::memcmp(gdsBacking.data() + 0x200, source.data(), 16) == 0, "the backing did not take the GDS bytes");
    check(!InstallGdsBacking(&other, otherBacking), "a second GDS backing was installed");
    execute(state, makePacket(0x50, {0x60100000, low(source.data()), high(source.data()), 0x300, 0, 8}));
    check(std::memcmp(gdsBacking.data() + 0x300, source.data(), 8) == 0, "DMA_DATA to GDS missed the backing");
    // A shader's store into the device's buffer: a store from the GDS waits for the drain.
    const std::uint32_t stored = 0xcafe;
    std::memcpy(gdsBacking.data() + 0x300, &stored, 4);
    NoteGdsShaderUse();
    const auto fromGds = makePacket(0x50, {0x20000000, 0x300, 0, low(destination.data()), high(destination.data()), 8});
    check(!ResolveStore(fromGds, state, 64).has_value(), "a store from the GDS was resolved while shader results may be on the way");
    execute(state, fromGds);
    check(destination[0] == stored && destination[1] == 22, "DMA_DATA from GDS did not read the backing");
    const auto store = ResolveStore(fromGds, state, 64);
    check(store.has_value() && std::memcmp(store->Bytes().data(), &stored, 4) == 0, "a store from the GDS was not resolved after the CP's access");
    ReleaseGdsBacking(&other);
    check(std::memcmp(gdsBacking.data() + 0x200, source.data(), 16) == 0, "a release by another owner changed the backing");
    ReleaseGdsBacking(&owner);
    std::memset(gdsBacking.data(), 0, gdsBacking.size());
    destination = {};
    execute(state, makePacket(0x50, {0x20000000, 0x300, 0, low(destination.data()), high(destination.data()), 8}));
    check(destination[0] == stored, "the released backing's bytes did not return to the CP");
    check(InstallGdsBacking(&other, otherBacking), "a backing was refused after the release");
    ReleaseGdsBacking(&other);
}

void testMemorySynchronization() {
    struct MemoryState {
        std::uint32_t source = 0;
        std::uint32_t destination = 0;
        bool read = false;
        bool written = false;
    };
    static MemoryState memory;
    memory = {};
    AgcDriver::QueueState state;
    struct FlushHookReset {
        ~FlushHookReset() { AgcDriver::GuestMemory::SetFlushHook(nullptr); }
    } reset;
    const auto resolve = [](std::uint64_t address, std::size_t bytes) {
        check(bytes == sizeof(std::uint32_t), "memory transfer resolved an unrelated range");
        if (address == reinterpret_cast<std::uintptr_t>(&memory.destination)) {
            memory.written = true;
        } else {
            check(address == reinterpret_cast<std::uintptr_t>(&memory.source), "memory transfer resolved an unrelated source");
            memory.source = 42;
            memory.read = true;
        }
    };
    AgcDriver::GuestMemory::SetFlushHook(resolve);
    execute(state, makePacket(0x37, {0x100, low(&memory.destination), high(&memory.destination), 17}));
    check(memory.written && !memory.read && memory.destination == 17, "WRITE_DATA did not synchronize its destination");
    for (const auto opcode : {0x40u, 0x50u}) {
        memory = {};
        const auto packet = opcode == 0x40
            ? makePacket(opcode, {0x101, low(&memory.source), high(&memory.source), low(&memory.destination), high(&memory.destination)})
            : makePacket(opcode, {0x60000000, low(&memory.source), high(&memory.source), low(&memory.destination), high(&memory.destination), 4});
        execute(state, packet);
        check(memory.read && memory.written && memory.destination == 42, "memory copy used stale data before range synchronization");
    }
    AgcDriver::GuestMemory::SetFlushHook([](std::uint64_t, std::size_t) {
        throw std::runtime_error("range synchronization failed");
    });
    expectFailure([&] { execute(state, makePacket(0x37, {0x100, low(&memory.destination), high(&memory.destination), 99})); }, "range synchronization failed");
    check(memory.destination == 42, "failed synchronization changed the destination");
}

void testWriteChangedKeepsUntouchedBytes() {
    alignas(256) static std::uint8_t guest[256];
    std::memset(guest, 0, sizeof(guest));
    std::vector<std::byte> original(sizeof(guest)), current(sizeof(guest));
    current[3] = std::byte{7};
    guest[100] = 0x55;
    AgcDriver::GuestMemory::WriteChanged(reinterpret_cast<std::uintptr_t>(guest), current, original);
    check(guest[3] == 7 && guest[100] == 0x55, "write-back rolled back a byte the GPU did not change");
}

void testEventWrite() {
    for (const auto eventType : {0x07u, 0x0fu, 0x10u}) {
        AgcDriver::Pm4::Validate(makePacket(0x46, {0x400u | eventType}), 0);
        for (std::uint32_t index = 0; index < 8; ++index) {
            if (index == 4) continue;
            expectFailure([&] { AgcDriver::Pm4::Validate(makePacket(0x46, {(index << 8u) | eventType}), 0); }, "partial-flush event index");
        }
        if (eventType == 0x07) {
            AgcDriver::Pm4::Validate(makePacket(0x46, {0x407}), 0x20);
        } else {
            expectFailure([&] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x400u | eventType}), 0x20); }, "compute queue");
        }
    }
    for (const auto eventType : {0x16u, 0x31u, 0x2au, 0x2cu, 0x2eu}) {
        for (const auto index : {0u, 7u}) {
            AgcDriver::Pm4::Validate(makePacket(0x46, {(index << 8u) | eventType}), 0);
        }
        for (std::uint32_t index = 1; index < 7; ++index) {
            expectFailure([&] { AgcDriver::Pm4::Validate(makePacket(0x46, {(index << 8u) | eventType}), 0); }, "cache-flush event index");
        }
        expectFailure([&] { AgcDriver::Pm4::Validate(makePacket(0x46, {eventType}), 0x20); }, "compute queue");
    }
    AgcDriver::Pm4::Validate(makePacket(0x46, {0x139, 0x1000, 0x2}), 0);
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x139, 0x1000, 0x2}), 0x20); }, "compute queue");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x039, 0x1000, 0x2}), 0); }, "counter dump event index");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x139, 0x1004, 0x2}), 0); }, "misaligned occlusion counter");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x139, 0, 0}), 0); }, "null or misaligned occlusion counter");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x139, 0x1000}), 0); }, "packet size");
    AgcDriver::Pm4::Validate(makePacket(0x46, {0x038}), 0);
    AgcDriver::Pm4::Validate(makePacket(0x46, {0x03a}), 0);
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x038}), 0x20); }, "compute queue");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x138, 0x1000, 0x2}), 0); }, "packet size");
    for (const auto bit : {0x40u, 0x80u, 0x800u, 0x80000000u}) {
        expectFailure([&] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x410u | bit}), 0); }, "reserved bits");
    }
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x410}, 1), 0); }, "header flags");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x410, 0, 0}), 0); }, "packet size");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x13a, 0, 0}), 0); }, "packet size");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x46, {0x0d}), 0); }, "event type 13");
}

void testGpuTimestampScale() {
    using AgcDriver::Pm4::ParseGpuTimestampScale;
    using AgcDriver::Pm4::ScaleGpuClockNs;
    check(ParseGpuTimestampScale(nullptr) == 100 && ParseGpuTimestampScale("115") == 115 && ParseGpuTimestampScale("1") == 1 && ParseGpuTimestampScale("1000") == 1000, "timestamp scale parsing");
    for (const char* rejected : {"", "0", "1001", "115%", "-5", "abc"}) {
        expectFailure([&] { ParseGpuTimestampScale(rejected); }, "APS5_GPU_TIMESTAMP_SCALE");
    }
    constexpr std::uint64_t origin = 5'000'000'000'000ull;
    check(ScaleGpuClockNs(origin, origin, 115) == origin, "the scaled clock starts at its origin");
    check(ScaleGpuClockNs(origin + 16'666'667, origin, 100) == origin + 16'666'667, "a 100 percent clock is the host clock");
    check(ScaleGpuClockNs(origin + 10'000'000, origin, 115) == origin + 11'500'000 && ScaleGpuClockNs(origin + 10'000'000, origin, 50) == origin + 5'000'000, "deltas scale by the percentage");
    check(ScaleGpuClockNs(origin + 101, origin, 115) == origin + 116, "sub-100 ns remainders scale too");
    const std::uint64_t day = 86'400'000'000'000ull;
    check(ScaleGpuClockNs(origin + day, origin, 1000) == origin + 10 * day, "a long run does not overflow");
    std::uint64_t previous = 0;
    for (std::uint64_t step = 0; step < 1000; ++step) {
        const auto value = ScaleGpuClockNs(origin + step * 7, origin, 115);
        check(value >= previous, "the scaled clock never goes backwards");
        previous = value;
    }
    expectFailure([&] { ScaleGpuClockNs(origin - 1, origin, 115); }, "backwards");
}

void testAcquireMem() {
    const auto captured = makePacket(0x58, {0x02007fc0, 0, 0, 0, 0, 10, 0x200});
    AgcDriver::Pm4::Validate(captured, 0);
    check(AgcDriver::Pm4::UsesGpuCacheBarrier(captured), "L1 acquire must preserve GPU render targets");
    for (const auto flags : {0u, 0x200u, 0x3ffu, 0x10200u, 0x20200u}) {
        auto packet = captured;
        packet[7] = flags;
        check(AgcDriver::Pm4::UsesGpuCacheBarrier(packet), "GPU cache acquire requires an unnecessary host writeback");
    }
    for (const auto flags : {0x400u, 0x800u, 0x1000u, 0x4000u, 0x8000u}) {
        auto packet = captured;
        packet[7] = flags;
        check(!AgcDriver::Pm4::UsesGpuCacheBarrier(packet), "L2 acquire lost host synchronization");
    }
    check(!AgcDriver::Pm4::UsesGpuCacheBarrier(makePacket(0x58, {0x00800000, 0xffffffff, 0, 0, 0, 10})), "legacy acquire lost host synchronization");
    expectFailure([] { AgcDriver::Pm4::UsesGpuCacheBarrier({}); }, "requires ACQUIRE_MEM");
    expectFailure([] { AgcDriver::Pm4::UsesGpuCacheBarrier(makePacket(0x58, {0, 0, 0, 0, 0, 0, 0x2000})); }, "cache discard");
    AgcDriver::Pm4::Validate(makePacket(0x58, {0x82007fc0, 1, 0, 0xffffffff, 0, 0xffff, 0x200}), 0);
    AgcDriver::Pm4::Validate(makePacket(0x58, {0x80000000, 0, 0, 0, 0, 10, 0x200}), 0x20);
    AgcDriver::Pm4::Validate(makePacket(0x58, {0x00800000, 0xffffffff, 0, 0, 0, 10}), 0);
    AgcDriver::Pm4::Validate(makePacket(0x58, {0x80800000, 16, 0, 0x1000, 0, 0}), 0x20);
    expectFailure([&] { AgcDriver::Pm4::Validate(captured, 0x20); }, "compute queue");
    const auto invalidWord = [&](std::size_t index, std::uint32_t value, const char* reason) {
        auto packet = captured;
        packet[index] = value;
        expectFailure([&] { AgcDriver::Pm4::Validate(packet, 0); }, reason);
    };
    invalidWord(0, captured[0] | 1u, "header flags");
    invalidWord(1, 4, "control flags");
    invalidWord(1, 0x00800000, "control flags");
    invalidWord(3, 1, "above 40 bits");
    invalidWord(5, 1, "above 40 bits");
    invalidWord(6, 0x10000, "poll interval");
    invalidWord(7, 0x40000, "GCR flags");
    invalidWord(7, 0x2000, "cache discard");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x58, {0, 2, 0, 0xffffffff, 0, 0, 0}), 0); }, "range exceeds");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x58, {0, 0, 0, 0, 0}), 0); }, "packet size");
    expectFailure([] { AgcDriver::Pm4::Validate(makePacket(0x58, {0, 0, 0, 0, 0, 0, 0, 0}), 0); }, "packet size");
}

void testStateEffects() {
    using AgcDriver::Pm4::StateEffect;
    const auto effect = [](std::uint32_t opcode, std::uint32_t flags = 0) { return AgcDriver::Pm4::PacketStateEffect(makePacket(opcode, {0}, flags)[0]); };
    for (const auto opcode : {0x11u, 0x12u, 0x13u, 0x26u, 0x2au, 0x2fu, 0x69u, 0x76u, 0x79u, 0x7au, 0x81u}) check(effect(opcode) == StateEffect::Registers, "a register packet is not classed as a register write");
    for (const auto opcode : {0x63u, 0x64u, 0x9fu}) check(effect(opcode) == StateEffect::Loads, "a register load is not classed as a load");
    for (const auto opcode : {0x15u, 0x16u, 0x27u, 0x2du, 0x35u, 0x37u, 0x3cu, 0x40u, 0x42u, 0x46u, 0x49u, 0x50u, 0x58u, 0x83u, 0x93u}) check(effect(opcode) == StateEffect::None, "a packet that leaves the state is classed as changing it");
    check(effect(0x10, 0x24) == StateEffect::Registers && effect(0x10, 0x68) == StateEffect::Registers && effect(0x10, 0x2c) == StateEffect::Registers && effect(0x10, 0x30) == StateEffect::Registers, "a custom state packet is not classed as a register write");
    check(effect(0x10) == StateEffect::None && AgcDriver::Pm4::PacketStateEffect(0xc004105cu) == StateEffect::None && AgcDriver::Pm4::PacketStateEffect(0xc0021018u) == StateEffect::None, "a NOP, flip or rendering wait is classed as changing the state");
    check(effect(0x28) == StateEffect::Unknown && effect(0x10, 0x50) == StateEffect::Unknown, "an unmodeled packet is not Unknown");
    std::array<std::uint32_t, 4> pairs{0x20, 5, 0x21, 6};
    const auto packet = makePacket(0x9f, {low(pairs.data()), high(pairs.data()), 0x80000000, 2});
    AgcDriver::QueueState executed, split;
    execute(executed, packet);
    const auto read = AgcDriver::Pm4::ReadRegisterPairs(packet);
    check(read == std::vector<std::uint32_t>(pairs.begin(), pairs.end()), "register load pairs read wrong");
    AgcDriver::Pm4::ApplyRegisterPairs(packet, split, read);
    check(executed.context == split.context, "a register load applied apart differs from Execute");
}

using AgcDriver::DriverDetail::DrawAhead;
using AgcDriver::DriverDetail::PreparedDraw;
using AgcDriver::DriverDetail::Submission;

std::uint64_t stateHash(const AgcDriver::QueueState& state) {
    std::uint64_t key = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    for (const auto* bank : {&state.context, &state.shader, &state.userConfig}) {
        mix(bank->size());
        for (const auto& [offset, value] : *bank) {
            mix(offset);
            mix(value);
        }
    }
    for (const auto value : {state.indexBase, std::uint64_t{state.indexType}, std::uint64_t{state.instanceCount}, std::uint64_t{state.indexBufferSize}, state.markers.size(), std::uint64_t{state.savedContext.has_value()}}) mix(value);
    return key;
}

void waitFor(const std::function<bool()>& condition, const char* reason) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!condition()) {
        check(std::chrono::steady_clock::now() < deadline, reason);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::vector<bool> runWorker(DrawAhead& ahead, const Submission& submission, AgcDriver::QueueState& state, const std::function<void(std::size_t)>& beforeDraw = {}) {
    ahead.Begin(submission, state);
    std::vector<bool> handed;
    const auto& commands = submission.commands;
    for (std::size_t cursor = 0; cursor < commands.size();) {
        const auto header = commands[cursor];
        const auto count = AgcDriver::Pm4::PacketWords(header);
        const auto packet = std::span(commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        cursor += count;
        if (AgcDriver::Pm4::DrawOpcode(opcode)) {
            if (beforeDraw) beforeDraw(handed.size());
            const auto prepared = ahead.TakeDraw();
            check(prepared == nullptr || prepared->drawKey == stateHash(state), "a handed-over draw was prepared from another state");
            handed.push_back(prepared != nullptr);
        } else if (AgcDriver::Pm4::RegisterLoadOpcode(opcode)) {
            const auto pairs = AgcDriver::Pm4::ReadRegisterPairs(packet);
            AgcDriver::Pm4::ApplyRegisterPairs(packet, state, pairs);
            ahead.ConfirmLoad(pairs);
        } else if (AgcDriver::Pm4::PacketStateEffect(header) == AgcDriver::Pm4::StateEffect::Registers) {
            execute(state, std::vector<std::uint32_t>(packet.begin(), packet.end()));
        }
    }
    ahead.End();
    return handed;
}

void testDrawAhead() {
    std::array<std::uint32_t, 4> firstPairs{0x30, 1, 0x31, 2};
    std::array<std::uint32_t, 2> shaderPairs{0x40, 3};
    std::array<std::uint32_t, 2> secondPairs{0x30, 7};
    std::uint32_t written = 0;
    Submission submission{};
    submission.queue = 0;
    for (const auto& packet : {
        makePacket(0x69, {0x30, 9}),
        makePacket(0x2d, {3, 2}),
        makePacket(0x9f, {low(firstPairs.data()), high(firstPairs.data()), 0x80000000, 2}),
        makePacket(0x76, {0x50, 11}),
        makePacket(0x2d, {3, 2}),
        makePacket(0x10, {1, 0}, 0x68),
        makePacket(0x63, {low(shaderPairs.data()), high(shaderPairs.data()), 0x80000000, 1}),
        makePacket(0x2d, {3, 2}),
        makePacket(0x37, {0x00000500, low(&written), high(&written), 5}),
        makePacket(0x10, {2, 0}, 0x68),
        makePacket(0x2f, {4}),
        makePacket(0x2d, {3, 2}),
        makePacket(0x9f, {low(secondPairs.data()), high(secondPairs.data()), 0x80000000, 1}),
        makePacket(0x10, {0x00636261}, 0x2c),
        makePacket(0x2d, {3, 2}),
        makePacket(0x10, {0}, 0x30),
        makePacket(0x79, {0x242, 4}),
        makePacket(0x2d, {3, 2})
    }) submission.commands.insert(submission.commands.end(), packet.begin(), packet.end());
    constexpr std::size_t draws = 6;

    std::atomic<std::size_t> prepares{0};
    std::atomic<bool> corruptSecondLoad{false};
    std::atomic<bool> released{true};
    const auto prepare = [&](const AgcDriver::QueueState& state, std::span<const std::uint32_t>, const Submission&) {
        auto prepared = std::make_shared<PreparedDraw>();
        prepared->drawKey = stateHash(state);
        ++prepares;
        return prepared;
    };
    const auto readPairs = [&](std::span<const std::uint32_t> packet) {
        auto pairs = AgcDriver::Pm4::ReadRegisterPairs(packet);
        if (corruptSecondLoad && packet[1] == low(secondPairs.data())) pairs[1] ^= 0xffu;
        return pairs;
    };
    const auto started = [&] { waitFor([&] { return released.load(); }, "the front end was never released"); };

    {
        DrawAhead ahead(prepare, readPairs, started);
        AgcDriver::QueueState state;
        const auto handed = runWorker(ahead, submission, state, [&](std::size_t index) {
            if (index == 0) waitFor([&] { return prepares.load() == draws; }, "the front end did not prepare every draw");
        });
        check(handed == std::vector<bool>(draws, true), "a draw prepared ahead was not handed over");
        const auto totals = ahead.Totals();
        check(totals.handed == draws && totals.loadsAhead == 3 && totals.loadMismatches == 0, "front end counters wrong");
        check(written == 0, "the front end executed a memory store");

        prepares = 0;
        const auto again = runWorker(ahead, submission, state, [&](std::size_t index) {
            if (index == 0) waitFor([&] { return prepares.load() == draws; }, "the front end did not prepare every draw again");
        });
        check(again == std::vector<bool>(draws, true), "a second submission's draws were not handed over");
    }
    {
        corruptSecondLoad = true;
        prepares = 0;
        DrawAhead ahead(prepare, readPairs, started);
        AgcDriver::QueueState state;
        const auto handed = runWorker(ahead, submission, state, [&](std::size_t index) {
            if (index == 0) waitFor([&] { return prepares.load() == draws; }, "the front end did not prepare every draw");
        });
        check(handed == std::vector<bool>{true, true, true, true, false, false}, "draws after a differing load were handed over");
        check(ahead.Totals().loadMismatches == 1 && ahead.Totals().poisonedDraws == 2, "a differing load was not counted");
        corruptSecondLoad = false;
    }
    {
        corruptSecondLoad = true;
        released = false;
        prepares = 0;
        DrawAhead ahead(prepare, [&](std::span<const std::uint32_t> packet) {
            check(packet[1] != low(firstPairs.data()) && packet[1] != low(shaderPairs.data()), "the front end read a load the worker had confirmed");
            return readPairs(packet);
        }, started);
        AgcDriver::QueueState state;
        corruptSecondLoad = false;
        const auto handed = runWorker(ahead, submission, state, [&](std::size_t index) {
            if (index != 3) return;
            released = true;
            waitFor([&] { return prepares.load() == 3; }, "the released front end did not prepare the remaining draws");
        });
        check(handed == std::vector<bool>{false, false, false, true, true, true}, "the worker's own draws or the prepared ones went wrong");
        const auto totals = ahead.Totals();
        check(totals.workerOwn == 3 && totals.loadsBehind == 2 && totals.loadsAhead == 1, "worker-first loads were not used by the front end");
    }
    {
        auto stopped = submission;
        const auto unknown = makePacket(0x28, {0});
        stopped.commands.insert(stopped.commands.begin() + 2 + 3, unknown.begin(), unknown.end());
        prepares = 0;
        DrawAhead ahead(prepare, readPairs, started);
        AgcDriver::QueueState state;
        const auto handed = runWorker(ahead, stopped, state, [&](std::size_t index) {
            if (index == 0) waitFor([&] { return ahead.Totals().stopped == 1; }, "the front end did not stop at an unknown packet");
        });
        check(handed == std::vector<bool>{true, false, false, false, false, false}, "draws after an unknown packet were prepared");
        prepares = 0;
        const auto failing = [&](std::span<const std::uint32_t> packet) -> std::vector<std::uint32_t> {
            if (packet[1] == low(shaderPairs.data())) throw std::runtime_error("unreadable");
            return AgcDriver::Pm4::ReadRegisterPairs(packet);
        };
        DrawAhead unreadable(prepare, failing, started);
        AgcDriver::QueueState other;
        const auto partial = runWorker(unreadable, submission, other, [&](std::size_t index) {
            if (index == 0) waitFor([&] { return unreadable.Totals().stopped == 1; }, "the front end did not stop at an unreadable load");
        });
        check(partial == std::vector<bool>{true, true, false, false, false, false}, "draws after an unreadable load were handed over");
    }
    {
        DrawAhead ahead([&](const AgcDriver::QueueState& state, std::span<const std::uint32_t> packet, const Submission& job) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            return prepare(state, packet, job);
        }, readPairs, started);
        AgcDriver::QueueState state;
        std::size_t handedTotal = 0;
        for (int round = 0; round < 200; ++round) {
            const auto handed = runWorker(ahead, submission, state);
            handedTotal += static_cast<std::size_t>(std::count(handed.begin(), handed.end(), true));
        }
        const auto totals = ahead.Totals();
        check(totals.handed == handedTotal && totals.handed + totals.workerOwn + totals.poisonedDraws == totals.draws && totals.draws == 200 * draws, "racing submissions lost a draw");
    }
}

void testDriverSubmission() {
    std::array<std::uint32_t, 2> source{0x10, 73};
    std::array<std::uint32_t, 1> destination{};
    std::vector<std::uint32_t> commands;
    for (const auto& packet : {
        makePacket(0x9f, {low(source.data()), high(source.data()), 0x80000000, 1}),
        makePacket(0x81, {0, 83}),
        makePacket(0x42, {0}),
        makePacket(0x46, {0x410}),
        makePacket(0x46, {0x407}),
        makePacket(0x46, {0x40f}),
        makePacket(0x46, {0x16}),
        makePacket(0x46, {0x731}),
        makePacket(0x46, {0x2a}),
        makePacket(0x46, {0x72c}),
        makePacket(0x46, {0x2e}),
        makePacket(0x58, {0x02007fc0, 0, 0, 0, 0, 10, 0x200}),
        makePacket(0x58, {0x00800000, 0xffffffff, 0, 0, 0, 10}),
        makePacket(0x83, {0, 1, low(destination.data()), high(destination.data())})
    }) commands.insert(commands.end(), packet.begin(), packet.end());
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&packet) == 0, "PM4 submission failed");
    AgcDriverWaitIdle_nid_postfix();
    check(destination[0] == 83, "worker did not execute PM4 memory operations");
    auto rejectedCommands = commands;
    const auto unsupportedEvent = makePacket(0x46, {0x0d});
    rejectedCommands.insert(rejectedCommands.end(), unsupportedEvent.begin(), unsupportedEvent.end());
    Packet rejectedPacket{rejectedCommands.data(), static_cast<std::uint32_t>(rejectedCommands.size()), 0, {}};
    destination[0] = 0;
    expectFailure([&] { sceAgcDriverSubmitDcb(&rejectedPacket); }, "EVENT_WRITE at DWORD");
    AgcDriverWaitIdle_nid_postfix();
    check(destination[0] == 0, "rejected event submission executed a prefix");
    destination[0] = 0;
    const auto emptyDraw = makePacket(0x2d, {0, 2});
    commands.insert(commands.end(), emptyDraw.begin(), emptyDraw.end());
    packet = Packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&packet) == 0, "auto draw submission failed");
    AgcDriverWaitIdle_nid_postfix();
    check(destination[0] == 83, "empty auto draw prevented command execution");
    destination[0] = 0;
    const auto draw = makePacket(0x2d, {3, 3});
    commands.insert(commands.end(), draw.begin(), draw.end());
    packet = Packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    expectFailure([&] { sceAgcDriverSubmitDcb(&packet); }, "DRAW_INDEX_AUTO at DWORD");
    AgcDriverWaitIdle_nid_postfix();
    check(destination[0] == 0, "rejected submission executed a prefix");
}

void testAsyncMemoryFailure() {
    auto commands = makePacket(0x37, {0x100, 0x1000, 0, 1});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&packet) == 0, "memory packet was not submitted");
    expectFailure([] { AgcDriverWaitIdle_nid_postfix(); }, "guest");
    expectFailure([] { AgcDriverSuspendPoint_nid_postfix(); }, "guest");
    expectFailure([&] { sceAgcDriverSubmitDcb(&packet); }, "guest");
    expectFailure([] { LibcRunShutdown_nid_postfix(); }, "guest");
}

}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "failure") {
            testAsyncMemoryFailure();
            std::puts("PM4 asynchronous memory failure propagated to idle, suspend, submit and shutdown");
            return 0;
        }
        testCatalog();
        testWriteChangedKeepsUntouchedBytes();
        testRegisters();
        testRegisterFile();
        testContextAndBases();
        testIndexedDraw();
        testAutoDraw();
        testIndirectDraw();
        testMemory();
        testCopies();
        testMemoryCopyDecode();
        testGdsBacking();
        testMemorySynchronization();
        testEventWrite();
        testGpuTimestampScale();
        testAcquireMem();
        testStateEffects();
        testDrawAhead();
        testDriverSubmission();
        LibcRunShutdown_nid_postfix();
        std::puts("PM4 catalog, registers, state, memory and submission tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
