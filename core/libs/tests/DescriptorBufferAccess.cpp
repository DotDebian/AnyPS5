#include "Optimization/DescriptorBindingBuilder.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace ShaderRecompiler;

namespace {

void Require(bool value, const char* what) {
    if (!value) throw std::runtime_error(what);
}

BufferResource Buffer(std::uint32_t source, bool read, bool written, bool atomic) {
    BufferResource buffer;
    buffer.source = source;
    buffer.read = read;
    buffer.written = written;
    buffer.atomic = atomic;
    return buffer;
}

}

int main() {
    ShaderInfo info;
    info.buffers = {Buffer(0, true, false, false), Buffer(1, false, true, false), Buffer(2, true, true, false), Buffer(3, true, true, true)};
    ResourceSnapshot snapshot;
    for (std::uint32_t i = 0; i < 4; ++i) {
        DescriptorValue value;
        value.dwords = {0x1000u * (i + 1u), 0u, 0x100u, 0x00005204u};
        value.dwordCount = 4;
        snapshot.buffers.push_back(value);
    }
    BindingAllocationResult allocation;
    allocation.layout.descriptors.push_back({DescriptorBindingKind::Buffers, {0, 1, 2, 3}});
    DescriptorBindingBuilder{}.Populate(allocation, info, IrShaderStage::Compute, 0, snapshot, {});
    Require(allocation.bindings.size() == 1 && allocation.bindings[0].role == DescriptorRole::GuestBuffers, "the buffers binding is missing");
    const auto& binding = allocation.bindings[0];
    Require(binding.bufferRead == std::vector<bool>{true, false, true, true}, "bufferRead does not follow the loads");
    Require(binding.bufferWritten == std::vector<bool>{false, true, true, true}, "bufferWritten does not follow the stores");
    Require(binding.bufferAtomic == std::vector<bool>{false, false, false, true}, "bufferAtomic does not follow the atomics");
    std::puts("descriptor buffer access tests passed");
    return 0;
}
