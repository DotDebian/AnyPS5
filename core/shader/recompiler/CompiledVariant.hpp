#ifndef CORE_SHADER_RECOMPILER_COMPILEDVARIANT_HPP
#define CORE_SHADER_RECOMPILER_COMPILEDVARIANT_HPP

#include "Recompiler.hpp"
#include "IntermediateRepresentation/IrMetadata/CompiledShaderInfo.hpp"
#include "Optimization/include/Optimization/BindingAllocator.hpp"
#include "Optimization/include/Optimization/ResourceMaterializer.hpp"

namespace ShaderRecompiler {

// One compiled variant of a source (a program under one binding layout and resource
// specialization): the SPIR-V and the metadata every result materialized from it shares. The
// result's bindings and push constants are empty here; materializeResult populates them per
// snapshot from `info` and the allocation's layout. Immutable once made, and what the shader disk
// cache (ShaderDiskCache.hpp) stores.
struct CompiledVariant {
    ResourceSpecialization specialization;
    BindingLayout layout;
    CompiledShaderInfo info;
    BindingAllocationResult bindings;
    RecompileResult result;
};

}

#endif
