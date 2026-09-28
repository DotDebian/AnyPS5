#include "prx/libSceAgc/Misc/include/ShaderFusion.hpp"

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"
#include "prx/libSceAgc/Shader/include/ShaderUtils.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"

struct SizeAlign {
 uint64_t m_size;
 uint8_t m_align;
};

namespace {

using namespace ShaderRegs;

constexpr std::uint32_t SCodeEnd = 0xbf9f0000u;
constexpr std::uint32_t SNop = 0xbf800000u;
constexpr std::uint32_t SSetpcS6 = 0xbe802006u;
constexpr std::uint8_t FusedCodeAlignmentLog2 = 8;
constexpr std::size_t FusedCodeAlignment = std::size_t{1} << FusedCodeAlignmentLog2;
constexpr char TrailerMagic[8] = {'b', 'a', 'r', 'e', 'f', 'o', 'o', 't'};
constexpr std::size_t TrailerCodeBytesOffset = sizeof(TrailerMagic) + 3 * sizeof(std::uint32_t);
constexpr std::uint32_t SpiShaderPgmChecksum = 0x080u;
constexpr std::uint32_t SpiShaderPgmRsrc1Gs = 0x08Au;
constexpr std::uint32_t SpiShaderPgmRsrc2Gs = 0x08Bu;
constexpr std::uint32_t Rsrc1VgprMask = 0x3fu;
constexpr std::uint32_t Rsrc1SgprMask = 0xfu << 6u;
constexpr std::uint32_t Rsrc2UserSgprMask = (0x1fu << 1u) | (1u << 27u);

struct Layout {
 std::uint32_t frontBytes;
 std::size_t backOffset;
 std::size_t codeBytes;
 std::vector<ShaderRegister> sh;
 std::vector<ShaderRegister> cx;
 std::size_t shOffset;
 std::size_t cxOffset;
 std::size_t specialsOffset;
 std::size_t inputsOffset;
 std::size_t outputsOffset;
 std::size_t totalBytes;
};

[[noreturn]] void Fail(const char* function, const std::string& message) {
 throw std::runtime_error(std::string(function) + ": " + message);
}

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
 return (value + alignment - 1) & ~(alignment - 1);
}

void ValidateHalf(const char* function, const Shader* shader, ShaderBinaryType type) {
 if (shader->file_header != SHADER_FILE_HEADER_MAGIC || shader->version != SHADER_VERSION) Fail(function, "invalid shader header or version");
 if (shader->type != static_cast<std::uint8_t>(type)) Fail(function, "unexpected shader half type " + std::to_string(shader->type));
 if (shader->code == nullptr || shader->shader_size == 0 || (shader->shader_size & 3u) != 0) Fail(function, "invalid shader half code");
 if (shader->embedded_constant_buffer_size_dqw != 0) Fail(function, "fusing halves with an embedded constant buffer is not implemented");
 if ((shader->num_sh_registers != 0 && shader->sh_registers == nullptr) || (shader->num_cx_registers != 0 && shader->cx_registers == nullptr)) Fail(function, "shader half registers missing");
}

std::uint32_t FrontProgramBytes(const char* function, const Shader* front) {
 const auto* bytes = static_cast<const std::uint8_t*>(const_cast<const void*>(front->code));
 const std::size_t size = front->shader_size;
 if (size < TrailerCodeBytesOffset + sizeof(std::uint32_t)) Fail(function, "front half has no program trailer");
 std::size_t trailer = size;
 for (std::size_t offset = size - TrailerCodeBytesOffset - sizeof(std::uint32_t) + 1; offset-- > 0;) {
  if (std::memcmp(bytes + offset, TrailerMagic, sizeof(TrailerMagic)) == 0) {
   trailer = offset;
   break;
  }
 }
 if (trailer == size) Fail(function, "front half has no program trailer");
 std::uint32_t codeBytes = 0;
 std::memcpy(&codeBytes, bytes + trailer + TrailerCodeBytesOffset, sizeof(codeBytes));
 if (codeBytes == 0 || (codeBytes & 3u) != 0 || codeBytes > trailer) Fail(function, "front half trailer has an invalid code size");
 std::size_t end = codeBytes / 4;
 const auto word = [&](std::size_t index) {
  std::uint32_t value = 0;
  std::memcpy(&value, bytes + index * 4, sizeof(value));
  return value;
 };
 while (end > 0 && word(end - 1) == SCodeEnd) --end;
 if (end == 0 || word(end - 1) != SSetpcS6) Fail(function, "front half does not end with s_setpc_b64 s[6:7]");
 return static_cast<std::uint32_t>((end - 1) * 4);
}

std::uint32_t MergeRsrc1(std::uint32_t back, std::uint32_t front) {
 const auto vgprs = std::max(back & Rsrc1VgprMask, front & Rsrc1VgprMask);
 const auto sgprs = std::max(back & Rsrc1SgprMask, front & Rsrc1SgprMask);
 return (back & ~(Rsrc1VgprMask | Rsrc1SgprMask)) | vgprs | sgprs;
}

std::uint32_t MergeRsrc2(const char* function, std::uint32_t back, std::uint32_t front) {
 if ((front & ~Rsrc2UserSgprMask & ~back) != 0) Fail(function, "front half requires hardware state the back half does not enable");
 return (back & ~Rsrc2UserSgprMask) | (front & Rsrc2UserSgprMask);
}

const ShaderRegister* FindRegister(const Shader* shader, std::uint32_t offset) {
 for (std::uint32_t i = 0; i < shader->num_sh_registers; ++i)
  if (shader->sh_registers[i].offset == offset) return &shader->sh_registers[i];
 return nullptr;
}

Layout ComputeLayout(const char* function, const Shader* front, const Shader* back) {
 ValidateHalf(function, back, ShaderBinaryType::GsBack);
 if (front->type == static_cast<std::uint8_t>(ShaderBinaryType::HsFront)) NotImplemented_nid_no_patch("hull shader half fusion");
 ValidateHalf(function, front, ShaderBinaryType::GsFront);
 Layout layout{};
 layout.frontBytes = FrontProgramBytes(function, front);
 layout.backOffset = AlignUp(layout.frontBytes, FusedCodeAlignment);
 layout.codeBytes = layout.backOffset + back->shader_size;
 const auto* frontRsrc1 = FindRegister(front, SpiShaderPgmRsrc1Gs);
 const auto* frontRsrc2 = FindRegister(front, SpiShaderPgmRsrc2Gs);
 if (frontRsrc1 == nullptr || frontRsrc2 == nullptr) Fail(function, "front half has no GS resource registers");
 layout.sh.assign(back->sh_registers, back->sh_registers + back->num_sh_registers);
 for (auto& reg : layout.sh) {
  if (reg.offset == SpiShaderPgmRsrc1Gs) reg.value = MergeRsrc1(reg.value, frontRsrc1->value);
  else if (reg.offset == SpiShaderPgmRsrc2Gs) reg.value = MergeRsrc2(function, reg.value, frontRsrc2->value);
 }
 for (std::uint32_t i = 0; i < front->num_sh_registers; ++i) {
  const auto& reg = front->sh_registers[i];
  if (reg.offset == SpiShaderPgmChecksum || FindRegister(back, reg.offset) != nullptr) continue;
  layout.sh.push_back(reg);
 }
 layout.cx.assign(back->cx_registers, back->cx_registers + back->num_cx_registers);
 for (std::uint32_t i = 0; i < front->num_cx_registers; ++i) {
  const auto& reg = front->cx_registers[i];
  if (std::none_of(layout.cx.begin(), layout.cx.end(), [&](const ShaderRegister& other) { return other.offset == reg.offset; })) layout.cx.push_back(reg);
 }
 if (layout.sh.size() > 0xff || layout.cx.size() > 0xff) Fail(function, "fused register count exceeds the shader header");
 layout.shOffset = AlignUp(layout.codeBytes, alignof(ShaderRegister));
 layout.cxOffset = layout.shOffset + layout.sh.size() * sizeof(ShaderRegister);
 layout.specialsOffset = AlignUp(layout.cxOffset + layout.cx.size() * sizeof(ShaderRegister), alignof(ShaderSpecialRegs));
 layout.inputsOffset = AlignUp(layout.specialsOffset + back->special_sizes_bytes, alignof(ShaderSemantic));
 layout.outputsOffset = layout.inputsOffset + front->num_input_semantics * sizeof(ShaderSemantic);
 layout.totalBytes = layout.outputsOffset + back->num_output_semantics * sizeof(ShaderSemantic);
 return layout;
}

void SetProgramAddress(std::vector<ShaderRegister>& regs, std::uint32_t loOffset, std::uint64_t address) {
 for (std::size_t i = 0; i + 1 < regs.size(); ++i) {
  if (regs[i].offset != loOffset || regs[i + 1].offset != loOffset + 1u) continue;
  regs[i].value = static_cast<std::uint32_t>(address >> 8u);
  regs[i + 1].value = (regs[i + 1].value & 0xFFFFFF00u) | static_cast<std::uint32_t>((address >> 40u) & 0xFFu);
  return;
 }
 throw std::runtime_error("sceAgcUnknownFuseShaderHalves: back half has no program address register pair");
}

}

extern "C" {

APS5_EXPORT("fd5Bp5tGTgo", sceAgcUnknownFuseShaderHalves);
int APS5_VABI sceAgcUnknownFuseShaderHalves(Shader* fused_result, const Shader* front, const Shader* back, void* scratch_mem) {
    constexpr auto fn = "sceAgcUnknownFuseShaderHalves";
    if (fused_result == nullptr || front == nullptr || back == nullptr || scratch_mem == nullptr) Fail(fn, "null argument");
    auto layout = ComputeLayout(fn, front, back);
    const auto base = reinterpret_cast<std::uintptr_t>(scratch_mem);
    if ((base & (FusedCodeAlignment - 1)) != 0 || (base & SHADER_BASE_ALIGN_MASK) != 0) Fail(fn, "fused shader memory is not a valid program address");
    auto* memory = static_cast<std::uint8_t*>(scratch_mem);
    std::memcpy(memory, const_cast<const void*>(front->code), layout.frontBytes);
    for (std::size_t offset = layout.frontBytes; offset < layout.backOffset; offset += sizeof(SNop)) std::memcpy(memory + offset, &SNop, sizeof(SNop));
    std::memcpy(memory + layout.backOffset, const_cast<const void*>(back->code), back->shader_size);
    SetProgramAddress(layout.sh, SPI_SHADER_PGM_LO_GS, base + layout.backOffset);
    if (PatchProgramAddressRegister(layout.sh.data(), static_cast<std::uint32_t>(layout.sh.size()), static_cast<std::uint8_t>(ShaderBinaryType::Gs), base) != 0) Fail(fn, "fused program address patch failed");
    auto* sh = reinterpret_cast<ShaderRegister*>(memory + layout.shOffset);
    auto* cx = reinterpret_cast<ShaderRegister*>(memory + layout.cxOffset);
    auto* specials = reinterpret_cast<ShaderSpecialRegs*>(memory + layout.specialsOffset);
    auto* inputs = reinterpret_cast<ShaderSemantic*>(memory + layout.inputsOffset);
    auto* outputs = reinterpret_cast<ShaderSemantic*>(memory + layout.outputsOffset);
    std::memcpy(sh, layout.sh.data(), layout.sh.size() * sizeof(ShaderRegister));
    std::memcpy(cx, layout.cx.data(), layout.cx.size() * sizeof(ShaderRegister));
    if (back->special_sizes_bytes != 0) std::memcpy(specials, back->specials, back->special_sizes_bytes);
    if (front->num_input_semantics != 0) std::memcpy(inputs, front->input_semantics, front->num_input_semantics * sizeof(ShaderSemantic));
    if (back->num_output_semantics != 0) std::memcpy(outputs, back->output_semantics, back->num_output_semantics * sizeof(ShaderSemantic));
    Shader fused = *back;
    fused.code = memory;
    fused.sh_registers = sh;
    fused.cx_registers = layout.cx.empty() ? nullptr : cx;
    fused.specials = back->special_sizes_bytes != 0 ? specials : nullptr;
    fused.input_semantics = front->num_input_semantics != 0 ? inputs : nullptr;
    fused.output_semantics = back->num_output_semantics != 0 ? outputs : nullptr;
    fused.shader_size = static_cast<std::uint32_t>(layout.codeBytes);
    fused.num_input_semantics = front->num_input_semantics;
    fused.scratch_size_dw_per_thread = std::max(front->scratch_size_dw_per_thread, back->scratch_size_dw_per_thread);
    fused.type = static_cast<std::uint8_t>(ShaderBinaryType::Gs);
    fused.num_sh_registers = static_cast<std::uint8_t>(layout.sh.size());
    fused.num_cx_registers = static_cast<std::uint8_t>(layout.cx.size());
    *fused_result = fused;
    AgcDriverRegisterShader_nid_postfix(fused_result);
    return 0;
}

APS5_EXPORT("dolOmWH+huQ", sceAgcUnknownGetFusedShaderSize);
int APS5_VABI sceAgcUnknownGetFusedShaderSize(SizeAlign* dst, const Shader* front, const Shader* back) {
    constexpr auto fn = "sceAgcUnknownGetFusedShaderSize";
    if (dst == nullptr || front == nullptr || back == nullptr) Fail(fn, "null argument");
    const auto layout = ComputeLayout(fn, front, back);
    dst->m_size = layout.totalBytes;
    dst->m_align = FusedCodeAlignmentLog2;
    return 0;
}

APS5_EXPORT("k0E7vkgqAuE", sceAgcCreateInterpolantMappingVsPs);
int APS5_VABI sceAgcCreateInterpolantMappingVsPs(ShaderRegister* regs, const Shader* vs, const Shader* ps) {
    (void)regs;
    (void)vs;
    (void)ps;
    NotImplemented_nid_no_patch(__func__);
    return 0;
}

}
