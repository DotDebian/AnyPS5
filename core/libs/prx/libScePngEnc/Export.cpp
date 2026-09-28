// libScePngEnc: PNG encoding of 8-bit RGBA / BGRA images with stb_image_write (public domain / MIT, 3rdparty/stb).
// The encode parameter block is laid out by analogy with the decoder's (image address, png address, sizes, geometry,
// pixel format); it is validated strictly and refused when it does not fit that layout.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STBI_WRITE_NO_STDIO
#include "stb_image_write.h"

#include "SceTypes.hpp"
#include "HitLog.hpp"
#include "prx/libc/include/General.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
constexpr std::int32_t kErrInvalidAddr = static_cast<std::int32_t>(0x80690101u);
constexpr std::int32_t kErrInvalidSize = static_cast<std::int32_t>(0x80690102u);
constexpr std::int32_t kErrInvalidParam = static_cast<std::int32_t>(0x80690103u);
constexpr std::int32_t kErrInvalidHandle = static_cast<std::int32_t>(0x80690104u);
constexpr std::int32_t kErrInvalidWorkMemory = static_cast<std::int32_t>(0x80690105u);
constexpr std::int32_t kErrEncodeError = static_cast<std::int32_t>(0x80690112u);
constexpr std::uint64_t kContextMagic = 0x504e47454e434f44ull;  // "PNGENCOD"
constexpr std::uint32_t kContextSize = 16;

// Guessed ScePngEncEncodeParam.
struct EncodeParam {
    const void* imageMemAddr;
    void* pngMemAddr;
    std::uint32_t imageMemSize;
    std::uint32_t pngMemSize;
    std::uint32_t imageWidth;
    std::uint32_t imageHeight;
    std::uint32_t imagePitch;
    std::uint16_t pixelFormat;
    std::uint16_t compressionLevel;
};

bool Readable(const void* p, std::size_t n) {
#ifdef _WIN32
    if (p == nullptr) return false;
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(p, &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_COMMIT) return false;
    if ((mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) return false;
    return reinterpret_cast<std::uintptr_t>(p) + n <= reinterpret_cast<std::uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
#else
    return p != nullptr;
#endif
}
}  // namespace

extern "C" {

int32_t APS5_VABI scePngEncQueryMemorySize(const void* param) {
 if (param == nullptr) return kErrInvalidParam;
 return static_cast<int32_t>(kContextSize);
}
int32_t APS5_VABI scePngEncCreate(const void* param, void* memory_address, uint32_t memory_size, void** handle) {
 if (param == nullptr || handle == nullptr) return kErrInvalidParam;
 if (memory_address == nullptr) return kErrInvalidAddr;
 if (memory_size < kContextSize) return kErrInvalidWorkMemory;
 const std::uint64_t magic = kContextMagic;
 std::memset(memory_address, 0, kContextSize);
 std::memcpy(memory_address, &magic, sizeof(magic));
 *handle = memory_address;
 return 0;
}
int32_t APS5_VABI scePngEncDelete(void* handle) {
 if (handle == nullptr) return kErrInvalidHandle;
 std::uint64_t magic = 0;
 std::memcpy(&magic, handle, sizeof(magic));
 if (magic != kContextMagic) return kErrInvalidHandle;
 std::memset(handle, 0, sizeof(magic));
 return 0;
}
int32_t APS5_VABI scePngEncEncode(void* handle, const void* param, void* output_info) {
 if (handle == nullptr) return kErrInvalidHandle;
 std::uint64_t magic = 0;
 std::memcpy(&magic, handle, sizeof(magic));
 if (magic != kContextMagic) return kErrInvalidHandle;
 if (param == nullptr) return kErrInvalidParam;
 if (!Readable(param, sizeof(EncodeParam))) return kErrInvalidAddr;
 const EncodeParam* p = static_cast<const EncodeParam*>(param);
 const std::uint64_t rowBytes = static_cast<std::uint64_t>(p->imageWidth) * 4;
 const std::uint64_t pitch = p->imagePitch != 0 ? p->imagePitch : rowBytes;
 const bool plausible = p->imageWidth > 0 && p->imageHeight > 0 && p->imageWidth <= 16384 && p->imageHeight <= 16384 && pitch >= rowBytes &&
                        p->pixelFormat <= 1 && p->imageMemSize >= pitch * (p->imageHeight - 1) + rowBytes && p->pngMemSize > 64 &&
                        Readable(p->imageMemAddr, pitch * (p->imageHeight - 1) + rowBytes) && Readable(p->pngMemAddr, p->pngMemSize);
 if (!plausible) {
  const std::uint8_t* raw = static_cast<const std::uint8_t*>(param);
  APS5_HIT("PNGENC", "scePngEncEncode: parameter block does not match the expected layout (%02x%02x%02x%02x%02x%02x%02x%02x ...), refusing",
           raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]);
  return kErrInvalidParam;
 }
 const std::uint8_t* src = static_cast<const std::uint8_t*>(p->imageMemAddr);
 std::vector<std::uint8_t> rgba(static_cast<std::size_t>(rowBytes) * p->imageHeight);
 for (std::uint32_t y = 0; y < p->imageHeight; ++y) {
  const std::uint8_t* s = src + static_cast<std::size_t>(y) * pitch;
  std::uint8_t* d = rgba.data() + static_cast<std::size_t>(y) * rowBytes;
  for (std::uint32_t x = 0; x < p->imageWidth; ++x, s += 4, d += 4) {
   if (p->pixelFormat == 1) { d[0] = s[2]; d[1] = s[1]; d[2] = s[0]; }
   else { d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; }
   d[3] = s[3];
  }
 }
 int len = 0;
 unsigned char* png = stbi_write_png_to_mem(rgba.data(), static_cast<int>(rowBytes), static_cast<int>(p->imageWidth), static_cast<int>(p->imageHeight), 4, &len);
 if (png == nullptr) return kErrEncodeError;
 if (static_cast<std::uint32_t>(len) > p->pngMemSize) {
  std::free(png);
  return kErrInvalidSize;
 }
 std::memcpy(p->pngMemAddr, png, static_cast<std::size_t>(len));
 std::free(png);
 if (output_info != nullptr && Readable(output_info, 4)) {
  const std::uint32_t size = static_cast<std::uint32_t>(len);
  std::memcpy(output_info, &size, 4);
 }
 APS5_HIT("PNGENC", "scePngEncEncode %ux%u -> %d bytes", p->imageWidth, p->imageHeight, len);
 return 0;
}
}
