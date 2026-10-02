#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DISPLAYFORMAT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DISPLAYFORMAT_HPP

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>

namespace AgcDriver {

// The Vulkan format of a display buffer's texels as the guest-memory path reads them
// (DecodeDisplayBuffer, and the GPU detile of the color transfer): the R8G8B8A8 base format keeps
// red in the low bits, any other base format blue; the 10-bit variant (bit 56) packs the same
// channel order in 2-10-10-10: A2B10G10R10 for the R8G8B8A8 base, A2R10G10B10 otherwise.
VkFormat DisplayTexelFormat(std::uint64_t pixelFormat);
bool DisplayTenBit(std::uint64_t pixelFormat);
bool DisplayRedLow(std::uint64_t pixelFormat);

// How a resident image (its storage format `storage`) presents a display buffer. Blit: the image
// holds the display's 8-bit texel format and can be a blit source, so the blit to the swapchain is
// exact. Convert: the image holds the display's 32-bit texels in another type (a 10-bit display,
// whose blit would round where the guest-memory path drops the low two bits of each channel; the
// other 8-bit channel order; an sRGB type the blit would decode): PresentationPass reads its raw
// texels and converts them as the guest-memory path does, so both paths present the same pixels.
// None: the image cannot present the buffer.
enum class ResidentPresent : std::uint8_t { None, Blit, Convert };
ResidentPresent ResidentPresentPath(VkFormat storage, std::uint64_t pixelFormat, bool blitSource);
std::optional<VkFormat> SwapchainFormat(std::span<const VkSurfaceFormatKHR> formats, const std::function<bool(VkFormat)>& usable);

}

#endif
