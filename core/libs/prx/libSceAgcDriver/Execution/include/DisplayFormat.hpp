#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DISPLAYFORMAT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DISPLAYFORMAT_HPP

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>

namespace AgcDriver {

VkFormat DisplayTexelFormat(std::uint64_t pixelFormat);
bool DisplayTenBit(std::uint64_t pixelFormat);
bool DisplayRedLow(std::uint64_t pixelFormat);

enum class ResidentPresent : std::uint8_t { None, Blit, Convert };
ResidentPresent ResidentPresentPath(VkFormat storage, std::uint64_t pixelFormat, bool blitSource);
std::optional<VkFormat> SwapchainFormat(std::span<const VkSurfaceFormatKHR> formats, const std::function<bool(VkFormat)>& usable);

}

#endif
