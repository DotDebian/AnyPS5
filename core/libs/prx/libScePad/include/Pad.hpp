#ifndef CORE_LIBS_PRX_LIBSCEPAD_PAD_HPP
#define CORE_LIBS_PRX_LIBSCEPAD_PAD_HPP

#include <cstdint>

constexpr int PAD_OK = 0;
constexpr int PAD_ERROR_INVALID_ARG = -2137915390;
constexpr int PAD_ERROR_INVALID_HANDLE = -2137915384;

constexpr int PAD_PORT_TYPE_STANDARD = 0;
constexpr int PAD_PORT_TYPE_SPECIAL = 2;
constexpr int PAD_PORT_TYPE_REMOTE = 16;
constexpr int PAD_USER_ID_SYSTEM = 0xff;
constexpr int PAD_HANDLE = 1;

constexpr int PAD_CONNECTION_TYPE_LOCAL = 0;
constexpr int PAD_DEVICE_CLASS_STANDARD = 0;

// ScePadDeviceType values reconstructed from DualSense HID telemetry; no current title queries them,
// they only describe the virtual device.
constexpr int PAD_DEVICE_TYPE_NONE = 0;
constexpr int PAD_DEVICE_TYPE_NAVIGATOR = 1;
constexpr int PAD_DEVICE_TYPE_DUAL_SHOCK_4 = 2;
constexpr int PAD_DEVICE_TYPE_DUAL_SENSE = 3;
constexpr int PAD_DEVICE_TYPE_DUAL_SENSE_EDGE = 4;

// ScePadConnectType values from the PS4/PS5 SDK.
constexpr int PAD_CONNECT_TYPE_UNREGISTERED = 0;
constexpr int PAD_CONNECT_TYPE_USB = 1;
constexpr int PAD_CONNECT_TYPE_BLUETOOTH = 2;

// Bluetooth public address of the virtual pad; DC:8C:37 is the Sony OUI a retail DualSense uses.
inline constexpr std::uint8_t PAD_BD_ADDRESS[6] = {0xDC, 0x8C, 0x37, 0x51, 0x01, 0xF4};

// Mirrors scePadGetInfo's ScePadInfo: per-port slot table of the pad service. The exact SDK field names
// are not public; this is the layout every out-param filler in this library writes and documents.
struct PadInfo {
    std::uint8_t maxConnectCount;                  // 0x00
    std::uint8_t connectedCount[2];                // 0x01 per personal port type (standard, special)
    std::uint8_t reserved0[2];                     // 0x03
    struct PadTypeInfo {
        std::uint8_t connectPort;                  // index within the port type
        std::uint8_t status;                       // 0 free, 1 connected
        std::uint8_t deviceType;                   // PAD_DEVICE_TYPE_*
        std::uint8_t connectType;                  // PAD_CONNECT_TYPE_*
        std::uint8_t bluetoothMacAddress[6];       // empty unless status == 1
    } padTypeInfo[8];                              // 0x05, 10 bytes per slot
    std::uint8_t reserved1[4];                     // 0x55
};
static_assert(sizeof(PadInfo) == 0x59);

#endif
