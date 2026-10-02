#include <cstddef>
#include <cstdint>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "SceTypes.hpp"
#include "prx//libc/include/General.hpp"
#include "prx/libScePad/include/Pad.hpp"
#include "prx/libScePad/include/PadState.hpp"

// APS5_TRACE_PAD_OUTPUT=1 prints the title's pad output requests (vibration, vibration mode, trigger
// effects, light bar): the first few of each kind, then a count per kind every few seconds.
namespace {
enum PadOutputKind { OutputVibration, OutputVibrationNonZero, OutputVibrationMode, OutputTriggerEffect, OutputLightBar, OutputKinds };
constexpr const char* OUTPUT_KIND_NAMES[OutputKinds] = {"SetVibration", "SetVibration(non-zero)", "SetVibrationMode", "SetTriggerEffect", "SetLightBar"};
constexpr std::uint64_t OUTPUT_TRACE_FULL = 8;
constexpr auto OUTPUT_TRACE_SUMMARY = std::chrono::seconds(5);

bool PadOutputTraceEnabled() {
 static const bool enabled = std::getenv("APS5_TRACE_PAD_OUTPUT") != nullptr;
 return enabled;
}

void TracePadOutput(PadOutputKind kind, int handle, int a, int b) {
 if (!PadOutputTraceEnabled()) return;
 static std::atomic<std::uint64_t> counts[OutputKinds];
 static std::atomic<std::int64_t> lastSummary{0};
 const auto count = ++counts[kind];
 if (count <= OUTPUT_TRACE_FULL) std::fprintf(stderr, "[padout] %s handle=%d %d %d (call %llu)\n", OUTPUT_KIND_NAMES[kind], handle, a, b, static_cast<unsigned long long>(count));
 const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
 auto last = lastSummary.load();
 if (last == 0) lastSummary.compare_exchange_strong(last, now);
 else if (now - last >= std::chrono::duration_cast<std::chrono::steady_clock::duration>(OUTPUT_TRACE_SUMMARY).count() && lastSummary.compare_exchange_strong(last, now)) {
  std::fprintf(stderr, "[padout] totals:");
  for (int index = 0; index < OutputKinds; index++) std::fprintf(stderr, " %s=%llu", OUTPUT_KIND_NAMES[index], static_cast<unsigned long long>(counts[index].load()));
  std::fprintf(stderr, "\n");
 }
}

constexpr int PAD_ERROR_DEVICE_NO_HANDLE = static_cast<int>(0x80920008);

bool g_opened = false;

bool ValidPort(int userId, int type, int index) {
    const bool personalPort = type == PAD_PORT_TYPE_STANDARD || type == PAD_PORT_TYPE_SPECIAL;
    const bool systemRemote = userId == PAD_USER_ID_SYSTEM && type == PAD_PORT_TYPE_REMOTE;
    return index == 0 && (personalPort || systemRemote);
}

}

extern "C" {

int APS5_VABI scePadClose_nid_postfix(int handle) {
 if (handle != PAD_HANDLE) {
  return PAD_ERROR_INVALID_HANDLE;
 }
 g_opened = false;
 return PAD_OK;
}

int APS5_VABI scePadDeviceClassGetExtendedInformation(int handle, PadDeviceClassExtendedInformation* info) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (info == nullptr) return PAD_ERROR_INVALID_ARG;
 std::memset(info, 0, sizeof(*info));
 info->deviceClass = PAD_DEVICE_CLASS_STANDARD;
 return PAD_OK;
}

int APS5_VABI scePadDeviceClassParseData(int handle, const PadData* data, PadDeviceClassData* class_data) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (data == nullptr || class_data == nullptr) return PAD_ERROR_INVALID_ARG;
 std::memset(class_data, 0, sizeof(*class_data));
 class_data->deviceClass = PAD_DEVICE_CLASS_STANDARD;
 class_data->dataValid = data->connected;
 return PAD_OK;
}

int APS5_VABI scePadGetControllerInformation(int handle, PadControllerInformation* info) {
 if (handle != PAD_HANDLE) {
  return PAD_ERROR_INVALID_HANDLE;
 }
 if (info == nullptr) {
  return PAD_ERROR_INVALID_ARG;
 }
 std::memset(info, 0, sizeof(*info));
 info->touchPadInfo.pixelDensity = 44.86f;
 info->touchPadInfo.resolution.x = 1920;
 info->touchPadInfo.resolution.y = 943;
 info->stickInfo.deadZoneLeft = 2;
 info->stickInfo.deadZoneRight = 2;
 info->connectionType = PAD_CONNECTION_TYPE_LOCAL;
 info->connectedCount = 1;
 info->connected = true;
 info->deviceClass = PAD_DEVICE_CLASS_STANDARD;
 return PAD_OK;
}

int APS5_VABI scePadGetHandle(int user_id, int type, int index) {
 if (!g_opened || !ValidPort(user_id, type, index)) return PAD_ERROR_DEVICE_NO_HANDLE;
 return PAD_HANDLE;
}

int APS5_VABI scePadGetInfo_nid_postfix(PadInfo* info) {
 if (info == nullptr) return PAD_ERROR_INVALID_ARG;
 std::memset(info, 0, sizeof(*info));
 info->maxConnectCount = 4;
 info->connectedCount[0] = 1;
 info->connectedCount[1] = 0;
 PadInfo::PadTypeInfo& slot = info->padTypeInfo[0];
 slot.connectPort = 0;
 slot.status = 1;
 slot.deviceType = PAD_DEVICE_TYPE_DUAL_SENSE;
 slot.connectType = PAD_CONNECT_TYPE_BLUETOOTH;
 std::memcpy(slot.bluetoothMacAddress, PAD_BD_ADDRESS, sizeof(slot.bluetoothMacAddress));
 return PAD_OK;
}

int APS5_VABI scePadGetTriggerEffectState(int handle, PadTriggerEffectStateInformation* info) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (info == nullptr) return PAD_ERROR_INVALID_ARG;
 std::memset(info, 0, sizeof(*info));
 return PAD_OK;
}

int APS5_VABI scePadGetExtControllerInformation(int handle, void* info) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (info == nullptr) return PAD_ERROR_INVALID_ARG;
 std::memset(info, 0, 0x2c);
 PadControllerInformation* base = static_cast<PadControllerInformation*>(info);
 base->touchPadInfo.pixelDensity = 44.86f;
 base->touchPadInfo.resolution.x = 1920;
 base->touchPadInfo.resolution.y = 943;
 base->stickInfo.deadZoneLeft = 2;
 base->stickInfo.deadZoneRight = 2;
 base->connectionType = PAD_CONNECTION_TYPE_LOCAL;
 base->connectedCount = 1;
 base->connected = true;
 base->deviceClass = PAD_DEVICE_CLASS_STANDARD;
 return PAD_OK;
}

int APS5_VABI scePadSetProcessPrivilege(int privilege) {
 (void)privilege;
 return PAD_OK;
}

int APS5_VABI scePadSetParticularMode(bool mode) {
 (void)mode;
 return PAD_OK;
}

int APS5_VABI scePadInit_nid_postfix(void) {
 Pad::Initialize();
 return PAD_OK;
}

int APS5_VABI scePadOpen_nid_postfix(int userId, int type, int index, const void* param) {
 (void)param;
 if (!ValidPort(userId, type, index)) {
  return PAD_ERROR_INVALID_ARG;
 }
 g_opened = true;
 return PAD_HANDLE;
}

int APS5_VABI scePadReadState(int handle, PadData* data);

int APS5_VABI scePadRead_nid_postfix(int handle, PadData* data, int num) {
    // The title drains the queued states; one current state is reported per call.
    if (data == nullptr || num <= 0) APS5_INVALID_ARG_EX;
    const int result = scePadReadState(handle, data);
    if (result != 0) return result;
    return 1;
}

int APS5_VABI scePadReadState(int handle, PadData* data) {
 if (handle != 1) APS5_INVALID_ARG_EX;
 if (data == nullptr) APS5_INVALID_ARG_EX;

 *data = Pad::ReadState();

 return 0;
}

int APS5_VABI scePadResetLightBar(int handle) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 Pad::SetLightBar(false, 0, 0, 0);
 return PAD_OK;
}

int APS5_VABI scePadResetOrientation(int handle) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 Pad::ResetOrientation();
 return PAD_OK;
}

int APS5_VABI scePadSetAngularVelocityDeadbandState(int handle, bool enable) {
 (void)handle;
 (void)enable;
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

int APS5_VABI scePadSetLightBar(int handle, const PadLightBarParam* param) {
 if (param != nullptr) TracePadOutput(OutputLightBar, handle, (param->r << 16) | (param->g << 8) | param->b, 0);
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (param == nullptr) return PAD_ERROR_INVALID_ARG;
 Pad::SetLightBar(true, param->r, param->g, param->b);
 return PAD_OK;
}

int APS5_VABI scePadSetMotionSensorState(int handle, bool enable) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 Pad::SetMotionEnabled(enable);
 return PAD_OK;
}

int APS5_VABI scePadSetTiltCorrectionState(int handle, bool enabled) {
 static std::atomic<bool> tiltCorrection{false};
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 tiltCorrection.store(enabled, std::memory_order_relaxed);
 return PAD_OK;
}

int APS5_VABI scePadSetTriggerEffect(int handle, const void* param) {
 if (param != nullptr) TracePadOutput(OutputTriggerEffect, handle, static_cast<const std::uint8_t*>(param)[0], 0);
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (param == nullptr) return PAD_ERROR_INVALID_ARG;
 const std::uint8_t* bytes = static_cast<const std::uint8_t*>(param);
 const std::uint8_t mask = bytes[0];
 if ((mask & ~0x3u) != 0) {
  return PAD_ERROR_INVALID_ARG;
 }
 std::uint32_t mode[2] = {};
 for (int trigger = 0; trigger < 2; ++trigger) std::memcpy(&mode[trigger], bytes + 8 + 56 * trigger, 4);
 if (mode[0] > 6 || mode[1] > 6) {
  return PAD_ERROR_INVALID_ARG;
 }
 for (int trigger = 0; trigger < 2; ++trigger) {
  if ((mask & (1u << trigger)) != 0) Pad::SetTriggerCommand(trigger, bytes + 8 + 56 * trigger);
 }
 return PAD_OK;
}

int APS5_VABI scePadSetVibration(int handle, const PadVibrationParam* param) {
 if (param != nullptr) {
  TracePadOutput(OutputVibration, handle, param->large_motor, param->small_motor);
  if (param->large_motor != 0 || param->small_motor != 0) TracePadOutput(OutputVibrationNonZero, handle, param->large_motor, param->small_motor);
 }
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (param == nullptr) return PAD_ERROR_INVALID_ARG;
 Pad::SetVibration(param->large_motor, param->small_motor);
 return PAD_OK;
}

int APS5_VABI scePadSetVibrationMode(int handle, int mode) {
 TracePadOutput(OutputVibrationMode, handle, mode, 0);
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (mode != 0 && mode != 1) return PAD_ERROR_INVALID_ARG;
 Pad::SetVibrationMode(mode);
 return PAD_OK;
}

int APS5_VABI scePadSetVibrationTriggerEffectWeakWhileEmbeddedMicInUse(bool enabled) {
 (void)enabled;
 return PAD_OK;
}

}
