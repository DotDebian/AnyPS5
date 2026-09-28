#include <cstddef>
#include <cstdint>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <thread>

#include "SceTypes.hpp"
#include "prx//libc/include/General.hpp"
#include "prx/libScePad/include/Pad.hpp"
#include "prx/libScePad/include/PadState.hpp"

extern "C" {

int APS5_VABI scePadClose_nid_postfix(int handle) {
 if (handle != PAD_HANDLE) {
  return PAD_ERROR_INVALID_HANDLE;
 }
 return PAD_OK;
}

int APS5_VABI scePadDeviceClassGetExtendedInformation(int handle, PadDeviceClassExtendedInformation* info) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (info == nullptr) return PAD_ERROR_INVALID_ARG;
 // Standard digital pad: device class only, every class-specific field zero.
 std::memset(info, 0, sizeof(*info));
 info->deviceClass = PAD_DEVICE_CLASS_STANDARD;
 return PAD_OK;
}

int APS5_VABI scePadDeviceClassParseData(int handle, const PadData* data, PadDeviceClassData* class_data) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (data == nullptr || class_data == nullptr) return PAD_ERROR_INVALID_ARG;
 std::memset(class_data, 0, sizeof(*class_data));
 class_data->deviceClass = PAD_DEVICE_CLASS_STANDARD;
 class_data->dataValid = data->connected; // no steering/guitar/drum payload for a standard pad
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
 (void)user_id;
 (void)type;
 (void)index;
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

// Port 0 holds the virtual DualSense while the host has no pad open; fields per the PadInfo layout.
int APS5_VABI scePadGetInfo_nid_postfix(PadInfo* info) {
 if (info == nullptr) return PAD_ERROR_INVALID_ARG;
 std::memset(info, 0, sizeof(*info));
 info->maxConnectCount = 4;
 info->connectedCount[0] = 1; // standard ports
 info->connectedCount[1] = 0; // special ports
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
 // Both triggers report the neutral "no feedback engaged" state; the host pad has no trigger readback.
 std::memset(info, 0, sizeof(*info));
 return PAD_OK;
}

int APS5_VABI scePadGetExtControllerInformation(int handle, void* info) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (info == nullptr) return PAD_ERROR_INVALID_ARG;
 // ScePadExtendedControllerInformation: base information followed by the class extension (all zero for a standard pad).
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
 if (index != 0) {
  return PAD_ERROR_INVALID_ARG;
 }
 const bool personalPort = (type == PAD_PORT_TYPE_STANDARD || type == PAD_PORT_TYPE_SPECIAL);
 const bool systemRemote = (userId == PAD_USER_ID_SYSTEM && type == PAD_PORT_TYPE_REMOTE);
 if (!personalPort && !systemRemote) {
  return PAD_ERROR_INVALID_ARG;
 }
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

// ScePadLightBarParam: u8 r, g, b at +0..2.
int APS5_VABI scePadSetLightBar(int handle, const PadLightBarParam* param) {
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

// ScePadTriggerEffectParam: u8 triggerMask (bit 0 = L2, bit 1 = R2), u8 padding[7], ScePadTriggerEffectCommandData command[2] (56 bytes each).
// Not one of the extended-info NIDs the audit called out by name, but wired to a real implementation here
// because it is the guest entry point for the adaptive-trigger plumbing added alongside it (Pad::SetTriggerCommand);
// leaving it NotImplemented would make the trigger effect state/output machinery unreachable from the guest.
int APS5_VABI scePadSetTriggerEffect(int handle, const void* param) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (param == nullptr) return PAD_ERROR_INVALID_ARG;
 const std::uint8_t* bytes = static_cast<const std::uint8_t*>(param);
 const std::uint8_t mask = bytes[0];
 if ((mask & ~0x3u) != 0) {
  return PAD_ERROR_INVALID_ARG;
 }
 std::uint32_t mode[2] = {};
 for (int trigger = 0; trigger < 2; ++trigger) std::memcpy(&mode[trigger], bytes + 8 + 56 * trigger, 4);
 // Modes 0..6 are the SDK ScePadTriggerEffectMode values; anything else means the struct was misread.
 if (mode[0] > 6 || mode[1] > 6) {
  return PAD_ERROR_INVALID_ARG;
 }
 for (int trigger = 0; trigger < 2; ++trigger) {
  if ((mask & (1u << trigger)) != 0) Pad::SetTriggerCommand(trigger, bytes + 8 + 56 * trigger);
 }
 return PAD_OK;
}

// ScePadVibrationParam: u8 largeMotor at +0, u8 smallMotor at +1 (vibration-mode motors only; the trigger
// motors are driven through scePadSetTriggerEffect).
// Also not in the audit's named list, but wired here for the same reason as scePadSetTriggerEffect: it is
// the only guest entry point for Pad::SetVibration, which the rumble plumbing added alongside it needs.
int APS5_VABI scePadSetVibration(int handle, const PadVibrationParam* param) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (param == nullptr) return PAD_ERROR_INVALID_ARG;
 Pad::SetVibration(param->large_motor, param->small_motor);
 return PAD_OK;
}

// mode: 0 = desktop (USB) vibration layout, 1 = embedded controller layout. Both drive the same host rumble.
// Same rationale as scePadSetVibration: the only guest entry point for Pad::SetVibrationMode.
int APS5_VABI scePadSetVibrationMode(int handle, int mode) {
 if (handle != PAD_HANDLE) return PAD_ERROR_INVALID_HANDLE;
 if (mode != 0 && mode != 1) return PAD_ERROR_INVALID_ARG;
 Pad::SetVibrationMode(mode);
 return PAD_OK;
}

int APS5_VABI scePadSetVibrationTriggerEffectWeakWhileEmbeddedMicInUse(bool enabled) {
 (void)enabled;
 NotImplemented_nid_no_patch(__func__);
 return 0;
}

}
