#ifndef CORE_LIBS_PRX_LIBSCEPAD_PADSTATE_HPP
#define CORE_LIBS_PRX_LIBSCEPAD_PADSTATE_HPP

#include <array>
#include <cstdint>
#include <exception>
#include "SceTypes.hpp"

struct PadTouchPoint {
    bool active = false;
    std::uint16_t x = 0;   // 0..1919
    std::uint16_t y = 0;   // 0..942
    std::uint8_t id = 0;

    bool operator==(const PadTouchPoint&) const = default;
};

struct PadInputState {
    std::uint32_t buttons = 0;
    std::array<std::uint8_t, 4> sticks{128, 128, 128, 128};
    std::uint8_t analogButtonsL2 = 0;
    std::uint8_t analogButtonsR2 = 0;
    bool touchLeft = false;
    bool touchRight = false;
    // Host gamepad extras (all neutral when no gamepad is present).
    bool hasMotion = false;                          // host pad delivers accelerometer + gyroscope
    std::array<float, 3> accel{0.0f, 9.80665f, 0.0f}; // m/s^2, +Y up when the pad lies flat
    std::array<float, 3> gyro{0.0f, 0.0f, 0.0f};      // rad/s
    std::array<PadTouchPoint, 2> touch{};             // host touchpad fingers
    std::uint8_t deviceKind = 0;                      // 0 none, 1 DualSense, 2 DualShock 4, 3 other pad
};

// Output requests issued by the guest (rumble, light bar, adaptive triggers), applied to the host pad by the window thread.
struct PadTriggerRequest {
    bool valid = false;               // an effect other than "off" is engaged
    std::uint8_t effect[11]{};        // DualSense native trigger effect block
    std::uint8_t fallback = 0;        // 0..255 rumble strength for pads without adaptive triggers
    std::uint32_t mode = 0;           // guest ScePadTriggerEffectMode
};

struct PadOutputState {
    std::uint32_t sequence = 0;       // bumped on every change
    std::uint8_t vibrationLarge = 0;
    std::uint8_t vibrationSmall = 0;
    bool lightBarValid = false;       // false: default colour
    std::uint8_t lightBar[3]{};
    PadTriggerRequest trigger[2];     // 0 = L2, 1 = R2
    bool triggerTouched = false;      // an effect struct was received at least once
    bool motionEnabled = true;
    int vibrationMode = 0;            // last ScePadSetVibrationMode argument (0/r0 = desktop, 1/r1 = embedded)
};

namespace Pad {
void Initialize();
PadData ReadState();
void SetVibration(std::uint8_t large, std::uint8_t small);
void SetVibrationMode(int mode);
void SetLightBar(bool valid, std::uint8_t r, std::uint8_t g, std::uint8_t b);
// trigger: 0 = L2, 1 = R2; command is a ScePadTriggerEffectCommandData (56 bytes: u32 mode, padding, command data at +8).
void SetTriggerCommand(int trigger, const std::uint8_t* command);
void ResetOrientation();
void SetMotionEnabled(bool enabled);
}

extern "C" void PadPublishInput_nid_postfix(const PadInputState& input);
extern "C" void PadReportInputFailure_nid_postfix(std::exception_ptr error);
// Copies the current output request when it differs from *seenSequence, updating it; returns true when a copy was made.
extern "C" bool PadFetchOutput_nid_postfix(std::uint32_t* seenSequence, PadOutputState* out);

#endif
