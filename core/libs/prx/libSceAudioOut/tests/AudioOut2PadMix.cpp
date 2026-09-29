#include "prx/libSceAudioOut/src/AudioOut2PadMix.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "failed: %s\n", what);
    std::abort();
}

static std::vector<float> Silence(std::uint32_t frames) {
    return std::vector<float>(static_cast<std::size_t>(frames) * AUDIO_OUT2_PAD_CHANNELS, 0.0f);
}

int main() {
    // Only the controller's port types with 1 or 2 channels leave the main mix.
    Require(AudioOut2RouteForPort(0x3, 1) == AudioOut2Route::PadSpeaker, "mono type 0x3 is the pad speaker");
    Require(AudioOut2RouteForPort(0x3, 2) == AudioOut2Route::PadSpeaker, "stereo type 0x3 is the pad speaker");
    Require(AudioOut2RouteForPort(0x6, 2) == AudioOut2Route::PadVibration, "stereo type 0x6 is the vibration");
    Require(AudioOut2RouteForPort(0x6, 1) == AudioOut2Route::PadVibration, "mono type 0x6 is the vibration");
    Require(AudioOut2RouteForPort(0x6, 8) == AudioOut2Route::Main, "an 8-channel type 0x6 stays in the main mix");
    Require(AudioOut2RouteForPort(0x3, 0) == AudioOut2Route::Main, "an undecoded format stays in the main mix");
    for (std::uint16_t type : {0x0, 0x1, 0x2, 0x4, 0x5, 0x7, 0x100}) {
        Require(AudioOut2RouteForPort(type, 1) == AudioOut2Route::Main, "other port types stay in the main mix");
        Require(AudioOut2RouteForPort(type, 2) == AudioOut2Route::Main, "other port types stay in the main mix");
    }

    // Device names as the PulseAudio, PipeWire and ALSA backends list them.
    Require(AudioOut2IsPadAudioDevice("DualSense wireless controller (PS5) Direct DualSense Wireless Controller"), "pulse name");
    Require(AudioOut2IsPadAudioDevice("DualSense Edge Wireless Controller, USB Audio"), "edge alsa name");
    Require(!AudioOut2IsPadAudioDevice("AD102 High Definition Audio Controller Digital Stereo (HDMI)"), "hdmi name");
    Require(!AudioOut2IsPadAudioDevice(nullptr), "null name");

    constexpr std::uint32_t frames = 3;
    const float unity[2] = {1.0f, 1.0f};

    // A mono speaker port feeds both speaker channels and nothing else, with its gain.
    {
        auto out = Silence(frames);
        const float speaker[frames] = {0.5f, -0.25f, 0.125f};
        const float half[1] = {0.5f};
        AudioOut2AccumulatePadPort(AudioOut2Route::PadSpeaker, speaker, 1, half, out.data(), frames);
        for (std::uint32_t frame = 0; frame < frames; frame++) {
            const float* pad = &out[frame * AUDIO_OUT2_PAD_CHANNELS];
            Require(pad[0] == speaker[frame] * 0.5f && pad[1] == speaker[frame] * 0.5f, "mono speaker on channels 1-2");
            Require(pad[2] == 0.0f && pad[3] == 0.0f, "speaker leaves the actuators alone");
        }
    }

    // The vibration port's left and right channels drive channels 3 and 4, each with its own gain.
    {
        auto out = Silence(frames);
        const float vibration[frames * 2] = {0.1f, 0.2f, 0.3f, 0.4f, -0.5f, -0.6f};
        const float gains[2] = {1.0f, 0.5f};
        AudioOut2AccumulatePadPort(AudioOut2Route::PadVibration, vibration, 2, gains, out.data(), frames);
        for (std::uint32_t frame = 0; frame < frames; frame++) {
            const float* pad = &out[frame * AUDIO_OUT2_PAD_CHANNELS];
            Require(pad[0] == 0.0f && pad[1] == 0.0f, "vibration leaves the speaker alone");
            Require(pad[2] == vibration[frame * 2] && pad[3] == vibration[frame * 2 + 1] * 0.5f, "vibration on channels 3-4");
        }
    }

    // Both ports sum into one grain, and the result is clamped to full scale.
    {
        auto out = Silence(frames);
        const float speaker[frames] = {0.75f, 0.75f, 0.0f};
        const float vibration[frames * 2] = {0.9f, -0.9f, 0.0f, 0.0f, 0.0f, 0.0f};
        AudioOut2AccumulatePadPort(AudioOut2Route::PadSpeaker, speaker, 1, unity, out.data(), frames);
        AudioOut2AccumulatePadPort(AudioOut2Route::PadSpeaker, speaker, 1, unity, out.data(), frames);
        AudioOut2AccumulatePadPort(AudioOut2Route::PadVibration, vibration, 2, unity, out.data(), frames);
        AudioOut2AccumulatePadPort(AudioOut2Route::PadVibration, vibration, 2, unity, out.data(), frames);
        AudioOut2FinishPadMix(out.data(), frames);
        Require(out[0] == 1.0f && out[1] == 1.0f, "speaker sum clamps at full scale");
        Require(out[2] == 1.0f && out[3] == -1.0f, "vibration sum clamps at full scale");
        Require(out[4] == 1.0f && out[6] == 0.0f, "second frame");
        for (std::uint32_t index = 8; index < frames * AUDIO_OUT2_PAD_CHANNELS; index++) Require(out[index] == 0.0f, "silent frame stays silent");
    }

    // Main-routed ports and missing data leave the pad mix untouched.
    {
        auto out = Silence(frames);
        const float data[frames * 2] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        AudioOut2AccumulatePadPort(AudioOut2Route::Main, data, 2, unity, out.data(), frames);
        AudioOut2AccumulatePadPort(AudioOut2Route::PadVibration, nullptr, 2, unity, out.data(), frames);
        AudioOut2AccumulatePadPort(AudioOut2Route::PadVibration, data, 8, unity, out.data(), frames);
        for (float sample : out) Require(sample == 0.0f, "nothing is mixed");
    }

    // Device layouts: quad everywhere but PulseAudio, where the actuators ride on 5.1's back pair.
    {
        const auto quad = AudioOut2PadLayoutForDriver("pipewire");
        Require(quad.channels == 4 && quad.position[0] == 0 && quad.position[1] == 1 && quad.position[2] == 2 && quad.position[3] == 3, "quad layout");
        Require(AudioOut2PadLayoutForDriver("alsa").channels == 4 && AudioOut2PadLayoutForDriver(nullptr).channels == 4, "alsa and unknown use quad");
        const auto pulse = AudioOut2PadLayoutForDriver("pulseaudio");
        Require(pulse.channels == 6 && pulse.position[0] == 0 && pulse.position[1] == 1 && pulse.position[2] == 4 && pulse.position[3] == 5, "pulse layout");

        const float pad[2 * AUDIO_OUT2_PAD_CHANNELS] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        float out[2 * AUDIO_OUT2_PAD_DEVICE_CHANNELS_MAX];
        for (float& sample : out) sample = 9.0f;
        AudioOut2WritePadFrames(pad, pulse, out, 2);
        const float expected[2 * 6] = {0.1f, 0.2f, 0.0f, 0.0f, 0.3f, 0.4f, 0.5f, 0.6f, 0.0f, 0.0f, 0.7f, 0.8f};
        for (int index = 0; index < 12; index++) Require(out[index] == expected[index], "pulse frames: speaker front, centre and LFE silent, actuators back");
        AudioOut2WritePadFrames(pad, quad, out, 2);
        for (int index = 0; index < 8; index++) Require(out[index] == pad[index], "quad frames are the pad mix");
    }

    std::puts("audio_out2_pad_mix: ok");
    return 0;
}
