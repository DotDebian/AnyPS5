#ifndef CORE_LIBS_PRX_LIBSCEAUDIOOUT_SRC_AUDIOOUT2PADMIX_HPP
#define CORE_LIBS_PRX_LIBSCEAUDIOOUT_SRC_AUDIOOUT2PADMIX_HPP

#include <cstdint>

// Routing of the AudioOut2 ports that belong to the controller rather than the TV.
//
// Astro Bot opens, next to its main bed (type 0x0, 8 channels) and object ports (0x100), a mono port
// of type 0x3 and a stereo port of type 0x6, each under an AudioOut2 user of its own. Their grains,
// recorded over the title and main menu, show what they carry: type 0x6 holds short bursts with 95%
// of their energy below 200 Hz (centroid 84 Hz, peaks at full scale), fired by menu moves and
// confirms: the DualSense's two voice-coil actuators. Type 0x3 carries small UI effects fired at the
// same moments (centroid 3.5 kHz, peaks around -8 dBFS): the controller's speaker. The title plays
// its own TV-side version of those effects on the other ports, so nothing is lost from the TV mix.
//
// A USB DualSense is also a 4-channel 48 kHz sound card: channels 1-2 feed the speaker and headset
// jack, channels 3-4 the left and right actuators. When that card is open, these ports play on it
// instead of the TV mix; every other port type (and every port when no card is found) stays in the
// main mix as before.
static constexpr std::uint16_t AUDIO_OUT2_PORT_TYPE_PAD_SPEAKER = 0x3;
static constexpr std::uint16_t AUDIO_OUT2_PORT_TYPE_PAD_VIBRATION = 0x6;

static constexpr std::uint32_t AUDIO_OUT2_PAD_CHANNELS = 4;
static constexpr std::uint32_t AUDIO_OUT2_PAD_SPEAKER_LEFT = 0;
static constexpr std::uint32_t AUDIO_OUT2_PAD_SPEAKER_RIGHT = 1;
static constexpr std::uint32_t AUDIO_OUT2_PAD_VIBRATION_LEFT = 2;
static constexpr std::uint32_t AUDIO_OUT2_PAD_VIBRATION_RIGHT = 3;

enum class AudioOut2Route {
    Main,
    PadSpeaker,
    PadVibration,
};

// Where a port of this type and channel count plays when the controller's sound card is open. Port
// types and layouts not listed above stay in the main mix.
AudioOut2Route AudioOut2RouteForPort(std::uint16_t type, std::uint32_t channels);

// Whether an SDL audio device name is a DualSense (or DualSense Edge) sound card.
bool AudioOut2IsPadAudioDevice(const char* name);

// Adds one grain of a pad-routed port (interleaved float, channels per frame, one gain per channel)
// onto the 4-channel pad mix: a mono speaker port feeds both speaker channels, a stereo one keeps its
// sides; the vibration port's left and right channels drive the left and right actuators (a mono
// one drives both). Main-routed ports are ignored.
void AudioOut2AccumulatePadPort(AudioOut2Route route, const float* data, std::uint32_t channels, const float* volume, float* out, std::uint32_t frames);

// How the 4 pad channels sit in the frames queued on the card. SDL's PulseAudio backend labels a
// 4-channel stream FL FR FC LFE (the WAVEEX order) instead of SDL's quad FL FR BL BR, and the sound
// server then folds channels 3-4 into the speaker (measured on the card's monitor: a tone on channel
// 3 came out on channels 1 and 2 at -3 dB, none on 3). There the card is opened as 5.1
// (FL FR FC LFE BL BR) with FC and LFE silent, which the server places on the card's FL FR RL RR
// unchanged; every other backend (PipeWire, ALSA) takes quad as it is.
static constexpr std::uint32_t AUDIO_OUT2_PAD_DEVICE_CHANNELS_MAX = 6;

struct AudioOut2PadLayout {
    std::uint32_t channels = AUDIO_OUT2_PAD_CHANNELS;
    // Device channel of each pad channel (speaker left, right, vibration left, right).
    std::uint32_t position[AUDIO_OUT2_PAD_CHANNELS] = {0, 1, 2, 3};
};

AudioOut2PadLayout AudioOut2PadLayoutForDriver(const char* driver);

// Spreads frames of the 4-channel pad mix into the device layout, zeroing the unused channels.
void AudioOut2WritePadFrames(const float* pad, const AudioOut2PadLayout& layout, float* out, std::uint32_t frames);

// Clamps the summed pad mix to full scale; the card takes the samples as they are, with no master
// gain, since each channel carries a single port.
void AudioOut2FinishPadMix(float* out, std::uint32_t frames);

#endif
