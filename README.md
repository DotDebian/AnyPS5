# AnyPS5: Astro Bot (PPSA21564) fork

This fork's `main` mirrors `astrobot`, my integration branch for running Astro Bot (PPSA21564) on Linux with AnyPS5: upstream `main` plus fixes that are still on their way upstream or specific to this setup. Pull requests to upstream are always cut from upstream `main`.

## Astro Bot status

Linux, measured on my machine (October 2026). Frame rates are the game's own, with no frame generation.

| Part | State | Frame rate |
| --- | --- | --- |
| Boot, PlayStation Studios video, logos | Renders | video at about 57 fps |
| Title screen | Renders, including the copyright line (console fonts or the Noto substitutes) | about 50 fps |
| NEW GAME menu | Renders | about 49 fps |
| Intro cinematic and space scene | Renders | 40 to 45 fps |
| Tutorial (Crash Site hub) | Playable with a DualSense | 6 to 13 fps |
| World map, controller ship flight | Renders | about 28 fps |
| Sky Garden | Reached and flown | about 4 fps |
| Snowy Canyon | Reached | about 3.5 fps |
| Windows | In game up to the first Gorilla Nebula level ([report](https://github.com/boykopovar/AnyPS5/discussions/357)) | |

Known issues: the frame rate in levels (the queue worker's CPU time per draw is the main limit there), a short white flash on the world map, and occasional minor glitches.

<img src="https://raw.githubusercontent.com/oneandonlydean/AnyPS5/410db37d25bec16eb67aabdde02e7c2ba932bdc3/readme/2026-10-04/video.jpg" width="400" alt="PlayStation Studios video"> <img src="https://raw.githubusercontent.com/oneandonlydean/AnyPS5/410db37d25bec16eb67aabdde02e7c2ba932bdc3/readme/2026-10-04/title.jpg" width="400" alt="Title screen">
<img src="https://raw.githubusercontent.com/oneandonlydean/AnyPS5/410db37d25bec16eb67aabdde02e7c2ba932bdc3/readme/2026-10-04/menu.jpg" width="400" alt="NEW GAME menu"> <img src="https://raw.githubusercontent.com/oneandonlydean/AnyPS5/410db37d25bec16eb67aabdde02e7c2ba932bdc3/readme/2026-10-04/intro-ships.jpg" width="400" alt="Intro space scene">
<img src="https://raw.githubusercontent.com/oneandonlydean/AnyPS5/410db37d25bec16eb67aabdde02e7c2ba932bdc3/readme/2026-10-04/hub.jpg" width="400" alt="Crash Site hub"> <img src="https://raw.githubusercontent.com/oneandonlydean/AnyPS5/410db37d25bec16eb67aabdde02e7c2ba932bdc3/readme/2026-10-04/map.jpg" width="400" alt="World map">
<img src="https://raw.githubusercontent.com/oneandonlydean/AnyPS5/410db37d25bec16eb67aabdde02e7c2ba932bdc3/readme/2026-10-04/garden.jpg" width="400" alt="Sky Garden"> <img src="https://raw.githubusercontent.com/oneandonlydean/AnyPS5/410db37d25bec16eb67aabdde02e7c2ba932bdc3/readme/2026-10-04/snowy.jpg" width="400" alt="Snowy Canyon">

## Upstream contributions

98 pull requests from this work are merged into [boykopovar/AnyPS5](https://github.com/boykopovar/AnyPS5), among them Linux write tracking (#120, #121), the shader disk and pipeline cache (#129), NGG geometry as mesh shaders (#133), resident 10-bit scanout (#278), flips that complete after the frame's GPU work (#417), large DMA copies on the GPU in queue order (#439), triangle fan geometry input (#440), occlusion counter dumps on the GPU (#463), cross-queue submission order (#465), exact reciprocals for `--to-intel` (#464) and DualSense output and audio (#179, #180). Open: #461, #489, #490.

---

# About

Tool for automatic executables porting to Linux and Windows.

Includes a [relinker](core/relinker) that converts executable to the target system's native format and implementations of [system prx libraries](core/libs/prx) suitable for dynamic linking. No emulation or separate runtime process.

[Usage](docs/user/USAGE.md), [Build instructions](docs/dev/BUILD.md), [Technical debt of the project](docs/dev/TechnicalDebt.md), [code style conventions](docs/dev/CONVENTIONS.md), [contributing](CONTRIBUTING.md)

## Status

[![libraries](https://boykopovar.github.io/AnyPS5/badge-libraries.svg)](https://boykopovar.github.io/AnyPS5/) [![shaders](https://boykopovar.github.io/AnyPS5/badge-shaders.svg)](https://boykopovar.github.io/AnyPS5/)

[![progress map](https://boykopovar.github.io/AnyPS5/progress.svg)](https://boykopovar.github.io/AnyPS5/)

<sub>* System libraries: percentage of the functions known to the project so far (declared in [core/libs/prx](core/libs/prx)), not of every PS5 system function. The total grows as more functions are declared.</sub>

[List of verified games](docs/user/COMPATIBILITY.md)

Dreaming Sarah (2D platformer) runs at a stable 60 fps on a GTX 1050 Ti / i5-7500 3.4GHz.

Unsupported or unexpected states strictly throw `std::runtime_error`. `what()` is printed to stderr and the process terminates.

The [shader recompiler](core/shader/recompiler/Recompiler.cpp) successfully produces SPIR-V (validated via [Spirv-Tools](3rdparty/SPIRV-Tools) when built with `ANYPS5_ENABLE_SPIRV_TOOLS`).

## Compatibility

See the [game compatibility list](docs/user/COMPATIBILITY.md) for tested games and known issues.

## Input mapping

SDL-mapped game controllers are supported, including analog sticks and triggers. Keyboard and mouse controls can be configured with an `anyps5-input.ini` file. See [input mapping](docs/user/INPUT_MAPPING.md) for the supported devices and configuration format.

## System fonts

Games that open the console's system font sets need font files in an `anyps5-fonts` directory beside the generated game executable; set `ANYPS5_SYSTEM_FONTS` to use another directory. Files dumped from the console are used under their own names (`SST-Roman.otf`, `SST-Bold.otf`, `SSTJpPro-Regular.otf`, ...). Without them, these openly licensed substitutes are used when present: `NotoSans-{Light,Regular,Medium,Bold}.ttf` and `NotoSans-{LightItalic,Italic,MediumItalic,BoldItalic}.ttf` (Latin and Vietnamese), `NotoSansMono-{Light,Regular,Medium,Bold}.ttf` (typewriter), `NotoSansThai-{Light,Regular,Medium,Bold}.ttf` (Thai) and `NotoSansCJK-{Light,Regular,Medium,Bold}.ttc` (Japanese and Chinese). Without either, opening a system font set fails and the game shows no text in those fonts.

## Disclaimer

This project is intended for interoperability, research, preservation, and compatibility purposes. It does not include, distribute, or require copyrighted software, firmware, cryptographic keys, or proprietary libraries. Users are responsible for ensuring that any binaries used with this project are obtained and used in accordance with applicable laws and their respective license terms.

## License

This project is licensed under the GNU General Public License version 2 only.
