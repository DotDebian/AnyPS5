# AnyPS5: Astro Bot (PPSA21564) fork

This fork's `main` mirrors `astrobot`, my integration branch for running Astro Bot (PPSA21564) on Linux with AnyPS5: upstream `main` plus fixes that are still on their way upstream or specific to this setup. Pull requests to upstream are always cut from upstream `main`.

## Astro Bot status

| Part | State |
| --- | --- |
| Boot, PlayStation Studios video, logos | Renders |
| Title screen | Renders; copyright line missing (needs the PS5 system fonts) |
| NEW GAME menu | Renders |
| Intro cinematic | Renders; some minor fixes still to make (it takes about 20 minutes at the current frame rate) |
| Tutorial (Crash Site) | Playable with a DualSense; tutorial videos render |
| World map, controller ship flight | Reached |
| Performance | Low: about 15 to 30 fps in the menus, 4 to 11 fps in the cinematic and gameplay |

Known issues: the copyright line, some artifacts and minor glitches in the cinematic and gameplay, and the frame rate.

## Upstream contributions

62 pull requests from this work are merged into [boykopovar/AnyPS5](https://github.com/boykopovar/AnyPS5), among them Linux write tracking (#120, #121), the shader disk and pipeline cache (#129), NGG geometry as mesh shaders (#133), structurizer cloning (#134), pixel input layout (#136, #137), geometry shader fusion (#124), recompiler fixes for v_fma_mix (#205) and SDWA results (#214), rendering fixes (#131, #132, #176, #177, #178, #181, #202, #204), DualSense output and audio (#179, #180) and Linux direct memory through memfd (#203). Open: #278, #297, #303, #304, #305, #306, #307, #322.

---

# About

Tool for automatic executables porting to Linux and Windows.

Includes a [relinker](core/relinker) that converts executable to the target system's native format and implementations of [system prx libraries](core/libs/prx) suitable for dynamic linking. No emulation or separate runtime process.



## Status

[![libraries](https://boykopovar.github.io/AnyPS5/badge-libraries.svg)](https://boykopovar.github.io/AnyPS5/) [![shaders](https://boykopovar.github.io/AnyPS5/badge-shaders.svg)](https://boykopovar.github.io/AnyPS5/)

[![progress map](https://boykopovar.github.io/AnyPS5/progress.svg)](https://boykopovar.github.io/AnyPS5/)

<sub>* System libraries: percentage of the functions known to the project so far (declared in [core/libs/prx](core/libs/prx)), not of every PS5 system function. The total grows as more functions are declared.</sub>

[List of verified games](docs/user/COMPATIBILITY.md)

Dreaming Sarah (2D platformer) runs at a stable 60 fps on a GTX 1050 Ti / i5-7500 3.4GHz.

Unsupported or unexpected states strictly throw `std::runtime_error`. `what()` is printed to stderr and the process terminates.

The [shader recompiler](core/shader/recompiler/Recompiler.cpp) successfully produces SPIR-V (validated via [Spirv-Tools](3rdparty/SPIRV-Tools) when built with `ANYPS5_ENABLE_SPIRV_TOOLS`).

[Technical debt of the project](docs/dev/TechnicalDebt.md), [code style conventions](docs/dev/CONVENTIONS.md), [contributing](CONTRIBUTING.md)

## Build

The relinker uses only the C++20 standard library and should build with any conforming compiler.

On Intel hosts, pass `--to-intel` to the relinker to lower supported AMD-only instructions in the executable and bundled `sce_module`/`sce_modules` PRX files. Unsupported instructions or stub jumps outside the x86-64 relative branch range produce an error.

[libc.prx](core/libs/prx/libc) implementations contain compiler-specific code. Linux builds work with GCC; on Windows, MinGW-w64 GCC 15.2.0 (`winlibs-gcc15`, `x86_64-ucrt-posix-seh`) is currently required.

The project targets maximum compiler portability (but now it is not implemented).

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
