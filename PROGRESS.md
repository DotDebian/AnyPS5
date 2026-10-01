# PPSA21564 (Astro Bot) on AnyPS5: progress

This fork's `main` mirrors `astrobot`, my integration branch for running Astro Bot (PPSA21564) natively on Linux
with AnyPS5. It carries upstream `main` plus the fixes that are still on their way upstream or are specific to this
setup. Upstream pull requests are cut from upstream `main`, never from this branch.

## Where the game is (2026-10-02)

| Part | State |
| --- | --- |
| Boot, PlayStation Studios video, logos | Renders; the video no longer tears; the Team Asobi logo dust effect is back |
| Title screen | Lit, crisp logo matching a real PS5 capture; copyright line missing (needs the PS5 system fonts) |
| NEW GAME menu | Textured save cards matching the PS5; DualSense light bar, rumble and haptic audio work |
| Intro cinematic (`intro_next`) | Galaxy with lit asteroids and moon, mothership crowd with visors and eyes, alien scene; crash landing and desert render but are still rough |
| First level (`hub_crashsite_tutorial`) | Loads; reaching gameplay is the current work |

Frame rate on an RTX 4090: about 29 fps intro, 15 title and menu, 17 galaxy, about 7 in the heavier cinematic scenes.
The cinematic runs slower than real time.

## Known differences from the PS5

- Copyright line on the title screen (system font set not available yet)
- Sky and haze in the galaxy a little darker than the PS5; robot eyes a little brighter
- Title logo glow halo slightly stronger than the PS5

## How changes are checked

Every change to `astrobot` passes a scripted visual regression run twice in a row before the branch moves: the game
is driven from boot through the title and menu into the cinematic, and the captured frames are compared with
reference frames (including cropped checks such as the moon), checked for single-frame flicker, checked for torn
video frames against the decoded source video, and timed per phase. References are only replaced deliberately,
after comparing with captures from a real PS5.

## Upstream contributions

58 pull requests merged into boykopovar/AnyPS5 so far, among them:

- Write tracking on Linux (#120, #121), shader disk and pipeline cache (#129), NGG geometry as mesh shaders (#133),
  structurizer cloning (#134), pixel input layout (#136, #137), geometry shader fusion (#124)
- Recompiler fixes that showed up as colour problems in PPSA21564: v_fma_mix precision (#205), SDWA dword results (#214)
- Rendering fixes: label waits (#178), DCC key moves (#177), base vertex (#176), misaligned buffers (#202, #204),
  2D array sampling (#131, #181), CB metadata passes (#132), MIN_LOD (#130)
- DualSense: HIDAPI output (#179) and speaker and vibration audio (#180)
- Direct memory through memfd on Linux (#203), json2, font, fibers, libc and libkernel additions (#44 to #63)

Open: #246 (key fill immediate clear follow-up to #163).
