# Smoke Sim (Offline)

Offline 3D smoke simulation and volumetric rendering aimed at believable point-source smoke (cigarette-like plume) with direct `mp4` / `gif` export.

## What It Does

- Simulates a 3D velocity + density + temperature volume.
- Uses semi-Lagrangian advection, buoyancy, vorticity confinement, and pressure projection.
- Raymarches the volume to PNG frames.
- Encodes outputs with ffmpeg in overwrite mode.
- Defaults to a Taichi engine (`main_taichi.py`) that can run on GPU.
- Uses system `ffmpeg` when available, otherwise falls back to `imageio-ffmpeg` from the venv.

## Quick Start

```bash
cd /home/lparker/git/fuerve/smoke_sim
make all
```

This bootstraps a local `.venv/` and installs dependencies automatically.

Engine options:

- `ENGINE=taichi` (default): GPU-capable Taichi backend.
- `ENGINE=numpy`: original NumPy backend.
- `ARCH=auto|cpu|cuda|vulkan|metal` (Taichi only, default `auto`).

By default, outputs land in `out/latest/` (`OUT=out/latest` in `Makefile`).
Use additional subdirectories with `OUT=out/<name>` to keep runs organized.

- `out/latest/frames/*.png`
- `out/latest/smoke.mp4`
- `out/latest/smoke.gif`

Config-driven runs:

```bash
make update CONFIG=config/cigarette.json
```

When `CONFIG=...` is set, values come from JSON by default and any explicit CLI overrides still win (for example `FRAMES=360` or `EXTRA_ARGS="--buoyancy 2.1"`).
The sample config emits both Discord and Facebook GIF variants by default.

## Build Targets

- `make frames` renders PNG frames only.
- `make mp4` encodes `$(OUT)/smoke.mp4` from existing frames.
- `make gif` encodes `$(OUT)/smoke.gif` from existing frames.
- `make all` renders once, then writes both `mp4` and `gif`.
- `make update` alias for `make all` (overwrite refresh).
- `make clean` deletes `out/` and `.venv/`.

All ffmpeg commands run with `-y`, so media outputs are overwritten.

## Useful Tweaks

Adjust on the command line or via `make` variables:

```bash
make all FRAMES=300 GRID=96 WIDTH=720 HEIGHT=720 RAY_STEPS=120
```

Wispy cigarette-style tuning (ambient floor airflow):

```bash
make all ENGINE=taichi ARCH=auto \
  FRAMES=300 GRID=96 WIDTH=720 HEIGHT=720 RAY_STEPS=120 SUBSTEPS=2 \
  OUT=out/cigarette \
  EXTRA_ARGS="--ambient-strength 1.1 --ambient-drift 0.045 --source-jitter 0.05 --extinction 2.1"
```

Loop-friendly clip (emit, stop, then dissipate):

```bash
make update ENGINE=taichi ARCH=vulkan OUT=out/final \
  FRAMES=300 GRID=44 WIDTH=320 HEIGHT=320 RAY_STEPS=84 SUBSTEPS=2 FPS=24 \
  EXTRA_ARGS="--emit-ratio 0.65 --auto-tail --tail-max-ratio 0.9 --tail-min-ratio 0.2 --tail-threshold 0.01 --loop-start-empty --bg-solid 0 0 0 --vignette 0"
```

Useful Taichi smoke-shape controls:

- `--ambient-strength` higher = more room turbulence and lateral drift.
- `--ambient-drift` higher = stronger consistent side movement.
- `--source-jitter` higher = source wanders more.
- `--snake-strength` higher = stronger side-to-side snaking.
- `--snake-speed` higher = faster meander.
- `--snake-wavelength` lower = tighter, more frequent bends.
- `--strand-strength` higher = more secondary wisps/paths at once.
- `--top-calm` higher = less lateral blowout near top.
- `--top-dissipation` higher = thinner/faster fade near top.
- `--bg-solid R G B` sets a flat background (for black: `--bg-solid 0 0 0`).
- `--vignette 0` disables corner darkening.
- `--source-density`, `--source-temp`, `--source-updraft` tune base column body without changing source width.
- `--emit-ratio` emits only for a fraction of base frames, then no new smoke.
- `--auto-tail` continues simulation after emission until fade-out.
- `--tail-max-ratio`, `--tail-min-ratio`, `--tail-threshold` control auto-tail duration.
- `--loop-start-empty` prepends an empty frame for cleaner gif looping.
- `--blackout-tail-ratio` applies fade-to-black over the final fraction of frames.
- `--blackout-head-ratio` applies fade-in from black over the initial fraction of frames.
- `--blackout-curve` controls fade shaping (`1` linear, larger = gentler near full brightness).
- `--extinction` lower = thinner, less opaque smoke.
- `--gif-profiles` selects GIF export presets: `default`, `discord`, `facebook`.
- `--gif-fps` / `--gif-scale-width` override profile defaults globally.

## JSON Config Files

`main_taichi.py` supports `--config <path/to/config.json>`.

- Nested categories are supported.
- Leaf keys map to CLI option names (`snake_speed` or `snake-speed` both work).
- Precedence is `script defaults < config file < CLI arguments`.

Sample config:

- `config/cigarette.json`

Direct CLI examples:

```bash
python3 main_taichi.py --config config/cigarette.json --mp4 --gif --overwrite
python3 main_taichi.py --config config/cigarette.json --frames 360 --ambient-strength 1.3 --mp4 --gif --overwrite
```

Platform-targeted GIF exports:

```bash
# write both out/final/smoke.discord.gif and out/final/smoke.facebook.gif
python3 main_taichi.py --out out/final --encode-only --gif --gif-profiles discord facebook --overwrite

# keep profile behavior but manually force tighter output
python3 main_taichi.py --out out/final --encode-only --gif --gif-profiles discord --gif-fps 10 --gif-scale-width 360 --overwrite
```

Force CPU (Taichi):

```bash
make all ENGINE=taichi ARCH=cpu
```

Fallback NumPy engine:

```bash
make all ENGINE=numpy
```

Direct CLI usage:

```bash
python3 main_taichi.py --out out/latest --frames 240 --grid 72 --arch auto --mp4 --gif --overwrite
```
