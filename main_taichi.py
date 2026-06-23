#!/usr/bin/env python3

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

LOCAL_TAICHI_CACHE = Path(__file__).resolve().parent / ".ti_cache"
LOCAL_XDG_CACHE = Path(__file__).resolve().parent / ".xdg_cache"
os.environ["XDG_CACHE_HOME"] = str(LOCAL_XDG_CACHE)
os.environ["TI_CACHE_FILE_PATH"] = str(LOCAL_TAICHI_CACHE)

try:
    import taichi as ti
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Taichi is required for main_taichi.py. Install with `pip install taichi`.") from exc


def resolve_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def maybe_encode_mp4(ffmpeg: str, frames_dir: Path, out_mp4: Path, fps: int, overwrite: bool) -> None:
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


GIF_PROFILE_PRESETS: dict[str, dict[str, int]] = {
    "default": {"fps": 15, "scale_width": 960},
    "discord": {"fps": 12, "scale_width": 480},
    "facebook": {"fps": 15, "scale_width": 640},
}


def resolve_gif_encode_settings(profile: str, gif_fps: int | None, gif_scale_width: int | None) -> tuple[int, int]:
    preset = GIF_PROFILE_PRESETS[profile]
    out_fps = int(gif_fps) if gif_fps is not None else int(preset["fps"])
    out_scale = int(gif_scale_width) if gif_scale_width is not None else int(preset["scale_width"])
    return max(1, out_fps), max(16, out_scale)


def maybe_encode_gif(
    ffmpeg: str,
    frames_dir: Path,
    out_gif: Path,
    fps: int,
    overwrite: bool,
    gif_fps: int,
    gif_scale_width: int,
) -> None:
    palette_path = out_gif.with_suffix(".palette.png")
    overwrite_flag = "-y" if overwrite else "-n"
    base_filter = f"fps={gif_fps},scale={gif_scale_width}:-1:flags=lanczos"

    cmd_palette = [
        ffmpeg,
        overwrite_flag,
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%05d.png"),
        "-vf",
        f"{base_filter},palettegen=stats_mode=diff",
        "-frames:v",
        "1",
        "-update",
        "1",
        str(palette_path),
    ]
    subprocess.run(cmd_palette, check=True)

    cmd_gif = [
        ffmpeg,
        overwrite_flag,
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%05d.png"),
        "-i",
        str(palette_path),
        "-lavfi",
        f"{base_filter}[x];[x][1:v]paletteuse=dither=sierra2_4a",
        str(out_gif),
    ]
    subprocess.run(cmd_gif, check=True)

    if palette_path.exists():
        palette_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Taichi-accelerated offline smoke simulation and render.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config file. Nested objects are supported; leaf keys map to CLI option names.",
    )
    parser.add_argument("--out", type=Path, default=Path("out/latest"), help="Output directory.")
    parser.add_argument("--frames", type=int, default=240, help="Number of frames to render.")
    parser.add_argument("--fps", type=int, default=30, help="Output framerate.")
    parser.add_argument("--grid", type=int, default=72, help="Grid resolution (N^3).")
    parser.add_argument("--width", type=int, default=540, help="Frame width in pixels.")
    parser.add_argument("--height", type=int, default=540, help="Frame height in pixels.")
    parser.add_argument("--ray-steps", type=int, default=104, help="Raymarch steps per frame.")
    parser.add_argument("--substeps", type=int, default=2, help="Simulation steps per frame.")
    parser.add_argument("--dt", type=float, default=0.115, help="Simulation delta time per frame.")
    parser.add_argument("--emit-ratio", type=float, default=1.0, help="Fraction of base frames that emit smoke.")
    parser.add_argument(
        "--auto-tail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue sim after emission stops until fade-out.",
    )
    parser.add_argument(
        "--tail-max-ratio",
        type=float,
        default=1.0,
        help="Maximum additional tail frames as a ratio of --frames when --auto-tail is enabled.",
    )
    parser.add_argument(
        "--tail-min-ratio",
        type=float,
        default=0.15,
        help="Minimum tail frames ratio before fade-out threshold can stop auto-tail.",
    )
    parser.add_argument(
        "--tail-threshold",
        type=float,
        default=0.012,
        help="Stop auto-tail when max density drops below this threshold.",
    )
    parser.add_argument(
        "--loop-start-empty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write an initial empty frame before simulation for cleaner gif looping.",
    )
    parser.add_argument(
        "--blackout-tail-ratio",
        type=float,
        default=0.0,
        help="Fade-to-black over this fraction of output frames at the end (0 disables).",
    )
    parser.add_argument(
        "--blackout-head-ratio",
        type=float,
        default=0.0,
        help="Fade-in from black over this fraction of output frames at the start (0 disables).",
    )
    parser.add_argument(
        "--blackout-curve",
        type=float,
        default=1.0,
        help="Power curve for blackout envelope (1 linear, >1 softer near full brightness).",
    )
    parser.add_argument("--pressure-iters", type=int, default=30, help="Jacobi pressure iterations.")
    parser.add_argument("--density-decay", type=float, default=0.997, help="Per-step density decay.")
    parser.add_argument("--temp-decay", type=float, default=0.994, help="Per-step temperature decay.")
    parser.add_argument("--velocity-decay", type=float, default=0.997, help="Per-step velocity decay.")
    parser.add_argument("--source-density", type=float, default=1.2, help="Density injected at the source per step.")
    parser.add_argument("--source-temp", type=float, default=1.7, help="Temperature injected at the source per step.")
    parser.add_argument("--source-updraft", type=float, default=2.55, help="Upward velocity injected at the source.")
    parser.add_argument("--buoyancy", type=float, default=0.68, help="Density-based buoyancy.")
    parser.add_argument("--thermal-buoyancy", type=float, default=0.42, help="Temperature-based buoyancy.")
    parser.add_argument("--vorticity", type=float, default=1.05, help="Vorticity confinement amount.")
    parser.add_argument("--extinction", type=float, default=2.25, help="Volume extinction strength.")
    parser.add_argument("--shadow", type=float, default=4.2, help="Top-light shadow strength.")
    parser.add_argument(
        "--bg-solid",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=None,
        help="Solid background color in linear RGB (0..1), e.g. --bg-solid 0 0 0.",
    )
    parser.add_argument(
        "--bg-top",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=(0.91, 0.93, 0.96),
        help="Top gradient background color in linear RGB (0..1).",
    )
    parser.add_argument(
        "--bg-bottom",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=(0.06, 0.07, 0.09),
        help="Bottom gradient background color in linear RGB (0..1).",
    )
    parser.add_argument("--vignette", type=float, default=0.22, help="Background vignette strength (0 disables).")
    parser.add_argument("--ambient-strength", type=float, default=1.0, help="Ambient room airflow strength.")
    parser.add_argument("--ambient-drift", type=float, default=0.038, help="Ambient side-drift amount.")
    parser.add_argument("--source-jitter", type=float, default=0.04, help="Source drift scale (fraction of grid width).")
    parser.add_argument("--snake-strength", type=float, default=0.060, help="Strength of coherent snake-like sway.")
    parser.add_argument("--snake-speed", type=float, default=0.62, help="Temporal speed of snake sway.")
    parser.add_argument("--snake-wavelength", type=float, default=0.33, help="Vertical wavelength of snake sway (normalized).")
    parser.add_argument(
        "--strand-strength",
        type=float,
        default=1.0,
        help="Strength of secondary micro-currents for multi-path wisps.",
    )
    parser.add_argument(
        "--top-calm",
        type=float,
        default=0.55,
        help="How much lateral turbulence is damped near the top (prevents fast billow).",
    )
    parser.add_argument(
        "--top-dissipation",
        type=float,
        default=0.10,
        help="Additional density fade near the top to keep wisps thin.",
    )
    parser.add_argument("--orbit", type=float, default=0.14, help="Camera orbit amount over the full clip.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--arch", choices=["auto", "cpu", "cuda", "vulkan", "metal"], default="auto")
    parser.add_argument(
        "--mp4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Encode an mp4 after rendering frames.",
    )
    parser.add_argument(
        "--gif",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Encode an animated gif after rendering frames.",
    )
    parser.add_argument(
        "--gif-profiles",
        nargs="+",
        choices=sorted(GIF_PROFILE_PRESETS),
        default=["default"],
        metavar="PROFILE",
        help="GIF export preset(s). Use multiple values to emit multiple GIFs, e.g. --gif-profiles discord facebook.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=None,
        help="Override GIF fps for all selected --gif-profiles.",
    )
    parser.add_argument(
        "--gif-scale-width",
        type=int,
        default=None,
        help="Override GIF output width (px) for all selected --gif-profiles.",
    )
    parser.add_argument(
        "--encode-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip simulation and encode using existing frames.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing frames and output media.",
    )
    return parser


def _normalize_config_key(key: str) -> str:
    return key.strip().replace("-", "_")


def _flatten_config_leaves(
    obj: dict[str, object], node_path: tuple[str, ...] = ()
) -> tuple[dict[str, object], dict[str, str]]:
    flattened: dict[str, object] = {}
    leaf_paths: dict[str, str] = {}

    def walk(node: object, path: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for raw_key, child in node.items():
                if not isinstance(raw_key, str):
                    joined = ".".join(path) if path else "<root>"
                    raise ValueError(f"Config object '{joined}' contains a non-string key.")
                walk(child, path + (raw_key,))
            return

        if not path:
            raise ValueError("Config root must be a JSON object with named keys.")

        key = _normalize_config_key(path[-1])
        key_path = ".".join(path)
        if key in flattened:
            raise ValueError(f"Config key collision on leaf '{key}': '{leaf_paths[key]}' and '{key_path}'.")
        flattened[key] = node
        leaf_paths[key] = key_path

    walk(obj, node_path)
    return flattened, leaf_paths


def _coerce_bool_config_value(value: object, key_path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Config key '{key_path}' must be a boolean value.")


def _coerce_config_value(action: argparse.Action, value: object, key_path: str) -> object:
    if value is None:
        return None

    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
        return _coerce_bool_config_value(value, key_path)

    if action.nargs is not None:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Config key '{key_path}' must be an array.")
        items = list(value)
        if isinstance(action.nargs, int) and len(items) != action.nargs:
            raise ValueError(f"Config key '{key_path}' must contain exactly {action.nargs} item(s).")
        if action.type is not None:
            try:
                items = [action.type(item) for item in items]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Config key '{key_path}' has invalid item type(s).") from exc
        if action.choices is not None:
            bad = [item for item in items if item not in action.choices]
            if bad:
                raise ValueError(f"Config key '{key_path}' contains invalid choice(s): {bad!r}.")
        return items

    converted = value
    if action.type is not None:
        try:
            converted = action.type(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config key '{key_path}' has an invalid value: {value!r}.") from exc

    if action.choices is not None and converted not in action.choices:
        raise ValueError(f"Config key '{key_path}' has invalid choice {converted!r}; expected {list(action.choices)!r}.")
    return converted


def load_config_defaults(config_path: Path, parser: argparse.ArgumentParser) -> dict[str, object]:
    path = config_path.expanduser()
    if not path.exists():
        raise ValueError(f"Config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")

    try:
        config_raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config JSON parse error in '{path}': {exc.msg} (line {exc.lineno}, column {exc.colno}).") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read config file '{path}': {exc}.") from exc

    if not isinstance(config_raw, dict):
        raise ValueError(f"Config file '{path}' must contain a top-level JSON object.")

    leaves, leaf_paths = _flatten_config_leaves(config_raw)
    actions_by_dest = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {"help", "config"} and action.option_strings
    }

    unknown: list[str] = []
    defaults: dict[str, object] = {}
    for leaf_key, raw_value in leaves.items():
        if leaf_key == "config":
            continue
        action = actions_by_dest.get(leaf_key)
        if action is None:
            unknown.append(leaf_paths[leaf_key])
            continue
        defaults[leaf_key] = _coerce_config_value(action, raw_value, leaf_paths[leaf_key])

    if unknown:
        unknown_str = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown config key(s): {unknown_str}.")

    return defaults


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    pre_args, _ = bootstrap.parse_known_args(argv)

    if pre_args.config is not None:
        try:
            config_defaults = load_config_defaults(pre_args.config, parser)
        except ValueError as exc:
            parser.error(str(exc))
        parser.set_defaults(**config_defaults)

    return parser.parse_args(argv)


def clamp_color_triplet(values: list[float] | tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        max(0.0, min(1.0, float(values[0]))),
        max(0.0, min(1.0, float(values[1]))),
        max(0.0, min(1.0, float(values[2]))),
    )


def apply_blackout_envelope(frames_dir: Path, head_ratio: float, tail_ratio: float, curve: float) -> None:
    frame_paths = sorted(frames_dir.glob("*.png"))
    total = len(frame_paths)
    if total == 0:
        return

    head_ratio = max(0.0, min(1.0, head_ratio))
    tail_ratio = max(0.0, min(1.0, tail_ratio))
    curve = max(0.05, curve)
    head_count = int(round(total * head_ratio))
    tail_count = int(round(total * tail_ratio))
    if head_count == 0 and tail_count == 0:
        return

    for idx, frame_path in enumerate(frame_paths):
        gain = 1.0

        if head_count > 0 and idx < head_count:
            head_t = (idx + 1) / float(head_count)
            head_t = max(0.0, min(1.0, head_t))
            gain *= head_t**curve

        if tail_count > 0 and idx >= total - tail_count:
            denom = max(1, tail_count - 1)
            tail_t = (total - 1 - idx) / float(denom)
            tail_t = max(0.0, min(1.0, tail_t))
            gain *= tail_t**curve

        if gain >= 0.9995:
            continue

        image = Image.open(frame_path).convert("RGB")
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels *= gain
        out = np.clip(pixels * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(out, mode="RGB").save(frame_path)

    print(f"[blackout] applied envelope head={head_count} frame(s), tail={tail_count} frame(s), curve={curve:.2f}")


def init_taichi(arch_name: str, seed: int) -> None:
    LOCAL_XDG_CACHE.mkdir(parents=True, exist_ok=True)
    LOCAL_TAICHI_CACHE.mkdir(parents=True, exist_ok=True)
    arch_map = {
        "cpu": ti.cpu,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "metal": ti.metal,
    }

    if arch_name == "auto":
        try:
            ti.init(arch=ti.gpu, random_seed=seed, offline_cache=False)
        except Exception:
            ti.init(arch=ti.cpu, random_seed=seed, offline_cache=False)
    else:
        ti.init(arch=arch_map[arch_name], random_seed=seed, offline_cache=False)


@ti.data_oriented
class SmokeSimTaichi:
    def __init__(
        self,
        grid: int,
        width: int,
        height: int,
        pressure_iterations: int,
        density_decay: float,
        temp_decay: float,
        velocity_decay: float,
        source_density: float,
        source_temp: float,
        source_updraft: float,
        buoyancy: float,
        thermal_buoyancy: float,
        vorticity_confinement: float,
        ambient_strength: float,
        ambient_drift: float,
        source_jitter: float,
        snake_strength: float,
        snake_speed: float,
        snake_wavelength: float,
        strand_strength: float,
        top_calm: float,
        top_dissipation: float,
        bg_top: tuple[float, float, float],
        bg_bottom: tuple[float, float, float],
        vignette_strength: float,
        seed: int,
    ) -> None:
        self.nx = grid
        self.ny = grid
        self.nz = grid
        self.width = width
        self.height = height
        self.pressure_iterations = pressure_iterations
        self.density_decay = density_decay
        self.temp_decay = temp_decay
        self.velocity_decay = velocity_decay
        self.source_density = source_density
        self.source_temp = source_temp
        self.source_updraft = source_updraft
        self.buoyancy = buoyancy
        self.thermal_buoyancy = thermal_buoyancy
        self.vorticity_confinement = vorticity_confinement
        self.ambient_strength = ambient_strength
        self.ambient_drift = ambient_drift
        self.source_jitter = source_jitter
        self.snake_strength = snake_strength
        self.snake_speed = snake_speed
        self.snake_wavelength = max(snake_wavelength, 0.05)
        self.strand_strength = strand_strength
        self.top_calm = top_calm
        self.top_dissipation = top_dissipation
        self.vignette_strength = max(0.0, vignette_strength)
        self.rng = np.random.default_rng(seed)

        shape = (self.nx, self.ny, self.nz)
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.vel_prev = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.curl = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        self.density = ti.field(dtype=ti.f32, shape=shape)
        self.density_prev = ti.field(dtype=ti.f32, shape=shape)
        self.temperature = ti.field(dtype=ti.f32, shape=shape)
        self.temperature_prev = ti.field(dtype=ti.f32, shape=shape)
        self.pressure = ti.field(dtype=ti.f32, shape=shape)
        self.pressure_tmp = ti.field(dtype=ti.f32, shape=shape)
        self.divergence = ti.field(dtype=ti.f32, shape=shape)
        self.curl_mag = ti.field(dtype=ti.f32, shape=shape)
        self.light_volume = ti.field(dtype=ti.f32, shape=shape)
        self.image = ti.Vector.field(3, dtype=ti.u8, shape=(self.height, self.width))
        self.bg_top = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.bg_bottom = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.bg_top[None] = ti.Vector([bg_top[0], bg_top[1], bg_top[2]])
        self.bg_bottom[None] = ti.Vector([bg_bottom[0], bg_bottom[1], bg_bottom[2]])

        self.source_radius = self.nx * 0.014
        self.source_y = self.ny * 0.10
        self.swirl_strength = 1.9
        self.injection_density = self.source_density
        self.injection_temp = self.source_temp
        self.injection_updraft = self.source_updraft
        self.source_x = self.nx * 0.5
        self.source_z = self.nz * 0.5
        self.sim_time = 0.0

    @ti.func
    def clamp(self, x, lo, hi):
        return ti.max(lo, ti.min(x, hi))

    @ti.func
    def trilerp_scalar(self, field, x, y, z):
        x = self.clamp(x, 0.0, self.nx - 1.001)
        y = self.clamp(y, 0.0, self.ny - 1.001)
        z = self.clamp(z, 0.0, self.nz - 1.001)

        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        z0 = ti.cast(ti.floor(z), ti.i32)
        x1 = ti.min(x0 + 1, self.nx - 1)
        y1 = ti.min(y0 + 1, self.ny - 1)
        z1 = ti.min(z0 + 1, self.nz - 1)

        tx = x - ti.cast(x0, ti.f32)
        ty = y - ti.cast(y0, ti.f32)
        tz = z - ti.cast(z0, ti.f32)

        c000 = field[x0, y0, z0]
        c100 = field[x1, y0, z0]
        c010 = field[x0, y1, z0]
        c110 = field[x1, y1, z0]
        c001 = field[x0, y0, z1]
        c101 = field[x1, y0, z1]
        c011 = field[x0, y1, z1]
        c111 = field[x1, y1, z1]

        c00 = c000 * (1.0 - tx) + c100 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx
        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty
        return c0 * (1.0 - tz) + c1 * tz

    @ti.func
    def trilerp_vector(self, field, x, y, z):
        x = self.clamp(x, 0.0, self.nx - 1.001)
        y = self.clamp(y, 0.0, self.ny - 1.001)
        z = self.clamp(z, 0.0, self.nz - 1.001)

        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        z0 = ti.cast(ti.floor(z), ti.i32)
        x1 = ti.min(x0 + 1, self.nx - 1)
        y1 = ti.min(y0 + 1, self.ny - 1)
        z1 = ti.min(z0 + 1, self.nz - 1)

        tx = x - ti.cast(x0, ti.f32)
        ty = y - ti.cast(y0, ti.f32)
        tz = z - ti.cast(z0, ti.f32)

        c000 = field[x0, y0, z0]
        c100 = field[x1, y0, z0]
        c010 = field[x0, y1, z0]
        c110 = field[x1, y1, z0]
        c001 = field[x0, y0, z1]
        c101 = field[x1, y0, z1]
        c011 = field[x0, y1, z1]
        c111 = field[x1, y1, z1]

        c00 = c000 * (1.0 - tx) + c100 * tx
        c10 = c010 * (1.0 - tx) + c110 * tx
        c01 = c001 * (1.0 - tx) + c101 * tx
        c11 = c011 * (1.0 - tx) + c111 * tx
        c0 = c00 * (1.0 - ty) + c10 * ty
        c1 = c01 * (1.0 - ty) + c11 * ty
        return c0 * (1.0 - tz) + c1 * tz

    @ti.kernel
    def inject_source(self, dt: ti.f32, cx: ti.f32, cy: ti.f32, cz: ti.f32, pulse: ti.f32):
        for i, j, k in self.density:
            dx = ti.cast(i, ti.f32) - cx
            dy = ti.cast(j, ti.f32) - cy
            dz = ti.cast(k, ti.f32) - cz
            d2 = dx * dx + dy * dy + dz * dz
            mask = ti.exp(-d2 / (2.0 * self.source_radius * self.source_radius))

            self.density[i, j, k] += self.injection_density * pulse * dt * mask
            self.temperature[i, j, k] += self.injection_temp * dt * mask
            self.vel[i, j, k][1] += self.injection_updraft * dt * mask
            self.vel[i, j, k][0] += self.swirl_strength * dt * mask * (dz / (self.source_radius + 1e-5))
            self.vel[i, j, k][2] -= self.swirl_strength * dt * mask * (dx / (self.source_radius + 1e-5))

    @ti.kernel
    def copy_state(self):
        for i, j, k in self.density:
            self.density_prev[i, j, k] = self.density[i, j, k]
            self.temperature_prev[i, j, k] = self.temperature[i, j, k]
            self.vel_prev[i, j, k] = self.vel[i, j, k]

    @ti.kernel
    def advect_velocity(self, dt: ti.f32):
        for i, j, k in self.vel:
            pos = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32), ti.cast(k, ti.f32)]) - dt * self.vel_prev[i, j, k]
            self.vel[i, j, k] = self.trilerp_vector(self.vel_prev, pos[0], pos[1], pos[2])

    @ti.kernel
    def advect_scalars(self, dt: ti.f32):
        for i, j, k in self.density:
            pos = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32), ti.cast(k, ti.f32)]) - dt * self.vel_prev[i, j, k]
            self.density[i, j, k] = self.trilerp_scalar(self.density_prev, pos[0], pos[1], pos[2])
            self.temperature[i, j, k] = self.trilerp_scalar(self.temperature_prev, pos[0], pos[1], pos[2])

    @ti.kernel
    def apply_buoyancy(self, dt: ti.f32):
        for i, j, k in self.density:
            buoy = self.buoyancy * self.density[i, j, k] + self.thermal_buoyancy * ti.max(self.temperature[i, j, k], 0.0)
            self.vel[i, j, k][1] += dt * buoy

    @ti.kernel
    def apply_ambient_flow(self, dt: ti.f32, t: ti.f32):
        for i, j, k in self.vel:
            x = ti.cast(i, ti.f32) / ti.cast(self.nx - 1, ti.f32)
            y = ti.cast(j, ti.f32) / ti.cast(self.ny - 1, ti.f32)
            z = ti.cast(k, ti.f32) / ti.cast(self.nz - 1, ti.f32)

            rise = ti.max(0.0, ti.min((y - 0.06) / 0.94, 1.0))
            top = ti.max(0.0, ti.min((y - 0.68) / 0.32, 1.0))
            lateral_calm = 1.0 - self.top_calm * top
            mid_belt = 0.55 + 0.45 * ti.exp(-((y - 0.56) * (y - 0.56)) / 0.08)
            layer = (0.16 + 0.84 * rise) * mid_belt

            wave_x = ti.sin(6.2831 * (0.55 * x + 0.35 * z) + t * 0.16)
            wave_z = ti.sin(6.2831 * (0.75 * z - 0.2 * x) - t * 0.11 + 1.7)
            eddy_x = ti.sin(6.2831 * (1.7 * y + 1.2 * z) + t * 0.31)
            eddy_z = ti.sin(6.2831 * (1.6 * y + 1.1 * x) - t * 0.27 + 0.9)

            # Coherent centerline meander so the column "snakes" as it rises.
            snake_phase = 6.2831 * (t * self.snake_speed - y / self.snake_wavelength)
            local_phase_a = 6.2831 * (0.45 * x + 0.32 * z)
            local_phase_b = 6.2831 * (0.21 * x - 0.51 * z)
            snake_amp = self.snake_strength * rise
            snake_x = snake_amp * (
                0.72 * ti.sin(snake_phase + local_phase_a + 0.35 * ti.sin(t * 0.17))
                + 0.28 * ti.sin(1.7 * snake_phase + local_phase_b + 0.9)
            )
            snake_z = snake_amp * (
                0.62 * ti.sin(0.86 * snake_phase - local_phase_b + 1.2)
                + 0.38 * ti.sin(1.45 * snake_phase + local_phase_a + 0.2)
            )

            strand_x = self.strand_strength * rise * (0.030 * ti.sin(6.2831 * (2.4 * y + 1.6 * z) + t * 0.54))
            strand_z = self.strand_strength * rise * (0.030 * ti.sin(6.2831 * (2.2 * y + 1.5 * x) - t * 0.49 + 0.7))

            vx = (self.ambient_drift + 0.070 * wave_x + 0.040 * eddy_x) * lateral_calm + snake_x + strand_x
            vz = (0.5 * self.ambient_drift + 0.060 * wave_z + 0.040 * eddy_z) * lateral_calm + snake_z + strand_z
            vy = (0.018 * ti.sin(6.2831 * (0.6 * x + 0.7 * z) + t * 0.22)) * (0.75 + 0.25 * lateral_calm)

            self.vel[i, j, k] += dt * self.ambient_strength * layer * ti.Vector([vx, vy, vz])

    @ti.kernel
    def compute_curl(self):
        for i, j, k in self.vel:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1 and 0 < k < self.nz - 1:
                du_dy = 0.5 * (self.vel[i, j + 1, k][0] - self.vel[i, j - 1, k][0])
                du_dz = 0.5 * (self.vel[i, j, k + 1][0] - self.vel[i, j, k - 1][0])
                dv_dx = 0.5 * (self.vel[i + 1, j, k][1] - self.vel[i - 1, j, k][1])
                dv_dz = 0.5 * (self.vel[i, j, k + 1][1] - self.vel[i, j, k - 1][1])
                dw_dx = 0.5 * (self.vel[i + 1, j, k][2] - self.vel[i - 1, j, k][2])
                dw_dy = 0.5 * (self.vel[i, j + 1, k][2] - self.vel[i, j - 1, k][2])

                c = ti.Vector([dw_dy - dv_dz, du_dz - dw_dx, dv_dx - du_dy])
                self.curl[i, j, k] = c
                self.curl_mag[i, j, k] = ti.sqrt(c.dot(c) + 1e-8)
            else:
                self.curl[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                self.curl_mag[i, j, k] = 0.0

    @ti.kernel
    def apply_vorticity(self, dt: ti.f32):
        for i, j, k in self.vel:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1 and 0 < k < self.nz - 1:
                n = ti.Vector(
                    [
                        0.5 * (self.curl_mag[i + 1, j, k] - self.curl_mag[i - 1, j, k]),
                        0.5 * (self.curl_mag[i, j + 1, k] - self.curl_mag[i, j - 1, k]),
                        0.5 * (self.curl_mag[i, j, k + 1] - self.curl_mag[i, j, k - 1]),
                    ]
                )
                n_len = ti.sqrt(n.dot(n) + 1e-8)
                n = n / n_len
                force = n.cross(self.curl[i, j, k])
                self.vel[i, j, k] += dt * self.vorticity_confinement * force

    @ti.kernel
    def compute_divergence(self):
        for i, j, k in self.vel:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1 and 0 < k < self.nz - 1:
                self.divergence[i, j, k] = 0.5 * (
                    self.vel[i + 1, j, k][0]
                    - self.vel[i - 1, j, k][0]
                    + self.vel[i, j + 1, k][1]
                    - self.vel[i, j - 1, k][1]
                    + self.vel[i, j, k + 1][2]
                    - self.vel[i, j, k - 1][2]
                )
            else:
                self.divergence[i, j, k] = 0.0

    @ti.kernel
    def clear_pressure(self):
        for i, j, k in self.pressure:
            self.pressure[i, j, k] = 0.0
            self.pressure_tmp[i, j, k] = 0.0

    @ti.kernel
    def jacobi_pressure(self):
        for i, j, k in self.pressure:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1 and 0 < k < self.nz - 1:
                self.pressure_tmp[i, j, k] = (
                    self.pressure[i + 1, j, k]
                    + self.pressure[i - 1, j, k]
                    + self.pressure[i, j + 1, k]
                    + self.pressure[i, j - 1, k]
                    + self.pressure[i, j, k + 1]
                    + self.pressure[i, j, k - 1]
                    - self.divergence[i, j, k]
                ) / 6.0
            else:
                self.pressure_tmp[i, j, k] = 0.0

    @ti.kernel
    def subtract_pressure_gradient(self):
        for i, j, k in self.vel:
            if 0 < i < self.nx - 1 and 0 < j < self.ny - 1 and 0 < k < self.nz - 1:
                gx = 0.5 * (self.pressure[i + 1, j, k] - self.pressure[i - 1, j, k])
                gy = 0.5 * (self.pressure[i, j + 1, k] - self.pressure[i, j - 1, k])
                gz = 0.5 * (self.pressure[i, j, k + 1] - self.pressure[i, j, k - 1])
                self.vel[i, j, k] -= ti.Vector([gx, gy, gz])
            else:
                self.vel[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def apply_decay_and_boundaries(self):
        for i, j, k in self.density:
            if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1 or k == 0 or k == self.nz - 1:
                self.vel[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                self.density[i, j, k] = 0.0
                self.temperature[i, j, k] = 0.0
            else:
                y = ti.cast(j, ti.f32) / ti.cast(self.ny - 1, ti.f32)
                top = ti.max(0.0, ti.min((y - 0.68) / 0.32, 1.0))
                lateral_damp = 1.0 - 0.30 * self.top_calm * top
                density_keep = ti.max(0.0, self.density_decay * (1.0 - self.top_dissipation * top))
                temp_keep = ti.max(0.0, self.temp_decay * (1.0 - 0.6 * self.top_dissipation * top))

                self.vel[i, j, k][0] *= self.velocity_decay * lateral_damp
                self.vel[i, j, k][1] *= self.velocity_decay
                self.vel[i, j, k][2] *= self.velocity_decay * lateral_damp
                self.density[i, j, k] = ti.max(self.density[i, j, k] * density_keep, 0.0)
                self.temperature[i, j, k] = ti.max(self.temperature[i, j, k] * temp_keep, 0.0)

    @ti.kernel
    def compute_light_volume(self, shadow_strength: ti.f32):
        for i, k in ti.ndrange(self.nx, self.nz):
            optical = 0.0
            for yy in range(self.ny):
                y = self.ny - 1 - yy
                optical += self.density[i, y, k] / ti.cast(self.ny, ti.f32)
                self.light_volume[i, y, k] = ti.exp(-shadow_strength * optical)

    @ti.kernel
    def render(self, angle: ti.f32, extinction: ti.f32, ray_steps: ti.i32):
        smoke_tint = ti.Vector([0.84, 0.87, 0.91])
        ember_tint = ti.Vector([1.0, 0.46, 0.15])
        bg_top = self.bg_top[None]
        bg_bottom = self.bg_bottom[None]
        cos_a = ti.cos(angle)
        sin_a = ti.sin(angle)

        near = -0.22
        far = 1.22
        step = (far - near) / ti.cast(ray_steps, ti.f32)
        w1 = ti.max(self.width - 1, 1)
        h1 = ti.max(self.height - 1, 1)

        for py, px in ti.ndrange(self.height, self.width):
            trans = 1.0
            rgb = ti.Vector([0.0, 0.0, 0.0])

            x01 = ti.cast(px, ti.f32) / ti.cast(w1, ti.f32)
            y01 = ti.cast(py, ti.f32) / ti.cast(h1, ti.f32)
            base_x = (-0.54 + x01 * 1.08) + 0.5
            base_y = 1.04 + y01 * (-1.10)

            for s in range(ray_steps):
                z = near + (ti.cast(s, ti.f32) + 0.5) * step
                local_x = base_x - 0.5
                local_z = z - 0.5
                xr = local_x * cos_a + local_z * sin_a + 0.5
                zr = -local_x * sin_a + local_z * cos_a + 0.5

                if 0.0 <= xr <= 1.0 and 0.0 <= base_y <= 1.0 and 0.0 <= zr <= 1.0:
                    dens = self.trilerp_scalar(self.density, xr * (self.nx - 1), base_y * (self.ny - 1), zr * (self.nz - 1))
                    temp = self.trilerp_scalar(
                        self.temperature, xr * (self.nx - 1), base_y * (self.ny - 1), zr * (self.nz - 1)
                    )
                    lit = self.trilerp_scalar(
                        self.light_volume, xr * (self.nx - 1), base_y * (self.ny - 1), zr * (self.nz - 1)
                    )

                    alpha = 1.0 - ti.exp(-dens * extinction * step)
                    alpha = ti.max(0.0, ti.min(alpha, 1.0))
                    warmth = ti.max(0.0, ti.min(temp * 0.085, 1.0))

                    color = smoke_tint * (0.35 + 0.65 * lit)
                    color += ember_tint * (0.24 * warmth)
                    rgb += trans * alpha * color
                    trans *= 1.0 - alpha

            bg = bg_top * (1.0 - y01) + bg_bottom * y01
            gx = -1.0 + 2.0 * x01
            gy = -1.0 + 2.0 * y01
            vignette = 1.0 - self.vignette_strength * ti.min(gx * gx + gy * gy, 1.0)
            bg *= vignette
            final_color = rgb + trans * bg

            out = ti.Vector.zero(ti.u8, 3)
            for c in ti.static(range(3)):
                v = ti.max(0.0, ti.min(final_color[c], 1.0))
                v = ti.pow(v, 1.0 / 2.2)
                out[c] = ti.cast(v * 255.0 + 0.5, ti.u8)
            self.image[py, px] = out

    def project(self) -> None:
        self.compute_divergence()
        self.clear_pressure()
        for _ in range(self.pressure_iterations):
            self.jacobi_pressure()
            self.pressure, self.pressure_tmp = self.pressure_tmp, self.pressure
        self.subtract_pressure_gradient()

    def step(self, dt: float, frame_idx: int, emit: bool = True) -> None:
        if emit:
            target = self.rng.normal(0.0, self.nx * self.source_jitter, size=2)
            self.source_x = 0.965 * self.source_x + 0.035 * (self.nx * 0.5 + float(target[0]))
            self.source_z = 0.965 * self.source_z + 0.035 * (self.nz * 0.5 + float(target[1]))
            drift_limit = self.nx * 0.11
            self.source_x = float(np.clip(self.source_x, self.nx * 0.5 - drift_limit, self.nx * 0.5 + drift_limit))
            self.source_z = float(np.clip(self.source_z, self.nz * 0.5 - drift_limit, self.nz * 0.5 + drift_limit))

            cx = self.source_x
            cy = self.source_y
            cz = self.source_z
            pulse = 0.88 + 0.12 * math.sin(frame_idx * 0.11)
            self.inject_source(dt, cx, cy, cz, pulse)

        self.copy_state()
        self.advect_velocity(dt)
        self.advect_scalars(dt)
        self.sim_time += dt
        self.apply_ambient_flow(dt, self.sim_time)
        self.apply_buoyancy(dt)
        self.compute_curl()
        self.apply_vorticity(dt)
        self.project()
        self.apply_decay_and_boundaries()

    def render_frame(self, angle: float, extinction: float, shadow_strength: float, ray_steps: int) -> np.ndarray:
        self.compute_light_volume(shadow_strength)
        self.render(angle, extinction, ray_steps)
        return self.image.to_numpy()

    def max_density(self) -> float:
        return float(np.max(self.density.to_numpy()))


def main() -> None:
    args = parse_args()
    init_taichi(args.arch, args.seed)
    print(f"[taichi] backend: {ti.cfg.arch}")

    if args.bg_solid is not None:
        bg_top = clamp_color_triplet(args.bg_solid)
        bg_bottom = bg_top
    else:
        bg_top = clamp_color_triplet(args.bg_top)
        bg_bottom = clamp_color_triplet(args.bg_bottom)

    out_dir = args.out
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite and not args.encode_only:
        for frame in frames_dir.glob("*.png"):
            frame.unlink()

    if not args.encode_only:
        emit_ratio = max(0.0, min(1.0, args.emit_ratio))
        tail_max_ratio = max(0.0, args.tail_max_ratio)
        tail_min_ratio = max(0.0, args.tail_min_ratio)
        tail_threshold = max(0.0, args.tail_threshold)

        sim = SmokeSimTaichi(
            grid=args.grid,
            width=args.width,
            height=args.height,
            pressure_iterations=args.pressure_iters,
            density_decay=args.density_decay,
            temp_decay=args.temp_decay,
            velocity_decay=args.velocity_decay,
            source_density=args.source_density,
            source_temp=args.source_temp,
            source_updraft=args.source_updraft,
            buoyancy=args.buoyancy,
            thermal_buoyancy=args.thermal_buoyancy,
            vorticity_confinement=args.vorticity,
            ambient_strength=args.ambient_strength,
            ambient_drift=args.ambient_drift,
            source_jitter=args.source_jitter,
            snake_strength=args.snake_strength,
            snake_speed=args.snake_speed,
            snake_wavelength=args.snake_wavelength,
            strand_strength=args.strand_strength,
            top_calm=args.top_calm,
            top_dissipation=args.top_dissipation,
            bg_top=bg_top,
            bg_bottom=bg_bottom,
            vignette_strength=args.vignette,
            seed=args.seed,
        )

        dt_sub = args.dt / float(max(1, args.substeps))
        emit_frames = max(0, int(round(args.frames * emit_ratio)))
        out_frame_idx = 0

        if args.loop_start_empty:
            image = sim.render_frame(
                angle=-args.orbit * 0.5,
                extinction=args.extinction,
                shadow_strength=args.shadow,
                ray_steps=args.ray_steps,
            )
            frame_path = frames_dir / f"{out_frame_idx:05d}.png"
            Image.fromarray(image, mode="RGB").save(frame_path)
            out_frame_idx += 1

        for frame_idx in range(args.frames):
            emit_on = frame_idx < emit_frames
            for _ in range(args.substeps):
                sim.step(dt_sub, frame_idx, emit=emit_on)

            angle = -args.orbit * 0.5 + (args.orbit * frame_idx / max(args.frames - 1, 1))
            image = sim.render_frame(
                angle=angle,
                extinction=args.extinction,
                shadow_strength=args.shadow,
                ray_steps=args.ray_steps,
            )
            frame_path = frames_dir / f"{out_frame_idx:05d}.png"
            Image.fromarray(image, mode="RGB").save(frame_path)
            out_frame_idx += 1

            if frame_idx % 10 == 0 or frame_idx == args.frames - 1:
                print(f"[sim] frame {frame_idx + 1}/{args.frames}")

        if args.auto_tail:
            max_tail_frames = int(round(args.frames * tail_max_ratio))
            min_tail_frames = int(round(args.frames * tail_min_ratio))
            tail_count = 0
            tail_finished = False
            while tail_count < max_tail_frames:
                sim_frame_idx = args.frames + tail_count
                for _ in range(args.substeps):
                    sim.step(dt_sub, sim_frame_idx, emit=False)

                angle = -args.orbit * 0.5 + (args.orbit * min(sim_frame_idx, args.frames - 1) / max(args.frames - 1, 1))
                image = sim.render_frame(
                    angle=angle,
                    extinction=args.extinction,
                    shadow_strength=args.shadow,
                    ray_steps=args.ray_steps,
                )
                frame_path = frames_dir / f"{out_frame_idx:05d}.png"
                Image.fromarray(image, mode="RGB").save(frame_path)
                out_frame_idx += 1
                tail_count += 1

                max_dens = sim.max_density()
                if tail_count % 10 == 0:
                    print(f"[tail] frame {tail_count}/{max_tail_frames}, max_density={max_dens:.5f}")
                if tail_count >= min_tail_frames and max_dens <= tail_threshold:
                    print(f"[tail] ended at frame {tail_count}/{max_tail_frames}, max_density={max_dens:.5f}")
                    tail_finished = True
                    break
            if not tail_finished and max_tail_frames > 0:
                print(f"[tail] reached max tail frames ({max_tail_frames}), max_density={sim.max_density():.5f}")
    else:
        frame_count = len(list(frames_dir.glob("*.png")))
        if frame_count == 0:
            raise RuntimeError("No frames found for --encode-only. Render frames first.")
        print(f"[encode] using {frame_count} existing frame(s)")

    if not args.encode_only and (args.blackout_head_ratio > 0.0 or args.blackout_tail_ratio > 0.0):
        apply_blackout_envelope(
            frames_dir=frames_dir,
            head_ratio=args.blackout_head_ratio,
            tail_ratio=args.blackout_tail_ratio,
            curve=args.blackout_curve,
        )

    ffmpeg = resolve_ffmpeg()
    if args.mp4:
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is unavailable. Install system ffmpeg or `pip install imageio-ffmpeg`.")
        maybe_encode_mp4(ffmpeg, frames_dir, out_dir / "smoke.mp4", args.fps, args.overwrite)
        print(f"[encode] wrote {out_dir / 'smoke.mp4'}")
    if args.gif:
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is unavailable. Install system ffmpeg or `pip install imageio-ffmpeg`.")
        unique_profiles: list[str] = []
        for profile in args.gif_profiles:
            if profile not in unique_profiles:
                unique_profiles.append(profile)
        for profile in unique_profiles:
            gif_fps, gif_scale_width = resolve_gif_encode_settings(profile, args.gif_fps, args.gif_scale_width)
            out_name = "smoke.gif" if profile == "default" else f"smoke.{profile}.gif"
            out_path = out_dir / out_name
            maybe_encode_gif(
                ffmpeg,
                frames_dir,
                out_path,
                args.fps,
                args.overwrite,
                gif_fps=gif_fps,
                gif_scale_width=gif_scale_width,
            )
            print(
                f"[encode] wrote {out_path} "
                f"(profile={profile}, gif_fps={gif_fps}, gif_scale_width={gif_scale_width})"
            )


if __name__ == "__main__":
    main()
