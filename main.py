#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def set_boundaries(field: np.ndarray, value: float = 0.0) -> None:
    field[0, :, :] = value
    field[-1, :, :] = value
    field[:, 0, :] = value
    field[:, -1, :] = value
    field[:, :, 0] = value
    field[:, :, -1] = value


def trilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    nx, ny, nz = field.shape
    x = np.clip(x, 0.0, nx - 1.001)
    y = np.clip(y, 0.0, ny - 1.001)
    z = np.clip(z, 0.0, nz - 1.001)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    z0 = np.floor(z).astype(np.int32)
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    z1 = np.minimum(z0 + 1, nz - 1)

    tx = x - x0
    ty = y - y0
    tz = z - z0

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


def sample_normalized(field: np.ndarray, xn: np.ndarray, yn: np.ndarray, zn: np.ndarray) -> np.ndarray:
    nx, ny, nz = field.shape
    return trilinear_sample(field, xn * (nx - 1), yn * (ny - 1), zn * (nz - 1))


class SmokeSim:
    def __init__(
        self,
        grid: int,
        pressure_iterations: int,
        density_decay: float,
        temp_decay: float,
        velocity_decay: float,
        buoyancy: float,
        thermal_buoyancy: float,
        vorticity_confinement: float,
        seed: int,
    ) -> None:
        self.nx = grid
        self.ny = grid
        self.nz = grid
        self.pressure_iterations = pressure_iterations
        self.density_decay = density_decay
        self.temp_decay = temp_decay
        self.velocity_decay = velocity_decay
        self.buoyancy = buoyancy
        self.thermal_buoyancy = thermal_buoyancy
        self.vorticity_confinement = vorticity_confinement
        self.rng = np.random.default_rng(seed)

        shape = (self.nx, self.ny, self.nz)
        self.u = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.w = np.zeros(shape, dtype=np.float32)
        self.density = np.zeros(shape, dtype=np.float32)
        self.temperature = np.zeros(shape, dtype=np.float32)
        self.pressure = np.zeros(shape, dtype=np.float32)
        self.pressure_tmp = np.zeros(shape, dtype=np.float32)
        self.divergence = np.zeros(shape, dtype=np.float32)

        x = np.arange(self.nx, dtype=np.float32)
        y = np.arange(self.ny, dtype=np.float32)
        z = np.arange(self.nz, dtype=np.float32)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")

        self.source_radius = self.nx * 0.07
        self.source_y = self.ny * 0.11
        self.swirl_strength = 4.5
        self.injection_density = 7.5
        self.injection_temp = 10.0
        self.injection_updraft = 8.0

    def inject_source(self, dt: float, frame_idx: int) -> None:
        jitter = self.rng.normal(0.0, self.nx * 0.01, size=2)
        cx = self.nx * 0.5 + jitter[0]
        cy = self.source_y
        cz = self.nz * 0.5 + jitter[1]

        d2 = (self.X - cx) ** 2 + (self.Y - cy) ** 2 + (self.Z - cz) ** 2
        mask = np.exp(-d2 / (2.0 * self.source_radius**2)).astype(np.float32)

        pulse = 0.8 + 0.2 * math.sin(frame_idx * 0.3)
        self.density += (self.injection_density * pulse * dt) * mask
        self.temperature += (self.injection_temp * dt) * mask
        self.v += (self.injection_updraft * dt) * mask

        # Add a little spin at the source so the plume breaks symmetry.
        self.u += (self.swirl_strength * dt) * mask * ((self.Z - cz) / (self.source_radius + 1e-5))
        self.w -= (self.swirl_strength * dt) * mask * ((self.X - cx) / (self.source_radius + 1e-5))

    def advect_all(self, dt: float) -> None:
        u0 = self.u.copy()
        v0 = self.v.copy()
        w0 = self.w.copy()
        d0 = self.density.copy()
        t0 = self.temperature.copy()

        bx = self.X - dt * u0
        by = self.Y - dt * v0
        bz = self.Z - dt * w0

        self.u[:] = trilinear_sample(u0, bx, by, bz)
        self.v[:] = trilinear_sample(v0, bx, by, bz)
        self.w[:] = trilinear_sample(w0, bx, by, bz)
        self.density[:] = trilinear_sample(d0, bx, by, bz)
        self.temperature[:] = trilinear_sample(t0, bx, by, bz)

    def apply_buoyancy(self, dt: float) -> None:
        ambient = 0.0
        self.v += dt * (
            self.buoyancy * self.density + self.thermal_buoyancy * np.maximum(self.temperature - ambient, 0.0)
        )

    def apply_vorticity_confinement(self, dt: float) -> None:
        du_dy = 0.5 * (np.roll(self.u, -1, axis=1) - np.roll(self.u, 1, axis=1))
        du_dz = 0.5 * (np.roll(self.u, -1, axis=2) - np.roll(self.u, 1, axis=2))
        dv_dx = 0.5 * (np.roll(self.v, -1, axis=0) - np.roll(self.v, 1, axis=0))
        dv_dz = 0.5 * (np.roll(self.v, -1, axis=2) - np.roll(self.v, 1, axis=2))
        dw_dx = 0.5 * (np.roll(self.w, -1, axis=0) - np.roll(self.w, 1, axis=0))
        dw_dy = 0.5 * (np.roll(self.w, -1, axis=1) - np.roll(self.w, 1, axis=1))

        curl_x = dw_dy - dv_dz
        curl_y = du_dz - dw_dx
        curl_z = dv_dx - du_dy
        curl_mag = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2 + 1e-8).astype(np.float32)

        n_x = 0.5 * (np.roll(curl_mag, -1, axis=0) - np.roll(curl_mag, 1, axis=0))
        n_y = 0.5 * (np.roll(curl_mag, -1, axis=1) - np.roll(curl_mag, 1, axis=1))
        n_z = 0.5 * (np.roll(curl_mag, -1, axis=2) - np.roll(curl_mag, 1, axis=2))
        n_len = np.sqrt(n_x**2 + n_y**2 + n_z**2 + 1e-8).astype(np.float32)
        n_x /= n_len
        n_y /= n_len
        n_z /= n_len

        force_x = n_y * curl_z - n_z * curl_y
        force_y = n_z * curl_x - n_x * curl_z
        force_z = n_x * curl_y - n_y * curl_x

        self.u += dt * self.vorticity_confinement * force_x
        self.v += dt * self.vorticity_confinement * force_y
        self.w += dt * self.vorticity_confinement * force_z

    def project(self) -> None:
        self.divergence[:] = 0.5 * (
            np.roll(self.u, -1, axis=0)
            - np.roll(self.u, 1, axis=0)
            + np.roll(self.v, -1, axis=1)
            - np.roll(self.v, 1, axis=1)
            + np.roll(self.w, -1, axis=2)
            - np.roll(self.w, 1, axis=2)
        )
        set_boundaries(self.divergence, 0.0)

        self.pressure.fill(0.0)
        self.pressure_tmp.fill(0.0)
        for _ in range(self.pressure_iterations):
            self.pressure_tmp[:] = (
                np.roll(self.pressure, 1, axis=0)
                + np.roll(self.pressure, -1, axis=0)
                + np.roll(self.pressure, 1, axis=1)
                + np.roll(self.pressure, -1, axis=1)
                + np.roll(self.pressure, 1, axis=2)
                + np.roll(self.pressure, -1, axis=2)
                - self.divergence
            ) / 6.0
            set_boundaries(self.pressure_tmp, 0.0)
            self.pressure, self.pressure_tmp = self.pressure_tmp, self.pressure

        self.u -= 0.5 * (np.roll(self.pressure, -1, axis=0) - np.roll(self.pressure, 1, axis=0))
        self.v -= 0.5 * (np.roll(self.pressure, -1, axis=1) - np.roll(self.pressure, 1, axis=1))
        self.w -= 0.5 * (np.roll(self.pressure, -1, axis=2) - np.roll(self.pressure, 1, axis=2))

    def enforce_boundaries_and_decay(self) -> None:
        set_boundaries(self.u, 0.0)
        set_boundaries(self.v, 0.0)
        set_boundaries(self.w, 0.0)

        self.u *= self.velocity_decay
        self.v *= self.velocity_decay
        self.w *= self.velocity_decay
        self.density *= self.density_decay
        self.temperature *= self.temp_decay

        np.maximum(self.density, 0.0, out=self.density)
        np.maximum(self.temperature, 0.0, out=self.temperature)

    def step(self, dt: float, frame_idx: int) -> None:
        self.inject_source(dt, frame_idx)
        self.advect_all(dt)
        self.apply_buoyancy(dt)
        self.apply_vorticity_confinement(dt)
        self.project()
        self.enforce_boundaries_and_decay()


def render_volume(
    density: np.ndarray,
    temperature: np.ndarray,
    width: int,
    height: int,
    ray_steps: int,
    angle: float,
    extinction: float,
    shadow_strength: float,
) -> np.ndarray:
    ny = density.shape[1]

    # Cheap top-down shadowing from integrated optical depth.
    top_to_bottom = np.flip(density, axis=1)
    optical_depth = np.cumsum(top_to_bottom, axis=1, dtype=np.float32) / float(ny)
    light_volume = np.flip(np.exp(-shadow_strength * optical_depth), axis=1)

    xs = np.linspace(-0.54, 0.54, width, dtype=np.float32)
    ys = np.linspace(1.04, -0.06, height, dtype=np.float32)
    base_x, base_y = np.meshgrid(xs + 0.5, ys, indexing="xy")

    transmittance = np.ones((height, width), dtype=np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.float32)

    smoke_tint = np.array([0.84, 0.87, 0.91], dtype=np.float32)
    ember_tint = np.array([1.0, 0.46, 0.15], dtype=np.float32)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    near = -0.22
    far = 1.22
    step = (far - near) / float(ray_steps)

    for i in range(ray_steps):
        z = near + (i + 0.5) * step

        local_x = base_x - 0.5
        local_z = z - 0.5
        xr = local_x * cos_a + local_z * sin_a + 0.5
        zr = -local_x * sin_a + local_z * cos_a + 0.5

        inside = (xr >= 0.0) & (xr <= 1.0) & (base_y >= 0.0) & (base_y <= 1.0) & (zr >= 0.0) & (zr <= 1.0)
        if not np.any(inside):
            continue

        dens = sample_normalized(density, xr, base_y, zr)
        temp = sample_normalized(temperature, xr, base_y, zr)
        lit = sample_normalized(light_volume, xr, base_y, zr)

        dens = np.where(inside, dens, 0.0)
        temp = np.where(inside, temp, 0.0)
        lit = np.where(inside, lit, 0.0)

        alpha = 1.0 - np.exp(-dens * extinction * step)
        alpha = clamp01(alpha)

        warmth = clamp01(temp * 0.085)
        color = smoke_tint[None, None, :] * (0.35 + 0.65 * lit[..., None])
        color += ember_tint[None, None, :] * (0.24 * warmth[..., None])

        contrib = (transmittance * alpha)[..., None] * color
        rgb += contrib
        transmittance *= 1.0 - alpha

    bg_top = np.array([0.91, 0.93, 0.96], dtype=np.float32)
    bg_bottom = np.array([0.06, 0.07, 0.09], dtype=np.float32)
    gy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    bg = (bg_top[None, None, :] * (1.0 - gy[..., None])) + (bg_bottom[None, None, :] * gy[..., None])
    bg = np.broadcast_to(bg, (height, width, 3)).copy()

    # Soft vignette so the plume stays visually centered.
    gx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    gy2 = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    vignette = 1.0 - 0.22 * np.clip(gx**2 + gy2**2, 0.0, 1.0)
    bg *= vignette[..., None]

    image = rgb + transmittance[..., None] * bg
    image = np.clip(image, 0.0, 1.0) ** (1.0 / 2.2)
    return (image * 255.0 + 0.5).astype(np.uint8)


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


def resolve_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline point-source smoke simulation and render.")
    parser.add_argument("--out", type=Path, default=Path("out/latest"), help="Output directory.")
    parser.add_argument("--frames", type=int, default=240, help="Number of frames to render.")
    parser.add_argument("--fps", type=int, default=30, help="Output framerate.")
    parser.add_argument("--grid", type=int, default=72, help="Grid resolution (N^3).")
    parser.add_argument("--width", type=int, default=540, help="Frame width in pixels.")
    parser.add_argument("--height", type=int, default=540, help="Frame height in pixels.")
    parser.add_argument("--ray-steps", type=int, default=104, help="Raymarch steps per frame.")
    parser.add_argument("--substeps", type=int, default=2, help="Simulation steps per frame.")
    parser.add_argument("--dt", type=float, default=0.115, help="Simulation delta time per frame.")
    parser.add_argument("--pressure-iters", type=int, default=30, help="Jacobi pressure iterations.")
    parser.add_argument("--density-decay", type=float, default=0.992, help="Per-step density decay.")
    parser.add_argument("--temp-decay", type=float, default=0.986, help="Per-step temperature decay.")
    parser.add_argument("--velocity-decay", type=float, default=0.995, help="Per-step velocity decay.")
    parser.add_argument("--buoyancy", type=float, default=1.1, help="Density-based buoyancy.")
    parser.add_argument("--thermal-buoyancy", type=float, default=0.9, help="Temperature-based buoyancy.")
    parser.add_argument("--vorticity", type=float, default=0.65, help="Vorticity confinement amount.")
    parser.add_argument("--extinction", type=float, default=3.15, help="Volume extinction strength.")
    parser.add_argument("--shadow", type=float, default=5.2, help="Top-light shadow strength.")
    parser.add_argument("--orbit", type=float, default=0.14, help="Camera orbit amount over the full clip.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--mp4", action="store_true", help="Encode an mp4 after rendering frames.")
    parser.add_argument("--gif", action="store_true", help="Encode an animated gif after rendering frames.")
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
    parser.add_argument("--encode-only", action="store_true", help="Skip simulation and encode using existing frames.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frames and output media.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite and not args.encode_only:
        for frame in frames_dir.glob("*.png"):
            frame.unlink()

    if not args.encode_only:
        sim = SmokeSim(
            grid=args.grid,
            pressure_iterations=args.pressure_iters,
            density_decay=args.density_decay,
            temp_decay=args.temp_decay,
            velocity_decay=args.velocity_decay,
            buoyancy=args.buoyancy,
            thermal_buoyancy=args.thermal_buoyancy,
            vorticity_confinement=args.vorticity,
            seed=args.seed,
        )

        dt_sub = args.dt / float(max(1, args.substeps))
        for frame_idx in range(args.frames):
            for _ in range(args.substeps):
                sim.step(dt_sub, frame_idx)

            angle = -args.orbit * 0.5 + (args.orbit * frame_idx / max(args.frames - 1, 1))
            image = render_volume(
                density=sim.density,
                temperature=sim.temperature,
                width=args.width,
                height=args.height,
                ray_steps=args.ray_steps,
                angle=angle,
                extinction=args.extinction,
                shadow_strength=args.shadow,
            )

            frame_path = frames_dir / f"{frame_idx:05d}.png"
            Image.fromarray(image, mode="RGB").save(frame_path)

            if frame_idx % 10 == 0 or frame_idx == args.frames - 1:
                print(f"[sim] frame {frame_idx + 1}/{args.frames}")
    else:
        frame_count = len(list(frames_dir.glob("*.png")))
        if frame_count == 0:
            raise RuntimeError("No frames found for --encode-only. Render frames first.")
        print(f"[encode] using {frame_count} existing frame(s)")

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
