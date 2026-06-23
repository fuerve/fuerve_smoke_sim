#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQ_FILE = ROOT / "requirements.txt"
DEPS_STAMP = VENV_DIR / ".deps-stamp"

DEFAULTS = {
    "arch": "auto",
    "out": "out/latest",
    "frames": 240,
    "fps": 30,
    "grid": 72,
    "width": 540,
    "height": 540,
    "ray_steps": 104,
    "substeps": 2,
}


def shell_join(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return " ".join(shlex.quote(part) for part in cmd)


def run_cmd(cmd: list[str]) -> None:
    print(f"+ {shell_join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_extra_args(extra_args: str) -> list[str]:
    if not extra_args.strip():
        return []
    return shlex.split(extra_args, posix=(os.name != "nt"))


def ensure_venv(python_exe: str | None) -> None:
    if VENV_PYTHON.exists():
        return
    base_python = python_exe or sys.executable
    run_cmd([base_python, "-m", "venv", str(VENV_DIR)])
    run_cmd([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])


def ensure_deps() -> None:
    req_mtime = REQ_FILE.stat().st_mtime
    stamp_mtime = DEPS_STAMP.stat().st_mtime if DEPS_STAMP.exists() else -1.0
    if stamp_mtime >= req_mtime:
        return
    run_cmd([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQ_FILE)])
    DEPS_STAMP.touch()


def build_engine_args(args: argparse.Namespace, use_config: bool) -> list[str]:
    if args.engine != "taichi":
        return []
    if use_config:
        if args.arch is None:
            return []
        return ["--arch", args.arch]
    return ["--arch", args.arch or DEFAULTS["arch"]]


def build_render_base_args(args: argparse.Namespace, use_config: bool, encode_only: bool) -> list[str]:
    cmd: list[str] = []
    if use_config:
        cmd += ["--config", str(args.config)]

    mapping = [("out", "out"), ("fps", "fps")]
    if not encode_only:
        mapping.extend(
            [
                ("frames", "frames"),
                ("grid", "grid"),
                ("width", "width"),
                ("height", "height"),
                ("ray_steps", "ray-steps"),
                ("substeps", "substeps"),
            ]
        )

    for attr, option_name in mapping:
        value = getattr(args, attr)
        if use_config:
            if value is not None:
                cmd += [f"--{option_name}", str(value)]
        else:
            fallback = DEFAULTS[attr]
            cmd += [f"--{option_name}", str(value if value is not None else fallback)]
    return cmd


def invoke_renderer(args: argparse.Namespace, encode_only: bool, encode_mp4: bool, encode_gif: bool) -> None:
    if args.config is not None and args.engine != "taichi":
        raise SystemExit("--config is only supported by the taichi engine.")

    use_config = args.config is not None and args.engine == "taichi"
    runner = "main_taichi.py" if args.engine == "taichi" else "main.py"

    cmd = [str(VENV_PYTHON), str(ROOT / runner)]
    cmd += build_engine_args(args, use_config)
    cmd += build_render_base_args(args, use_config, encode_only)
    if encode_only:
        cmd.append("--encode-only")
    if encode_mp4:
        cmd.append("--mp4")
    if encode_gif:
        cmd.append("--gif")
    cmd.append("--overwrite")
    cmd += parse_extra_args(args.extra_args)
    run_cmd(cmd)


def clean() -> None:
    for target in (ROOT / "out", VENV_DIR):
        if target.exists():
            print(f"- removing {target}")
            shutil.rmtree(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-platform task runner for fuerve_smoke_sim.")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["venv", "deps", "frames", "mp4", "gif", "all", "update", "clean"],
        help="Task to run (make-like targets).",
    )
    parser.add_argument("--python", default=None, help="Python executable used to create .venv if missing.")
    parser.add_argument("--engine", choices=["taichi", "numpy"], default="taichi")
    parser.add_argument("--arch", choices=["auto", "cpu", "cuda", "vulkan", "metal"], default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--grid", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--ray-steps", dest="ray_steps", type=int, default=None)
    parser.add_argument("--substeps", type=int, default=None)
    parser.add_argument(
        "--extra-args",
        default="",
        help="Additional args forwarded to main.py/main_taichi.py, for example \"--gif-profiles discord facebook\".",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target == "clean":
        clean()
        return

    ensure_venv(args.python)
    if args.target == "venv":
        return

    ensure_deps()
    if args.target == "deps":
        return

    if args.target == "frames":
        invoke_renderer(args, encode_only=False, encode_mp4=False, encode_gif=False)
        return

    if args.target == "mp4":
        invoke_renderer(args, encode_only=False, encode_mp4=False, encode_gif=False)
        invoke_renderer(args, encode_only=True, encode_mp4=True, encode_gif=False)
        return

    if args.target == "gif":
        invoke_renderer(args, encode_only=False, encode_mp4=False, encode_gif=False)
        invoke_renderer(args, encode_only=True, encode_mp4=False, encode_gif=True)
        return

    if args.target in {"all", "update"}:
        invoke_renderer(args, encode_only=False, encode_mp4=True, encode_gif=True)
        return

    raise SystemExit(f"Unsupported target: {args.target}")


if __name__ == "__main__":
    main()
