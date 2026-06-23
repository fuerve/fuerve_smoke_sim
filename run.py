#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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


def run_cmd_capture(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)


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


def parse_requirements(requirements_path: Path) -> list[str]:
    packages: list[str] = []
    pattern = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        line = line.split(";", 1)[0].strip()
        match = pattern.match(line)
        if match:
            packages.append(match.group(1))
    return packages


def doctor() -> bool:
    checks: list[tuple[str, str, str]] = []

    def add(status: str, name: str, detail: str) -> None:
        checks.append((status, name, detail))

    py_ver = sys.version_info
    if py_ver >= (3, 10):
        add("PASS", "Python (host)", f"{sys.executable} ({py_ver.major}.{py_ver.minor}.{py_ver.micro})")
    else:
        add(
            "FAIL",
            "Python (host)",
            f"{sys.executable} ({py_ver.major}.{py_ver.minor}.{py_ver.micro}); Python 3.10+ is recommended.",
        )

    if VENV_PYTHON.exists():
        cp = run_cmd_capture([str(VENV_PYTHON), "--version"])
        if cp.returncode == 0:
            add("PASS", "Virtualenv", f"found at {VENV_PYTHON} ({(cp.stdout or cp.stderr).strip()})")
        else:
            detail = (cp.stderr or cp.stdout).strip() or "unable to query venv python version"
            add("FAIL", "Virtualenv", detail)
    else:
        add("FAIL", "Virtualenv", f"missing at {VENV_PYTHON}; run scripts/setup.sh or scripts/setup.cmd")

    req_packages = parse_requirements(REQ_FILE)
    if VENV_PYTHON.exists():
        missing_packages: list[str] = []
        for pkg in req_packages:
            cp = run_cmd_capture([str(VENV_PYTHON), "-m", "pip", "show", pkg])
            if cp.returncode != 0:
                missing_packages.append(pkg)
        if missing_packages:
            add("FAIL", "Dependencies", f"missing package(s): {', '.join(missing_packages)}")
        else:
            add("PASS", "Dependencies", f"all {len(req_packages)} requirement package(s) detected in .venv")
    else:
        add("WARN", "Dependencies", "skipped (virtualenv missing)")

    if DEPS_STAMP.exists():
        req_mtime = REQ_FILE.stat().st_mtime
        stamp_mtime = DEPS_STAMP.stat().st_mtime
        if stamp_mtime >= req_mtime:
            add("PASS", "Dependency stamp", f"{DEPS_STAMP} is up to date")
        else:
            add("WARN", "Dependency stamp", f"{DEPS_STAMP} is older than {REQ_FILE.name}; run `python3 run.py deps`")
    else:
        add("WARN", "Dependency stamp", f"{DEPS_STAMP} not found")

    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        add("PASS", "ffmpeg", f"system ffmpeg found at {sys_ffmpeg}")
    elif VENV_PYTHON.exists():
        cp = run_cmd_capture(
            [str(VENV_PYTHON), "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"]
        )
        ffmpeg_path = cp.stdout.strip() if cp.returncode == 0 else ""
        if ffmpeg_path and Path(ffmpeg_path).exists():
            add("PASS", "ffmpeg", f"imageio-ffmpeg fallback found at {ffmpeg_path}")
        else:
            detail = (cp.stderr or cp.stdout).strip() or "ffmpeg not found via imageio-ffmpeg fallback"
            add("FAIL", "ffmpeg", detail)
    else:
        add("FAIL", "ffmpeg", "system ffmpeg not found and virtualenv fallback unavailable")

    taichi_probe = None
    if VENV_PYTHON.exists():
        probe_code = r"""
import json
result = {"import_ok": False, "import_error": "", "arches": {}}
try:
    import taichi as ti
    result["import_ok"] = True
except Exception as exc:
    result["import_error"] = str(exc)
if result["import_ok"]:
    for name in ["cpu", "vulkan", "cuda", "metal"]:
        if not hasattr(ti, name):
            result["arches"][name] = "unsupported"
            continue
        try:
            ti.init(arch=getattr(ti, name), random_seed=1, offline_cache=False)
            result["arches"][name] = "ok"
        except Exception as exc:
            result["arches"][name] = f"error:{exc}"
        finally:
            try:
                ti.reset()
            except Exception:
                pass
print(json.dumps(result))
"""
        cp = run_cmd_capture([str(VENV_PYTHON), "-c", probe_code])
        if cp.returncode == 0:
            candidates = [line.strip() for line in cp.stdout.splitlines() if line.strip()]
            for line in reversed(candidates):
                if not (line.startswith("{") and line.endswith("}")):
                    continue
                try:
                    taichi_probe = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

    if taichi_probe is None:
        add("WARN", "Taichi", "probe unavailable (taichi emitted no parseable JSON status)")
    elif not taichi_probe.get("import_ok", False):
        add("FAIL", "Taichi import", taichi_probe.get("import_error", "unknown import failure"))
    else:
        arches: dict[str, str] = taichi_probe.get("arches", {})
        cpu_state = str(arches.get("cpu", "unknown"))
        if cpu_state == "ok":
            add("PASS", "Taichi CPU backend", "cpu backend initialized")
        else:
            add("FAIL", "Taichi CPU backend", cpu_state)

        gpu_states = {k: str(v) for k, v in arches.items() if k in {"vulkan", "cuda", "metal"}}
        gpu_ok = sorted([k for k, v in gpu_states.items() if v == "ok"])
        if gpu_ok:
            add("PASS", "Taichi GPU backends", f"available: {', '.join(gpu_ok)}")
        else:
            detail = ", ".join([f"{k}={v}" for k, v in sorted(gpu_states.items())]) or "no GPU backends checked"
            add("WARN", "Taichi GPU backends", detail)

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    print("== Doctor Report ==")
    for status, name, detail in checks:
        counts[status] += 1
        print(f"[{status}] {name}: {detail}")

    print(f"Summary: {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")
    return counts["FAIL"] == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-platform task runner for fuerve_smoke_sim.")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["venv", "deps", "frames", "mp4", "gif", "all", "update", "clean", "doctor"],
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
    if args.target == "doctor":
        ok = doctor()
        raise SystemExit(0 if ok else 1)

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
