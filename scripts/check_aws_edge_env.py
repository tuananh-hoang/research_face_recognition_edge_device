"""Collect environment details for AWS edge-constrained simulation runs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "" or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def detect_container_memory_limit() -> dict:
    candidates = [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    for path in candidates:
        raw = _read_text(path)
        value = _parse_int(raw)
        if value is None:
            continue
        # Very large values usually mean "no practical cgroup limit".
        if value > 1 << 60:
            continue
        return {
            "bytes": value,
            "mb": round(value / 1024 / 1024, 2),
            "source": path,
        }
    return {"bytes": None, "mb": None, "source": None}


def detect_container_cpu_quota() -> dict:
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) >= 2 and parts[0] != "max":
            quota = _parse_int(parts[0])
            period = _parse_int(parts[1])
            if quota and period:
                return {
                    "quota": quota,
                    "period": period,
                    "cpus": quota / period,
                    "source": "/sys/fs/cgroup/cpu.max",
                }

    quota = _parse_int(_read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period = _parse_int(_read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if quota and quota > 0 and period:
        return {
            "quota": quota,
            "period": period,
            "cpus": quota / period,
            "source": "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        }
    return {"quota": None, "period": None, "cpus": None, "source": None}


def running_inside_docker() -> bool:
    if Path("/.dockerenv").exists():
        return True
    cgroup = _read_text("/proc/1/cgroup") or ""
    return "docker" in cgroup or "containerd" in cgroup


def import_status(module_name: str) -> dict:
    try:
        module = importlib.import_module(module_name)
        return {
            "importable": True,
            "version": getattr(module, "__version__", None),
            "error": "",
        }
    except Exception as exc:
        return {
            "importable": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def available_ram() -> dict:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "available_mb": round(mem.available / 1024 / 1024, 2),
            "total_mb": round(mem.total / 1024 / 1024, 2),
        }
    except Exception:
        return {"available_mb": None, "total_mb": None}


def collect_environment() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.replace("\n", " "),
        "cpu_count": os.cpu_count(),
        "available_ram": available_ram(),
        "container_memory_limit": detect_container_memory_limit(),
        "container_cpu_quota": detect_container_cpu_quota(),
        "inside_docker": running_inside_docker(),
        "modules": {
            "insightface": import_status("insightface"),
            "onnxruntime": import_status("onnxruntime"),
            "cv2": import_status("cv2"),
        },
    }


def print_summary(info: dict) -> None:
    print("AWS edge-constrained simulation environment")
    print("-" * 60)
    print(f"platform             : {info['platform']}")
    print(f"machine              : {info['machine']}")
    print(f"python               : {info['python_version']}")
    print(f"cpu_count            : {info['cpu_count']}")
    print(f"available_ram_mb     : {info['available_ram'].get('available_mb')}")
    print(f"total_ram_mb         : {info['available_ram'].get('total_mb')}")
    print(f"container_memory_mb  : {info['container_memory_limit'].get('mb')}")
    print(f"container_cpu_quota  : {info['container_cpu_quota'].get('cpus')}")
    print(f"inside_docker        : {info['inside_docker']}")
    for name, status in info["modules"].items():
        suffix = f" ({status['version']})" if status.get("version") else ""
        print(f"{name:<21}: {status['importable']}{suffix}")
        if status.get("error"):
            print(f"  error              : {status['error']}")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "outputs" / "aws_edge_env.json"))
    args = parser.parse_args()

    info = collect_environment()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print_summary(info)
    print(f"Saved environment JSON: {output_path}")


if __name__ == "__main__":
    main()
