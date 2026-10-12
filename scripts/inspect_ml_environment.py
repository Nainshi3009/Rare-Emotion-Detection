from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys


def _package_version(package: str, module: str) -> str:
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        try:
            installed = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            installed = False
        return "installed, version unavailable" if installed else "not installed"
    return version


def _memory_info() -> dict[str, float | None]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "total_gib": round(memory.total / (1024**3), 2),
            "available_gib": round(memory.available / (1024**3), 2),
        }
    except ImportError:
        return {"total_gib": None, "available_gib": None}


def _graphics_adapters() -> list[str]:
    if platform.system() != "Windows":
        return []
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _processor_name() -> str:
    if platform.system() != "Windows":
        return platform.processor() or "unknown"
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return platform.processor() or "unknown"
    return result.stdout.strip() or platform.processor() or "unknown"


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.architecture()[0]})")
    print(f"Platform: {platform.platform()}")
    print(f"CPU: {_processor_name()}")
    print(f"Logical CPU count: {os.cpu_count() or 'unknown'}")
    adapters = _graphics_adapters()
    print(f"System graphics adapter(s): {', '.join(adapters) if adapters else 'unavailable'}")
    memory = _memory_info()
    print(f"RAM total: {memory['total_gib'] if memory['total_gib'] is not None else 'unavailable'} GiB")
    print(
        "RAM available: "
        f"{memory['available_gib'] if memory['available_gib'] is not None else 'unavailable'} GiB"
    )

    torch_version = _package_version("torch", "torch")
    transformers_version = _package_version("transformers", "transformers")
    accelerate_version = _package_version("accelerate", "accelerate")
    print(f"PyTorch: {torch_version}")
    print(f"Transformers: {transformers_version}")
    print(f"Accelerate: {accelerate_version}")

    try:
        import torch
    except ImportError:
        print("CUDA available: no (PyTorch not installed)")
        print("CUDA version: unavailable")
        print("CUDA GPU name: unavailable")
        return 0

    cuda_available = bool(torch.cuda.is_available())
    cuda_version = torch.version.cuda
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    print(f"CUDA available: {'yes' if cuda_available else 'no'}")
    print(f"CUDA version: {cuda_version if cuda_version else 'unavailable (CPU-only PyTorch build)'}")
    print(f"CUDA GPU name: {gpu_name if gpu_name else 'unavailable'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
