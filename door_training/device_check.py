"""Report a machine's capabilities for YOLOv8 training.

Run on the target (GPU) machine — no dependencies required (uses stdlib +
nvidia-smi if present; torch/ultralytics are reported only if already installed):

    python device_check.py

Copy the printed report back here so we can pick batch size / estimate runtime.
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sys


def line(label, value):
    print(f"  {label:<22} {value}")


def total_ram_gb():
    try:
        import psutil  # optional
        return psutil.virtual_memory().total / 1024**3
    except Exception:
        pass
    try:
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1024**3
        else:  # POSIX
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except Exception:
        return None


def nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            stderr=subprocess.STDOUT, text=True, timeout=15)
        return [l.strip() for l in out.strip().splitlines() if l.strip()]
    except Exception as e:
        return [f"nvidia-smi not available ({e.__class__.__name__})"]


def main():
    print("=" * 60)
    print("DEVICE CAPABILITY REPORT (for YOLOv8 training)")
    print("=" * 60)

    print("\n[System]")
    line("OS", f"{platform.system()} {platform.release()} ({platform.machine()})")
    line("Python", sys.version.split()[0])
    line("CPU cores (logical)", os.cpu_count())
    ram = total_ram_gb()
    line("RAM", f"{ram:.1f} GB" if ram else "unknown")
    try:
        free = shutil.disk_usage(os.getcwd()).free / 1024**3
        line("Free disk (cwd)", f"{free:.1f} GB")
    except Exception:
        pass

    print("\n[GPU — nvidia-smi]")
    for g in nvidia_smi():
        print(f"  {g}")

    print("\n[PyTorch]")
    try:
        import torch
        line("torch", torch.__version__)
        line("CUDA build", torch.version.cuda)
        line("CUDA available", torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            line("GPU (torch)", p.name)
            line("VRAM", f"{p.total_memory/1024**3:.2f} GB")
            line("Compute capability", f"{p.major}.{p.minor}")
            line("Multiprocessors", p.multi_processor_count)
    except ImportError:
        print("  torch NOT installed  (install: pip install ultralytics)")

    print("\n[Ultralytics]")
    try:
        import ultralytics
        line("ultralytics", ultralytics.__version__)
    except ImportError:
        print("  ultralytics NOT installed  (pip install ultralytics)")

    print("\n" + "=" * 60)
    print("Share this whole report back to decide batch size & runtime.")
    print("=" * 60)


if __name__ == "__main__":
    main()
