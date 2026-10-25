"""Observed process memory and explicit resource limits for CPU feature work."""

import ctypes
import os
import sys
import time


def process_rss_bytes():
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (key, ctypes.c_size_t) for key in
                ("peak", "rss", "quota_peak", "quota", "page_peak", "page", "swap", "swap_peak")
            ]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters),
                                               ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters),
                                         counters.cb):
            raise OSError("Cannot measure process memory")
        return int(counters.rss)
    import resource
    # ru_maxrss is a peak rather than current RSS, a conservative limit check.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def enforce_resources(config, started):
    rss = process_rss_bytes()
    if rss > config["max_rss_bytes"]:
        raise MemoryError(f"Observed RSS {rss} exceeds max_rss_bytes={config['max_rss_bytes']}")
    if time.monotonic() - started > config["max_train_hours"] * 3600:
        raise TimeoutError("Elapsed runtime exceeded max_train_hours; reduce work or revise budget")
    return rss


def ensure_allocation(config, started, additional_bytes, label):
    """Reject a projected peak before allocating resident arrays or intermediates."""
    rss = enforce_resources(config, started)
    if rss + additional_bytes > config["max_rss_bytes"]:
        raise MemoryError(
            f"Planned {label} allocation/working set {additional_bytes} plus RSS {rss} "
            f"exceeds max_rss_bytes={config['max_rss_bytes']}"
        )
    return rss
