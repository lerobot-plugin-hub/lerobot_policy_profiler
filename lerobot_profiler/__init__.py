"""Public entrypoint for the LeRobot zero-modification profiler plugin."""

from .config import ProfilerConfig
from .core import LeRobotProfiler, auto_register, get_profiler, record_function


auto_register()


__all__ = [
    "LeRobotProfiler",
    "ProfilerConfig",
    "get_profiler",
    "record_function",
]
