"""Environment-driven configuration for the LeRobot profiler plugin."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.lower() == "true"


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None:
        return default
    return int(value)


@dataclass(frozen=True)
class ProfilerConfig:
    """Profiler configuration parsed from environment variables."""

    enabled: bool = False
    backend: str = "torch"
    wait: int = 5
    warmup: int = 2
    active: int = 8
    repeat: int = -1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True
    with_flops: bool = True
    output_subdir: str = "profiler_logs"
    nsys_command: Optional[str] = None
    nsight_enable_cuda_profiler: bool = True
    nsight_emit_nvtx: bool = True
    nsight_step_prefix: str = "lerobot_step"

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "ProfilerConfig":
        source = os.environ if env is None else env
        return cls(
            enabled=_env_bool(source, "LEROBOT_PROFILE", False),
            backend=source.get("LEROBOT_PROFILE_BACKEND", "torch").lower(),
            wait=_env_int(source, "LEROBOT_PROFILE_WAIT", 5),
            warmup=_env_int(source, "LEROBOT_PROFILE_WARMUP", 2),
            active=_env_int(source, "LEROBOT_PROFILE_ACTIVE", 8),
            repeat=_env_int(source, "LEROBOT_PROFILE_REPEAT", -1),
            record_shapes=_env_bool(source, "LEROBOT_PROFILE_SHAPES", True),
            profile_memory=_env_bool(source, "LEROBOT_PROFILE_MEMORY", True),
            with_stack=_env_bool(source, "LEROBOT_PROFILE_STACK", True),
            with_flops=_env_bool(source, "LEROBOT_PROFILE_FLOPS", True),
            output_subdir=source.get("LEROBOT_PROFILE_OUTPUT_SUBDIR", "profiler_logs"),
            nsys_command=source.get("LEROBOT_PROFILE_NSYS_CMD"),
            nsight_enable_cuda_profiler=_env_bool(
                source,
                "LEROBOT_PROFILE_NSIGHT_CUDA_PROFILER",
                True,
            ),
            nsight_emit_nvtx=_env_bool(source, "LEROBOT_PROFILE_NSIGHT_NVTX", True),
            nsight_step_prefix=source.get("LEROBOT_PROFILE_NSIGHT_STEP_PREFIX", "lerobot_step"),
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain dict for backward-compatible inspection."""
        return asdict(self)
