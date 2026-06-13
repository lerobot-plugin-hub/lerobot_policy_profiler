"""Profiler backend abstractions.

The plugin starts with PyTorch profiler support, while keeping the integration
surface small enough to add Nsight Systems or other profilers later.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Protocol

from .config import ProfilerConfig


class ProfileSession(Protocol):
    """Runtime session created by a profiler backend."""

    profiler: Any

    def __enter__(self) -> "ProfileSession":
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        ...

    def step(self) -> None:
        ...

    def record_function(self, name: str) -> Generator[None, None, None]:
        ...


class ProfilerBackend(Protocol):
    """Factory contract for profiler backends."""

    name: str

    def create_session(
        self,
        cfg: Any,
        accelerator: Any,
        runtime: "RuntimeProfileConfig",
    ) -> ProfileSession:
        ...


class RuntimeProfileConfig:
    """Resolved runtime information passed to backend implementations."""

    def __init__(
        self,
        config: ProfilerConfig,
        output_dir: Path,
        total_steps: int,
        device: Any,
        logger: Any,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.total_steps = total_steps
        self.device = device
        self.logger = logger


class TorchProfileSession:
    """Adapter around ``torch.profiler.profile``."""

    def __init__(self, profiler: Any) -> None:
        self.profiler = profiler

    def __enter__(self) -> "TorchProfileSession":
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.profiler.__exit__(exc_type, exc_val, exc_tb)
        return False

    def step(self) -> None:
        self.profiler.step()

    @contextmanager
    def record_function(self, name: str) -> Generator[None, None, None]:
        from torch.profiler import record_function as torch_record_function

        with torch_record_function(name):
            yield


class TorchProfilerBackend:
    """PyTorch profiler backend."""

    name = "torch"

    def create_session(
        self,
        cfg: Any,
        accelerator: Any,
        runtime: RuntimeProfileConfig,
    ) -> TorchProfileSession:
        del cfg, accelerator

        import torch
        from torch.profiler import ProfilerActivity, profile, schedule

        config = runtime.config
        repeat = config.repeat
        if repeat == -1:
            steps_per_cycle = config.wait + config.warmup + config.active
            repeat = max(1, runtime.total_steps // max(steps_per_cycle, 1))

        runtime.output_dir.mkdir(parents=True, exist_ok=True)

        runtime.logger.info("输出目录: %s", runtime.output_dir)
        runtime.logger.info(
            "Backend: torch, schedule: wait=%s, warmup=%s, active=%s, repeat=%s",
            config.wait,
            config.warmup,
            config.active,
            repeat,
        )
        runtime.logger.info("总步数: %s, 设备: %s", runtime.total_steps, runtime.device)
        runtime.logger.info("TensorBoard: tensorboard --logdir=%s", runtime.output_dir)

        activities = [ProfilerActivity.CPU]
        if runtime.device is not None and getattr(runtime.device, "type", "") == "cuda":
            activities.append(ProfilerActivity.CUDA)

        profiler = profile(
            activities=activities,
            schedule=schedule(
                wait=config.wait,
                warmup=config.warmup,
                active=config.active,
                repeat=repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(runtime.output_dir)),
            record_shapes=config.record_shapes,
            profile_memory=config.profile_memory,
            with_stack=config.with_stack,
            with_flops=config.with_flops,
        )
        return TorchProfileSession(profiler)


class NsightProfileSession:
    """Nsight Systems session using CUDA Profiler API and NVTX ranges."""

    profiler = None

    def __init__(self, runtime: RuntimeProfileConfig) -> None:
        self._runtime = runtime
        self._current_step = 0
        self._warned_cuda_api = False
        self._warned_nvtx = False

    def __enter__(self) -> "NsightProfileSession":
        self._runtime.output_dir.mkdir(parents=True, exist_ok=True)
        self._runtime.logger.info(
            "Backend: nsight 已启用，输出目录: %s",
            self._runtime.output_dir,
        )
        self._runtime.logger.info(
            "建议使用: %s",
            self._runtime.config.nsys_command or self.default_nsys_command(),
        )
        if self._runtime.config.nsight_enable_cuda_profiler:
            self._call_cuda_profiler("cudaProfilerStart")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._runtime.config.nsight_enable_cuda_profiler:
            self._call_cuda_profiler("cudaProfilerStop")
        self._runtime.logger.info("Backend: nsight 会话结束")
        return False

    def step(self) -> None:
        if not self._runtime.config.nsight_emit_nvtx:
            return

        name = f"{self._runtime.config.nsight_step_prefix}_{self._current_step}"
        self._current_step += 1
        nvtx = self._get_nvtx()
        if nvtx is None:
            return

        try:
            nvtx.range_push(name)
            nvtx.range_pop()
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime state.
            self._warn_nvtx_once(exc)

    @contextmanager
    def record_function(self, name: str) -> Generator[None, None, None]:
        if not self._runtime.config.nsight_emit_nvtx:
            yield
            return

        nvtx = self._get_nvtx()
        if nvtx is None:
            yield
            return

        pushed = False
        try:
            nvtx.range_push(name)
            pushed = True
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime state.
            self._warn_nvtx_once(exc)

        try:
            yield
        finally:
            if pushed:
                try:
                    nvtx.range_pop()
                except Exception as exc:  # pragma: no cover - depends on CUDA runtime state.
                    self._warn_nvtx_once(exc)

    def default_nsys_command(self) -> str:
        """Return the recommended command for CUDA Profiler API capture."""
        return (
            "nsys profile --capture-range=cudaProfilerApi "
            "--capture-range-end=stop -o lerobot_profile <your lerobot-train command>"
        )

    def _call_cuda_profiler(self, method_name: str) -> None:
        try:
            import torch

            cuda = getattr(torch, "cuda", None)
            if cuda is None or not cuda.is_available():
                return

            cudart = cuda.cudart()
            getattr(cudart, method_name)()
        except Exception as exc:  # pragma: no cover - CUDA may be unavailable in CI.
            if not self._warned_cuda_api:
                self._runtime.logger.warning("Nsight CUDA Profiler API 不可用: %s", exc)
                self._warned_cuda_api = True

    def _get_nvtx(self) -> Any:
        try:
            import torch

            cuda = getattr(torch, "cuda", None)
            if cuda is None or not cuda.is_available():
                return None

            return getattr(cuda, "nvtx", None)
        except Exception as exc:  # pragma: no cover - CUDA may be unavailable in CI.
            self._warn_nvtx_once(exc)
            return None

    def _warn_nvtx_once(self, exc: Exception) -> None:
        if not self._warned_nvtx:
            self._runtime.logger.warning("Nsight NVTX 标记不可用: %s", exc)
            self._warned_nvtx = True


class NsightProfilerBackend:
    """Nsight Systems profiling backend."""

    name = "nsight"

    def create_session(
        self,
        cfg: Any,
        accelerator: Any,
        runtime: RuntimeProfileConfig,
    ) -> NsightProfileSession:
        del cfg, accelerator
        return NsightProfileSession(runtime)


NsysProfileSession = NsightProfileSession
NsysProfilerBackend = NsightProfilerBackend


def create_backend(config: ProfilerConfig) -> ProfilerBackend:
    """Create the selected profiler backend."""
    if config.backend == "torch":
        return TorchProfilerBackend()
    if config.backend in {"nsight", "nsys"}:
        return NsightProfilerBackend()
    raise ValueError(f"Unsupported profiler backend: {config.backend}")


def resolve_torch_device(accelerator: Any = None) -> Any:
    """Resolve device lazily so importing the package does not require torch."""
    if accelerator is not None and hasattr(accelerator, "device"):
        return accelerator.device

    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
