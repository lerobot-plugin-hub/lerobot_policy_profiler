"""Core monkey-patching and lifecycle management for the profiler plugin."""

from __future__ import annotations

import functools
import inspect
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Generator, Optional

from .backends import (
    ProfileSession,
    ProfilerBackend,
    RuntimeProfileConfig,
    create_backend,
    resolve_torch_device,
)
from .config import ProfilerConfig


class LeRobotProfiler:
    """
    LeRobot zero-modification profiler manager.

    The manager owns monkey patches and delegates profiler implementation details
    to pluggable backends such as PyTorch profiler or future Nsight Systems.
    """

    _instance: Optional["LeRobotProfiler"] = None
    _patched: bool = False

    def __new__(cls) -> "LeRobotProfiler":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "LeRobotProfiler":
        """获取全局单例。"""
        return cls._instance or cls()

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._config = ProfilerConfig.from_env()
        self.enabled = self._config.enabled
        self._backend: Optional[ProfilerBackend] = None
        self._session: Optional[ProfileSession] = None
        self.profiler: Any = None
        self._original_train: Optional[Callable[..., Any]] = None
        self._original_update_policy: Optional[Callable[..., Any]] = None
        self._pending_cfg: Any = None
        self._pending_accelerator: Any = None
        self._session_started = False
        self._current_step = 0
        self._total_steps = 0
        self._logger = logging.getLogger("lerobot_policy_profiler")

        if self.enabled:
            try:
                self._backend = create_backend(self._config)
                self._apply_monkey_patches()
                self._logger.info(
                    "LeRobot Policy Profiler 已初始化（%s backend）",
                    self._backend.name,
                )
            except Exception as exc:  # pragma: no cover - defensive guard for env mistakes.
                self._logger.error("Profiler backend 初始化失败: %s", exc)
                self.enabled = False

        self._initialized = True

    @property
    def config(self) -> Dict[str, Any]:
        """从环境变量读取的配置，返回 dict 便于调试和兼容旧调用方。"""
        return self._config.as_dict()

    def _apply_monkey_patches(self) -> None:
        """应用所有 Monkey Patches。"""
        if self._patched:
            return

        try:
            from lerobot.scripts import lerobot_train

            self._original_train = lerobot_train.train
            self._original_update_policy = getattr(lerobot_train, "update_policy", None)

            lerobot_train.train = self._create_patched_train(self._original_train)

            if self._original_update_policy:
                lerobot_train.update_policy = self._create_patched_update_policy(
                    self._original_update_policy
                )

            self._patched = True
            self._logger.info("已应用 Monkey Patches:")
            self._logger.info("  - lerobot_train.train")
            if self._original_update_policy:
                self._logger.info("  - lerobot_train.update_policy")
        except ImportError as exc:
            self._logger.warning("无法导入 lerobot_train: %s", exc)
            self.enabled = False
        except Exception as exc:  # pragma: no cover - defensive guard for host app compatibility.
            self._logger.error("应用 Monkey Patch 失败: %s", exc)
            self.enabled = False

    def _create_patched_train(self, original_train: Callable[..., Any]) -> Callable[..., Any]:
        """创建 patch 后的 train 函数。"""

        @functools.wraps(original_train)
        def patched_train(*args: Any, **kwargs: Any) -> Any:
            if not self.enabled:
                return original_train(*args, **kwargs)

            self._logger.info("=" * 70)
            self._logger.info("LeRobot Policy Profiler 开始工作（零修改模式）")
            self._logger.info("=" * 70)

            try:
                cfg, accelerator = self._extract_train_args(args, kwargs, original_train)

                if cfg is None:
                    cfg = self._extract_cli_cfg()
                    self._logger.info(
                        "未从 train 参数获取 cfg，改用 CLI fallback: output_dir=%s steps=%s",
                        cfg.output_dir,
                        cfg.steps,
                    )

                self._pending_cfg = cfg
                self._pending_accelerator = accelerator
                return original_train(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - profiler must never break training.
                self._logger.error("Profiler 包装执行错误: %s", exc)
                self._logger.debug("Profiler 包装执行堆栈", exc_info=True)
                raise
            finally:
                self._close_active_session()

        return patched_train

    def _extract_train_args(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        original_func: Callable[..., Any],
    ) -> tuple[Any, Any]:
        """通过反射提取 train 函数的 cfg 和 accelerator 参数。"""
        try:
            sig = inspect.signature(original_func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            return bound_args.arguments.get("cfg"), bound_args.arguments.get("accelerator")
        except Exception as exc:
            self._logger.warning("反射获取参数失败: %s", exc)
            return None, None

    def _extract_cli_cfg(self) -> Any:
        """Extract minimal profiler config from CLI args when LeRobot wraps train()."""
        output_dir = _read_cli_value("--output_dir", "./outputs")
        steps_value = _read_cli_value("--steps", "10000")
        try:
            steps = int(steps_value)
        except (TypeError, ValueError):
            steps = 10000
        return SimpleNamespace(output_dir=output_dir, steps=steps)

    def _initialize_profiler(self, cfg: Any, accelerator: Any = None) -> None:
        """初始化选定 backend 的 profiler session。"""
        try:
            if not self._is_main_process(accelerator):
                self._logger.info("检测到非主进程，跳过 Profiler 初始化")
                self._session = None
                self.profiler = None
                self.enabled = False
                return

            output_dir = getattr(cfg, "output_dir", "./outputs")
            if isinstance(output_dir, Path):
                output_dir = str(output_dir)

            total_steps = int(getattr(cfg, "steps", 10000))
            self._total_steps = total_steps

            runtime = RuntimeProfileConfig(
                config=self._config,
                output_dir=Path(output_dir) / self._config.output_subdir,
                total_steps=total_steps,
                device=self._resolve_device(accelerator),
                logger=self._logger,
            )

            if self._backend is None:
                self._backend = create_backend(self._config)
            self._session = self._backend.create_session(cfg, accelerator, runtime)
            self.profiler = self._session.profiler
            self._logger.info("Profiler backend '%s' 已创建并准备就绪", self._backend.name)
        except Exception as exc:  # pragma: no cover - profiler must never break training.
            self._logger.error("初始化 Profiler 失败: %s", exc)
            self._logger.debug("初始化 Profiler 堆栈", exc_info=True)
            self._session = None
            self.profiler = None
            self.enabled = False

    def _create_patched_update_policy(self, original_func: Callable[..., Any]) -> Callable[..., Any]:
        """创建 patch 后的 update_policy 函数，添加 record_function 标记并推进 step。"""

        @functools.wraps(original_func)
        def patched_update_policy(*args: Any, **kwargs: Any) -> Any:
            if not self.enabled:
                return original_func(*args, **kwargs)

            self._start_session_if_needed()
            if self._session is None:
                return original_func(*args, **kwargs)

            with self.record_function("training_update"):
                result = original_func(*args, **kwargs)
            self.step()
            return result

        return patched_update_policy

    def _start_session_if_needed(self) -> None:
        """Lazy-start profiler after LeRobot has validated and created output_dir."""
        if self._session_started or self._session is not None:
            return

        cfg = self._pending_cfg or self._extract_cli_cfg()
        self._initialize_profiler(cfg, self._pending_accelerator)
        if self._session is not None:
            self._session.__enter__()
            self._session_started = True
            self._logger.info("Profiler 已启动")

    def _close_active_session(self) -> None:
        """Close a lazy-started profiler session at the end of training."""
        if self._session is None or not self._session_started:
            return

        self._logger.info("正在关闭 Profiler 并生成最终报告...")
        self._session.__exit__(None, None, None)
        self._logger.info("Profiler 已关闭")
        self._logger.info("总共记录了 %s 个步骤", self._current_step)
        self._session_started = False

    def _is_main_process(self, accelerator: Any = None) -> bool:
        """尽量只在主进程启用 Profiler。"""
        if accelerator is None:
            return True

        if hasattr(accelerator, "is_main_process"):
            return bool(accelerator.is_main_process)

        if hasattr(accelerator, "process_index"):
            return int(accelerator.process_index) == 0

        return True

    def _resolve_device(self, accelerator: Any = None) -> Any:
        """通过 Accelerator 或 torch 状态推断设备。"""
        return resolve_torch_device(accelerator)

    def record_function(self, name: str) -> Generator[None, None, None]:
        """Delegate record markers to the active backend session."""
        if self._session is None:
            return _noop_record_function(name)
        return self._session.record_function(name)

    def step(self) -> None:
        """推进 Profiler 到下一个 step。"""
        if self._session is not None:
            self._session.step()
            self._current_step += 1

    def __enter__(self) -> "LeRobotProfiler":
        """进入 Context Manager。"""
        if self._session is not None:
            self._session.__enter__()
            self._logger.info("Profiler 已启动")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """退出 Context Manager。"""
        if self._session is not None:
            self._logger.info("正在关闭 Profiler 并生成最终报告...")
            self._session.__exit__(exc_type, exc_val, exc_tb)
            self._logger.info("Profiler 已关闭")
            self._logger.info("总共记录了 %s 个步骤", self._current_step)
        return False


@contextmanager
def _noop_record_function(name: str) -> Generator[None, None, None]:
    del name
    yield


def _read_cli_value(name: str, default: str) -> str:
    """Read ``--name=value`` or ``--name value`` from current argv."""
    prefix = f"{name}="
    argv = sys.argv[1:]
    for index, item in enumerate(argv):
        if item.startswith(prefix):
            return item[len(prefix) :]
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
    return default


_global_profiler: Optional[LeRobotProfiler] = None


def get_profiler() -> LeRobotProfiler:
    """获取全局 Profiler 实例。"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = LeRobotProfiler()
    return _global_profiler


@contextmanager
def record_function(name: str) -> Generator[None, None, None]:
    """安全的 record_function 上下文管理器。"""
    with get_profiler().record_function(name):
        yield


def auto_register() -> None:
    """模块导入时创建全局 Profiler 实例并应用 Monkey Patches。"""
    profiler = get_profiler()

    if profiler.enabled:
        print("[lerobot_policy_profiler] 插件已加载并通过 Monkey Patch 注入")
        backend_name = profiler._backend.name if profiler._backend is not None else "unknown"
        print(f"[lerobot_policy_profiler] Backend: {backend_name}")
        print("[lerobot_policy_profiler] 提示: 训练开始时会自动启用 Profiler")
