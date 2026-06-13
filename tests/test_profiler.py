import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[1]


class FakeDevice:
    def __init__(self, device_type):
        self.type = device_type

    def __str__(self):
        return self.type


@pytest.fixture
def fake_torch(monkeypatch):
    torch_module = types.ModuleType("torch")
    profiler_module = types.ModuleType("torch.profiler")
    calls = []

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    @contextmanager
    def fake_record_function(name):
        calls.append(name)
        yield

    profiler_module.record_function = fake_record_function
    profiler_module.ProfilerActivity = types.SimpleNamespace(CPU="cpu", CUDA="cuda")
    profiler_module.schedule = Mock(return_value="schedule")
    profiler_module.tensorboard_trace_handler = Mock(return_value="trace_handler")
    profiler_module.profile = Mock(return_value=Mock())

    torch_module.cuda = FakeCuda()
    torch_module.device = FakeDevice
    torch_module.profiler = profiler_module

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.profiler", profiler_module)
    return types.SimpleNamespace(torch=torch_module, profiler=profiler_module, calls=calls)


@pytest.fixture
def profiler_module(monkeypatch, fake_torch):
    monkeypatch.delenv("LEROBOT_PROFILE", raising=False)
    monkeypatch.syspath_prepend(str(PLUGIN_ROOT))
    import lerobot_profiler
    import lerobot_profiler.core as core

    core.LeRobotProfiler._instance = None
    core.LeRobotProfiler._patched = False
    core._global_profiler = None
    yield lerobot_profiler
    core.LeRobotProfiler._instance = None
    core.LeRobotProfiler._patched = False
    core._global_profiler = None


def test_config_reads_environment():
    from lerobot_profiler.config import ProfilerConfig

    config = ProfilerConfig.from_env(
        {
            "LEROBOT_PROFILE": "true",
            "LEROBOT_PROFILE_BACKEND": "nsys",
            "LEROBOT_PROFILE_WAIT": "10",
            "LEROBOT_PROFILE_WARMUP": "3",
            "LEROBOT_PROFILE_ACTIVE": "15",
            "LEROBOT_PROFILE_REPEAT": "2",
            "LEROBOT_PROFILE_SHAPES": "false",
            "LEROBOT_PROFILE_MEMORY": "false",
            "LEROBOT_PROFILE_STACK": "false",
            "LEROBOT_PROFILE_FLOPS": "false",
            "LEROBOT_PROFILE_OUTPUT_SUBDIR": "custom_logs",
            "LEROBOT_PROFILE_NSYS_CMD": "nsys profile",
            "LEROBOT_PROFILE_NSIGHT_CUDA_PROFILER": "false",
            "LEROBOT_PROFILE_NSIGHT_NVTX": "false",
            "LEROBOT_PROFILE_NSIGHT_STEP_PREFIX": "train_step",
        }
    )

    assert config.as_dict() == {
        "enabled": True,
        "backend": "nsys",
        "wait": 10,
        "warmup": 3,
        "active": 15,
        "repeat": 2,
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "with_flops": False,
        "output_subdir": "custom_logs",
        "nsys_command": "nsys profile",
        "nsight_enable_cuda_profiler": False,
        "nsight_emit_nvtx": False,
        "nsight_step_prefix": "train_step",
    }


def test_extract_train_args_supports_positional_and_keyword_args(profiler_module):
    def train(cfg, accelerator=None):
        return cfg, accelerator

    profiler = profiler_module.LeRobotProfiler()
    cfg = object()
    accelerator = object()

    assert profiler._extract_train_args((cfg,), {"accelerator": accelerator}, train) == (
        cfg,
        accelerator,
    )


def test_extract_cli_cfg_supports_equals_and_space_args(monkeypatch, profiler_module):
    profiler = profiler_module.LeRobotProfiler()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-train",
            "--output_dir=lerobot_train/",
            "--steps",
            "100",
        ],
    )

    cfg = profiler._extract_cli_cfg()

    assert cfg.output_dir == "lerobot_train/"
    assert cfg.steps == 100


@pytest.mark.parametrize(
    "accelerator, expected",
    [
        (None, True),
        (types.SimpleNamespace(is_main_process=True), True),
        (types.SimpleNamespace(is_main_process=False), False),
        (types.SimpleNamespace(process_index=0), True),
        (types.SimpleNamespace(process_index=1), False),
    ],
)
def test_is_main_process_handles_accelerator_shapes(profiler_module, accelerator, expected):
    profiler = profiler_module.LeRobotProfiler()

    assert profiler._is_main_process(accelerator) is expected


def test_resolve_device_uses_accelerator_device_when_available(profiler_module):
    profiler = profiler_module.LeRobotProfiler()
    device = FakeDevice("cuda")
    accelerator = types.SimpleNamespace(device=device)

    assert profiler._resolve_device(accelerator) is device


def test_step_advances_profiler_when_initialized(profiler_module):
    profiler = profiler_module.LeRobotProfiler()
    fake_session = Mock()
    profiler._session = fake_session
    profiler._current_step = 0

    profiler.step()

    fake_session.step.assert_called_once_with()
    assert profiler._current_step == 1


def test_patched_train_lazy_starts_profiler_from_update_policy(monkeypatch, profiler_module):
    events = []

    class FakeSession:
        profiler = object()

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            events.append("exit")
            return False

        def step(self):
            events.append("step")

        @contextmanager
        def record_function(self, name):
            events.append(("record", name))
            yield

    class FakeBackend:
        name = "fake"

        def create_session(self, cfg, accelerator, runtime):
            events.append(("create", runtime.output_dir.name, runtime.total_steps))
            return FakeSession()

    profiler = profiler_module.LeRobotProfiler()
    profiler.enabled = True
    profiler._backend = FakeBackend()
    monkeypatch.setattr(sys, "argv", ["lerobot-train", "--output_dir=run_dir", "--steps=3"])

    def update_policy():
        events.append("update")
        return "updated"

    patched_update = profiler._create_patched_update_policy(update_policy)

    def train():
        events.append("train")
        return patched_update()

    result = profiler._create_patched_train(train)()

    assert result == "updated"
    assert events == [
        "train",
        ("create", "profiler_logs", 3),
        "enter",
        ("record", "training_update"),
        "update",
        "step",
        "exit",
    ]


def test_record_function_noops_when_profiler_is_missing(profiler_module):
    with profiler_module.record_function("custom_operation"):
        observed = "ran"

    assert observed == "ran"


def test_record_function_delegates_to_torch_when_profiler_exists(profiler_module, fake_torch):
    profiler = profiler_module.get_profiler()

    class FakeSession:
        @contextmanager
        def record_function(self, name):
            fake_torch.calls.append(name)
            yield

    profiler._session = FakeSession()

    with profiler_module.record_function("custom_operation"):
        observed = "ran"

    assert observed == "ran"
    assert fake_torch.calls == ["custom_operation"]


def test_lerobot_policy_profiler_shim_imports_public_api(profiler_module):
    import lerobot_policy_profiler

    assert lerobot_policy_profiler.get_profiler is profiler_module.get_profiler
    assert lerobot_policy_profiler.record_function is profiler_module.record_function


def test_create_backend_supports_torch_nsight_and_nsys_alias():
    from lerobot_profiler.backends import NsightProfilerBackend, TorchProfilerBackend, create_backend
    from lerobot_profiler.config import ProfilerConfig

    assert isinstance(create_backend(ProfilerConfig(backend="torch")), TorchProfilerBackend)
    assert isinstance(create_backend(ProfilerConfig(backend="nsight")), NsightProfilerBackend)
    assert isinstance(create_backend(ProfilerConfig(backend="nsys")), NsightProfilerBackend)


def test_create_backend_rejects_unknown_backend():
    from lerobot_profiler.backends import create_backend
    from lerobot_profiler.config import ProfilerConfig

    with pytest.raises(ValueError, match="Unsupported profiler backend"):
        create_backend(ProfilerConfig(backend="unknown"))


def test_unknown_backend_does_not_break_when_profiler_disabled(monkeypatch, fake_torch):
    monkeypatch.setenv("LEROBOT_PROFILE", "false")
    monkeypatch.setenv("LEROBOT_PROFILE_BACKEND", "unknown")
    monkeypatch.syspath_prepend(str(PLUGIN_ROOT))

    import lerobot_profiler.core as core

    core.LeRobotProfiler._instance = None
    core._global_profiler = None

    profiler = core.LeRobotProfiler()

    assert profiler.enabled is False
    assert profiler._backend is None


def test_nsight_backend_noops_without_cuda(tmp_path):
    from lerobot_profiler.backends import NsightProfilerBackend, RuntimeProfileConfig
    from lerobot_profiler.config import ProfilerConfig

    runtime = RuntimeProfileConfig(
        config=ProfilerConfig(backend="nsight"),
        output_dir=tmp_path / "nsys_logs",
        total_steps=10,
        device=None,
        logger=Mock(),
    )

    session = NsightProfilerBackend().create_session(cfg=object(), accelerator=None, runtime=runtime)

    with session:
        session.step()
        with session.record_function("training_update"):
            observed = "ran"

    assert observed == "ran"
    assert runtime.output_dir.exists()


def test_nsight_backend_uses_cuda_profiler_api_and_nvtx(monkeypatch, tmp_path):
    from lerobot_profiler.backends import NsightProfilerBackend, RuntimeProfileConfig
    from lerobot_profiler.config import ProfilerConfig

    torch_module = types.ModuleType("torch")
    profiler_module = types.ModuleType("torch.profiler")
    calls = []

    class FakeNvtx:
        @staticmethod
        def range_push(name):
            calls.append(("push", name))

        @staticmethod
        def range_pop():
            calls.append(("pop", None))

    class FakeCudart:
        @staticmethod
        def cudaProfilerStart():
            calls.append(("cudaProfilerStart", None))

        @staticmethod
        def cudaProfilerStop():
            calls.append(("cudaProfilerStop", None))

    class FakeCuda:
        nvtx = FakeNvtx()

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def cudart():
            return FakeCudart()

    torch_module.cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.profiler", profiler_module)

    runtime = RuntimeProfileConfig(
        config=ProfilerConfig(
            backend="nsight",
            nsight_enable_cuda_profiler=True,
            nsight_emit_nvtx=True,
            nsight_step_prefix="train_step",
        ),
        output_dir=tmp_path / "nsight_logs",
        total_steps=10,
        device=FakeDevice("cuda"),
        logger=Mock(),
    )

    session = NsightProfilerBackend().create_session(cfg=object(), accelerator=None, runtime=runtime)

    with session:
        with session.record_function("training_update"):
            observed = "ran"
        session.step()

    assert observed == "ran"
    assert calls == [
        ("cudaProfilerStart", None),
        ("push", "training_update"),
        ("pop", None),
        ("push", "train_step_0"),
        ("pop", None),
        ("cudaProfilerStop", None),
    ]
