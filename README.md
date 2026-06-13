# LeRobot Policy Profiler Plugin (Zero-Modification)

English | [中文](README_zh.md)

This policy plugin injects profiling into LeRobot training with **Monkey Patch + runtime reflection**, without requiring changes to LeRobot source files.

## Features

- **Zero code modification**: No LeRobot source file needs to be changed.
- **Plug and play**: Install the package and enable it with environment variables.
- **Automatic injection**: Designed to be loaded by LeRobot's `register_third_party_plugins()` mechanism.
- **Runtime reflection**: Automatically resolves runtime config, output directory, device and training steps.
- **Extensible backend design**: Supports PyTorch Profiler and Nsight Systems (`nsight` / `nsys`) backends.
- **Safe record markers**: Provides a `record_function` API for marking custom code regions.
- **TensorBoard visualization**: Generates PyTorch Profiler trace files compatible with TensorBoard.

## Installation And Usage

### Step 1: Install The Plugin

```bash
cd lerobot_profiler_plugin
pip install -e .
```

The installed distribution name is `lerobot_policy_profiler` so it can match LeRobot's third-party plugin discovery rules. User code should import the public API from `lerobot_policy_profiler`; `lerobot_profiler` is kept as a compatibility implementation package.

### Step 2: Run Training

```bash
export LEROBOT_PROFILE=true

lerobot-train \
  --policy.path path/to/policy_config.json \
  --dataset.repo_id your-dataset-id \
  --steps 5000
```

### Step 3: Open Results

```bash
tensorboard --logdir=outputs/train/YYYY-MM-DD/HH-MM-SS_*/profiler_logs/
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `LEROBOT_PROFILE` | `false` | Enable or disable profiling (`true` / `false`) |
| `LEROBOT_PROFILE_BACKEND` | `torch` | Profiler backend, currently `torch`, `nsight` or `nsys` |
| `LEROBOT_PROFILE_WAIT` | `5` | Number of initial steps to skip |
| `LEROBOT_PROFILE_WARMUP` | `2` | Number of warmup steps |
| `LEROBOT_PROFILE_ACTIVE` | `8` | Number of active profiling steps |
| `LEROBOT_PROFILE_REPEAT` | `-1` | Number of schedule repeats; `-1` means auto-calculate from total steps |
| `LEROBOT_PROFILE_SHAPES` | `true` | Record tensor shapes |
| `LEROBOT_PROFILE_MEMORY` | `true` | Record memory allocation |
| `LEROBOT_PROFILE_STACK` | `true` | Record stack traces |
| `LEROBOT_PROFILE_FLOPS` | `true` | Record FLOPs |
| `LEROBOT_PROFILE_OUTPUT_SUBDIR` | `profiler_logs` | Output subdirectory name for profile files |
| `LEROBOT_PROFILE_NSYS_CMD` | empty | Recommended launch command hint for Nsight Systems |
| `LEROBOT_PROFILE_NSIGHT_CUDA_PROFILER` | `true` | Whether to call CUDA Profiler API start/stop |
| `LEROBOT_PROFILE_NSIGHT_NVTX` | `true` | Whether to emit NVTX ranges for Nsight Systems |
| `LEROBOT_PROFILE_NSIGHT_STEP_PREFIX` | `lerobot_step` | Prefix for Nsight step markers |

> Do not paste full runtime configs into public issues or logs without reviewing them first. Fields such as `LEROBOT_PROFILE_NSYS_CMD` and `output_dir` may contain private dataset paths, cluster paths, host names or internal commands.

## Examples

### Basic Usage

```bash
export LEROBOT_PROFILE=true
lerobot-train --config your_config.yaml
```

### Nsight Systems Backend

Nsight Systems should launch the training process from outside. The plugin then calls `cudaProfilerStart()` / `cudaProfilerStop()` inside the process and emits NVTX ranges for custom timeline markers.

```bash
export LEROBOT_PROFILE=true
export LEROBOT_PROFILE_BACKEND=nsight
export LEROBOT_PROFILE_NSIGHT_CUDA_PROFILER=true
export LEROBOT_PROFILE_NSIGHT_NVTX=true

nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o lerobot_profile \
  lerobot-train --config your_config.yaml
```

The `nsys` backend name is kept as a compatible alias:

```bash
export LEROBOT_PROFILE_BACKEND=nsys
```

### Advanced Configuration

```bash
export LEROBOT_PROFILE=true
export LEROBOT_PROFILE_BACKEND=torch
export LEROBOT_PROFILE_WAIT=10
export LEROBOT_PROFILE_WARMUP=3
export LEROBOT_PROFILE_ACTIVE=15
export LEROBOT_PROFILE_STACK=false

lerobot-train --config your_config.yaml
```

### Add Custom Record Markers

```python
from lerobot_policy_profiler import record_function


class MyPolicy:
    def forward(self, batch):
        with record_function("custom_encoder"):
            features = self.encoder(batch)

        with record_function("custom_decoder"):
            output = self.decoder(features)

        return output
```

## How It Works

### Architecture Flow

```text
lerobot-train CLI starts
└── main()
    └── register_third_party_plugins()
        └── discovers the lerobot_policy_profiler distribution
            └── imports the policy shim package and forwards to the compatibility implementation package
                └── __init__.auto_register()
                    └── core.LeRobotProfiler.__init__()
                        ├── config.ProfilerConfig.from_env()
                        ├── backends.create_backend()
                        └── _apply_monkey_patches()
                            ├── replaces lerobot_train.train
                            └── replaces lerobot_train.update_policy

train(cfg, accelerator)
└── patched_train()
    ├── inspect.signature() extracts runtime args
    ├── _initialize_profiler(cfg, accelerator)
    │   └── backend.create_session()
    └── with profiler:
        └── original train()
            └── update_policy()
                ├── record_function("training_update")
                └── profiler.step()
```

### Key Techniques

#### 1. Monkey Patch

```python
self._original_train = lerobot_train.train
lerobot_train.train = self._create_patched_train(self._original_train)
```

#### 2. Runtime Reflection

```python
sig = inspect.signature(original_func)
bound_args = sig.bind(*args, **kwargs)
cfg = bound_args.arguments.get("cfg")
```

#### 3. Context Manager Lifecycle

```python
with profiler:
    result = original_train(...)
```

#### 4. Backend Extension Point

```python
class MyProfilerBackend:
    name = "my_backend"

    def create_session(self, cfg, accelerator, runtime):
        return MyProfileSession(runtime)
```

To add a backend, implement `create_session()` and a session object with `__enter__`, `__exit__`, `step()` and `record_function()`, then register it in `backends.create_backend()`.

## Output Example

### TensorBoard Timeline

```text
Timeline
├── data_loading_and_preprocessing [50ms]
│   ├── DataLoader.iter
│   └── preprocessor.process
├── training_update [120ms]
│   ├── Policy.forward
│   ├── backward
│   └── optimizer.step
└── logging [5ms]
```

### Generated Files

```text
outputs/train/
└── YYYY-MM-DD/
    └── 14-30-00_act_training/
        ├── profiler_logs/
        │   ├── batch_0.json.trace.json.gz
        │   ├── batch_1.json.trace.json.gz
        │   └── ...
        ├── checkpoints/
        └── ...
```

## Notes

### Runtime Overhead

- Enabling profiling usually adds 5% to 15% runtime overhead.
- Enable it only when diagnosing performance issues.
- Disable it in production with `export LEROBOT_PROFILE=false`.

### Disk Usage

- Each active profiling cycle generates one trace file.
- File size depends on model complexity and enabled record options.
- A typical trace file is around 10MB to 100MB per cycle.

### Compatibility

- PyTorch >= 2.0.
- Python >= 3.8.
- Works with training scripts based on `lerobot-train`.
- Supports single-GPU and multi-GPU distributed training; only the main process is profiled by default.

## Troubleshooting

### Problem 1: No Profile Files Are Generated

Possible causes:

- Environment variables are not set correctly.
- Training has too few steps; at least `wait + warmup + active` steps are required.
- The profiler failed to initialize.

Solutions:

```bash
echo "$LEROBOT_PROFILE"
lerobot-train --steps 1000
```

The logs should include:

```text
[lerobot_policy_profiler] 插件已加载并通过 Monkey Patch 注入
```

### Problem 2: TensorBoard Cannot Open Trace Files

```bash
pip install "tensorboard>=2.10"
tensorboard --logdir=path/to/profiler_logs/
rm -rf ~/.tensorboard-cache
```

### Problem 3: Profiling Causes OOM

```bash
export LEROBOT_PROFILE_MEMORY=false
export LEROBOT_PROFILE_SHAPES=false
export LEROBOT_PROFILE_STACK=false
export LEROBOT_PROFILE_ACTIVE=4
```

## Development

### Local Test

```bash
cd lerobot_profiler_plugin
pip install -e .

python -c "
from lerobot_policy_profiler import get_profiler
p = get_profiler()
print(f'Profiler enabled: {p.enabled}')
print(f'Config: {p.config}')
"
```

Review debug output before sharing it publicly to avoid exposing private paths, host names, internal dataset IDs or custom Nsight commands.

```bash
export LEROBOT_PROFILE=true
lerobot-train --config test_config.yaml --steps 100
```

### Build Package

```bash
cd lerobot_profiler_plugin
scripts/build_package.sh
```

The script removes old `build/`, `dist/` and `*.egg-info/` artifacts, then runs `python -m build` to generate both source distribution and wheel files under `dist/`.

Useful options:

- `PYTHON=/path/to/python scripts/build_package.sh`: choose a specific Python interpreter.
- `RUN_TESTS=1 scripts/build_package.sh`: run pytest before building.
- `CLEAN=0 scripts/build_package.sh`: keep existing build artifacts.

### Project Structure

```text
lerobot_profiler_plugin/
├── setup.py
├── pyproject.toml
├── MANIFEST.in
├── requirements-dev.txt
├── README.md
├── scripts/
│   └── build_package.sh
├── tests/
│   └── test_profiler.py
├── lerobot_policy_profiler/
│   └── __init__.py
└── lerobot_profiler/
    ├── __init__.py
    ├── config.py
    ├── backends.py
    └── core.py
```

Core modules:

- `__init__.py`: Public API exports and auto-registration entrypoint.
- `lerobot_policy_profiler/__init__.py`: LeRobot policy plugin discovery shim and recommended public import package.
- `config.py`: Environment variable parsing and `ProfilerConfig`.
- `backends.py`: PyTorch backend, Nsight Systems backend and session abstraction.
- `core.py`: `LeRobotProfiler`, Monkey Patch logic, runtime reflection and lifecycle management.
- `tests/test_profiler.py`: Tests for configuration, backend factory, lifecycle and `record_function`.

### Extension Ideas

Add new hook points:

```python
def _create_patched_dataloader(self, original_dl):
    def patched_dl(*args, **kwargs):
        with record_function("dataloader_iteration"):
            return original_dl(*args, **kwargs)

    return patched_dl
```

Support more configuration sources:

- Read global switches and profiler parameters from environment variables.
- Resolve output directory, training steps and runtime device from the `cfg` object.
- Extend later to YAML, TOML or native LeRobot configuration fields.
