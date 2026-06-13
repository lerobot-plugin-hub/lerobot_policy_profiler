# LeRobot Policy Profiler Plugin (Zero-Modification)

中文 | [English](README.md)

通过 **Monkey Patch + 反射机制**，尽量不修改任何 LeRobot 源码即可为 policy 训练注入 PyTorch Profiler。

## 特性

- **零代码修改**: 无需修改任何 LeRobot 源文件。
- **即插即用**: 只需 `pip install` 并设置环境变量。
- **自动注入**: 利用 `register_third_party_plugins()` 机制自动加载。
- **智能反射**: 自动检测设备、配置、输出目录等运行时信息。
- **Backend 可扩展**: 当前支持 PyTorch Profiler 与 Nsight Systems (`nsight` / `nsys`) backend。
- **安全包装**: 提供 `record_function` API 用于标记代码区域。
- **完整可视化**: 输出 TensorBoard 兼容的 Trace 文件。

## 安装与使用

### 第一步：安装插件

```bash
cd lerobot_profiler_plugin
pip install -e .
```

安装后的发行包名是 `lerobot_policy_profiler`，用于匹配 LeRobot 第三方插件发现规则；日常代码建议从 `lerobot_policy_profiler` 导入公共 API；`lerobot_profiler` 仅作为兼容实现包保留。

### 第二步：运行训练

```bash
export LEROBOT_PROFILE=true

lerobot-train \
  --policy.path path/to/policy_config.json \
  --dataset.repo_id your-dataset-id \
  --steps 5000
```

### 第三步：查看结果

```bash
tensorboard --logdir=outputs/train/YYYY-MM-DD/HH-MM-SS_*/profiler_logs/
```

## 环境变量配置

| 变量名 | 默认值 | 说明 |
| --- | --- | --- |
| `LEROBOT_PROFILE` | `false` | 是否启用 Profiler (`true`/`false`) |
| `LEROBOT_PROFILE_BACKEND` | `torch` | Profiler 后端，当前可选 `torch` / `nsight` / `nsys` |
| `LEROBOT_PROFILE_WAIT` | `5` | 跳过前 N 步（等待阶段） |
| `LEROBOT_PROFILE_WARMUP` | `2` | 预热 N 步 |
| `LEROBOT_PROFILE_ACTIVE` | `8` | 记录 N 步详细数据 |
| `LEROBOT_PROFILE_REPEAT` | `-1` | 重复次数（`-1` 表示根据总步数自动计算） |
| `LEROBOT_PROFILE_SHAPES` | `true` | 记录 Tensor 形状 |
| `LEROBOT_PROFILE_MEMORY` | `true` | 记录内存分配 |
| `LEROBOT_PROFILE_STACK` | `true` | 记录调用栈 |
| `LEROBOT_PROFILE_FLOPS` | `true` | 记录 FLOPs |
| `LEROBOT_PROFILE_OUTPUT_SUBDIR` | `profiler_logs` | profile 输出子目录名 |
| `LEROBOT_PROFILE_NSYS_CMD` | 空 | Nsight Systems 推荐启动命令提示 |
| `LEROBOT_PROFILE_NSIGHT_CUDA_PROFILER` | `true` | 是否调用 CUDA Profiler API 的 start/stop |
| `LEROBOT_PROFILE_NSIGHT_NVTX` | `true` | 是否向 Nsight Systems 写入 NVTX range |
| `LEROBOT_PROFILE_NSIGHT_STEP_PREFIX` | `lerobot_step` | Nsight step 标记前缀 |

> 开源 issue 或公开日志中请勿直接粘贴包含私有数据集路径、集群路径、主机名或内部命令的完整配置。`LEROBOT_PROFILE_NSYS_CMD` 和 `output_dir` 等字段可能包含本地环境信息。

## 使用示例

### 基础用法

```bash
export LEROBOT_PROFILE=true
lerobot-train --config your_config.yaml
```

### Nsight Systems 后端

Nsight Systems 需要从进程外启动训练命令，插件会在进程内负责 `cudaProfilerStart()` / `cudaProfilerStop()` 与 NVTX range 标记。

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

也可以使用兼容别名：

```bash
export LEROBOT_PROFILE_BACKEND=nsys
```

### 高级配置

```bash
export LEROBOT_PROFILE=true
export LEROBOT_PROFILE_BACKEND=torch
export LEROBOT_PROFILE_WAIT=10
export LEROBOT_PROFILE_WARMUP=3
export LEROBOT_PROFILE_ACTIVE=15
export LEROBOT_PROFILE_STACK=false

lerobot-train --config your_config.yaml
```

### 在自定义策略中使用 Hook

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

## 工作原理

### 架构流程图

```text
lerobot-train CLI 启动
└── main()
    └── register_third_party_plugins()
        └── 发现 lerobot_policy_profiler 发行包
            └── 导入 policy shim 包并转发到兼容实现包
                └── __init__.auto_register()
                    └── core.LeRobotProfiler.__init__()
                        ├── config.ProfilerConfig.from_env()
                        ├── backends.create_backend()
                        └── _apply_monkey_patches()
                            ├── 替换 lerobot_train.train
                            └── 替换 lerobot_train.update_policy

train(cfg, accelerator)
└── patched_train()
    ├── inspect.signature() 反射提取参数
    ├── _initialize_profiler(cfg, accelerator)
    │   └── backend.create_session()
    └── with profiler:
        └── 原始 train()
            └── update_policy()
                ├── record_function("training_update")
                └── profiler.step()
```

### 核心技术点

#### 1. Monkey Patch 机制

```python
self._original_train = lerobot_train.train
lerobot_train.train = self._create_patched_train(self._original_train)
```

#### 2. 反射参数提取

```python
sig = inspect.signature(original_func)
bound_args = sig.bind(*args, **kwargs)
cfg = bound_args.arguments.get("cfg")
```

#### 3. Context Manager 协议

```python
with profiler:
    result = original_train(...)
```

#### 4. Backend 扩展点

```python
class MyProfilerBackend:
    name = "my_backend"

    def create_session(self, cfg, accelerator, runtime):
        return MyProfileSession(runtime)
```

新增后端时只需要实现 `create_session()` 与 session 的 `__enter__`、`__exit__`、`step()`、`record_function()`，再在 `backends.create_backend()` 中注册。

## 输出示例

### TensorBoard Timeline 视图

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

### 生成的文件结构

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

## 注意事项

### 性能影响

- 启用 Profiler 通常会带来 5% 到 15% 的性能开销。
- 建议只在调试性能问题时启用。
- 生产环境请禁用：`export LEROBOT_PROFILE=false`。

### 磁盘空间

- 每个 active 周期会生成一个 trace 文件。
- 文件大小取决于模型复杂度和记录选项。
- 典型文件大小为 10MB 到 100MB per cycle。

### 兼容性要求

- PyTorch >= 2.0。
- Python >= 3.8。
- 适用于基于 `lerobot-train` 的训练脚本。
- 支持单 GPU 和多 GPU 分布式训练；默认仅主进程启用 Profiler。

## 故障排查

### 问题 1：没有生成 Profile 文件

可能原因：

- 环境变量未正确设置。
- 训练步数太少，至少需要 `wait + warmup + active` 步。
- Profiler 未成功初始化。

解决方案：

```bash
echo "$LEROBOT_PROFILE"
lerobot-train --steps 1000
```

日志中应看到：

```text
[lerobot_policy_profiler] 插件已加载并通过 Monkey Patch 注入
```

### 问题 2：TensorBoard 无法打开 Trace 文件

```bash
pip install "tensorboard>=2.10"
tensorboard --logdir=path/to/profiler_logs/
rm -rf ~/.tensorboard-cache
```

### 问题 3：Profiling 导致 OOM

```bash
export LEROBOT_PROFILE_MEMORY=false
export LEROBOT_PROFILE_SHAPES=false
export LEROBOT_PROFILE_STACK=false
export LEROBOT_PROFILE_ACTIVE=4
```

## 开发指南

### 本地测试

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

公开粘贴调试输出前，请先确认配置中没有私有路径、主机名、内部数据集 ID 或自定义 Nsight 命令。

```bash
export LEROBOT_PROFILE=true
lerobot-train --config test_config.yaml --steps 100
```

### 打包

```bash
cd lerobot_profiler_plugin
scripts/build_package.sh
```

脚本会先删除旧的 `build/`、`dist/` 和 `*.egg-info/` 产物，再执行 `python -m build`，在 `dist/` 下生成源码包和 wheel 包。

常用选项：

- `PYTHON=/path/to/python scripts/build_package.sh`: 指定 Python 解释器。
- `RUN_TESTS=1 scripts/build_package.sh`: 打包前先运行 pytest。
- `CLEAN=0 scripts/build_package.sh`: 保留已有构建产物。

### 代码结构说明

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

核心实现包含：

- `__init__.py`: 公共 API 导出与自动注册入口。
- `lerobot_policy_profiler/__init__.py`: LeRobot policy 插件发现 shim，也是推荐的公共导入包。
- `config.py`: 环境变量解析与 `ProfilerConfig`。
- `backends.py`: PyTorch backend、Nsight Systems backend 与 session 抽象。
- `core.py`: `LeRobotProfiler`、Monkey Patch、反射参数提取和生命周期管理。
- `tests/test_profiler.py`: 配置、backend 工厂、核心生命周期和 `record_function` 测试。

### 扩展开发

添加新的 Hook 点：

```python
def _create_patched_dataloader(self, original_dl):
    def patched_dl(*args, **kwargs):
        with record_function("dataloader_iteration"):
            return original_dl(*args, **kwargs)

    return patched_dl
```

支持更多配置来源：

- 从环境变量读取全局开关和 profiler 参数。
- 从 `cfg` 对象反射获取输出目录、训练步数和运行设备。
- 后续可以扩展为读取 YAML、TOML 或 LeRobot 原生配置字段。
