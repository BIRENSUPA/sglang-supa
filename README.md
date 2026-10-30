# sglang_supa

`sglang_supa` 是一个面向 SGLang 的第三方平台插件包，用于注册并接入 `supa` 平台。
它的目标是以插件方式扩展上游 SGLang，在尽量不修改上游源码的前提下完成平台发现、运行时 Hook、注意力后端接入、JIT Kernel 替换以及部分量化相关能力的适配。

当前仓库整体处于 **bring-up / 适配接入阶段**：平台注册、Hook 注入、自定义 attention backend、部分 JIT kernel 替换链路已经具备，但图执行、分段编译后端等部分能力仍是占位实现。

## 项目介绍

这个仓库本质上是一个 SGLang 平台插件示例与实际适配骨架，核心职责包括：

- 向 SGLang 注册平台名 `supa`
- 在 SGLang 启动时自动注册插件 Hook
- 注入 `supa` 专属 attention backend
- 替换部分上游模块路径到插件自有实现
- 承载 `jit_kernel` 下的内核、测试与相关 Hook
- 为后续接入 SUPA / BIRENSUPA 运行时、设备能力、图执行和编译后端提供落点

从包名和平台名上看，两者并不相同：

- Python 包名：`sglang_supa`
- SGLang 平台名：`supa`

因此用户在 SGLang 中选择平台时，通常使用的是 `supa`，而不是 `sglang_supa`。

## 主要功能

当前代码已经覆盖的能力包括：

- **平台注册**
  - 通过 `pyproject.toml` 中的 entry points 注册 `sglang.srt.platforms` 和 `sglang.srt.plugins`
- **平台激活检测**
  - 支持通过环境变量 `SUPA_AVAILABLE=1` 强制启用
  - 默认会尝试探测 `supa_runtime` 模块是否存在
- **SupaPlatform 接入**
  - 默认 attention backend 设置为 `supa`
  - 暴露 KV Pool / Allocator / GraphRunner / PiecewiseBackend 等平台接口
  - 提供多平台分发 key：`supa`
- **Hook 注册与运行时替换**
  - 注册 `supa` attention backend
  - 注册 MLA 和 chunked prefix cache 对应 backend
  - 将 `sglang_supa.jit_kernel` 别名到 `sglang.jit_kernel`
  - 替换部分上游函数到插件实现
- **JIT Kernel 适配**
  - 提供 `clamp_position`、`kvcache`、`norm` 等 kernel hook 接入
  - 提供独立测试目录与 `.clangd` 生成入口
- **量化相关 Hook**
  - 在 `sglang_supa/srt/layers/quantization/` 下提供 AWQ、FP8 等模块 Hook 注册逻辑

## 当前实现状态

当前仓库不是完整可用的生产后端，而是一个可继续演进的适配骨架。需要特别注意：

- `SupaGraphRunner` 目前仍是占位类
- `SupaPiecewiseBackend` 仍会抛出 `NotImplementedError`
- `sglang_supa/memory.py` 当前主要复用上游 SGLang 的内存池与分配器实现
- 设备能力与运行时集成边界主要集中在 `sglang_supa/device.py`
- 一些 SGLang 路径通过 `sglang_supa/srt/...` 镜像目录进行替换，以避免直接修改上游源码

如果你的目标是继续完成 SUPA 平台 bring-up，通常需要优先补全以下部分：

- 真实设备与运行时检测
- `torch.supa` 相关设备控制路径
- 图执行与编译后端
- 更完整的 kernel 与量化能力实现
- 与本地 SGLang 版本的接口对齐

## 目录结构

下面是仓库中比较重要的目录和文件说明：

```text
sglang_supa/
├── pyproject.toml                  # Python 包配置与 SGLang entry points
├── README.md                       # 项目说明文档
├── CLAUDE.md                       # 仓库开发说明
└── sglang_supa/
    ├── entrypoints.py              # 平台检测与激活入口
    ├── platform.py                 # SupaPlatform 主平台定义
    ├── device.py                   # 设备/运行时抽象边界
    ├── memory.py                   # KV Pool 与分配器封装
    ├── hooks.py                    # 插件 Hook 总注册入口
    ├── ops.py                      # 多平台算子注册示例
    ├── graph_runner.py             # 图执行占位实现
    ├── compile_backend.py          # 分段编译后端占位实现
    ├── attention/                  # supa attention backend 实现
    ├── jit_kernel/                 # JIT kernel、测试、benchmark 与 csrc 代码
    └── srt/                        # 对上游 SGLang 模块路径的镜像与替换实现
        ├── layers/
        │   ├── attention/
        │   └── quantization/
        ├── distributed/
        └── models/
```

### 关键模块说明

#### 1. `sglang_supa/entrypoints.py`

负责平台激活：

- `detect_supa_hardware()`：探测是否满足 `supa` 平台启用条件
- `activate_supa()`：探测成功后返回 `sglang_supa.platform.SupaPlatform`

默认逻辑比较轻量，适合作为 bring-up 阶段的入口判断。

#### 2. `sglang_supa/platform.py`

这是与 SGLang 平台接口对接的核心类，主要提供：

- 默认 attention backend 为 `supa`
- KV Pool / Allocator 类型
- graph runner 类型
- piecewise backend 类型
- dispatch key `supa`

#### 3. `sglang_supa/device.py`

这是设备与供应商运行时的主要边界层，当前定义了：

- `device_name = "supa"`
- `device_type = "cuda"`
- `torch.device("supa")`
- `torch.supa.set_device(...)`
- 分布式 backend 返回 `gloo`

如果平台初始化、设备切换或 runtime 集成出现问题，通常应先从这里排查。

#### 4. `sglang_supa/hooks.py`

插件扩展的总入口，主要完成：

- JIT Kernel 包路径别名
- 量化模块 Hook 注册
- models utils Hook 注册
- attention backend 注册
- `MultiPlatformOp` 的 `supa` forward 注册

这是理解“插件如何接管上游行为”的关键文件。

#### 5. `sglang_supa/attention/`

这里包含 `supa` 专属 attention backend 的核心实现，是当前仓库中最实质性的功能区域之一。

#### 6. `sglang_supa/jit_kernel/`

包含 JIT Kernel 相关实现、测试、benchmark 和 csrc 代码，是底层 GPU / kernel 开发的主要区域。

#### 7. `sglang_supa/srt/`

该目录镜像了部分上游 SGLang 模块路径，便于通过 HookRegistry 将上游调用重定向到插件实现。

## 安装方法

### 0. Full-stack 依赖

#### Full-stack Install

```shell
cd full-stack
bash ./install.sh -p all
pip install --upgrade pip setuptools wheel
pip install suda*.whl
pip install torch_supa-1.0.0.21000*.whl
pip install triton-3.4*.whl
pip install suattention-0.0.1.21000*.whl
pip install flashinfer*.whl
pip install flash_mla-1.0.0.21000*.whl
# pip install deep_gemm*.whl
pip install deep_ep-2.0.0.21000*.whl
pip install flashattn_train-3.0.0.0.21000*.whl
suda init
```

<!-- #### Flash-attention Install

```shell
git clone --recursive https://github.com/birensupa/flash-attention.git
cd flash-attention/pack_train
python3 setup.py bdist_wheel
pip install dist/*.whl --force
``` -->

### 1. 克隆仓库

SGLang 官方代码作为 submodule 包含在 `3rdparty/sglang` 中，`--recursive` 会自动拉取。

```bash
git clone --recursive -b develop_br200 https://github.com/birensupa/sglang-supa.git
cd sglang-supa
git submodule update --init
```

### 2. 安装 SGLang + sglang_supa

```bash
bash install.sh
```

`install.sh` 会自动完成：patch 上游 SGLang（去掉 Rust 扩展和 NVIDIA 依赖）→ 安装 SGLang 代码 → 安装 sglang_supa 及全部 SUPA 依赖。

### 3. 安装 sgl-kernel

```bash
cd sgl-kernel
python3 setup.py bdist_wheel && pip install dist/*.whl --force
cd ..
```

### 4. 验证安装

```bash
SUPA_AVAILABLE=1 SGLANG_PLATFORM=supa \
python3 -c "from sglang.srt.platforms import current_platform; print(type(current_platform).__name__, current_platform.device_name)"
```

输出 `SupaPlatform supa` 即安装成功。然后可以测试离线推理脚本：

```bash
python3 /home/ubuntu/sglang-supa/examples/cmodel/run_llm.py
```

## 使用方法

通过环境变量选择 SUPA 平台：

```bash
export SUPA_AVAILABLE=1
export SGLANG_PLATFORM=supa
```

如果当前环境还没有 `supa_runtime`，`SUPA_AVAILABLE=1` 可以强制通过平台探测，便于 bring-up 阶段验证插件链路。

启动服务：

```bash
sglang serve --model-path <model>
```

## 工作机制简述

仓库当前的插件接入流程大致如下：

1. SGLang 通过 entry points 发现 `supa` 平台插件
2. `activate_supa()` 判断当前环境是否启用 `supa`
3. 若启用，则实例化 `SupaPlatform`
4. 插件 Hook 在导入阶段完成注册
5. attention backend、JIT Kernel 路径和部分上游函数被替换为插件实现
6. 运行时通过 `sglang_supa/srt/...` 和 `jit_kernel/...` 提供平台特化能力

## 适用场景

这个仓库适合以下场景：

- 为 SGLang 接入新的第三方硬件平台
- 以插件方式验证平台 bring-up 流程
- 调试 HookRegistry、attention backend 和 kernel 替换链路
- 开发和测试 `supa` 专属 JIT kernel
- 逐步替换上游默认实现为平台专属实现

## 许可证

如仓库未额外声明，请以项目实际许可证文件为准。
