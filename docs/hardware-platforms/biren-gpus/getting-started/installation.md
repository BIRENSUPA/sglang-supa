---
title: "安装指南"
description: "壁仞 SUPA GPU 上 SGLang SUPA 的完整安装指南，包含组件版本对应关系、环境准备、安装与安装验证。"
---

SGLang SUPA 是面向壁仞 BR2XX 系列 GPU 的 SGLang 插件。该插件使 SGLang 框架可以在壁仞 GPU 上运行。

本文介绍 SGLang SUPA 的安装方式。如遇到问题，请提交 issue。

## 基础要求

* Ubuntu 22.04、Ubuntu 24.04 或兼容的 Linux 环境
* Python >= 3.10
* BR2XX 硬件与 BIRENSUPA SDK
* Docker >= 20.10.7（仅使用 Docker 镜像安装时需要）

## 版本依赖

下表列出 SGLang SUPA 当前版本的核心依赖。各组件版本需严格匹配，请勿混用不同的版本集合。

| 依赖 | 版本 | 说明 |
|---|---|---|
| SGLang | 0.5.20 | 上游 SGLang 基线版本 |
| PyTorch | 2.12.0 | CPU wheel 或 TorchSUPA wheel |
| TorchSUPA (torch_supa) | 2.12.0 | 提供 SUPA 的 PyTorch 运行时与 PrivateUse1 设备支持 |
| transformers | 5.12.1 | HuggingFace Transformers |
| BIRENSUPA SDK | 配套交付版本 | 包含驱动、运行时、BRCC 编译器与 BCCL 通信库 |

### 配套算子依赖

以下算子包通过配套交付的 wheel 包安装。

| 依赖 | 说明 |
|---|---|
| flashattn_train | Flash Attention 训练内核 |
| flashinfer | FlashInfer 推理内核 |
| suattention | SUPA Attention 算子库 |
| deep_ep | DeepEP 通信库，用于 MoE 专家并行 |
| flash_mla | Flash MLA 解码内核 |
| triton | Triton 编译器，SUPA 适配版本 |
| tilelang | TileLang 算子编程框架 |

## 环境准备

SGLang SUPA 提供以下三种安装方式。请根据部署与开发需求选择。

### 方式一：官方 Docker 镜像

官方镜像已预装 SGLang SUPA、TorchSUPA 与运行时依赖。

> **注意：** 当前暂未发布镜像，请关注官方发布动态。

如需自行构建镜像，执行以下命令：

```bash
cd sglang-supa/docker
docker build -t sglang-supa:latest -f Dockerfile .
```

启动容器：

```bash
docker run -it --rm --privileged --network=host --ipc=host --shm-size=16g \
    --device=/dev/biren0 --device=/dev/biren1 \
    --device=/dev/biren2 --device=/dev/biren3 \
    --device=/dev/biren4 --device=/dev/biren5 \
    --device=/dev/biren6 --device=/dev/biren7 \
    --volume ~/.cache/:/root/.cache/ \
    sglang-supa:latest \
    bash
```

> **提示：** SGLang 默认监听 `http://127.0.0.1:30000`。如需修改，请使用 `--host` 与 `--port` 参数。

### 方式二：wheel 包安装

该方式适用于物理机环境，或已部署 BIRENSUPA 的环境。

#### Python 环境

Python 版本需为 3.10 及以上。建议使用 [conda](https://github.com/conda/conda) 创建隔离环境。

```bash
conda create --name sglang_supa python=3.10
conda activate sglang_supa
```

#### 安装 BIRENSUPA SDK

安装 SDK 并设置环境变量。

```bash
sudo bash birensupa-sdk-xxxx.run

source /usr/local/birensupa/all/latest/scripts/brsw_set_env.sh  # 设置 BIRENSUPA 环境
suda init
. "$HOME/.gstub/suda.sh"
suda load
```

验证 SUPA 环境：

```bash
# 检查壁仞 GPU 设备是否可见
brsmi

# 验证 PyTorch 可访问 SUPA 运行时
python -c "import torch; import torch_supa; print(torch.supa.is_available())"
```

#### 安装 PyTorch 和 TorchSUPA

```bash
pip install torch==2.12.0
pip install torch_supa-*.whl  # TorchSUPA wheel，从配套交付包获取
```

TorchSUPA (torch_supa) 通过 PyTorch 的 PrivateUse1 机制注册壁仞 SUPA GPU 设备。注册后，可通过 `torch.cuda` 与 `torch.supa` 接口访问该设备。

#### 安装配套算子依赖和 SGLang SUPA

```bash
pip install \
    flashattn_train-*.whl \
    flashinfer-*.whl \
    suattention-*.whl \
    deep_ep-*.whl \
    flash_mla-*.whl \
    triton-*.whl \
    tilelang-*.whl \
    torch_supa-*.whl \
    sglang_supa-*.whl
```

> **注意：** 当前暂未发布配套算子及相关软件包，请关注官方发布动态。

### 方式三：源码构建

该方式适用于需要修改代码或重新编译 SUPA 扩展的场景。使用前，请先完成[方式二：wheel 包安装](#方式二wheel-包安装)中 BIRENSUPA SDK、PyTorch、TorchSUPA 与配套算子依赖的安装，仅跳过 `sglang_supa-*.whl` 一项。

```bash
git clone <sglang-supa-repo-url>
cd sglang-supa
pip install -e "python[all]"
```

开发模式从源码编译 C++/SUPA 扩展。如需指定并行编译任务数，请设置 `MAX_JOBS` 参数。

```bash
MAX_JOBS=8 pip install -e "python[all]" --no-build-isolation
```

## 系统设置

以下设置用于获得稳定的推理性能。建议在正式部署前完成配置。

### CPU 性能模式

默认电源策略可能影响性能。建议将调速器切换为 `performance`。

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor  # should show: performance
```

### 关闭 NUMA 平衡

NUMA 平衡会引入额外的内存页迁移。建议关闭该功能。

```bash
sudo sysctl -w kernel.numa_balancing=0

# Verify
cat /proc/sys/kernel/numa_balancing  # should show: 0
```

## 安装验证

安装完成后，执行以下命令确认 SUPA 环境可用。

```bash
# 检查壁仞 GPU 设备是否可见
brsmi

# 验证 PyTorch 可访问 SUPA 运行时
python -c "import torch; import torch_supa; print(torch.supa.is_available())"
```

若 `brsmi` 正常列出设备，且上述命令输出 `True`，则表示安装成功。

若 `brsmi` 命令不存在，或未列出任何设备，请参见[常见问题](../faq.md)中的「运行时设备不可见」。

## 下一步

完成安装后，请参见[快速开始](quick_start.md)，了解启动推理服务、发送测试请求与停止服务的方法。
