---
title: "快速开始"
description: "在壁仞 SUPA GPU 上运行 SGLang SUPA 的快速上手指南，包含启动服务与发送测试请求的示例。"
---

> **说明：** 本页仅介绍最简部署流程。完整的环境准备与安装说明请参见[安装指南](installation.md)。

## 前置条件

### 支持的设备

- 壁仞 BR2XX 系列 GPU

执行以下命令，确认系统已识别设备：

```bash
brsmi
```

### 软件依赖

请确认已安装以下组件：

- 壁仞 SUPA 驱动与运行时
- `torch_supa`、`supa_core`、`supa_bindings` Python wheel 包
- SGLang SUPA（本仓库）

如尚未安装，请先完成[安装指南](installation.md)中的安装步骤。

执行以下命令，验证 SUPA 环境：

```bash
python -c "import torch; import torch_supa; print('SUPA available:', torch.supa.is_available())"
```

输出 `SUPA available: True` 表示环境就绪。

## 启动服务

SGLang SUPA 自动从 HuggingFace 下载模型。若模型已下载到本地，请通过 `--model-path /path/to/model` 指定本地路径。

```bash
# Set HF_ENDPOINT to a mirror site if network is not available
export HF_ENDPOINT=https://hf-mirror.com

# Set your own HF_TOKEN to download restricted models
export HF_TOKEN=<secret>

# Start SGLang server
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 30000
```

检测到壁仞 GPU 时，SGLang SUPA 自动选择 SUPA 后端。如需强制指定设备，请使用 `--device supa`。

服务启动需要数分钟。出现以下输出表示服务已就绪：

```text
INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
The server is fired up and ready to roll!
```

相关环境变量的完整说明，参见[环境变量](../reference/environment_variables.md)。

### 多卡张量并行

使用 `--tp` 指定并行度，将模型分布到多张 GPU 上：

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tp 4 \
  --host 127.0.0.1 \
  --port 30000
```

壁仞硬件上的多卡服务使用 BCCL 作为分布式通信后端。

启动后，请按下文[验证服务](#验证服务)执行健康检查并发送 `/generate` 请求，确认多卡服务正常工作。

## 验证服务

日志出现 `The server is fired up and ready to roll!` 后，服务即可接收请求。

### 健康检查

```bash
curl http://127.0.0.1:30000/health
```

成功时返回 HTTP 200 状态码，响应体为空。

### 发送测试请求（原生接口）

```bash
curl -X POST http://localhost:30000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 16
        }
    }'
```

响应中 `text` 字段包含 `Paris`，表示服务工作正常。

### OpenAI 兼容接口

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

响应内容包含 `Paris`，表示服务工作正常。

## 停止服务

```bash
SGLANG_PID=$(pgrep -f "sglang.launch_server")
kill -SIGINT $SGLANG_PID
```

发送信号后请稍候，等待服务正常退出。

## 下一步

- [支持的模型](../reference/support_models.md)：已在壁仞 GPU 上验证的模型列表。
- [支持的功能](../reference/support_features.md)：功能特性矩阵与服务器参数。
- [常见问题](../faq.md)：故障排查与常见问题。
