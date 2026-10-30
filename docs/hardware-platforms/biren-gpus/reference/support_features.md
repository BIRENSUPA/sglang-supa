---
title: "支持的功能"
description: "壁仞 SUPA GPU 支持功能说明"
---

本节介绍 SGLang SUPA 在壁仞 SUPA GPU 上支持的功能。如遇问题或有疑问，请提交 issue。

## 核心功能

| 功能 | 状态 | 说明 |
|---|---|---|
| BF16 / FP16 推理 | 支持 | 根据模型配置自动识别。 |
| Tensor Parallelism (TP) | 支持 | 基于 BCCL 后端实现。 |
| Paged Attention | 支持 | |
| Continuous Batching | 支持 | |
| Radix Cache（前缀缓存） | 支持 | |
| Chunked Prefill | 支持 | |
| OpenAI 兼容接口 | 支持 | `/v1/chat/completions`、`/v1/completions`。 |
| 流式输出 | 支持 | |

## 量化

| 方法 | 状态 | 说明 |
|---|---|---|
| FP8 (W8A8) | 开发中 | |
| INT8 (W8A8) | 开发中 | |
| GPTQ | 规划中 | |
| AWQ | 规划中 | |

## 并行策略

| 策略 | 状态 | 说明 |
|---|---|---|
| Tensor Parallelism (TP) | 支持 | 基于 BCCL 实现。 |
| Data Parallelism (DP) | 支持 | |
| Pipeline Parallelism (PP) | 规划中 | |
| Expert Parallelism (EP) | 规划中 | 用于 MoE 模型。 |

## 关键服务器参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model-path` | None | 模型权重路径。 |
| `--tp-size` | 1 | 张量并行度。 |
| `--dtype` | auto | 数据类型，可选 auto、float16、bfloat16。 |
| `--host` | 127.0.0.1 | 服务监听地址。 |
| `--port` | 30000 | 服务监听端口。 |
| `--mem-fraction-static` | None | KV Cache 占用的显存比例。 |
| `--max-running-requests` | None | 最大并发请求数。 |
| `--context-length` | None | 覆盖模型的上下文长度。 |
| `--chunked-prefill-size` | None | Prefill 的分块大小。 |
| `--device` | auto | 设备类型，SUPA 环境下自动识别。 |
