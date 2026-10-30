---
title: "术语表"
description: "壁仞 SUPA GPU 平台术语表"
---

## 硬件

| 术语 | 说明 |
|---|---|
| BR2XX | 本插件支持的壁仞硬件系列，包括 BR200 等型号。 |
| GCU | General Computing Unit，壁仞 GPU 的计算核心，功能类似 NVIDIA SM。 |
| HBM | High Bandwidth Memory，壁仞 GPU 使用的高带宽显存。 |

## 软件

| 术语 | 说明 |
|---|---|
| SUPA | 壁仞统一并行计算平台，功能类似 NVIDIA CUDA。 |
| BIRENSUPA | 壁仞软件栈的统称，包含运行时、工具链和配套开发库。 |
| TorchSUPA (torch_supa) | 提供 SUPA PyTorch 运行时和 PrivateUse1 设备支持的依赖包。 |
| BCCL | BIRENSUPA 集合通信库，功能类似 NVIDIA NCCL。 |
| BRCC | BIRENSUPA 编译器。 |
| SUDA | 将 CUDA 兼容代码转发到 SUPA 的工具链。 |
| SubLAS | BIRENSUPA 线性代数库，提供 SUPA 上的矩阵乘法和分组矩阵乘法。 |
| brsmi | 壁仞系统管理接口命令行工具，功能类似 nvidia-smi。 |
| PrivateUse1 | PyTorch 为第三方设备后端预留的 dispatch key。壁仞通过该机制接入 SUPA GPU。 |

## 推理

| 术语 | 说明 |
|---|---|
| TP | Tensor Parallelism，张量并行。将模型层拆分到多个 GPU 上并行计算。 |
| EP | Expert Parallelism，专家并行。用于 MoE 模型的并行策略。 |
| KV Cache | Transformer Attention 的 Key-Value 缓存，存储在 GPU 显存中。 |
| Paged Attention | 分页式 KV Cache 管理机制，可减少显存碎片。 |
| Continuous Batching | 连续批处理策略。无需等待整批请求完成，即可添加或移除请求。 |
| Radix Cache | 前缀感知的缓存策略，可在多个请求间共享公共前缀。 |
