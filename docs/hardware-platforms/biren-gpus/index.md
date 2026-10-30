---
title: SGLang SUPA 用户指南
description: 壁仞 BR2XX GPU 上的 SGLang 推理引擎文档
---

# SGLang SUPA 用户指南

SGLang SUPA 是面向壁仞 BR2XX 系列 GPU 的 SGLang 插件。借助该插件，SGLang 框架可直接在壁仞 GPU 上运行，无需侵入式修改上游 SGLang 源码。

| 文档信息 | |
|---|---|
| 适用版本 | SGLang 0.5.20 / PyTorch 2.12.0 |
| 更新日期 | 2026-09-23 |

## 快速链接

- **[安装指南](getting-started/installation.md)** — 环境准备与安装步骤
- **[快速开始](getting-started/quick_start.md)** — 快速启动推理服务
- **[支持的模型](reference/support_models.md)** — 已在壁仞 GPU 上验证的模型列表
- **[支持的功能](reference/support_features.md)** — 功能特性矩阵与服务器参数说明
- **[环境变量](reference/environment_variables.md)** — SUPA 相关环境变量参考
- **[术语表](reference/glossary.md)** — 文档中术语的统一说明
- **[常见问题](faq.md)** — 常见问题与故障排查

## 概述

SGLang SUPA 支持在壁仞 GPU 上运行主流开源大语言模型。主要特性如下：

- **高性能 LLM 推理**：在壁仞 BR2XX 系列 GPU 上执行推理计算。
- **PyTorch PrivateUse1 集成**：通过 TorchSUPA (torch_supa) 提供设备支持。
- **BCCL 分布式通信**：支持多 GPU 张量并行推理。
- **OpenAI 兼容接口**：可直接替换现有部署中的服务接口。
