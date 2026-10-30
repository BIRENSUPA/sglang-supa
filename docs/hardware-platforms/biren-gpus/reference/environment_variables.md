---
title: "环境变量"
description: "SGLang SUPA 环境变量参考"
---

环境变量需在启动 Python 进程前设置。最晚也应在首次导入 `torch`、`torch_supa`
或 `sglang_supa` 之前完成设置。未特别说明时，布尔开关以 `0` 表示关闭，`1` 表示开启。

## 壁仞硬件环境变量

用于控制 SUPA/BR2XX 设备的可见性。

| 变量 | 默认值 | 作用 |
|---|---|---|
| `SUPA_VISIBLE_DEVICES` | 未设置 | 指定当前进程可见的 SUPA 设备，例如 `0` 或 `0,1`。 |

## SGLang SUPA 环境变量

| 变量 | 默认值 | 作用 |
|---|---|---|
| `PATCH_LOG_LEVEL` | `critical` | 设置 SUPA 补丁的日志级别。排障时建议设为 `debug`。 |

## 通用环境变量

下表为常用的上游 SGLang 与 HuggingFace 环境变量。

| 变量 | 默认值 | 作用 |
|---|---|---|
| `HF_TOKEN` | 未设置 | HuggingFace access token，用于下载受限模型。 |
| `HF_ENDPOINT` | `https://huggingface.co` | HuggingFace 镜像站地址。国内环境建议设为 `https://hf-mirror.com`。 |
| `SGLANG_SET_CPU_AFFINITY` | `0` | 设为 `1` 启用 CPU 亲和性绑定。建议在生产环境开启。 |

修改环境变量后，请重新启动 SGLang 进程。运行期间的修改通常不会影响已初始化的组件。
