---
title: "常见问题"
description: "SGLang SUPA 常见问题与故障排查"
---

欢迎贡献更多内容。

## 1. 编译时找不到 CMake、SUDA 或 BRCC

**原因：** 未加载 BIRENSUPA SDK 环境。

**解决方法：**

请先加载 BIRENSUPA SDK 环境，确认 `cmake`、`suda` 和 `brcc` 已在 `PATH` 中：

```bash
source /usr/local/birensupa/all/latest/scripts/brsw_set_env.sh
suda init
. "$HOME/.gstub/suda.sh"
suda load
```

随后检查 TorchSUPA (torch_supa) 版本是否与交付包一致。

## 2. 运行时设备不可见

**错误信息：**

```
RuntimeError: No SUPA devices are available
```

或

```text
>>> import torch; import torch_supa; print(torch.supa.is_available())
False
```

**解决方法：**

1. 检查 [`SUPA_VISIBLE_DEVICES`](reference/environment_variables.md#壁仞硬件环境变量) 环境变量的设置：
   ```bash
   echo $SUPA_VISIBLE_DEVICES
   ```

2. 确认壁仞驱动已加载，且设备可见：
   ```bash
   brsmi
   ```
   若 `brsmi` 不存在或未列出设备，请重新安装壁仞驱动。

3. 容器环境中，请确认设备已正确映射：
   ```bash
   docker run --privileged --device=/dev/biren0 ...
   ```

## 3. 插件未生效

**原因：** 上游 SGLang 与 `sglang_supa` 未安装在同一 Python 环境。

**解决方法：**

请确认上游 SGLang 与 `sglang_supa` 安装在同一 Python 环境，并检查插件入口点：

```bash
python -c "import sglang_supa; print(sglang_supa.__version__)"
```

若补丁未输出日志，请设置 [`PATCH_LOG_LEVEL`](reference/environment_variables.md#sglang-supa-环境变量)`=debug` 后重试。

## 4. 显存不足

**错误信息：**

```
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

**解决方法：**

1. 检查 GPU 显存使用情况：
   ```bash
   brsmi
   ```

2. 若 GPU 已被其他进程占用，请通过 `--base-gpu-id` 指定空闲设备。

3. 通过 `--tp` 将模型分布到多张 GPU 上。

4. 调整 `--mem-fraction-static`，减少 KV Cache 的显存占用。

## 5. BCCL 通信超时

**错误信息：**

```
RuntimeError: BCCL timeout or BCCL error during collective operation
```

**原因：** 多 GPU 通信失败。该问题通常源于网络配置或驱动异常。

**解决方法：**

1. 确认所有 GPU 均可见：
   ```bash
   brsmi
   ```

2. 确认 BCCL 已正确安装。BCCL 包含在 BIRENSUPA 全栈 LKG 中。

3. 适当增加超时时间：
   ```bash
   python -m sglang.launch_server ... --dist-timeout 600
   ```

## 6. 模型下载失败

**原因：** 受网络限制，无法访问 HuggingFace 服务器。

**解决方法：**

- 使用 HuggingFace 镜像站：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

- 配置代理：
  ```bash
  export http_proxy=http://your-proxy:port
  export https_proxy=http://your-proxy:port
  ```

- 手动下载模型，再指定本地路径：
  ```bash
  python -m sglang.launch_server --model-path /path/to/local/model ...
  ```
