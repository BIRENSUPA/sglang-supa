#!/usr/bin/env bash
set -e

# 1. 更新submodules
git submodule update --init --recursive

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SGLANG_PYPROJECT="$SCRIPT_DIR/3rdparty/sglang/python/pyproject.toml"
SGLANG_USE_WHEEL="${SGLANG_USE_WHEEL:-0}"
SGLANG_WHEEL_URL="sglang==0.5.18"

# 2. patch SGLang: 去掉 Rust gRPC 扩展和上游 NVIDIA 依赖声明
if [[ "$SGLANG_USE_WHEEL" != "1" ]]; then
python3 -c "
import pathlib
p = pathlib.Path('$SGLANG_PYPROJECT')
lines = p.read_text().splitlines(keepends=True)
out, skip = [], False
for line in lines:
    if 'setuptools-rust' in line:
        line = line.replace('\"setuptools-rust>=1.10\", ', '')
    if line.startswith('dependencies = ['):
        out.append('dependencies = []\n')
        skip = not line.rstrip().endswith(']')
        continue
    if skip:
        if line.startswith(']'):
            skip = False
        continue
    if line.startswith('[[tool.setuptools-rust'):
        skip = True
        continue
    out.append(line)
p.write_text(''.join(out))
"
fi

# 3. 安装 SGLang
if [[ "$SGLANG_USE_WHEEL" == "1" ]]; then
    echo "Installing SGLang from wheel: $SGLANG_WHEEL_URL"
    pip install --no-deps "$SGLANG_WHEEL_URL"
else
    echo "Installing SGLang from local checkout: $SCRIPT_DIR/3rdparty/sglang/python/"
    pip install --no-deps -e "$SCRIPT_DIR/3rdparty/sglang/python/"
fi

# 4. 安装 sglang_supa（带全部 SUPA 依赖）
pip install -e "$SCRIPT_DIR[dev,all]"

# 5. 测试supa platform
SUPA_AVAILABLE=1 SGLANG_PLATFORM=supa \
python3 -c "from sglang.srt.platforms import current_platform; print(type(current_platform).__name__, current_platform.device_name)"
