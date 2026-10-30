import os

# The plugin tests target SUPA. Set the platform before importing SGLang's
# lazy platform singleton so quantization modules use the SUPA path.
os.environ.setdefault("SUPA_AVAILABLE", "1")
os.environ.setdefault("SGLANG_PLATFORM", "supa")

from sglang.srt.plugins import load_plugins
from sglang.srt.plugins.hook_registry import HookRegistry

load_plugins()

_PLUGIN_SENTINEL = "sglang.srt.layers.activation._jit_act_supported"
if _PLUGIN_SENTINEL not in HookRegistry._patched:
    raise RuntimeError(
        "The sglang_supa plugin entrypoint did not finish loading. "
        "Install the project with its test dependencies and inspect the "
        "load_plugins() error logged above."
    )
