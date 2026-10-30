# CHANGELOG

版本格式：`{sglang_version}+sglang-supa_v{major}.{minor}.{patch}`
- **sglang_version** = 当前绑定的上游 sglang release tag
- **supa{X}.{Y}.{Z}** = 插件自身迭代版本（见 `pyproject.toml` `[tool.sglang-supa]`）

详细升级记录见 [`docs/upgrades/`](docs/upgrades/) 目录。

---

## 0.5.12.post1+sglang-supa_v0.1.0 (2026-05-29)

初始版本，迁移至新版本管理规范。基于 sglang v0.5.12.post1，包含 DeepSeek V4 完整适配、jit_kernel deepseek_v4 支持，以及版本管理工具链（setup.py 修复、CI 一致性检查、升级日志体系）。

详见 [docs/upgrades/UPGRADE_0.5.12.post1_sglang-supa_v0.1.0.md](docs/upgrades/UPGRADE_0.5.12.post1_sglang-supa_v0.1.0.md)
