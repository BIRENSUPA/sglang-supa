#!/usr/bin/env python3
"""Validate the SUPA hook manifest without importing plugin modules."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "hooks" / "hooks_manifest.yaml"
REQUIRED_FIELDS = {
    "id",
    "registration",
    "category",
    "mechanism",
    "targets",
    "purpose",
    "activation_phase",
    "process_scope",
    "import_contract",
    "ordering",
    "failure_policy",
    "idempotence",
    "tests",
    "test_status",
    "temporary",
    "removal_condition",
}
ALLOWED_MECHANISMS = {
    "direct_monkey_patch",
    "hook_registry_around",
    "hook_registry_replace",
    "patch_to",
    "registry_extension",
    "sys_modules_alias",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a mapping")
    return data


def _registration_location(registration: str) -> tuple[Path, str] | None:
    if ":" not in registration:
        return None
    module_name, function_name = registration.split(":", 1)
    if not module_name.startswith("sglang_supa"):
        return None
    relative = Path(*module_name.split("."))
    module_file = REPO_ROOT / relative.with_suffix(".py")
    if not module_file.exists():
        module_file = REPO_ROOT / relative / "__init__.py"
    return module_file, function_name


def _defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def validate_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        data = _load_yaml(path)
    except Exception as error:
        return [f"cannot load {path}: {error}"], warnings

    phases = data.get("activation_phases", {})
    contracts = data.get("import_contract_profiles", {})
    hooks = data.get("hooks")
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(hooks, list):
        return errors + ["hooks must be a list"], warnings
    for profile_name, profile in contracts.items():
        if not isinstance(profile, dict) or not {
            "target_may_be_preimported",
            "stale_binding_strategy",
            "frozen_side_effects",
        }.issubset(profile):
            errors.append(f"import contract profile {profile_name!r} is incomplete")

    seen_ids: set[str] = set()
    function_cache: dict[Path, set[str]] = {}
    for index, hook in enumerate(hooks):
        label = f"hooks[{index}]"
        if not isinstance(hook, dict):
            errors.append(f"{label} must be a mapping")
            continue
        hook_id = hook.get("id", label)
        missing = sorted(REQUIRED_FIELDS - hook.keys())
        if missing:
            errors.append(f"{hook_id}: missing fields: {', '.join(missing)}")
        if hook_id in seen_ids:
            errors.append(f"{hook_id}: duplicate id")
        seen_ids.add(hook_id)

        if hook.get("mechanism") not in ALLOWED_MECHANISMS:
            errors.append(f"{hook_id}: unsupported mechanism {hook.get('mechanism')!r}")
        if hook.get("activation_phase") not in phases:
            errors.append(f"{hook_id}: unknown activation_phase")
        if hook.get("import_contract") not in contracts:
            errors.append(f"{hook_id}: unknown import_contract")
        for text_field in ("purpose", "process_scope", "failure_policy", "test_status"):
            if not isinstance(hook.get(text_field), str) or not hook[text_field].strip():
                errors.append(f"{hook_id}: {text_field} must be a non-empty string")

        ordering = hook.get("ordering")
        if not isinstance(ordering, dict) or not {
            "before",
            "after",
        }.issubset(ordering):
            errors.append(f"{hook_id}: ordering needs before and after lists")
        elif not isinstance(ordering["before"], list) or not isinstance(
            ordering["after"], list
        ):
            errors.append(f"{hook_id}: ordering before/after must be lists")

        targets = hook.get("targets")
        if not isinstance(targets, list) or not targets:
            errors.append(f"{hook_id}: targets must be a non-empty list")
        else:
            for target_index, target in enumerate(targets):
                if not isinstance(target, dict) or not target.get("path"):
                    errors.append(f"{hook_id}: targets[{target_index}] needs path")
                elif "operation" not in target and "hook_type" not in target:
                    errors.append(
                        f"{hook_id}: targets[{target_index}] needs operation or hook_type"
                    )
                elif hook.get("mechanism") == "hook_registry_around" and target.get(
                    "hook_type"
                ) != "AROUND":
                    errors.append(
                        f"{hook_id}: HookRegistry AROUND target needs hook_type=AROUND"
                    )
                elif hook.get("mechanism") == "hook_registry_replace" and target.get(
                    "hook_type"
                ) != "REPLACE":
                    errors.append(
                        f"{hook_id}: HookRegistry REPLACE target needs hook_type=REPLACE"
                    )

        idempotence = hook.get("idempotence")
        if not isinstance(idempotence, dict) or not {
            "load_plugins",
            "direct_registration_call",
        }.issubset(idempotence):
            errors.append(f"{hook_id}: incomplete idempotence contract")
        elif not isinstance(idempotence["load_plugins"], bool) or not isinstance(
            idempotence["direct_registration_call"], bool
        ):
            errors.append(f"{hook_id}: idempotence values must be booleans")

        if not isinstance(hook.get("temporary"), bool):
            errors.append(f"{hook_id}: temporary must be a boolean")
        elif hook["temporary"] and hook.get("removal_condition") in {
            None,
            "",
            "not_applicable",
        }:
            errors.append(f"{hook_id}: temporary hook needs a removal_condition")

        registration = hook.get("registration", "")
        location = _registration_location(registration)
        if location is None:
            errors.append(f"{hook_id}: invalid registration {registration!r}")
        else:
            source_path, function_name = location
            if not source_path.exists():
                errors.append(f"{hook_id}: registration module not found: {source_path}")
            else:
                if source_path not in function_cache:
                    function_cache[source_path] = _defined_functions(source_path)
                functions = function_cache[source_path]
                if function_name not in functions:
                    errors.append(
                        f"{hook_id}: registration function {function_name!r} "
                        f"not found in {source_path.relative_to(REPO_ROOT)}"
                    )

        tests = hook.get("tests")
        if not isinstance(tests, list):
            errors.append(f"{hook_id}: tests must be a list")
        elif not tests:
            warnings.append(f"{hook_id}: no focused test declared")
        else:
            for test in tests:
                if not (REPO_ROOT / test).is_file():
                    errors.append(f"{hook_id}: test does not exist: {test}")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--strict-tests",
        action="store_true",
        help="Treat manifest entries without a focused test as errors.",
    )
    args = parser.parse_args()
    errors, warnings = validate_manifest(args.manifest)
    if args.strict_tests:
        errors.extend(warnings)
        warnings = []
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print(f"Validated {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
