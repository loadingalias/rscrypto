#!/usr/bin/env python3
"""Read config/target-matrix.toml and emit shell/json views for CI/scripts."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import sys

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"python >= 3.11 required (tomllib missing): {exc}")


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def load_manifest() -> dict:
    path = repo_root() / "config" / "target-matrix.toml"
    with path.open("rb") as f:
        return tomllib.load(f)


def bash_array(name: str, values: list[str]) -> str:
    quoted = " ".join(shlex.quote(v) for v in values)
    return f"{name}=({quoted})"


def print_shell(manifest: dict) -> None:
    groups = manifest["groups"]
    print("# Generated from config/target-matrix.toml")
    print(bash_array("WIN_TARGETS", groups["win"]))
    print(bash_array("LINUX_TARGETS", groups["linux"]))
    print(bash_array("IBM_TARGETS", groups.get("ibm", [])))
    print(bash_array("NOSTD_TARGETS", groups["no_std"]))
    print(bash_array("WASM_TARGETS", groups["wasm"]))


def get_json_view(manifest: dict, key: str):
    if key == "commit_ci":
        return manifest["ci"]["commit"]
    if key == "commit_no_std":
        return manifest["groups"]["no_std"]
    if key == "commit_wasm":
        return manifest["groups"]["wasm"]
    if key == "tune_arches":
        return manifest["tune"]["arches"]
    raise KeyError(f"unknown json key: {key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["shell", "json"], required=True)
    parser.add_argument("--key", default="")
    args = parser.parse_args()

    manifest = load_manifest()

    if args.format == "shell":
        print_shell(manifest)
        return 0

    if not args.key:
        raise SystemExit("--key is required for --format json")
    try:
        value = get_json_view(manifest, args.key)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    json.dump(value, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
