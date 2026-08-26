#!/usr/bin/env python3
"""Validate features that must remain explicit, standalone capabilities."""

from __future__ import annotations

import pathlib
import sys
import tomllib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPLICIT_ONLY_FEATURES = ("websocket-sha1",)


def local_feature(reference: str, features: dict[str, list[str]]) -> str | None:
    if reference.startswith("dep:"):
        return None
    name = reference.split("/", maxsplit=1)[0].removesuffix("?")
    return name if name in features else None


def reaches(
    source: str,
    target: str,
    features: dict[str, list[str]],
    visited: set[str] | None = None,
) -> bool:
    if source == target:
        return True
    seen = set() if visited is None else visited
    if source in seen:
        return False
    seen.add(source)
    return any(
        dependency is not None and reaches(dependency, target, features, seen)
        for dependency in (local_feature(item, features) for item in features[source])
    )


def main() -> int:
    with (REPO_ROOT / "Cargo.toml").open("rb") as manifest:
        features = tomllib.load(manifest)["features"]

    errors: list[str] = []
    for target in EXPLICIT_ONLY_FEATURES:
        if target not in features:
            errors.append(f"missing explicit-only feature: {target}")
            continue
        if features[target]:
            errors.append(f"{target} must not activate dependencies: {features[target]}")

        for source in features:
            if source != target and reaches(source, target, features):
                errors.append(f"{source} must not activate explicit-only feature {target}")

    if errors:
        print("feature boundary validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("feature boundaries ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
