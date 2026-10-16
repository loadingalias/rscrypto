#!/usr/bin/env -S uv run python
"""Query and validate rscrypto's benchmark identity catalog."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / ".config" / "benchmark-matrix.json"


class CatalogError(ValueError):
  pass


def normalize(value: str) -> str:
  return re.sub(r"[^a-z0-9]", "", value.lower())


def load_catalog() -> dict:
  with CATALOG_PATH.open(encoding="utf-8") as source:
    catalog = json.load(source)
  validate_catalog(catalog)
  return catalog


def validate_catalog(catalog: dict) -> None:
  if catalog.get("schema") != 2:
    raise CatalogError("benchmark catalog schema must be 2")

  benches = catalog.get("benches")
  algorithms = catalog.get("algorithms")
  groups = catalog.get("groups")
  if not all(isinstance(value, dict) and value for value in (benches, algorithms, groups)):
    raise CatalogError("benchmark catalog maps must be non-empty objects")

  # Targets, algorithms, and groups share one selector name space; `all` is built in.
  names = {"all": "the built-in all selector"}
  for kind, entries in (("target", benches), ("algorithm", algorithms), ("group", groups)):
    for name in entries:
      key = normalize(name)
      if key in names:
        raise CatalogError(f"{kind} {name} has the same selector name as {names[key]}")
      names[key] = f"{kind} {name}"

  for name, algorithm in algorithms.items():
    bench = algorithm.get("bench")
    if bench not in benches:
      raise CatalogError(f"algorithm {name} references unknown bench {bench}")
    if not isinstance(algorithm.get("filter"), str) or not algorithm["filter"]:
      raise CatalogError(f"algorithm {name} needs a non-empty filter")

  valid_kinds = {"criterion", "gungraun"}
  for name, bench in benches.items():
    if bench.get("kind") not in valid_kinds:
      raise CatalogError(f"bench {name} has invalid kind {bench.get('kind')}")
    binary = bench.get("binary")
    if not isinstance(binary, str) or not binary:
      raise CatalogError(f"bench {name} needs a binary")
    manifest = Path(bench.get("manifest", "Cargo.toml"))
    if manifest.is_absolute() or '..' in manifest.parts or manifest.name != 'Cargo.toml':
      raise CatalogError(f"bench {name} has invalid manifest path")
    if not (ROOT / manifest).is_file():
      raise CatalogError(f"bench {name} references missing {manifest}")
    if not (ROOT / manifest.parent / "benches" / f"{binary}.rs").is_file():
      raise CatalogError(f"bench {name} references missing {manifest.parent}/benches/{binary}.rs")
    cargo = tomllib.loads((ROOT / manifest).read_text())
    targets = {entry['name']: entry for entry in cargo.get('bench', [])}
    if binary not in targets:
      raise CatalogError(f"bench {name} is not declared by {manifest}")
    features = bench.get("features")
    if not isinstance(features, list) or not features or any(not isinstance(item, str) or not item for item in features):
      raise CatalogError(f"bench {name} needs a non-empty feature list")
    if not set(targets[binary].get('required-features', [])) <= set(features):
      raise CatalogError(f"bench {name} omits required features")
    if not isinstance(bench.get("required"), bool):
      raise CatalogError(f"bench {name} needs a Boolean required field")
    external = bench.get("external")
    if external is not None and (not isinstance(external, str) or not external):
      raise CatalogError(f"bench {name} needs a non-empty external prerequisite")
    if external and bench["required"]:
      raise CatalogError(f"bench {name} needs {external}, so default runs cannot include it")

  for group, members in groups.items():
    if (not isinstance(members, list) or not members or len(set(members)) != len(members)
        or any(member not in algorithms and benches.get(member, {}).get("kind") != "criterion" for member in members)):
      raise CatalogError(f"group {group} must list distinct algorithms or Criterion targets")

  presets = catalog.get("profile_presets")
  if not isinstance(presets, dict) or not presets:
    raise CatalogError("profile_presets must be a non-empty object")
  for name, preset in presets.items():
    if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", name):
      raise CatalogError(f"invalid profile preset name: {name}")
    if not isinstance(preset, dict) or set(preset) != {"bench", "case", "diagnostic"}:
      raise CatalogError(f"profile preset {name} must contain bench, case, and diagnostic")
    bench = benches.get(preset.get("bench"))
    if bench is None or bench["kind"] != "criterion":
      raise CatalogError(f"profile preset {name} references an unknown Criterion bench")
    case = preset.get("case")
    if not isinstance(case, str) or not case:
      raise CatalogError(f"profile preset {name} needs one exact case")
    if not isinstance(preset.get("diagnostic"), bool):
      raise CatalogError(f"profile preset {name} needs a Boolean diagnostic field")

  binaries = {bench["binary"] for bench in benches.values()}
  classes = catalog.get("case_classes")
  if not isinstance(classes, dict):
    raise CatalogError("case_classes must be an object")
  for binary, rules in classes.items():
    if binary not in binaries or not isinstance(rules, dict):
      raise CatalogError(f"invalid case classification for {binary}")
    for category, patterns in rules.items():
      if category not in {"diagnostic", "expensive"} or not isinstance(patterns, list) or not patterns:
        raise CatalogError(f"invalid case category for {binary}: {category}")
      for pattern in patterns:
        try:
          re.compile(pattern)
        except (TypeError, re.error) as error:
          raise CatalogError(f"invalid case classification pattern: {pattern}") from error


def case_class(catalog: dict, binary: str, case: str) -> str:
  rules = catalog["case_classes"].get(binary, {})
  for category in ("diagnostic", "expensive"):
    if any(re.search(pattern, case) for pattern in rules.get(category, [])):
      return category
  return "ordinary"


def resolve_selector(catalog: dict, selector: str) -> list[str] | None:
  """Expand one selector to target and algorithm names, or return None if it is unknown."""
  key = normalize(selector)
  if key == "all":
    return [name for name, bench in catalog["benches"].items() if bench["kind"] == "criterion" and bench["required"]]
  for kind in ("benches", "algorithms", "groups"):
    for name, entry in catalog[kind].items():
      if normalize(name) == key:
        return list(entry) if kind == "groups" else [name]
  return None


def scope(catalog: dict, name: str) -> tuple[str, str]:
  """Return the target and the case scope that one target or algorithm name selects."""
  if name in catalog["benches"]:
    return name, ""
  algorithm = catalog["algorithms"][name]
  return algorithm["bench"], algorithm["filter"]
