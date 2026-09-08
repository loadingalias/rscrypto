#!/usr/bin/env python3
"""Query and validate rscrypto's benchmark identity catalog."""

from __future__ import annotations

import json
import re
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
  if catalog.get("schema") != 1:
    raise CatalogError("benchmark catalog schema must be 1")

  benches = catalog.get("benches")
  algorithms = catalog.get("algorithms")
  selectors = catalog.get("selectors")
  if not all(isinstance(value, dict) and value for value in (benches, algorithms, selectors)):
    raise CatalogError("benchmark catalog maps must be non-empty objects")

  normalized_algorithms: dict[str, str] = {}
  for name, algorithm in algorithms.items():
    key = normalize(name)
    if key in normalized_algorithms:
      raise CatalogError(f"algorithm names normalize to the same selector: {name}, {normalized_algorithms[key]}")
    normalized_algorithms[key] = name
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
    if not (ROOT / "benches" / f"{binary}.rs").is_file():
      raise CatalogError(f"bench {name} references missing benches/{binary}.rs")
    features = bench.get("features")
    if not isinstance(features, list) or not features or any(not isinstance(item, str) or not item for item in features):
      raise CatalogError(f"bench {name} needs a non-empty feature list")
    if not isinstance(bench.get("required"), bool):
      raise CatalogError(f"bench {name} needs a Boolean required field")

  for selector, names in selectors.items():
    if normalize(selector) != selector:
      raise CatalogError(f"selector key must already be normalized: {selector}")
    if not isinstance(names, list) or not names or any(name not in algorithms for name in names):
      raise CatalogError(f"selector {selector} references an unknown algorithm")
    if selector in normalized_algorithms:
      raise CatalogError(f"family selector shadows an exact algorithm: {selector}")

  all_algorithms = set(selectors.get("all", []))
  expected_algorithms = set(algorithms) - {"aead-diag"}
  if all_algorithms != expected_algorithms:
    raise CatalogError("the all selector must contain every non-diagnostic algorithm exactly once")

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
  key = normalize(selector)
  for name in catalog["algorithms"]:
    if normalize(name) == key:
      return [name]
  return catalog["selectors"].get(key)
