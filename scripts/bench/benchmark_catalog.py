#!/usr/bin/env python3
"""Query and validate rscrypto's benchmark identity catalog."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / ".config" / "benchmark-matrix.json"


class CatalogError(ValueError):
  pass


def normalize(value: str) -> str:
  return re.sub(r"[^a-z0-9]", "", value.lower())


def csv(values: list[str]) -> str:
  return ",".join(values)


def csv_values(value: str) -> list[str]:
  return [part.strip() for part in value.split(",") if part.strip()]


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
  crates = catalog.get("crates")
  aliases = catalog.get("bench_aliases")
  if not all(isinstance(value, dict) and value for value in (benches, algorithms, selectors, crates, aliases)):
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
    if algorithm.get("crate") not in crates:
      raise CatalogError(f"algorithm {name} references unknown crate {algorithm.get('crate')}")
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

  for crate, defaults in crates.items():
    if not isinstance(defaults, list) or any(name not in benches for name in defaults):
      raise CatalogError(f"crate {crate} references an unknown default bench")

  for selector, names in selectors.items():
    if normalize(selector) != selector:
      raise CatalogError(f"selector key must already be normalized: {selector}")
    if not isinstance(names, list) or not names or any(name not in algorithms for name in names):
      raise CatalogError(f"selector {selector} references an unknown algorithm")

  all_algorithms = set(selectors.get("all", []))
  expected_algorithms = set(algorithms) - {"aead-diag"}
  if all_algorithms != expected_algorithms:
    raise CatalogError("the all selector must contain every non-diagnostic algorithm exactly once")

  for alias, targets in aliases.items():
    if not isinstance(targets, list) or not targets or any(target not in benches for target in targets):
      raise CatalogError(f"bench alias {alias} references an unknown bench")


def resolve_selector(catalog: dict, selector: str) -> list[str] | None:
  key = normalize(selector)
  selected = catalog["selectors"].get(key)
  if selected is not None:
    return selected
  for name in catalog["algorithms"]:
    if normalize(name) == key:
      return [name]
  return None


def merged_features(catalog: dict, benches: list[str]) -> list[str]:
  if not benches:
    return catalog["default_features"]
  merged: list[str] = []
  for name in benches:
    bench = catalog["benches"].get(name)
    if bench is None:
      raise CatalogError(f"unknown bench: {name}")
    for feature in bench["features"]:
      if feature not in merged:
        merged.append(feature)
  return merged


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)
  subparsers.add_parser("validate")

  resolve = subparsers.add_parser("resolve-selector")
  resolve.add_argument("selector")

  plan = subparsers.add_parser("plan-algorithm")
  plan.add_argument("algorithm")
  plan.add_argument("filter", nargs="?", default="")

  plans = subparsers.add_parser("plan-algorithms")
  plans.add_argument("algorithms")
  plans.add_argument("filter", nargs="?", default="")

  crate = subparsers.add_parser("crate-for-algorithm")
  crate.add_argument("algorithm")

  crates = subparsers.add_parser("crates-for-algorithms")
  crates.add_argument("algorithms")

  defaults = subparsers.add_parser("default-benches")
  defaults.add_argument("crate")

  features = subparsers.add_parser("features")
  features.add_argument("benches", nargs="?", default="")

  binary = subparsers.add_parser("binary")
  binary.add_argument("bench")

  expand = subparsers.add_parser("expand-benches")
  expand.add_argument("benches")

  subparsers.add_parser("required-benches")
  subparsers.add_parser("criterion-binaries")

  kind = subparsers.add_parser("require-kind")
  kind.add_argument("bench")
  kind.add_argument("kind")

  args = parser.parse_args()

  try:
    catalog = load_catalog()
    if args.command == "validate":
      print(f"validated {CATALOG_PATH.relative_to(ROOT)}")
    elif args.command == "resolve-selector":
      selected = resolve_selector(catalog, args.selector)
      if selected is None:
        return 3
      print(csv(selected))
    elif args.command == "plan-algorithm":
      algorithm = catalog["algorithms"].get(args.algorithm)
      if algorithm is None:
        raise CatalogError(f"unknown algorithm: {args.algorithm}")
      filter_value = args.filter or algorithm["filter"]
      print(f"{algorithm['crate']}|{algorithm['bench']}|{filter_value}")
    elif args.command == "plan-algorithms":
      for name in csv_values(args.algorithms):
        algorithm = catalog["algorithms"].get(name)
        if algorithm is None:
          raise CatalogError(f"unknown algorithm: {name}")
        filter_value = args.filter or algorithm["filter"]
        print(f"{algorithm['crate']}|{algorithm['bench']}|{filter_value}")
    elif args.command == "crate-for-algorithm":
      algorithm = catalog["algorithms"].get(args.algorithm)
      if algorithm is None:
        raise CatalogError(f"unknown algorithm: {args.algorithm}")
      print(algorithm["crate"])
    elif args.command == "crates-for-algorithms":
      selected: list[str] = []
      for name in csv_values(args.algorithms):
        algorithm = catalog["algorithms"].get(name)
        if algorithm is None:
          raise CatalogError(f"unknown algorithm: {name}")
        if algorithm["crate"] not in selected:
          selected.append(algorithm["crate"])
      print(csv(selected))
    elif args.command == "default-benches":
      print(csv(catalog["crates"].get(args.crate, [])))
    elif args.command == "features":
      print(csv(merged_features(catalog, csv_values(args.benches))))
    elif args.command == "binary":
      bench = catalog["benches"].get(args.bench)
      if bench is None:
        raise CatalogError(f"unknown bench: {args.bench}")
      print(bench["binary"])
    elif args.command == "expand-benches":
      expanded: list[str] = []
      for token in csv_values(args.benches):
        for target in catalog["bench_aliases"].get(token, [token]):
          if target not in expanded:
            expanded.append(target)
      print(csv(expanded))
    elif args.command == "required-benches":
      print(csv([name for name, bench in catalog["benches"].items() if bench["required"]]))
    elif args.command == "criterion-binaries":
      binaries: list[str] = []
      for bench in catalog["benches"].values():
        if bench["kind"] != "criterion":
          continue
        if bench["binary"] not in binaries:
          binaries.append(bench["binary"])
      print(csv(binaries))
    elif args.command == "require-kind":
      bench = catalog["benches"].get(args.bench)
      if bench is None or bench["kind"] != args.kind:
        actual = "missing" if bench is None else bench["kind"]
        raise CatalogError(f"bench {args.bench} has kind {actual}, expected {args.kind}")
    return 0
  except (CatalogError, json.JSONDecodeError) as error:
    print(f"benchmark catalog error: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
