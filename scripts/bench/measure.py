"""Measure one deduplicated case union per build configuration."""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil

from execution import execute, read_json, unchanged, write_json
import settings


def positive(value) -> bool:
  return type(value) in (int, float) and math.isfinite(value) and value > 0


def estimates(path: Path) -> None:
  data = read_json(path)
  for name in ("mean", "median"):
    estimate = data[name]
    interval = estimate["confidence_interval"]
    values = [estimate["point_estimate"], estimate["standard_error"],
              interval["lower_bound"], interval["upper_bound"], interval["confidence_level"]]
    if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
      raise ValueError(f"invalid estimates: {path}")
    if interval["lower_bound"] > interval["upper_bound"] or not 0 < interval["confidence_level"] < 1:
      raise ValueError(f"invalid confidence interval: {path}")


def measurements(home: Path) -> dict[str, Path]:
  records = {}
  for record in home.glob("**/new/benchmark.json"):
    case = read_json(record)["full_id"]
    if case in records:
      raise ValueError(f"duplicate measurement identity: {case}")
    current = record.parent
    sample = read_json(current / "sample.json")
    iters, times = sample["iters"], sample["times"]
    if len(iters) < 2 or len(iters) != len(times) or not all(map(positive, [*iters, *times])):
      raise ValueError(f"missing or invalid statistical samples for {case!r}")
    estimates(current / "estimates.json")
    if not positive(read_json(current / "estimates.json")["mean"]["point_estimate"]):
      raise ValueError(f"invalid mean estimate for {case!r}")
    for name in ("benchmark.json", "sample.json", "estimates.json"):
      if (current / name).read_bytes() != (current.parent / "base" / name).read_bytes():
        raise ValueError(f"baseline was not updated for {case!r}: {name}")
    records[case] = current
  return records


def verify(root: Path) -> dict:
  plan = read_json(root / "plan.json")
  if not plan:
    raise ValueError("no configurations selected")
  result = {}
  for entry in plan:
    home = root / entry["home"]
    if entry["home"] in result or not home.resolve().is_relative_to((root / "criterion").resolve()):
      raise ValueError("invalid or duplicate measurement directory")
    records = measurements(home)
    if set(records) != set(entry["cases"]) or len(entry["cases"]) != len(records):
      raise ValueError("measurement artifacts do not match the execution plan")
    for case, current in records.items():
      if len(read_json(current / "sample.json")["iters"]) != entry["settings"]["sample_size"]:
        raise ValueError(f"observed sample count differs from effective settings: {case}")
      if case in entry["baselines"]:
        estimates(current.parent / "change/estimates.json")
    result[entry["home"]] = records
  if len(list((root / "criterion").glob("**/new/benchmark.json"))) != sum(map(len, result.values())):
    raise ValueError("unplanned measurement artifacts")
  return result


def measure(root: Path, plan: list[dict], baseline: Path | None) -> None:
  nominal = sum(len(entry["cases"]) * (entry["settings"]["warmup_ms"] + entry["settings"]["measure_ms"]) / 1000
                for entry in plan)
  budget = settings.load()["max_run_seconds"]
  if nominal >= budget - min(5, budget / 10):
    raise ValueError("requested sampling windows exceed the run budget; narrow the selection")
  if baseline:
    previous = verify(baseline)
    for entry in plan:
      for case in entry["cases"]:
        source = previous.get(entry["home"], {}).get(case)
        if source:
          relative = source.parent.relative_to(baseline / entry["home"])
          shutil.copytree(source.parent / "base", root / entry["home"] / relative / "base")
          entry["baselines"].append(case)
    if not any(entry["baselines"] for entry in plan):
      raise ValueError("selected baseline has no matching case/build configurations")
  write_json(root / "plan.json", plan)
  for entry in plan:
    unchanged(entry["artifact"])
    home = root / entry["home"]
    if list(home.glob("**/new/*")):
      raise ValueError(f"measurement directory is not fresh: {home}")
    cases = root / (entry["id"] + ".cases.json")
    write_json(cases, entry["cases"])
    command = [entry["artifact"]["path"], "--bench", *settings.arguments(entry["settings"])]
    entry["command"] = command
    write_json(root / "plan.json", plan)
    execute(command, root / "output.txt", env=dict(os.environ) | {
      "CRITERION_HOME": str(home), "RSCRYPTO_BENCH_CASES": str(cases),
    })
    unchanged(entry["artifact"])
