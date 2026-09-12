"""One Criterion configuration, with invocation-wide overrides."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / ".config/criterion.json"


def load(overrides=None) -> dict:
  settings = json.loads(CONFIG.read_text())
  if overrides:
    settings.update(overrides)
  integers = {"sample_size": 10, "warmup_ms": 1, "measure_ms": 1, "nresamples": 1000, "max_run_seconds": 1}
  for key, minimum in integers.items():
    if type(settings[key]) is not int or settings[key] < minimum:
      raise ValueError(f"{CONFIG}: {key} must be an integer >= {minimum}")
  if settings["max_run_seconds"] > 5400:
    raise ValueError("benchmark run budget cannot exceed 90 minutes")
  for key in ("confidence_level", "significance_level", "noise_threshold"):
    if type(settings[key]) not in (int, float) or not 0 < settings[key] < 1:
      raise ValueError(f"{CONFIG}: {key} must be between zero and one")
  return settings


def arguments(settings: dict) -> list[str]:
  return ["--warm-up-time", f"{settings['warmup_ms'] / 1000:.3f}",
          "--measurement-time", f"{settings['measure_ms'] / 1000:.3f}",
          "--sample-size", str(settings["sample_size"]), "--nresamples", str(settings["nresamples"]),
          "--confidence-level", str(settings["confidence_level"]),
          "--significance-level", str(settings["significance_level"]),
          "--noise-threshold", str(settings["noise_threshold"]), "--noplot"]
