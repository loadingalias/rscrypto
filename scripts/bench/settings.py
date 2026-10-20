"""One Criterion configuration, with invocation-wide overrides."""

from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / ".config/criterion.json"
MAX_RUN_SECONDS = 600
PROFILE_CAPTURE_DEFAULT_SECONDS = 5
PROFILE_CAPTURE_MAX_SECONDS = 15
PROFILE_PREPARE_TIMEOUT_MINUTES = 20
PROFILE_CAPTURE_TIMEOUT_MINUTES = 20


def load(overrides=None) -> dict:
  settings = json.loads(CONFIG.read_text())
  if overrides:
    settings.update(overrides)
  quick = settings.pop("quick", False)
  if type(quick) is not bool:
    raise ValueError("quick must be Boolean")
  if quick and {"sample_size", "warmup_ms"} & set(overrides or {}):
    raise ValueError("quick mode takes no sample_size or warmup_ms override")
  integers = {"sample_size": 10, "warmup_ms": 1, "measure_ms": 1, "nresamples": 1000,
              "max_run_seconds": 1, "max_comparison_seconds": MAX_RUN_SECONDS}
  for key, minimum in integers.items():
    if type(settings[key]) is not int or settings[key] < minimum:
      raise ValueError(f"{CONFIG}: {key} must be an integer >= {minimum}")
  if settings["max_run_seconds"] > MAX_RUN_SECONDS:
    raise ValueError("benchmark run budget cannot exceed 10 minutes")
  for key in ("confidence_level", "significance_level", "noise_threshold"):
    if type(settings[key]) not in (int, float) or not 0 < settings[key] < 1:
      raise ValueError(f"{CONFIG}: {key} must be between zero and one")
  if quick:
    settings.pop("sample_size")
    settings.pop("warmup_ms")
    settings["quick"] = True
  return settings


def arguments(settings: dict) -> list[str]:
  sampling = (["--quick"] if settings.get("quick") else
              ["--warm-up-time", f"{settings['warmup_ms'] / 1000:.3f}", "--sample-size", str(settings["sample_size"])])
  return [*sampling,
          "--measurement-time", f"{settings['measure_ms'] / 1000:.3f}",
          "--nresamples", str(settings["nresamples"]),
          "--confidence-level", str(settings["confidence_level"]),
          "--significance-level", str(settings["significance_level"]),
          "--noise-threshold", str(settings["noise_threshold"]), "--noplot"]
