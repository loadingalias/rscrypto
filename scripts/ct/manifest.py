"""Shared CT target and measurement selection from ct.toml."""

def binsec_required_targets(ct: dict) -> set[str]:
  return {
    target.get("name", "")
    for target in ct.get("target", [])
    if target.get("claim") in {"ct-intended", "ct-claimed"} and target.get("binsec") == "required"
  }


def binsec_kernel_targets(ct: dict, kernel: dict) -> set[str]:
  targets = kernel.get("targets", [])
  if "*" in targets:
    return binsec_required_targets(ct)
  return set(targets)


def primitive_supports_physical_timing(primitive: dict, target: str | None) -> bool:
  if target is None:
    return True
  return target not in set(primitive.get("physical_timing_unsupported_targets", []))


def is_diagnostic_dudect_case(case: dict) -> bool:
  return case.get("gate") == "diagnostic"


def replay_cases(cases: dict[str, dict], selection: str) -> list[str]:
  """Resolve an exact case or the complete required ML-DSA kernel suite."""
  if selection == "mldsa":
    selected = sorted(name for name, case in cases.items()
                      if case.get("primitive") == "signature.mldsa.secret_kernels"
                      and not is_diagnostic_dudect_case(case))
    if not selected:
      raise ValueError("ML-DSA replay requires the manifest's required kernel cases")
    return selected
  if selection not in cases:
    raise ValueError(f"unknown CT diagnostic case: {selection}")
  return [selection]


def required_dudect_cases(ct: dict, target: str | None = None, *, cases: list[dict] | None = None) -> list[dict]:
  primitives = {primitive.get("id", ""): primitive for primitive in ct.get("primitive", [])}
  return [
    case
    for case in (ct.get("dudect_case", []) if cases is None else cases)
    if not is_diagnostic_dudect_case(case)
    and primitive_supports_physical_timing(primitives.get(case.get("primitive"), {}), target)
  ]


def target_record(ct: dict, target: str) -> dict | None:
  for row in ct.get("target", []):
    if row.get("name") == target:
      return row
  return None


def dudect_sample_count(case, *, smoke=False, override=None, fallback=20000):
  value = override if override is not None else case["smoke_samples"] if smoke else case.get("samples", fallback)
  if isinstance(value, bool) or not isinstance(value, int) or value < 2:
    raise ValueError("DudeCT sample budget must be an integer of at least two")
  return value
