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


def dudect_case_gate(case: dict, target: str | None = None) -> str:
  """Return the case's gate on `target`; `diagnostic_targets` demotes a required case on named targets only."""
  if target is not None and target in case.get("diagnostic_targets", ()):
    return "diagnostic"
  return str(case.get("gate", "required"))


def dudect_case_reason(case: dict, target: str | None = None) -> str | None:
  if target is not None and target in case.get("diagnostic_targets", ()):
    return case.get("diagnostic_targets_reason")
  return case.get("reason") or case.get("notes")


def resolve_dudect_case(case: dict, target: str | None) -> dict:
  """Copy a manifest case with its gate and reason resolved for `target`."""
  return {**case, "gate": dudect_case_gate(case, target), "reason": dudect_case_reason(case, target)}


def is_diagnostic_dudect_case(case: dict, target: str | None = None) -> bool:
  return dudect_case_gate(case, target) == "diagnostic"


REPLAY_GROUPS = ("mldsa", "mldsa-probe")


def replay_cases(cases: dict[str, dict], selection: str, target: str | None = None) -> list[str]:
  """Resolve an exact case, the required ML-DSA kernel suite on `target`, or the ML-DSA probes."""
  if selection == "mldsa":
    selected = sorted(name for name, case in cases.items()
                      if case.get("primitive") == "signature.mldsa.secret_kernels"
                      and not is_diagnostic_dudect_case(case, target))
    if not selected:
      raise ValueError("ML-DSA replay requires the manifest's required kernel cases")
    return selected
  if selection == "mldsa-probe":
    selected = sorted(name for name, case in cases.items()
                      if name.startswith("mldsa_probe_") and is_diagnostic_dudect_case(case, target))
    if not selected:
      raise ValueError("ML-DSA probe replay requires the manifest's diagnostic probe cases")
    return selected
  if selection not in cases:
    raise ValueError(f"unknown CT diagnostic case: {selection}")
  return [selection]


def required_dudect_cases(ct: dict, target: str | None = None, *, cases: list[dict] | None = None) -> list[dict]:
  primitives = {primitive.get("id", ""): primitive for primitive in ct.get("primitive", [])}
  return [
    case
    for case in (ct.get("dudect_case", []) if cases is None else cases)
    if not is_diagnostic_dudect_case(case, target)
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
