"""TOML reader compatibility for CT tooling."""

try:
  import tomllib
except ModuleNotFoundError:
  try:
    import tomli as tomllib
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "rscrypto CT scripts require Python 3.11+ for tomllib, "
      "or Python 3.10 with the tomli package installed"
    ) from exc
