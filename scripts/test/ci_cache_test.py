#!/usr/bin/env python3
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
ACTION = "loadingalias/cargo-rail-action"
REVISION = "e706767c343acb6a2ab1917f243b1616ae9eb05b"


def step(steps, name):
  return next(item for item in steps if item.get("name") == name)


workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
jobs = workflow["jobs"]
assert workflow["env"]["CARGO_RAIL_CACHE"] == "off"
called_secrets = workflow["on"]["workflow_call"]["secrets"]
assert set(called_secrets) == {
  "CARGO_RAIL_R2_READ_ACCESS_KEY_ID",
  "CARGO_RAIL_R2_READ_SECRET_ACCESS_KEY",
  "CARGO_RAIL_R2_WRITE_ACCESS_KEY_ID",
  "CARGO_RAIL_R2_WRITE_SECRET_ACCESS_KEY",
}
assert called_secrets["CARGO_RAIL_R2_READ_ACCESS_KEY_ID"]["required"] == "true"
assert called_secrets["CARGO_RAIL_R2_READ_SECRET_ACCESS_KEY"]["required"] == "true"
assert called_secrets["CARGO_RAIL_R2_WRITE_ACCESS_KEY_ID"]["required"] == "false"
assert called_secrets["CARGO_RAIL_R2_WRITE_SECRET_ACCESS_KEY"]["required"] == "false"
native = jobs["native"]
rows = native["strategy"]["matrix"]["include"]
assert {row["platform"] for row in rows if row.get("cache") == "true"} == {
  "x86_64-linux",
  "aarch64-linux",
  "x86_64-win",
}
cross = jobs["cross-build"]
assert set(cross["strategy"]["matrix"]["target"]) == {
  "riscv64gc-unknown-linux-gnu",
  "powerpc64le-unknown-linux-gnu",
  "s390x-unknown-linux-gnu",
}

for name, steps, label in (
  ("cross-build", cross["steps"], "cross-${{ matrix.target }}"),
  ("native", native["steps"], "${{ matrix.platform }}"),
):
  authority = step(steps, "Select shared compiler-cache authority")
  assert "pull_request.head.repo.full_name == github.repository" in authority["if"], name
  if name == "native":
    assert "matrix.cache == true" in authority["if"]
  assert "refs/heads/main" in authority["env"]["CACHE_ACCESS_KEY_ID"], name
  assert "CARGO_RAIL_R2_WRITE_ACCESS_KEY_ID" in authority["env"]["CACHE_ACCESS_KEY_ID"], name
  assert "CARGO_RAIL_R2_READ_ACCESS_KEY_ID" in authority["env"]["CACHE_ACCESS_KEY_ID"], name
  assert "CARGO_RAIL_CACHE=\\n" in authority["run"], name
  assert "AWS_SESSION_TOKEN=\\n" in authority["run"], name
  assert "AWS_WEB_IDENTITY_TOKEN_FILE=\\n" in authority["run"], name
  assert "AWS_EC2_METADATA_DISABLED=true\\n" in authority["run"], name
  assert "enabled=false" in authority["run"] and "enabled=true" in authority["run"], name

  cache = step(steps, "Configure shared compiler cache")
  assert cache["uses"] == f"{ACTION}/cache@{REVISION}", name
  assert cache["if"] == "steps.cache-authority.outputs.enabled == 'true'", name
  assert cache["with"] == {
    "remote": "${{ vars.CARGO_RAIL_CACHE_URL }}",
    "mode": "${{ github.event_name == 'push' && github.ref == 'refs/heads/main' && 'read-write' || 'read' }}",
    "max-size": "10GiB",
    "root-portability": "remap",
    "verify-remote": "true",
  }, name

  collect = step(steps, "Collect compiler-cache measurements")
  assert collect["uses"] == f"{ACTION}/cache/collect@{REVISION}", name
  assert collect["if"] == "always() && steps.cache.outcome == 'success'", name
  assert collect["with"]["job"] == label, name

report = jobs["cache-report"]
assert set(report["needs"]) == {"cross-build", "native"}
report_step = step(report["steps"], "Report compiler-cache results")
assert report_step["uses"] == f"{ACTION}/cache/report@{REVISION}"
assert set(json.loads(report_step["with"]["expected-jobs"])) == {
  "cross-riscv64gc-unknown-linux-gnu",
  "cross-powerpc64le-unknown-linux-gnu",
  "cross-s390x-unknown-linux-gnu",
  "x86_64-linux",
  "aarch64-linux",
  "x86_64-win",
}

release = yaml.load((ROOT / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader)
assert set(release["jobs"]["ci"]["secrets"]) == {
  "CARGO_RAIL_R2_READ_ACCESS_KEY_ID",
  "CARGO_RAIL_R2_READ_SECRET_ACCESS_KEY",
}

for name in ("bench.yml", "ct.yml", "fuzz.yml", "profile.yml", "release.yml"):
  cold = yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)
  assert cold["env"]["CARGO_RAIL_CACHE"] == "off", name

print("CI cache authority regressions passed")
