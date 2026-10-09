"""Execution-affecting environment evidence shared by measurements and profiles."""

from __future__ import annotations

import os
import hashlib
from pathlib import Path

BUILD_KEYS = {
  "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTDOCFLAGS", "RUSTC", "RUSTC_WRAPPER",
  "RUSTC_WORKSPACE_WRAPPER", "RUSTUP_TOOLCHAIN", "CARGO_HOME", "CC", "CXX", "AR",
  "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "HOST_CC", "HOST_CXX", "HOST_CFLAGS",
  "HOST_CXXFLAGS", "TARGET_CC", "TARGET_CXX", "TARGET_CFLAGS", "TARGET_CXXFLAGS",
}
BUILD_PREFIXES = ("CARGO_BUILD_", "CARGO_PROFILE_", "CARGO_TARGET_", "CC_", "CXX_", "AR_", "CFLAGS_", "CXXFLAGS_")
RUNTIME_KEYS = (
  "RAYON_NUM_THREADS", "RAYON_RS_NUM_CPUS", "RSCRYPTO_FORCE_AVX512",
  "RSCRYPTO_BENCH_DISABLE_IFMA", "RSCRYPTO_BLAKE3_BENCH_ISA",
  "RSCRYPTO_CRC16_CCITT_FORCE", "RSCRYPTO_CRC16_IBM_FORCE", "RSCRYPTO_CRC24_FORCE",
  "RSCRYPTO_CRC32_FORCE", "RSCRYPTO_CRC64_FORCE", "RSCRYPTO_FUZZ_CORPUS", "RSCRYPTO_TEST_THREADS",
  "NEXTEST_TEST_THREADS",
  "RSCRYPTO_BLAKE3_B3SUM", "RSCRYPTO_BLAKE3_B3SUM_RECEIPT",
)
RUNTIME_FILES = ("RSCRYPTO_BLAKE3_B3SUM", "RSCRYPTO_BLAKE3_B3SUM_RECEIPT")


def collect(environment=None) -> dict:
  env = os.environ if environment is None else environment
  artifacts = {}
  for key in RUNTIME_FILES:
    value = env.get(key)
    if value is None:
      continue
    path = Path(value)
    sha256 = None
    if path.is_file():
      with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    artifacts[key] = {"path": value, "sha256": sha256}
  return {
    "schema": 1,
    "build": {key: value for key, value in sorted(env.items())
              if key in BUILD_KEYS or key.startswith(BUILD_PREFIXES)},
    "runtime": {key: env.get(key) for key in RUNTIME_KEYS},
    "runtime_artifacts": artifacts,
  }
