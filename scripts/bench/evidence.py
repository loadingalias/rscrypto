"""Execution-affecting environment evidence shared by measurements and profiles."""

from __future__ import annotations

import os

BUILD_KEYS = {
  "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTDOCFLAGS", "RUSTC", "RUSTC_WRAPPER",
  "RUSTC_WORKSPACE_WRAPPER", "RUSTUP_TOOLCHAIN", "CARGO_HOME", "CC", "CXX", "AR",
  "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "HOST_CC", "HOST_CXX", "HOST_CFLAGS",
  "HOST_CXXFLAGS", "TARGET_CC", "TARGET_CXX", "TARGET_CFLAGS", "TARGET_CXXFLAGS",
}
BUILD_PREFIXES = ("CARGO_BUILD_", "CARGO_PROFILE_", "CARGO_TARGET_", "CC_", "CXX_", "AR_", "CFLAGS_", "CXXFLAGS_")
RUNTIME_KEYS = (
  "RAYON_NUM_THREADS", "RAYON_RS_NUM_CPUS", "RSCRYPTO_FORCE_AVX512",
  "RSCRYPTO_CRC16_CCITT_FORCE", "RSCRYPTO_CRC16_IBM_FORCE", "RSCRYPTO_CRC24_FORCE",
  "RSCRYPTO_CRC32_FORCE", "RSCRYPTO_CRC64_FORCE",
)


def collect(environment=None) -> dict:
  env = os.environ if environment is None else environment
  return {
    "schema": 1,
    "build": {key: value for key, value in sorted(env.items())
              if key in BUILD_KEYS or key.startswith(BUILD_PREFIXES)},
    "runtime": {key: env.get(key) for key in RUNTIME_KEYS},
  }
