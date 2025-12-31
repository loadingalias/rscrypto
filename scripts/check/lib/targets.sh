#!/usr/bin/env bash
# Target definitions for cross-platform checks

# Windows targets (via cargo-xwin)
WIN_TARGETS=(
  "x86_64-pc-windows-msvc"
  "aarch64-pc-windows-msvc"
)

# Linux targets (via zig)
LINUX_TARGETS=(
  "x86_64-unknown-linux-gnu"
  "aarch64-unknown-linux-gnu"
  "x86_64-unknown-linux-musl"
  "aarch64-unknown-linux-musl"
)

# Constrained targets (no_std / bare metal)
NOSTD_TARGETS=(
  "thumbv6m-none-eabi"
  "riscv32imac-unknown-none-elf"
  "aarch64-unknown-none"
  "x86_64-unknown-none"
)

# WASM targets
WASM_TARGETS=(
  "wasm32-unknown-unknown"
  "wasm32-wasip1"
  # "wasm64-unknown-unknown"  # Tier 3: requires -Z build-std (nightly), tested in weekly CI
)
