#!/usr/bin/env bash
# Target definitions for cross-platform checks.

WIN_TARGETS=(
  "x86_64-pc-windows-msvc"
  "aarch64-pc-windows-msvc"
)

LINUX_TARGETS=(
  "x86_64-unknown-linux-gnu"
  "aarch64-unknown-linux-gnu"
  "x86_64-unknown-linux-musl"
  "aarch64-unknown-linux-musl"
)

NOSTD_TARGETS=(
  "thumbv6m-none-eabi"
  "riscv32imac-unknown-none-elf"
  "aarch64-unknown-none"
  "x86_64-unknown-none"
)

WASM_TARGETS=(
  "wasm32-unknown-unknown"
  "wasm32-wasip1"
)
