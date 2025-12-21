#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# zig-cc.sh - Zig CC wrapper for cross-compilation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Used by check.sh for local cross-target compilation checks.
# Translates Rust target triples to Zig target triples.
#
# Why this exists:
#   - cc-rs passes --target=<rust-triple> to the C compiler
#   - Zig uses different triple format (e.g., x86_64-linux-gnu vs x86_64-unknown-linux-gnu)
#   - This wrapper intercepts the --target flag and translates it for zig
#
# Usage (automatic via CC env var in check.sh):
#   CC="path/to/zig-cc.sh" cargo check --target x86_64-unknown-linux-gnu
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Extract the target from --target= argument, pass everything else through
TARGET=""
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == --target=* ]]; then
    TARGET="${arg#--target=}"
  else
    ARGS+=("$arg")
  fi
done

# Translate Rust target triple to Zig target triple
case "$TARGET" in
  # Linux GNU
  x86_64-unknown-linux-gnu)    ZIG_TARGET="x86_64-linux-gnu" ;;
  aarch64-unknown-linux-gnu)   ZIG_TARGET="aarch64-linux-gnu" ;;

  # Linux MUSL
  x86_64-unknown-linux-musl)   ZIG_TARGET="x86_64-linux-musl" ;;
  aarch64-unknown-linux-musl)  ZIG_TARGET="aarch64-linux-musl" ;;

  # Windows MSVC
  x86_64-pc-windows-msvc)      ZIG_TARGET="x86_64-windows-msvc" ;;
  aarch64-pc-windows-msvc)     ZIG_TARGET="aarch64-windows-msvc" ;;

  # Windows GNU
  x86_64-pc-windows-gnu)       ZIG_TARGET="x86_64-windows-gnu" ;;

  # macOS
  aarch64-apple-darwin)        ZIG_TARGET="aarch64-macos" ;;
  x86_64-apple-darwin)         ZIG_TARGET="x86_64-macos" ;;

  # WASM
  wasm32-unknown-unknown)      ZIG_TARGET="wasm32-freestanding" ;;
  wasm64-unknown-unknown)      ZIG_TARGET="wasm64-freestanding" ;;

  # Embedded / no_std
  thumbv6m-none-eabi)          ZIG_TARGET="thumb-freestanding-eabi" ;;
  aarch64-unknown-none)        ZIG_TARGET="aarch64-freestanding" ;;
  x86_64-unknown-none)         ZIG_TARGET="x86_64-freestanding" ;;
  riscv32imac-unknown-none-elf) ZIG_TARGET="riscv32-freestanding" ;;

  # No target specified - pass through to zig cc directly
  "")
    exec zig cc "${ARGS[@]}"
    ;;

  # Unknown target - fail explicitly
  *)
    echo "zig-cc.sh: Unknown target triple: $TARGET" >&2
    echo "Add mapping to scripts/check/zig-cc.sh if this is a valid target" >&2
    exit 1
    ;;
esac

exec zig cc -target "$ZIG_TARGET" "${ARGS[@]}"
