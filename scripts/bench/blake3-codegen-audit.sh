#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-x86_64-unknown-linux-gnu}"
OUT_DIR="${2:-target/blake3-codegen/${TARGET}}"

mkdir -p "$OUT_DIR"

echo "BLAKE3 codegen audit"
echo "  target: $TARGET"
echo "  out:    $OUT_DIR"

run_asm() {
  local symbol="$1"
  local out_file="$2"
  echo "  - cargo asm: $symbol"
  cargo asm -p hashes --lib --target "$TARGET" "$symbol" --rust > "$OUT_DIR/$out_file"
}

run_asm_or_emit_fallback() {
  local symbol="$1"
  local out_file="$2"
  local fallback_pattern="$3"
  echo "  - cargo asm: $symbol"
  if cargo asm -p hashes --lib --target "$TARGET" "$symbol" --rust > "$OUT_DIR/$out_file" 2>"$OUT_DIR/$out_file.err"; then
    rm -f "$OUT_DIR/$out_file.err"
    return
  fi

  echo "    cargo asm parse failed; falling back to --emit=asm extraction"
  cargo rustc -p hashes --lib --release --target "$TARGET" -- --emit=asm >/dev/null

  local asm_file
  asm_file="$(ls -t target/"$TARGET"/release/deps/hashes-*.s | head -n1)"
  if [[ -z "$asm_file" ]]; then
    echo "No assembly output found for fallback extraction" >&2
    exit 1
  fi

  rg -n "$fallback_pattern" "$asm_file" > "$OUT_DIR/$out_file.index"
  local first_line
  first_line="$(head -n1 "$OUT_DIR/$out_file.index" | cut -d: -f1)"
  if [[ -z "$first_line" ]]; then
    echo "Fallback pattern not found in $asm_file: $fallback_pattern" >&2
    exit 1
  fi

  local start end
  start=$((first_line))
  end=$((first_line + 200))
  sed -n "${start},${end}p" "$asm_file" > "$OUT_DIR/$out_file"
}

run_llvm_lines() {
  local filter="$1"
  local out_file="$2"
  echo "  - cargo llvm-lines: $filter"
  cargo llvm-lines -p hashes --lib --release --target "$TARGET" | rg "$filter" > "$OUT_DIR/$out_file"
}

case "$TARGET" in
  x86_64-unknown-linux-gnu)
    run_asm "hashes::crypto::blake3::digest_one_chunk_root_hash_words_x86" "digest_one_chunk_root_hash_words_x86.s"
    run_asm "hashes::crypto::blake3::single_chunk_output" "single_chunk_output.s"
    run_asm "hashes::crypto::blake3::x86_64::chunk_compress_blocks_sse41" "chunk_compress_blocks_sse41.s"
    run_llvm_lines "blake3::" "llvm-lines-blake3.txt"
    ;;
  s390x-unknown-linux-gnu)
    run_asm "hashes::crypto::blake3::kernels::hash_many_contiguous_s390x_vector_wrapper" \
      "hash_many_contiguous_s390x_vector_wrapper.s"
    run_llvm_lines "blake3::(kernels::hash_many_contiguous_s390x_vector|kernels::hash_one_chunk_unchecked|compress)" \
      "llvm-lines-blake3-hot.txt"
    ;;
  powerpc64le-unknown-linux-gnu)
    run_asm_or_emit_fallback "hashes::crypto::blake3::kernels::hash_many_contiguous_power_vsx_wrapper" \
      "hash_many_contiguous_power_vsx_wrapper.s" \
      "hash_many_contiguous_power_vsx_wrapper"
    run_llvm_lines "blake3::(kernels::hash_many_contiguous_power_vsx|kernels::hash_one_chunk_unchecked|compress)" \
      "llvm-lines-blake3-hot.txt"
    ;;
  riscv64gc-unknown-linux-gnu)
    run_asm "hashes::crypto::blake3::kernels::hash_many_contiguous_riscv_v_wrapper" \
      "hash_many_contiguous_riscv_v_wrapper.s"
    run_llvm_lines "blake3::(kernels::hash_many_contiguous_riscv_v|kernels::hash_one_chunk_unchecked|compress)" \
      "llvm-lines-blake3-hot.txt"
    ;;
  *)
    echo "Unsupported target for this audit script: $TARGET" >&2
    echo "Supported targets: x86_64-unknown-linux-gnu, s390x-unknown-linux-gnu, powerpc64le-unknown-linux-gnu, riscv64gc-unknown-linux-gnu" >&2
    exit 2
    ;;
esac

echo "Done."
echo "Artifacts in $OUT_DIR"
ls -1 "$OUT_DIR"
