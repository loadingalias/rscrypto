//! Argon2 BlaMka compression kernels.
//!
//! The compression function `G(X, Y)` per RFC 9106 §3.6 is the wall-clock
//! bottleneck for Argon2. Each arch ships its best kernel through the
//! common [`CompressFn`] contract so the fill engine can call through a
//! function pointer resolved once per platform.
//!
//! This module holds the **portable** kernel. Per-architecture kernels live
//! in sibling files (`aarch64.rs`, `x86_64.rs`, …) and are gated by
//! `#[cfg(target_arch = ...)]`. The runtime dispatcher is [`super::dispatch`].

#![allow(clippy::indexing_slicing)]
// The portable kernel body is generic over the block word count (128) —
// every index is bounds-proven by the fixed `[u64; BLOCK_WORDS]` shape.

use super::{BLOCK_WORDS, P_LANE_WORDS};

/// Architecture-agnostic signature for an Argon2 BlaMka compression kernel.
///
/// Computes `dst = G(x, y)` (when `xor_into == false`) or
/// `dst ^= G(x, y)` (when `xor_into == true`, RFC 9106 v1.3 pass > 0).
///
/// Marked `unsafe fn` because per-arch SIMD kernels require their
/// `target_feature` precondition to hold at the call site. The dispatcher
/// in [`super::dispatch`] only returns a pointer whose required caps are
/// present in `crate::platform::caps()`, so call sites that route through
/// the dispatcher uphold the contract by construction.
pub(super) type CompressFn =
  unsafe fn(dst: &mut [u64; BLOCK_WORDS], x: &[u64; BLOCK_WORDS], y: &[u64; BLOCK_WORDS], xor_into: bool);

// ─── BlaMka primitives (portable) ──────────────────────────────────────────

macro_rules! gb_direct {
  ($a:expr, $b:expr, $c:expr, $d:expr) => {{
    let t1 = ($a & 0xFFFF_FFFFu64).wrapping_mul($b & 0xFFFF_FFFFu64);
    $a = $a.wrapping_add($b).wrapping_add(t1.wrapping_mul(2));
    $d = ($d ^ $a).rotate_right(32);

    let t2 = ($c & 0xFFFF_FFFFu64).wrapping_mul($d & 0xFFFF_FFFFu64);
    $c = $c.wrapping_add($d).wrapping_add(t2.wrapping_mul(2));
    $b = ($b ^ $c).rotate_right(24);

    let t3 = ($a & 0xFFFF_FFFFu64).wrapping_mul($b & 0xFFFF_FFFFu64);
    $a = $a.wrapping_add($b).wrapping_add(t3.wrapping_mul(2));
    $d = ($d ^ $a).rotate_right(16);

    let t4 = ($c & 0xFFFF_FFFFu64).wrapping_mul($d & 0xFFFF_FFFFu64);
    $c = $c.wrapping_add($d).wrapping_add(t4.wrapping_mul(2));
    $b = ($b ^ $c).rotate_right(63);
  }};
}

macro_rules! p_direct {
  (
    $v0:expr, $v1:expr, $v2:expr, $v3:expr,
    $v4:expr, $v5:expr, $v6:expr, $v7:expr,
    $v8:expr, $v9:expr, $v10:expr, $v11:expr,
    $v12:expr, $v13:expr, $v14:expr, $v15:expr $(,)?
  ) => {{
    gb_direct!($v0, $v4, $v8, $v12);
    gb_direct!($v1, $v5, $v9, $v13);
    gb_direct!($v2, $v6, $v10, $v14);
    gb_direct!($v3, $v7, $v11, $v15);
    gb_direct!($v0, $v5, $v10, $v15);
    gb_direct!($v1, $v6, $v11, $v12);
    gb_direct!($v2, $v7, $v8, $v13);
    gb_direct!($v3, $v4, $v9, $v14);
  }};
}

/// Portable Argon2 BlaMka compression.
///
/// Reference implementation per RFC 9106 §3.6. This is the correctness
/// oracle for every per-arch SIMD kernel — the forced-kernel tests in
/// `tests/argon2_vectors.rs` and the sibling kernel files must produce
/// identical byte output for any pair `(x, y)` of input blocks.
///
/// # Safety
///
/// The `unsafe fn` signature is syntactic — this implementation uses no
/// unsafe and has no precondition beyond the fixed-size-array contract.
/// It is marked unsafe only to fit the shared [`CompressFn`] type used by
/// SIMD kernels with `target_feature` preconditions.
#[inline(always)]
pub(super) unsafe fn compress_portable(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  // R = X XOR Y
  let mut r = [0u64; BLOCK_WORDS];
  for i in 0..BLOCK_WORDS {
    r[i] = x[i] ^ y[i];
  }

  let mut q = r;

  // Row pass: apply P to each 16-word row (8 rows of 16 u64s).
  for chunk in q.chunks_exact_mut(P_LANE_WORDS) {
    p_direct!(
      chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7], chunk[8], chunk[9], chunk[10],
      chunk[11], chunk[12], chunk[13], chunk[14], chunk[15],
    );
  }

  // Column pass: each 16-word column is two u64s per row at the same
  // register index.
  for col in 0usize..8 {
    let base = col * 2;
    p_direct!(
      q[base],
      q[base + 1],
      q[base + 16],
      q[base + 17],
      q[base + 32],
      q[base + 33],
      q[base + 48],
      q[base + 49],
      q[base + 64],
      q[base + 65],
      q[base + 80],
      q[base + 81],
      q[base + 96],
      q[base + 97],
      q[base + 112],
      q[base + 113],
    );
  }

  // Final XOR with R, merged into the dst store so we avoid writing the
  // intermediate block twice.
  if xor_into {
    for i in 0..BLOCK_WORDS {
      dst[i] ^= q[i] ^ r[i];
    }
  } else {
    for i in 0..BLOCK_WORDS {
      dst[i] = q[i] ^ r[i];
    }
  }
}
