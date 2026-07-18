//! scrypt password hashing (RFC 7914).
//!
//! Memory-hard key derivation combining PBKDF2-HMAC-SHA256 with a
//! Salsa20/8 core, via the BlockMix / ROMix construction of
//! Percival & Josefsson (RFC 7914 §3–§5). Parameters are driven by
//! [`ScryptParams`], whose defaults use `log_n = 17`, `r = 8`, and `p = 1`.
//!
//! The implementation reuses [`crate::Pbkdf2Sha256`] for the setup /
//! finalisation legs and is portable Rust throughout.
//!
//! # Examples
//!
//! ```rust
//! use rscrypto::{Scrypt, ScryptParams};
//!
//! let params = ScryptParams::new(10, 8, 1).expect("valid params");
//!
//! let password = b"correct horse battery staple";
//! let salt = b"random-salt-1234";
//!
//! let mut hash = [0u8; 32];
//! Scrypt::derive(&params, password, salt, &mut hash).expect("derive");
//!
//! assert!(Scrypt::verify(&params, password, salt, &hash).is_ok());
//! assert!(Scrypt::verify(&params, b"wrong", salt, &hash).is_err());
//! ```
//!
//! # Security
//!
//! - [`Scrypt::MIN_SALT_LEN`] documents the 16-byte production recommendation. The algorithmic API
//!   accepts any salt length; policy enforcement is the caller's responsibility.
//! - [`ScryptParams::new`] rejects invalid cost profiles; output length is taken from the caller's
//!   destination slice.
//! - Allocation failure surfaces as [`ScryptError::AllocationFailed`] rather than a panic.
//! - Working buffers (B, V, scratch) are zeroised on drop.
//! - [`Scrypt::verify`] is constant-time with respect to the reference tag bytes.
//!
//! # Compliance
//!
//! scrypt is **not FIPS 140-3 approved**. NIST SP 800-132 only covers
//! PBKDF2 for password-based key derivation; deployments under FIPS
//! policy should use [`crate::Pbkdf2Sha256`]. This module is suitable
//! for OWASP-aligned password hashing outside strict FIPS boundaries.
//!
//! Requires `alloc` — the memory matrix (`N · 128 · r` bytes) cannot be
//! stack-allocated. Bare-metal / heap-less targets should select
//! [`crate::Pbkdf2Sha256`] (alloc-free) or the `argon2` / `phc-strings`
//! features under the same caveats.

#![allow(clippy::indexing_slicing)]
// `unwrap_used` applies to slice→array conversions whose lengths are fixed
// by construction (BLOCK_SIZE / BLOCK_WORDS slicing); every site is bounded
// by compile-time constants and cannot fail at runtime.
#![allow(clippy::unwrap_used)]

use alloc::vec::Vec;
use core::fmt;

use super::pbkdf2::Pbkdf2Sha256;
use crate::traits::{VerificationError, ct};

// ─── Constants ──────────────────────────────────────────────────────────────

/// Salsa20/8 block size in bytes.
pub const BLOCK_SIZE: usize = 64;

/// Salsa20/8 block size in 32-bit words.
const BLOCK_WORDS: usize = BLOCK_SIZE / 4; // 16

/// OWASP baseline cost profile: `log_n = 17`, `r = 8`, `p = 1`.
const DEFAULT_LOG_N: u8 = 17;
const DEFAULT_R: u32 = 8;
const DEFAULT_P: u32 = 1;
#[cfg(feature = "phc-strings")]
const DEFAULT_OUTPUT_LEN: usize = 32;

/// Minimum salt length (bytes) recommended for production deployments.
///
/// The scrypt algorithm accepts arbitrary salt lengths; this is a policy
/// constant exposed for callers and not enforced at derivation time (RFC 7914
/// §11 test vectors use empty / short salts).
pub const MIN_SALT_LEN: usize = 16;

/// Minimum output length in bytes per RFC 7914 §2.
pub const MIN_OUTPUT_LEN: usize = 1;

/// Upper bound on `r * p` per RFC 7914 §6.
const MAX_R_TIMES_P: u64 = (1u64 << 30) - 1;

// ─── Error ──────────────────────────────────────────────────────────────────

/// Invalid scrypt parameter, input length, or resource constraint.
///
/// Surfaced at construction or derivation time — never at `verify` time (a parameter
/// error during verification would leak information about the stored hash,
/// so verify collapses these into [`crate::VerificationError`]).
///
/// # Examples
///
/// ```rust
/// use rscrypto::{ScryptParams, auth::scrypt::ScryptError};
///
/// assert_eq!(
///   ScryptParams::new(0, 8, 1).unwrap_err(),
///   ScryptError::InvalidLogN
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ScryptError {
  /// `log_n` must be in `1..=63` (N must be a power of two greater than 1).
  InvalidLogN,
  /// `r` must be at least 1.
  InvalidR,
  /// `p` must be at least 1 and satisfy `r · p ≤ 2^30 − 1`.
  InvalidP,
  /// `output_len` must be at least [`MIN_OUTPUT_LEN`].
  InvalidOutputLen,
  /// The requested working-set size overflows the address space of the
  /// target. Typically means `log_n` is too large for a 32-bit target.
  ResourceOverflow,
  /// The allocator refused to provide the working-set buffers.
  AllocationFailed,
  /// The platform entropy source failed while generating a PHC salt.
  #[cfg(all(feature = "phc-strings", feature = "getrandom"))]
  EntropyUnavailable,
  /// Password generation parameters exceed the verifier's resource limits.
  #[cfg(feature = "phc-strings")]
  VerificationLimitTooLow,
}

impl fmt::Display for ScryptError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::InvalidLogN => "scrypt log_n must be in 1..=63",
      Self::InvalidR => "scrypt r must be at least 1",
      Self::InvalidP => "scrypt p must be at least 1 and satisfy r * p <= 2^30 - 1",
      Self::InvalidOutputLen => "scrypt output length must be at least 1",
      Self::ResourceOverflow => "scrypt parameters exceed the target's address space",
      Self::AllocationFailed => "scrypt working-set allocation failed",
      #[cfg(all(feature = "phc-strings", feature = "getrandom"))]
      Self::EntropyUnavailable => "scrypt entropy source unavailable",
      #[cfg(feature = "phc-strings")]
      Self::VerificationLimitTooLow => "scrypt verification limits do not admit the generation parameters",
    })
  }
}

impl core::error::Error for ScryptError {}

// ─── Parameters ─────────────────────────────────────────────────────────────

/// Validated scrypt cost parameters.
///
/// # Examples
///
/// ```rust
/// use rscrypto::{Scrypt, ScryptParams};
///
/// let params = ScryptParams::new(10, 8, 1)?;
///
/// let mut out = [0u8; 32];
/// Scrypt::derive(&params, b"password", b"salty-salty-salt", &mut out)?;
/// # Ok::<(), rscrypto::ScryptError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScryptParams {
  log_n: u8,
  r: u32,
  p: u32,
}

impl Default for ScryptParams {
  fn default() -> Self {
    Self {
      log_n: DEFAULT_LOG_N,
      r: DEFAULT_R,
      p: DEFAULT_P,
    }
  }
}

impl ScryptParams {
  /// Construct an RFC 7914 parameter profile.
  pub const fn new(log_n: u8, r: u32, p: u32) -> Result<Self, ScryptError> {
    if log_n < 1 || log_n > 63 {
      return Err(ScryptError::InvalidLogN);
    }
    if r < 1 {
      return Err(ScryptError::InvalidR);
    }
    if p < 1 {
      return Err(ScryptError::InvalidP);
    }
    let rp = (r as u64).strict_mul(p as u64);
    if rp > MAX_R_TIMES_P {
      return Err(ScryptError::InvalidP);
    }
    Ok(Self { log_n, r, p })
  }

  /// `log_n` (base-2 log of the CPU/memory cost).
  #[must_use]
  pub const fn get_log_n(&self) -> u8 {
    self.log_n
  }

  /// Block-size parameter `r`.
  #[must_use]
  pub const fn get_r(&self) -> u32 {
    self.r
  }

  /// Parallelisation parameter `p`.
  #[must_use]
  pub const fn get_p(&self) -> u32 {
    self.p
  }
}

/// Finite memory and work ceilings for scrypt password verification.
#[cfg(feature = "phc-strings")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScryptVerificationLimits {
  max_memory_bytes: u128,
  max_work: u128,
}

#[cfg(feature = "phc-strings")]
impl ScryptVerificationLimits {
  /// Derive ceilings from the largest deployment profile the verifier admits.
  #[must_use]
  pub const fn for_profile(params: ScryptParams) -> Self {
    let n = 1u128 << params.log_n;
    let r = params.r as u128;
    let p = params.p as u128;
    Self {
      max_memory_bytes: 128u128
        .strict_mul(r)
        .strict_mul(n.strict_add(p.strict_mul(2)).strict_add(1)),
      max_work: n.strict_mul(r).strict_mul(p),
    }
  }

  const fn allows(&self, params: ScryptParams) -> bool {
    let usage = Self::for_profile(params);
    usage.max_memory_bytes <= self.max_memory_bytes && usage.max_work <= self.max_work
  }
}

#[cfg(feature = "phc-strings")]
impl Default for ScryptVerificationLimits {
  fn default() -> Self {
    Self::for_profile(ScryptParams::default())
  }
}

// ─── Kernel dispatch ───────────────────────────────────────────────────────

/// Salsa20/8 kernel identifier.
///
/// SIMD kernels plug into this enum while the portable kernel remains the
/// reference backend and `portable-only` escape hatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelId {
  /// x86_64 SSE2 Salsa20/8 BlockMix implementation.
  #[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
  X86Sse2,
  /// Pure-Rust portable implementation (always available).
  Portable,
}

impl KernelId {
  /// Kernel name for diagnostics and per-kernel test plumbing.
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      #[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
      Self::X86Sse2 => "x86-sse2",
      Self::Portable => "portable",
    }
  }
}

/// All compiled-in kernels, in preference order.
#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
pub const ALL_KERNELS: &[KernelId] = &[KernelId::X86Sse2, KernelId::Portable];

/// All compiled-in kernels, in preference order.
#[cfg(not(all(target_arch = "x86_64", not(miri), not(feature = "portable-only"))))]
pub const ALL_KERNELS: &[KernelId] = &[KernelId::Portable];

/// Capabilities required for `kernel`.
#[must_use]
pub const fn required_caps(kernel: KernelId) -> crate::platform::Caps {
  match kernel {
    #[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
    KernelId::X86Sse2 => crate::platform::Caps::from_words([0; 4]),
    KernelId::Portable => crate::platform::Caps::from_words([0; 4]),
  }
}

/// Runtime dispatch for the active scrypt BlockMix backend.
#[inline]
#[allow(dead_code)] // Reserved for Phase 4 dispatch; referenced by tests to pin the contract.
fn active_kernel() -> KernelId {
  #[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
  {
    KernelId::X86Sse2
  }
  #[cfg(not(all(target_arch = "x86_64", not(miri), not(feature = "portable-only"))))]
  {
    KernelId::Portable
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
mod x86_sse2 {
  use core::arch::x86_64::*;

  const PIVOT_ABCD: [usize; 16] = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11];
  const INVERSE_PIVOT_ABCD: [usize; 16] = {
    let mut index = [0usize; 16];
    let mut i = 0usize;
    while i < 16 {
      let mut inverse = 0usize;
      while inverse < 16 {
        if PIVOT_ABCD[inverse] == i {
          index[i] = inverse;
          break;
        }
        inverse += 1;
      }
      i += 1;
    }
    index
  };

  #[inline]
  fn shuffle_words(blocks: &mut [u8], pivot: &[usize; 16]) {
    debug_assert_eq!(blocks.len() % super::BLOCK_SIZE, 0);
    for chunk in blocks.chunks_exact_mut(super::BLOCK_SIZE) {
      let mut words = [0u32; super::BLOCK_WORDS];
      for (src, word) in chunk.chunks_exact(4).zip(words.iter_mut()) {
        let arr: [u8; 4] = src.try_into().unwrap();
        *word = u32::from_le_bytes(arr);
      }
      for (i, dst) in chunk.chunks_exact_mut(4).enumerate() {
        dst.copy_from_slice(&words[pivot[i]].to_le_bytes());
      }
    }
  }

  #[inline]
  fn shuffle_in(blocks: &mut [u8]) {
    shuffle_words(blocks, &PIVOT_ABCD);
  }

  #[inline]
  fn shuffle_out(blocks: &mut [u8]) {
    shuffle_words(blocks, &INVERSE_PIVOT_ABCD);
  }

  #[inline]
  fn block_mix(input: &[u8], output: &mut [u8]) {
    debug_assert_eq!(input.len() % 128, 0);
    debug_assert_eq!(input.len(), output.len());
    let last = &input[input.len().strict_sub(super::BLOCK_SIZE)..];

    // SAFETY: Load/store and SSE2 Salsa20/8 rounds because:
    // 1. x86_64 guarantees SSE2 availability for all supported CPUs.
    // 2. `last` is exactly the final 64-byte block, so offsets 0, 16, 32, and 48 are in-bounds.
    // 3. Each loop `chunk` is exactly 64 bytes, so the same four 16-byte loads are in-bounds.
    // 4. `output` length equals `input` length, and `pos` is either an even or odd 64-byte slot inside
    //    it.
    // 5. Unaligned SSE2 loads/stores accept arbitrary byte alignment and `__m128i` has no invalid bit
    //    patterns.
    unsafe {
      macro_rules! rol32x {
        ($v:expr, $amt:literal) => {{
          let v = $v;
          _mm_or_si128(_mm_slli_epi32(v, $amt), _mm_srli_epi32(v, 32 - $amt))
        }};
      }

      let mut a = _mm_loadu_si128(last.as_ptr().cast());
      let mut b = _mm_loadu_si128(last.as_ptr().add(16).cast());
      let mut c = _mm_loadu_si128(last.as_ptr().add(32).cast());
      let mut d = _mm_loadu_si128(last.as_ptr().add(48).cast());

      for (i, chunk) in input.chunks_exact(super::BLOCK_SIZE).enumerate() {
        let pos = if i & 1 == 0 {
          (i >> 1).strict_mul(super::BLOCK_SIZE)
        } else {
          (i >> 1).strict_mul(super::BLOCK_SIZE).strict_add(input.len() >> 1)
        };

        a = _mm_xor_si128(a, _mm_loadu_si128(chunk.as_ptr().cast()));
        b = _mm_xor_si128(b, _mm_loadu_si128(chunk.as_ptr().add(16).cast()));
        c = _mm_xor_si128(c, _mm_loadu_si128(chunk.as_ptr().add(32).cast()));
        d = _mm_xor_si128(d, _mm_loadu_si128(chunk.as_ptr().add(48).cast()));

        let saved_a = a;
        let saved_b = b;
        let saved_c = c;
        let saved_d = d;

        let mut round = 0usize;
        while round < 8 {
          b = _mm_xor_si128(b, rol32x!(_mm_add_epi32(a, d), 7));
          c = _mm_xor_si128(c, rol32x!(_mm_add_epi32(b, a), 9));
          d = _mm_xor_si128(d, rol32x!(_mm_add_epi32(c, b), 13));
          a = _mm_xor_si128(a, rol32x!(_mm_add_epi32(d, c), 18));

          d = _mm_shuffle_epi32(d, 0b00_11_10_01);
          c = _mm_shuffle_epi32(c, 0b01_00_11_10);
          b = _mm_shuffle_epi32(b, 0b10_01_00_11);
          core::mem::swap(&mut b, &mut d);

          round = round.strict_add(1);
        }

        a = _mm_add_epi32(a, saved_a);
        b = _mm_add_epi32(b, saved_b);
        c = _mm_add_epi32(c, saved_c);
        d = _mm_add_epi32(d, saved_d);

        _mm_storeu_si128(output.as_mut_ptr().add(pos).cast(), a);
        _mm_storeu_si128(output.as_mut_ptr().add(pos.strict_add(16)).cast(), b);
        _mm_storeu_si128(output.as_mut_ptr().add(pos.strict_add(32)).cast(), c);
        _mm_storeu_si128(output.as_mut_ptr().add(pos.strict_add(48)).cast(), d);
      }
    }
  }

  #[inline(always)]
  fn integerify_pivot_low64(chunk: &[u8]) -> u64 {
    debug_assert_eq!(chunk.len() % super::BLOCK_SIZE, 0);
    let last = &chunk[chunk.len().strict_sub(super::BLOCK_SIZE)..];
    let lo = u32::from_le_bytes(last[0..4].try_into().unwrap()) as u64;
    let hi = u32::from_le_bytes(last[52..56].try_into().unwrap()) as u64;
    lo | (hi << 32)
  }

  #[inline]
  fn xor_into(x: &[u8], y: &[u8], out: &mut [u8]) {
    for ((o, a), b) in out.iter_mut().zip(x.iter()).zip(y.iter()) {
      *o = *a ^ *b;
    }
  }

  pub(super) fn ro_mix(chunk: &mut [u8], v: &mut [u8], scratch: &mut [u8], n: usize) {
    debug_assert_eq!(chunk.len(), scratch.len());
    debug_assert_eq!(v.len(), n.strict_mul(chunk.len()));
    debug_assert_eq!(n & 1, 0);

    let len = chunk.len();
    shuffle_in(chunk);

    for v_chunk in v.chunks_exact_mut(len) {
      v_chunk.copy_from_slice(chunk);
      block_mix(v_chunk, chunk);
    }

    let n_mask = (n as u64).wrapping_sub(1);
    for _ in 0..n {
      let j = (integerify_pivot_low64(chunk) & n_mask) as usize;
      let v_start = j.strict_mul(len);
      xor_into(chunk, &v[v_start..v_start.strict_add(len)], scratch);
      block_mix(scratch, chunk);
    }

    shuffle_out(chunk);
  }
}

// ─── Salsa20/8 core ─────────────────────────────────────────────────────────

/// A 64-byte (16 × u32) Salsa20/8 block.
#[repr(align(16))]
#[derive(Clone, Copy)]
struct SalsaBlock([u32; BLOCK_WORDS]);

impl SalsaBlock {
  #[inline(always)]
  const fn zero() -> Self {
    Self([0u32; BLOCK_WORDS])
  }
}

/// Salsa20/8 core permutation (RFC 7914 §3).
///
/// Runs 4 double-rounds (column + row) on the 16-word block in place,
/// then adds the original input back word-wise. Additions are `u32`
/// modular-wraparound per the spec.
#[inline(always)]
#[allow(clippy::too_many_lines)]
fn salsa20_8(block: &mut SalsaBlock) {
  let input = block.0;
  let mut y = block.0;

  // 4 iterations = 8 rounds of Salsa20.
  let mut round = 0u32;
  while round < 4 {
    // ── Column round ─────────────────────────────────────────────────
    y[4] ^= y[0].wrapping_add(y[12]).rotate_left(7);
    y[8] ^= y[4].wrapping_add(y[0]).rotate_left(9);
    y[12] ^= y[8].wrapping_add(y[4]).rotate_left(13);
    y[0] ^= y[12].wrapping_add(y[8]).rotate_left(18);

    y[9] ^= y[5].wrapping_add(y[1]).rotate_left(7);
    y[13] ^= y[9].wrapping_add(y[5]).rotate_left(9);
    y[1] ^= y[13].wrapping_add(y[9]).rotate_left(13);
    y[5] ^= y[1].wrapping_add(y[13]).rotate_left(18);

    y[14] ^= y[10].wrapping_add(y[6]).rotate_left(7);
    y[2] ^= y[14].wrapping_add(y[10]).rotate_left(9);
    y[6] ^= y[2].wrapping_add(y[14]).rotate_left(13);
    y[10] ^= y[6].wrapping_add(y[2]).rotate_left(18);

    y[3] ^= y[15].wrapping_add(y[11]).rotate_left(7);
    y[7] ^= y[3].wrapping_add(y[15]).rotate_left(9);
    y[11] ^= y[7].wrapping_add(y[3]).rotate_left(13);
    y[15] ^= y[11].wrapping_add(y[7]).rotate_left(18);

    // ── Row round ────────────────────────────────────────────────────
    y[1] ^= y[0].wrapping_add(y[3]).rotate_left(7);
    y[2] ^= y[1].wrapping_add(y[0]).rotate_left(9);
    y[3] ^= y[2].wrapping_add(y[1]).rotate_left(13);
    y[0] ^= y[3].wrapping_add(y[2]).rotate_left(18);

    y[6] ^= y[5].wrapping_add(y[4]).rotate_left(7);
    y[7] ^= y[6].wrapping_add(y[5]).rotate_left(9);
    y[4] ^= y[7].wrapping_add(y[6]).rotate_left(13);
    y[5] ^= y[4].wrapping_add(y[7]).rotate_left(18);

    y[11] ^= y[10].wrapping_add(y[9]).rotate_left(7);
    y[8] ^= y[11].wrapping_add(y[10]).rotate_left(9);
    y[9] ^= y[8].wrapping_add(y[11]).rotate_left(13);
    y[10] ^= y[9].wrapping_add(y[8]).rotate_left(18);

    y[12] ^= y[15].wrapping_add(y[14]).rotate_left(7);
    y[13] ^= y[12].wrapping_add(y[15]).rotate_left(9);
    y[14] ^= y[13].wrapping_add(y[12]).rotate_left(13);
    y[15] ^= y[14].wrapping_add(y[13]).rotate_left(18);

    round = round.strict_add(1);
  }

  let mut i = 0usize;
  while i < BLOCK_WORDS {
    block.0[i] = input[i].wrapping_add(y[i]);
    i = i.strict_add(1);
  }
}

#[inline(always)]
fn xor_block_into(dst: &mut SalsaBlock, src: &SalsaBlock) {
  for (d, s) in dst.0.iter_mut().zip(src.0.iter()) {
    *d ^= *s;
  }
}

// ─── BlockMix (RFC 7914 §4) ────────────────────────────────────────────────

/// Apply BlockMix to `src` (length `2 · r` blocks) and write the shuffled
/// output into `dst` (same length). `src` is read-only; the caller retains
/// the original buffer.
///
/// ROMix alternates the `(src, dst)` roles between its two working buffers
/// so each BlockMix's output lands in whichever buffer is next in the
/// ping-pong — eliminating the `data.copy_from_slice(scratch)` that a
/// self-contained contract would require on every call (2N × 128·r bytes
/// per ROMix, ~256 MiB at OWASP shape).
///
/// BlockMix chains Salsa20/8 through the 2r input blocks (each initial
/// state is the previous output XOR'd with the next input block), then
/// shuffles the outputs as `(Y_0, Y_2, …, Y_{2r−2}, Y_1, Y_3, …, Y_{2r−1})`.
#[inline]
fn block_mix_into(src: &[SalsaBlock], dst: &mut [SalsaBlock], r: usize) {
  let two_r = r.strict_mul(2);
  debug_assert_eq!(src.len(), two_r);
  debug_assert_eq!(dst.len(), two_r);

  let mut x = src[two_r.strict_sub(1)];
  for (i, block_in) in src.iter().enumerate() {
    xor_block_into(&mut x, block_in);
    salsa20_8(&mut x);
    // Y_i → shuffle: even i → i/2; odd i → r + i/2.
    let out = if i & 1 == 0 { i >> 1 } else { r.strict_add(i >> 1) };
    dst[out] = x;
  }
}

// ─── ROMix (RFC 7914 §5) ────────────────────────────────────────────────────

/// Interpret the first 8 bytes of a 64-byte block as a little-endian u64.
///
/// `Integerify(B[0..2r-1])` per RFC 7914 §5 picks the last block and reads
/// it as a little-endian integer; modulo `N` (a power of two) only the low
/// `log_n` bits survive. With `log_n ≤ 32` the second word is masked away
/// by `n_mask`; for `log_n ∈ 33..=63` both words contribute. Since
/// `integerify_low64` works on native-order `u32`s that were decoded from
/// LE bytes, the result matches the RFC integer on both LE and BE hosts.
#[inline(always)]
fn integerify_low64(block: &SalsaBlock) -> u64 {
  (block.0[0] as u64) | ((block.0[1] as u64) << 32)
}

/// Full ROMix pass (RFC 7914 §5) operating on a `2r`-block chunk.
///
/// Alternates `chunk` and `scratch` as BlockMix's `(src, dst)` pair to
/// avoid a per-iteration 128·r memcpy. `ScryptParams::validate` rejects
/// `log_n = 0`, so `N = 1 << log_n ≥ 2` is always even — the pair-unrolled
/// loops below cover each (even, odd) iteration in one step and guarantee
/// the "live" X lives in `chunk` after every `N` iterations. The second
/// loop reuses the same alignment, so no conditional final copy is needed.
///
/// `v` must have exactly `n · 2r` blocks; `scratch` must have `2r`.
fn ro_mix(chunk: &mut [SalsaBlock], v: &mut [SalsaBlock], scratch: &mut [SalsaBlock], n: usize, r: usize) {
  let two_r = r.strict_mul(2);
  debug_assert_eq!(chunk.len(), two_r);
  debug_assert_eq!(v.len(), n.strict_mul(two_r));
  debug_assert_eq!(scratch.len(), two_r);
  // `n = 1 << log_n` with `log_n ≥ 1` (enforced by `ScryptParams::validate`),
  // so `n` is always a positive power of two ≥ 2 — always even. The
  // pair-unrolled loops below rely on this invariant.
  debug_assert_eq!(n & 1, 0, "n must be even (log_n ≥ 1)");

  let pairs = n >> 1;

  // First loop: V_i ← X ; X ← BlockMix(X). Each pair does one
  // (chunk → scratch) then one (scratch → chunk), covering two ROMix
  // iterations without any memcpy. `n / 2` pairs = `n` iterations; X
  // ends in `chunk`.
  let mut v_off = 0usize;
  for _ in 0..pairs {
    v[v_off..v_off.strict_add(two_r)].copy_from_slice(chunk);
    block_mix_into(chunk, scratch, r);
    v_off = v_off.strict_add(two_r);

    v[v_off..v_off.strict_add(two_r)].copy_from_slice(scratch);
    block_mix_into(scratch, chunk, r);
    v_off = v_off.strict_add(two_r);
  }

  // Second loop: X ← BlockMix(X XOR V[Integerify(X) mod N]). Same
  // ping-pong; each pair XORs V[j] into the current X-buffer before
  // BlockMix targets the other buffer. X ends in `chunk`, which the
  // caller re-serialises for the final PBKDF2 leg.
  let n_mask = (n as u64).wrapping_sub(1);
  for _ in 0..pairs {
    // Even iteration: X lives in `chunk`; write into `scratch`.
    let j = (integerify_low64(&chunk[two_r.strict_sub(1)]) & n_mask) as usize;
    let v_off = j.strict_mul(two_r);
    for k in 0..two_r {
      xor_block_into(&mut chunk[k], &v[v_off.strict_add(k)]);
    }
    block_mix_into(chunk, scratch, r);

    // Odd iteration: X lives in `scratch`; write into `chunk`.
    let j = (integerify_low64(&scratch[two_r.strict_sub(1)]) & n_mask) as usize;
    let v_off = j.strict_mul(two_r);
    for k in 0..two_r {
      xor_block_into(&mut scratch[k], &v[v_off.strict_add(k)]);
    }
    block_mix_into(scratch, chunk, r);
  }
}

// ─── Zeroisation helpers ────────────────────────────────────────────────────

#[inline]
fn zeroize_u32_slice_no_fence(words: &mut [u32]) {
  let mut chunks = words.chunks_exact_mut(16);
  for chunk in &mut chunks {
    // SAFETY: chunk has exactly 16 initialized u32s and [u32; 16] has the
    // same alignment requirement as u32.
    unsafe { core::ptr::write_volatile(chunk.as_mut_ptr().cast::<[u32; 16]>(), [0u32; 16]) };
  }
  for w in chunks.into_remainder() {
    // SAFETY: w is a valid, aligned, dereferenceable pointer to initialized u32.
    unsafe { core::ptr::write_volatile(w, 0) };
  }
}

#[inline]
fn zeroize_blocks_no_fence(blocks: &mut [SalsaBlock]) {
  for block in blocks {
    zeroize_u32_slice_no_fence(&mut block.0);
  }
}

/// Zeroising working set. Holds every buffer allocated during a single
/// scrypt call so `Drop` wipes them on every exit path.
struct ScryptState {
  b_bytes: Vec<u8>,
  b_u32: Vec<SalsaBlock>,
  v: Vec<SalsaBlock>,
  scratch: Vec<SalsaBlock>,
}

impl ScryptState {
  fn new(total_b_blocks: usize, v_blocks: usize, scratch_blocks: usize) -> Result<Self, ScryptError> {
    let b_bytes_len = total_b_blocks
      .checked_mul(BLOCK_SIZE)
      .ok_or(ScryptError::ResourceOverflow)?;

    let b_bytes = alloc_u8_vec(b_bytes_len)?;
    let b_u32 = alloc_block_vec(total_b_blocks)?;
    let v = alloc_block_vec(v_blocks)?;
    let scratch = alloc_block_vec(scratch_blocks)?;

    Ok(Self {
      b_bytes,
      b_u32,
      v,
      scratch,
    })
  }
}

impl Drop for ScryptState {
  fn drop(&mut self) {
    ct::zeroize_no_fence(&mut self.b_bytes);
    zeroize_blocks_no_fence(&mut self.b_u32);
    zeroize_blocks_no_fence(&mut self.v);
    zeroize_blocks_no_fence(&mut self.scratch);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
struct ScryptByteState {
  b: Vec<u8>,
  v: Vec<u8>,
  scratch: Vec<u8>,
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
impl ScryptByteState {
  fn new(b_len: usize, v_len: usize, scratch_len: usize) -> Result<Self, ScryptError> {
    Ok(Self {
      b: alloc_u8_vec(b_len)?,
      v: alloc_u8_vec(v_len)?,
      scratch: alloc_u8_vec(scratch_len)?,
    })
  }
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
impl Drop for ScryptByteState {
  fn drop(&mut self) {
    ct::zeroize_no_fence(&mut self.b);
    ct::zeroize_no_fence(&mut self.v);
    ct::zeroize_no_fence(&mut self.scratch);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

fn alloc_u8_vec(len: usize) -> Result<Vec<u8>, ScryptError> {
  let mut v: Vec<u8> = Vec::new();
  v.try_reserve_exact(len).map_err(|_| ScryptError::AllocationFailed)?;
  v.resize(len, 0);
  Ok(v)
}

fn alloc_block_vec(len: usize) -> Result<Vec<SalsaBlock>, ScryptError> {
  let mut v: Vec<SalsaBlock> = Vec::new();
  v.try_reserve_exact(len).map_err(|_| ScryptError::AllocationFailed)?;
  v.resize(len, SalsaBlock::zero());
  Ok(v)
}

// ─── Full scrypt ────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct ScryptShape {
  n: usize,
  r: usize,
  p: usize,
  two_r: usize,
  total_b_blocks: usize,
  v_blocks: usize,
}

#[inline]
fn scrypt_shape(params: &ScryptParams) -> Result<ScryptShape, ScryptError> {
  let log_n = params.log_n as u32;
  // Guard against the 32-bit target where `1 << log_n` would wrap.
  if log_n >= (usize::BITS) {
    return Err(ScryptError::ResourceOverflow);
  }
  let n: usize = 1usize.checked_shl(log_n).ok_or(ScryptError::ResourceOverflow)?;
  let r: usize = params.r as usize;
  let p: usize = params.p as usize;

  let two_r = r.checked_mul(2).ok_or(ScryptError::ResourceOverflow)?;
  let total_b_blocks = p.checked_mul(two_r).ok_or(ScryptError::ResourceOverflow)?;
  let v_blocks = n.checked_mul(two_r).ok_or(ScryptError::ResourceOverflow)?;

  let b_bytes = total_b_blocks
    .checked_mul(BLOCK_SIZE)
    .ok_or(ScryptError::ResourceOverflow)?;
  let v_bytes = v_blocks.checked_mul(BLOCK_SIZE).ok_or(ScryptError::ResourceOverflow)?;
  let scratch_bytes = two_r.checked_mul(BLOCK_SIZE).ok_or(ScryptError::ResourceOverflow)?;
  let portable_bytes = b_bytes
    .checked_mul(2)
    .and_then(|bytes| bytes.checked_add(v_bytes))
    .and_then(|bytes| bytes.checked_add(scratch_bytes))
    .ok_or(ScryptError::ResourceOverflow)?;
  if portable_bytes > isize::MAX as usize {
    return Err(ScryptError::ResourceOverflow);
  }

  Ok(ScryptShape {
    n,
    r,
    p,
    two_r,
    total_b_blocks,
    v_blocks,
  })
}

fn scrypt_hash_portable(
  params: &ScryptParams,
  password: &[u8],
  salt: &[u8],
  out: &mut [u8],
) -> Result<(), ScryptError> {
  if out.len() < MIN_OUTPUT_LEN {
    return Err(ScryptError::InvalidOutputLen);
  }
  let shape = scrypt_shape(params)?;
  let mut state = ScryptState::new(shape.total_b_blocks, shape.v_blocks, shape.two_r)?;

  // Pre-compute the HMAC prefix state from `password` once; both PBKDF2
  // legs use the same key and `Pbkdf2Sha256::new` hashes the password
  // plus runs the inner/outer compress eagerly. Reusing the state saves
  // one password hash and two compress calls on the second leg.
  let prf = Pbkdf2Sha256::new(password);

  // Step 1: (B_0 || … || B_{p-1}) ← PBKDF2-HMAC-SHA256(P, S, 1, p·128·r).
  prf
    .derive(salt, 1, &mut state.b_bytes)
    .map_err(|_| ScryptError::InvalidOutputLen)?;

  // Decode byte form into little-endian u32 blocks.
  for (block, chunk) in state.b_u32.iter_mut().zip(state.b_bytes.chunks_exact(BLOCK_SIZE)) {
    for (word, bytes) in block.0.iter_mut().zip(chunk.chunks_exact(4)) {
      let arr: [u8; 4] = bytes.try_into().unwrap();
      *word = u32::from_le_bytes(arr);
    }
  }

  // Step 2: for each p-chunk, apply ROMix.
  for chunk_idx in 0..shape.p {
    let chunk_start = chunk_idx.strict_mul(shape.two_r);
    let chunk_end = chunk_start.strict_add(shape.two_r);
    let chunk = &mut state.b_u32[chunk_start..chunk_end];
    ro_mix(chunk, &mut state.v, &mut state.scratch, shape.n, shape.r);
  }

  // Re-serialise the mixed B back into the byte buffer for the final
  // PBKDF2 leg (the spec treats B as a byte string at this point).
  for (block, chunk) in state.b_u32.iter().zip(state.b_bytes.chunks_exact_mut(BLOCK_SIZE)) {
    for (word, bytes) in block.0.iter().zip(chunk.chunks_exact_mut(4)) {
      bytes.copy_from_slice(&word.to_le_bytes());
    }
  }

  // Step 3: DK ← PBKDF2-HMAC-SHA256(P, B, 1, dkLen).
  prf
    .derive(&state.b_bytes, 1, out)
    .map_err(|_| ScryptError::InvalidOutputLen)?;

  // `state` wipes every working buffer on drop; `prf` zeroises its HMAC
  // prefix state on drop per `Pbkdf2Sha256::Drop`.
  Ok(())
}

#[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
fn scrypt_hash_x86_sse2(
  params: &ScryptParams,
  password: &[u8],
  salt: &[u8],
  out: &mut [u8],
) -> Result<(), ScryptError> {
  if out.len() < MIN_OUTPUT_LEN {
    return Err(ScryptError::InvalidOutputLen);
  }
  let shape = scrypt_shape(params)?;
  let r128 = shape.r.checked_mul(128).ok_or(ScryptError::ResourceOverflow)?;
  let b_len = shape.p.checked_mul(r128).ok_or(ScryptError::ResourceOverflow)?;
  let v_len = shape.n.checked_mul(r128).ok_or(ScryptError::ResourceOverflow)?;
  let mut state = ScryptByteState::new(b_len, v_len, r128)?;

  let prf = Pbkdf2Sha256::new(password);

  prf
    .derive(salt, 1, &mut state.b)
    .map_err(|_| ScryptError::InvalidOutputLen)?;

  for chunk in state.b.chunks_exact_mut(r128) {
    x86_sse2::ro_mix(chunk, &mut state.v, &mut state.scratch, shape.n);
  }

  prf
    .derive(&state.b, 1, out)
    .map_err(|_| ScryptError::InvalidOutputLen)?;

  Ok(())
}

fn scrypt_hash(params: &ScryptParams, password: &[u8], salt: &[u8], out: &mut [u8]) -> Result<(), ScryptError> {
  match active_kernel() {
    #[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
    KernelId::X86Sse2 => scrypt_hash_x86_sse2(params, password, salt, out),
    KernelId::Portable => scrypt_hash_portable(params, password, salt, out),
  }
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// scrypt password-hashing (RFC 7914).
///
/// Provides raw derivation and constant-time verification. Use
/// `ScryptPassword` (feature `phc-strings`) for generated, bounded PHC password records.
///
/// # Examples
///
/// ```rust
/// use rscrypto::{Scrypt, ScryptParams};
///
/// // Small CI-friendly params — production deployments should use
/// let params = ScryptParams::new(10, 8, 1).unwrap();
///
/// let mut hash = [0u8; 32];
/// Scrypt::derive(&params, b"password", b"random-salt-1234", &mut hash).unwrap();
/// assert!(Scrypt::verify(&params, b"password", b"random-salt-1234", &hash).is_ok());
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Scrypt;

impl Scrypt {
  /// Algorithm identifier used in PHC strings and diagnostics.
  pub const ALGORITHM: &'static str = "scrypt";

  /// Minimum salt length (bytes) recommended for production deployments.
  /// The algorithmic path does not enforce this; see [`MIN_SALT_LEN`].
  pub const MIN_SALT_LEN: usize = MIN_SALT_LEN;

  /// Minimum output length (bytes) accepted by the hasher.
  pub const MIN_OUTPUT_LEN: usize = MIN_OUTPUT_LEN;

  /// Derive bytes from `password` and `salt` into `out`.
  ///
  /// # Errors
  ///
  /// Returns [`ScryptError`] if the output length, resource shape, or
  /// working-set allocation is invalid.
  pub fn derive(params: &ScryptParams, password: &[u8], salt: &[u8], out: &mut [u8]) -> Result<(), ScryptError> {
    scrypt_hash(params, password, salt, out)
  }

  /// Verify `expected` against a freshly-computed hash in constant time.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] on any mismatch, malformed
  /// input, or parameter error.
  #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
  pub fn verify(params: &ScryptParams, password: &[u8], salt: &[u8], expected: &[u8]) -> Result<(), VerificationError> {
    if expected.len() < MIN_OUTPUT_LEN {
      return Err(VerificationError::new());
    }
    let mut actual = Vec::new();
    actual
      .try_reserve_exact(expected.len())
      .map_err(|_| VerificationError::new())?;
    actual.resize(expected.len(), 0);
    let hash_failed = Self::derive(params, password, salt, &mut actual).is_err();

    let bytes_match = ct::constant_time_eq(&actual, expected);
    ct::zeroize(&mut actual);

    let success = !hash_failed & bytes_match;
    if core::hint::black_box(success) {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }
}

/// scrypt password generation and bounded PHC verification.
#[cfg(feature = "phc-strings")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScryptPassword {
  generation: ScryptParams,
  limits: ScryptVerificationLimits,
}

#[cfg(feature = "phc-strings")]
impl Default for ScryptPassword {
  fn default() -> Self {
    let generation = ScryptParams::default();
    Self {
      generation,
      limits: ScryptVerificationLimits::for_profile(generation),
    }
  }
}

#[cfg(feature = "phc-strings")]
impl ScryptPassword {
  /// Use `generation` for new hashes and derive matching verification limits.
  ///
  /// # Errors
  ///
  /// Returns [`ScryptError::ResourceOverflow`] if the profile cannot fit the
  /// target address space.
  pub fn new(generation: ScryptParams) -> Result<Self, ScryptError> {
    scrypt_shape(&generation)?;
    Ok(Self {
      generation,
      limits: ScryptVerificationLimits::for_profile(generation),
    })
  }

  /// Use distinct generation parameters and finite verification limits.
  pub fn with_limits(generation: ScryptParams, limits: ScryptVerificationLimits) -> Result<Self, ScryptError> {
    scrypt_shape(&generation)?;
    if !limits.allows(generation) {
      return Err(ScryptError::VerificationLimitTooLow);
    }
    Ok(Self { generation, limits })
  }

  /// Hash a password with an OS-generated salt and canonical scrypt PHC encoding.
  #[cfg(all(feature = "phc-strings", feature = "getrandom"))]
  pub fn hash_password(&self, password: &[u8]) -> Result<alloc::string::String, ScryptError> {
    let mut salt = [0u8; PASSWORD_SALT_LEN];
    getrandom::fill(&mut salt).map_err(|_| ScryptError::EntropyUnavailable)?;
    let mut verifier = crate::secret::ZeroizingBytes::<PASSWORD_OUTPUT_LEN>::zeroed();
    Scrypt::derive(&self.generation, password, &salt, verifier.as_mut_array())?;
    Ok(password_phc::encode(self.generation, &salt, verifier.as_array()))
  }

  /// Verify a password after approving all encoded resource requests.
  #[cfg(feature = "phc-strings")]
  #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
  pub fn verify_password(
    &self,
    password: &[u8],
    encoded: &str,
  ) -> Result<crate::auth::PasswordStatus, VerificationError> {
    let approved = password_phc::approve(encoded, self.limits).map_err(|_| VerificationError::new())?;
    let mut actual = crate::secret::ZeroizingBytes::<PASSWORD_OUTPUT_LEN>::zeroed();
    Scrypt::derive(&approved.params, password, approved.salt(), actual.as_mut_array())
      .map_err(|_| VerificationError::new())?;
    let verified = ct::constant_time_eq(actual.as_array(), &approved.expected);
    if !core::hint::black_box(verified) {
      return Err(VerificationError::new());
    }
    if approved.params == self.generation && approved.salt_len as usize == PASSWORD_SALT_LEN {
      Ok(crate::auth::PasswordStatus::Current)
    } else {
      Ok(crate::auth::PasswordStatus::NeedsRehash)
    }
  }
}

#[cfg(feature = "phc-strings")]
const PASSWORD_SALT_LEN: usize = 16;
#[cfg(feature = "phc-strings")]
const MIN_PHC_SALT_LEN: usize = 8;
#[cfg(feature = "phc-strings")]
const MAX_PHC_SALT_LEN: usize = 48;
#[cfg(feature = "phc-strings")]
const PASSWORD_OUTPUT_LEN: usize = DEFAULT_OUTPUT_LEN;

#[cfg(feature = "phc-strings")]
mod password_phc {
  #[cfg(any(feature = "getrandom", test))]
  use alloc::string::String;

  use super::{
    MAX_PHC_SALT_LEN, MIN_PHC_SALT_LEN, PASSWORD_OUTPUT_LEN, ScryptParams, ScryptVerificationLimits, scrypt_shape,
  };
  use crate::auth::phc::{self, PhcError};

  pub(super) struct ApprovedPhc {
    pub params: ScryptParams,
    pub salt: [u8; MAX_PHC_SALT_LEN],
    pub salt_len: u8,
    pub expected: [u8; PASSWORD_OUTPUT_LEN],
  }

  impl ApprovedPhc {
    pub fn salt(&self) -> &[u8] {
      &self.salt[..self.salt_len as usize]
    }
  }

  fn next_param<'a>(params: &mut phc::PhcParamIter<'a>, expected: &str) -> Result<&'a str, PhcError> {
    let (key, value) = params.next().ok_or(PhcError::MissingParam)??;
    if key != expected {
      return Err(if matches!(key, "ln" | "r" | "p") {
        PhcError::DuplicateParam
      } else {
        PhcError::UnknownParam
      });
    }
    Ok(value)
  }

  pub(super) fn approve(encoded: &str, limits: ScryptVerificationLimits) -> Result<ApprovedPhc, PhcError> {
    let parts = phc::parse(encoded)?;
    if parts.algorithm != "scrypt" {
      return Err(PhcError::AlgorithmMismatch);
    }
    if parts.version.is_some() {
      return Err(PhcError::UnsupportedVersion);
    }

    let mut values = phc::PhcParamIter::new(parts.parameters);
    let log_n = phc::parse_param_u32(next_param(&mut values, "ln")?)?;
    let r = phc::parse_param_u32(next_param(&mut values, "r")?)?;
    let p = phc::parse_param_u32(next_param(&mut values, "p")?)?;
    if let Some(extra) = values.next() {
      let (key, _) = extra?;
      return Err(if matches!(key, "ln" | "r" | "p") {
        PhcError::DuplicateParam
      } else {
        PhcError::UnknownParam
      });
    }
    if log_n > u8::MAX as u32 {
      return Err(PhcError::ParamOutOfRange);
    }
    let params = ScryptParams::new(log_n as u8, r, p).map_err(|_| PhcError::ParamOutOfRange)?;
    scrypt_shape(&params).map_err(|_| PhcError::ParamOutOfRange)?;
    if !limits.allows(params) {
      return Err(PhcError::ParamOutOfRange);
    }

    let salt_len = phc::base64_decoded_len(parts.salt_b64.len());
    let output_len = phc::base64_decoded_len(parts.hash_b64.len());
    if !(MIN_PHC_SALT_LEN..=MAX_PHC_SALT_LEN).contains(&salt_len) || output_len != PASSWORD_OUTPUT_LEN {
      return Err(PhcError::InvalidLength);
    }

    let mut salt = [0u8; MAX_PHC_SALT_LEN];
    let decoded_salt_len = phc::base64_decode_into(parts.salt_b64, &mut salt)?;
    let mut expected = [0u8; PASSWORD_OUTPUT_LEN];
    let decoded_output_len = phc::base64_decode_into(parts.hash_b64, &mut expected)?;
    if decoded_salt_len != salt_len || decoded_output_len != PASSWORD_OUTPUT_LEN {
      return Err(PhcError::InvalidLength);
    }

    Ok(ApprovedPhc {
      params,
      salt,
      salt_len: decoded_salt_len as u8,
      expected,
    })
  }

  #[cfg(any(feature = "getrandom", test))]
  pub(super) fn encode(params: ScryptParams, salt: &[u8], verifier: &[u8; PASSWORD_OUTPUT_LEN]) -> String {
    let mut out = String::with_capacity(112);
    out.push_str("$scrypt$ln=");
    phc::push_u32_decimal(&mut out, u32::from(params.get_log_n()));
    out.push_str(",r=");
    phc::push_u32_decimal(&mut out, params.get_r());
    out.push_str(",p=");
    phc::push_u32_decimal(&mut out, params.get_p());
    out.push('$');
    phc::base64_encode_into(salt, &mut out);
    out.push('$');
    phc::base64_encode_into(verifier, &mut out);
    out
  }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use alloc::vec;

  use super::*;

  const RFC_V1_EXPECTED: [u8; 64] = [
    0x77, 0xd6, 0x57, 0x62, 0x38, 0x65, 0x7b, 0x20, 0x3b, 0x19, 0xca, 0x42, 0xc1, 0x8a, 0x04, 0x97, 0xf1, 0x6b, 0x48,
    0x44, 0xe3, 0x07, 0x4a, 0xe8, 0xdf, 0xdf, 0xfa, 0x3f, 0xed, 0xe2, 0x14, 0x42, 0xfc, 0xd0, 0x06, 0x9d, 0xed, 0x09,
    0x48, 0xf8, 0x32, 0x6a, 0x75, 0x3a, 0x0f, 0xc8, 0x1f, 0x17, 0xe8, 0xd3, 0xe0, 0xfb, 0x2e, 0x0d, 0x36, 0x28, 0xcf,
    0x35, 0xe2, 0x0c, 0x38, 0xd1, 0x89, 0x06,
  ];

  fn small_params() -> ScryptParams {
    ScryptParams::new(4, 1, 1).unwrap()
  }

  #[test]
  fn rfc7914_vector_1_empty_inputs() {
    let mut output = [0u8; 64];
    Scrypt::derive(&small_params(), b"", b"", &mut output).unwrap();
    assert_eq!(output, RFC_V1_EXPECTED);
  }

  fn oracle_scrypt(password: &[u8], salt: &[u8], log_n: u8, r: u32, p: u32, output_len: usize) -> alloc::vec::Vec<u8> {
    let params = scrypt::Params::new(log_n, r, p).unwrap();
    let mut output = vec![0u8; output_len];
    scrypt::scrypt(password, salt, &params, &mut output).unwrap();
    output
  }

  #[test]
  fn matches_the_oracle_across_shapes_and_output_lengths() {
    let cases: &[(u8, u32, u32, usize)] = &[(4, 1, 1, 16), (5, 2, 1, 32), (6, 2, 2, 64)];
    for &(log_n, r, p, output_len) in cases {
      let params = ScryptParams::new(log_n, r, p).unwrap();
      let mut actual = vec![0u8; output_len];
      Scrypt::derive(&params, b"password", b"salty-salty-salt", &mut actual).unwrap();
      assert_eq!(
        actual,
        oracle_scrypt(b"password", b"salty-salty-salt", log_n, r, p, output_len),
        "mismatch log_n={log_n} r={r} p={p} output_len={output_len}",
      );
    }
  }

  #[cfg(all(target_arch = "x86_64", not(miri), not(feature = "portable-only")))]
  #[test]
  fn sse2_backend_matches_portable() {
    let cases: &[(u8, u32, u32, usize)] = &[(4, 1, 1, 32), (5, 2, 1, 48), (6, 2, 2, 32), (7, 8, 1, 64)];
    for &(log_n, r, p, output_len) in cases {
      let params = ScryptParams::new(log_n, r, p).unwrap();
      let mut portable = vec![0u8; output_len];
      let mut sse2 = vec![0u8; output_len];
      scrypt_hash_portable(&params, b"password", b"salty-salty-salt", &mut portable).unwrap();
      scrypt_hash_x86_sse2(&params, b"password", b"salty-salty-salt", &mut sse2).unwrap();
      assert_eq!(sse2, portable);
    }
  }

  #[test]
  fn raw_verify_accepts_only_the_exact_inputs() {
    let params = small_params();
    let mut expected = [0u8; 32];
    Scrypt::derive(&params, b"password", b"random-salt-1234", &mut expected).unwrap();

    assert!(Scrypt::verify(&params, b"password", b"random-salt-1234", &expected).is_ok());
    assert!(Scrypt::verify(&params, b"wrong", b"random-salt-1234", &expected).is_err());
    assert!(Scrypt::verify(&params, b"password", b"other-salt-00000", &expected).is_err());

    for position in 0..expected.len() {
      let mut tampered = expected;
      tampered[position] ^= 1;
      assert!(Scrypt::verify(&params, b"password", b"random-salt-1234", &tampered).is_err());
    }
  }

  #[test]
  fn params_are_valid_by_construction() {
    assert_eq!(ScryptParams::new(0, 1, 1), Err(ScryptError::InvalidLogN));
    assert_eq!(ScryptParams::new(64, 1, 1), Err(ScryptError::InvalidLogN));
    assert_eq!(ScryptParams::new(4, 0, 1), Err(ScryptError::InvalidR));
    assert_eq!(ScryptParams::new(4, 1, 0), Err(ScryptError::InvalidP));
    assert_eq!(ScryptParams::new(4, 1 << 15, 1 << 15), Err(ScryptError::InvalidP));
    assert!(ScryptParams::new(4, 1, 1).is_ok());
  }

  #[test]
  fn derive_rejects_empty_output() {
    let mut output = [];
    assert_eq!(
      Scrypt::derive(&small_params(), b"password", b"salt", &mut output),
      Err(ScryptError::InvalidOutputLen)
    );
  }

  #[cfg(target_pointer_width = "64")]
  #[test]
  fn derive_rejects_impossible_memory_shape_before_allocation() {
    let params = ScryptParams::new(63, 1 << 20, 1).unwrap();
    let mut output = [0u8; 32];
    assert_eq!(
      Scrypt::derive(&params, b"password", b"salt", &mut output),
      Err(ScryptError::ResourceOverflow)
    );
    #[cfg(feature = "phc-strings")]
    assert_eq!(ScryptPassword::new(params), Err(ScryptError::ResourceOverflow));
  }

  #[test]
  fn error_contract_is_complete() {
    fn assert_copy<T: Copy>() {}
    fn assert_err<T: core::error::Error>() {}
    assert_copy::<ScryptError>();
    assert_err::<ScryptError>();

    let variants = [
      ScryptError::InvalidLogN,
      ScryptError::InvalidR,
      ScryptError::InvalidP,
      ScryptError::InvalidOutputLen,
      ScryptError::ResourceOverflow,
      ScryptError::AllocationFailed,
      #[cfg(all(feature = "phc-strings", feature = "getrandom"))]
      ScryptError::EntropyUnavailable,
      #[cfg(feature = "phc-strings")]
      ScryptError::VerificationLimitTooLow,
    ];
    for error in variants {
      assert!(!alloc::format!("{error}").is_empty());
    }
  }

  #[test]
  fn kernel_contract_includes_the_portable_fallback() {
    assert!(ALL_KERNELS.contains(&KernelId::Portable));
    assert!(required_caps(KernelId::Portable).is_empty());
    assert_eq!(KernelId::Portable.as_str(), "portable");
  }

  #[test]
  fn salsa20_8_is_deterministic_and_non_identity() {
    let original = SalsaBlock([0x1234_5678; BLOCK_WORDS]);
    let mut first = original;
    let mut second = original;
    salsa20_8(&mut first);
    salsa20_8(&mut second);
    assert_eq!(first.0, second.0);
    assert_ne!(first.0, original.0);
  }

  #[cfg(all(feature = "phc-strings", not(miri)))]
  mod phc_tests {
    use alloc::format;

    use super::*;
    use crate::auth::{PasswordStatus, phc::PhcError};

    fn encode(params: ScryptParams, password: &[u8], salt: &[u8]) -> alloc::string::String {
      let mut verifier = [0u8; PASSWORD_OUTPUT_LEN];
      Scrypt::derive(&params, password, salt, &mut verifier).unwrap();
      password_phc::encode(params, salt, &verifier)
    }

    #[test]
    fn canonical_password_record_round_trips() {
      let params = small_params();
      let password = ScryptPassword::new(params).unwrap();
      let encoded = encode(params, b"password", &[0xaa; 16]);

      assert!(encoded.starts_with("$scrypt$ln=4,r=1,p=1$"));
      assert_eq!(
        password.verify_password(b"password", &encoded),
        Ok(PasswordStatus::Current)
      );
      assert!(password.verify_password(b"wrong", &encoded).is_err());
    }

    #[test]
    fn accepted_older_profile_requests_rehash() {
      let generation = ScryptParams::new(5, 1, 1).unwrap();
      let password = ScryptPassword::new(generation).unwrap();
      let encoded = encode(small_params(), b"password", &[0xbb; 16]);

      assert_eq!(
        password.verify_password(b"password", &encoded),
        Ok(PasswordStatus::NeedsRehash)
      );
    }

    #[test]
    fn accepted_noncurrent_salt_length_requests_rehash() {
      let params = small_params();
      let password = ScryptPassword::new(params).unwrap();
      let encoded = encode(params, b"password", &[0xbb; 8]);

      assert_eq!(
        password.verify_password(b"password", &encoded),
        Ok(PasswordStatus::NeedsRehash)
      );
    }

    #[test]
    fn approval_rejects_resource_requests_before_base64_decode() {
      let limits = ScryptVerificationLimits::for_profile(small_params());
      let expensive = "$scrypt$ln=5,r=1,p=1$*$*";
      assert_eq!(
        password_phc::approve(expensive, limits).err(),
        Some(PhcError::ParamOutOfRange)
      );

      let admitted = "$scrypt$ln=4,r=1,p=1$*$*";
      assert_eq!(
        password_phc::approve(admitted, limits).err(),
        Some(PhcError::InvalidLength)
      );
    }

    #[test]
    fn limits_bound_both_memory_and_work() {
      let limits = ScryptVerificationLimits::for_profile(small_params());
      assert!(limits.allows(small_params()));
      assert!(!limits.allows(ScryptParams::new(5, 1, 1).unwrap()));
      assert!(!limits.allows(ScryptParams::new(4, 1, 2).unwrap()));
    }

    #[test]
    fn parser_accepts_only_the_canonical_scrypt_protocol() {
      let limits = ScryptVerificationLimits::for_profile(small_params());
      let salt = "AAAAAAAAAAAAAAAAAAAAAA";
      let hash = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
      let cases = [
        format!("$argon2id$v=19$m=32,t=2,p=1$${salt}$${hash}"),
        format!("$scrypt$v=1$ln=4,r=1,p=1$${salt}$${hash}"),
        format!("$scrypt$r=1,ln=4,p=1$${salt}$${hash}"),
        format!("$scrypt$ln=4,ln=4,p=1$${salt}$${hash}"),
        format!("$scrypt$ln=4,r=1,x=1$${salt}$${hash}"),
        format!("$scrypt$ln=04,r=1,p=1$${salt}$${hash}"),
      ];
      for encoded in cases {
        assert!(password_phc::approve(&encoded, limits).is_err(), "{encoded}");
      }
    }

    #[cfg(feature = "getrandom")]
    #[test]
    fn generated_records_use_fresh_salts() {
      let password = ScryptPassword::new(small_params()).unwrap();
      let first = password.hash_password(b"password").unwrap();
      let second = password.hash_password(b"password").unwrap();

      assert_ne!(first, second);
      assert_eq!(
        password.verify_password(b"password", &first),
        Ok(PasswordStatus::Current)
      );
      assert_eq!(
        password.verify_password(b"password", &second),
        Ok(PasswordStatus::Current)
      );
    }
  }
}
