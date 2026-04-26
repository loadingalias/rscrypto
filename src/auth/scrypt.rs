//! scrypt password hashing (RFC 7914).
//!
//! Memory-hard key derivation combining PBKDF2-HMAC-SHA256 with a
//! Salsa20/8 core, via the BlockMix / ROMix construction of
//! Percival & Josefsson (RFC 7914 §3–§5). Parameters are driven by
//! [`ScryptParams`], whose defaults track OWASP 2024 (`log_n = 17`,
//! `r = 8`, `p = 1`, `output = 32 bytes`).
//!
//! The implementation reuses [`crate::Pbkdf2Sha256`] for the setup /
//! finalisation legs and is portable Rust throughout. Platform-specific
//! Salsa20/8 kernels (SSE2 / AVX2 / NEON / VSX / simd128 / RVV) plug into
//! the existing [`KernelId`] dispatch in a later phase.
//!
//! # Examples
//!
//! ```rust
//! use rscrypto::{Scrypt, ScryptParams};
//!
//! let params = ScryptParams::new()
//!   .log_n(10)
//!   .r(8)
//!   .p(1)
//!   .output_len(32)
//!   .build()
//!   .expect("valid params");
//!
//! let password = b"correct horse battery staple";
//! let salt = b"random-salt-1234";
//!
//! let mut hash = [0u8; 32];
//! Scrypt::hash(&params, password, salt, &mut hash).expect("hash");
//!
//! assert!(Scrypt::verify(&params, password, salt, &hash).is_ok());
//! assert!(Scrypt::verify(&params, b"wrong", salt, &hash).is_err());
//! ```
//!
//! # Security
//!
//! - [`Scrypt::MIN_SALT_LEN`] documents the 16-byte OWASP minimum. The algorithmic API accepts any
//!   salt length; policy enforcement is the caller's responsibility.
//! - [`ScryptParams::validate`] enforces RFC 7914 bounds (`log_n` in `1..=63`, `r ≥ 1`, `p ≥ 1`, `r
//!   · p ≤ 2^30 − 1`, `output_len ≥ 1`).
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

/// OWASP 2024 defaults: `log_n = 17`, `r = 8`, `p = 1`, `output_len = 32`.
const DEFAULT_LOG_N: u8 = 17;
const DEFAULT_R: u32 = 8;
const DEFAULT_P: u32 = 1;
const DEFAULT_OUTPUT_LEN: u32 = 32;

/// Minimum salt length (bytes) recommended for production deployments.
///
/// The scrypt algorithm accepts arbitrary salt lengths; this is a policy
/// constant exposed for callers and not enforced at `hash`-time (RFC 7914
/// §11 test vectors use empty / short salts).
pub const MIN_SALT_LEN: usize = 16;

/// Minimum output length in bytes per RFC 7914 §2.
pub const MIN_OUTPUT_LEN: usize = 1;

/// Upper bound on `r * p` per RFC 7914 §6.
const MAX_R_TIMES_P: u64 = (1u64 << 30) - 1;

// ─── Error ──────────────────────────────────────────────────────────────────

/// Invalid scrypt parameter, input length, or resource constraint.
///
/// Surfaced at `build` or `hash` time — never at `verify` time (a parameter
/// error during verification would leak information about the stored hash,
/// so verify collapses these into [`crate::VerificationError`]).
///
/// # Examples
///
/// ```rust
/// use rscrypto::{ScryptParams, auth::scrypt::ScryptError};
///
/// assert_eq!(
///   ScryptParams::new().log_n(0).build().unwrap_err(),
///   ScryptError::InvalidLogN,
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
    })
  }
}

impl core::error::Error for ScryptError {}

// ─── Parameters ─────────────────────────────────────────────────────────────

/// Validated scrypt cost-parameter set.
///
/// Constructed via [`ScryptParams::new`] and the `log_n` / `r` / `p` /
/// `output_len` setters; call [`ScryptParams::build`] to validate and
/// produce a `Result<ScryptParams, ScryptError>`. Every field is
/// stack-allocated — `ScryptParams` is `Copy` and cheap to clone.
///
/// # Examples
///
/// ```rust
/// use rscrypto::{Scrypt, ScryptParams};
///
/// let params = ScryptParams::new()
///   .log_n(10)
///   .r(8)
///   .p(1)
///   .output_len(32)
///   .build()
///   .expect("valid params");
///
/// let mut out = [0u8; 32];
/// Scrypt::hash(&params, b"password", b"salty-salty-salt", &mut out).unwrap();
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScryptParams {
  log_n: u8,
  r: u32,
  p: u32,
  output_len: u32,
}

impl fmt::Debug for ScryptParams {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("ScryptParams")
      .field("log_n", &self.log_n)
      .field("r", &self.r)
      .field("p", &self.p)
      .field("output_len", &self.output_len)
      .finish()
  }
}

impl Default for ScryptParams {
  fn default() -> Self {
    Self::new()
  }
}

impl ScryptParams {
  /// Create a new parameter builder pre-populated with OWASP 2024 defaults
  /// (`log_n = 17`, `r = 8`, `p = 1`, `output = 32`). Override via setters,
  /// then call [`ScryptParams::build`] to validate.
  #[must_use]
  pub const fn new() -> Self {
    Self {
      log_n: DEFAULT_LOG_N,
      r: DEFAULT_R,
      p: DEFAULT_P,
      output_len: DEFAULT_OUTPUT_LEN,
    }
  }

  /// Set `log_n` (the base-2 log of the CPU/memory cost `N`). Must be
  /// in `1..=63`.
  #[must_use]
  pub const fn log_n(mut self, lg_n: u8) -> Self {
    self.log_n = lg_n;
    self
  }

  /// Set the block-size parameter `r`. Must be `≥ 1`.
  #[must_use]
  pub const fn r(mut self, r: u32) -> Self {
    self.r = r;
    self
  }

  /// Set the parallelisation parameter `p`. Must satisfy
  /// `1 ≤ p` and `r · p ≤ 2^30 − 1`.
  #[must_use]
  pub const fn p(mut self, p: u32) -> Self {
    self.p = p;
    self
  }

  /// Set the derived-key length in bytes. Must be `≥ 1`.
  ///
  /// Capped at `2^32 - 1` bytes by the `u32` field type. RFC 7914 §2 permits
  /// up to `(2^32 - 1) × 32 ≈ 137 GiB`; in practice deployments never need
  /// more than a few hundred bytes, so the `u32` ceiling is a deliberate
  /// design choice rather than a spec limitation.
  #[must_use]
  pub const fn output_len(mut self, t: u32) -> Self {
    self.output_len = t;
    self
  }

  /// Validate every field against RFC 7914 bounds and return the finalised
  /// parameter set.
  ///
  /// # Errors
  ///
  /// Returns [`ScryptError`] if any parameter is out of range.
  pub const fn build(self) -> Result<Self, ScryptError> {
    match self.validate() {
      Ok(()) => Ok(self),
      Err(e) => Err(e),
    }
  }

  /// Run validation without consuming the builder — returns the first error.
  ///
  /// # Errors
  ///
  /// Returns [`ScryptError`] on the first invalid field.
  pub const fn validate(&self) -> Result<(), ScryptError> {
    if self.log_n < 1 || self.log_n > 63 {
      return Err(ScryptError::InvalidLogN);
    }
    if self.r < 1 {
      return Err(ScryptError::InvalidR);
    }
    if self.p < 1 {
      return Err(ScryptError::InvalidP);
    }
    // `r` and `p` are both `u32`, so `r * p` always fits in `u64`
    // (max `(2^32-1)^2 ≈ 2^64 - 2^33 + 1`). No overflow check needed.
    let rp = (self.r as u64) * (self.p as u64);
    if rp > MAX_R_TIMES_P {
      return Err(ScryptError::InvalidP);
    }
    if (self.output_len as usize) < MIN_OUTPUT_LEN {
      return Err(ScryptError::InvalidOutputLen);
    }
    Ok(())
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

  /// Derived-key length in bytes.
  #[must_use]
  pub const fn get_output_len(&self) -> u32 {
    self.output_len
  }
}

/// Operational limits for verifying scrypt PHC strings from untrusted storage.
///
/// PHC strings encode their own CPU/memory parameters and output length. Use
/// `Scrypt::verify_string_with_policy` when those encoded parameters can be
/// controlled by another tenant, database row, network peer, or migration
/// input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScryptVerifyPolicy {
  /// Maximum encoded `log_n` value.
  pub max_log_n: u8,
  /// Maximum encoded block-size parameter `r`.
  pub max_r: u32,
  /// Maximum encoded parallelisation parameter `p`.
  pub max_p: u32,
  /// Maximum encoded output length in bytes.
  pub max_output_len: usize,
}

impl ScryptVerifyPolicy {
  /// Build a policy from explicit upper bounds.
  #[must_use]
  pub const fn new(max_log_n: u8, max_r: u32, max_p: u32, max_output_len: usize) -> Self {
    Self {
      max_log_n,
      max_r,
      max_p,
      max_output_len,
    }
  }

  /// Return `true` when `params` and `output_len` are within this policy.
  #[must_use]
  pub const fn allows(&self, params: &ScryptParams, output_len: usize) -> bool {
    params.log_n <= self.max_log_n
      && params.r <= self.max_r
      && params.p <= self.max_p
      && output_len <= self.max_output_len
  }
}

impl Default for ScryptVerifyPolicy {
  fn default() -> Self {
    Self::new(DEFAULT_LOG_N, DEFAULT_R, DEFAULT_P, DEFAULT_OUTPUT_LEN as usize)
  }
}

// ─── Kernel dispatch (Phase 2: Portable only) ──────────────────────────────

/// Salsa20/8 kernel identifier.
///
/// Phase 2 ships only the portable kernel. Phase 4 SIMD kernels
/// (SSE2 / AVX2 / NEON / VSX / simd128 / RVV) plug into this enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelId {
  /// Pure-Rust portable implementation (always available).
  Portable,
}

impl KernelId {
  /// Kernel name for diagnostics and forced-kernel test plumbing.
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
    }
  }
}

/// All compiled-in kernels, in preference order.
pub const ALL_KERNELS: &[KernelId] = &[KernelId::Portable];

/// Capabilities required for `kernel`. Phase 2 returns empty caps; Phase 4
/// kernels will return the architecture-specific feature set they need.
#[must_use]
pub const fn required_caps(kernel: KernelId) -> crate::platform::Caps {
  match kernel {
    KernelId::Portable => crate::platform::Caps::from_words([0; 4]),
  }
}

/// Runtime dispatch. Phase 2 has only a portable kernel; Phase 4 SIMD
/// kernels slot in here without any other API changes.
#[inline]
#[allow(dead_code)] // Reserved for Phase 4 dispatch; referenced by tests to pin the contract.
fn active_kernel() -> KernelId {
  KernelId::Portable
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

fn scrypt_hash(params: &ScryptParams, password: &[u8], salt: &[u8], out: &mut [u8]) -> Result<(), ScryptError> {
  params.validate()?;
  if out.len() != params.output_len as usize {
    return Err(ScryptError::InvalidOutputLen);
  }

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

  let mut state = ScryptState::new(total_b_blocks, v_blocks, two_r)?;

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
  for chunk_idx in 0..p {
    let chunk_start = chunk_idx.strict_mul(two_r);
    let chunk_end = chunk_start.strict_add(two_r);
    let chunk = &mut state.b_u32[chunk_start..chunk_end];
    ro_mix(chunk, &mut state.v, &mut state.scratch, n, r);
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

// ─── Public API ─────────────────────────────────────────────────────────────

/// scrypt password-hashing (RFC 7914).
///
/// Mirrors the UX of [`crate::Argon2id`]: `hash`, `hash_array`, `verify`
/// for raw tags, and PHC-string helpers behind `feature = "phc-strings"`.
///
/// # Examples
///
/// ```rust
/// use rscrypto::{Scrypt, ScryptParams};
///
/// // Small CI-friendly params — production deployments should use
/// // `ScryptParams::new()` for the OWASP 2024 defaults.
/// let params = ScryptParams::new()
///   .log_n(10)
///   .r(8)
///   .p(1)
///   .output_len(32)
///   .build()
///   .unwrap();
///
/// let hash = Scrypt::hash_array::<32>(&params, b"password", b"random-salt-1234").unwrap();
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

  /// Hash `password` with `salt` into `out`.
  ///
  /// # Errors
  ///
  /// Returns [`ScryptError`] if parameters are out of range, `out.len()`
  /// does not match `params.output_len`, or the working-set allocation
  /// fails.
  pub fn hash(params: &ScryptParams, password: &[u8], salt: &[u8], out: &mut [u8]) -> Result<(), ScryptError> {
    scrypt_hash(params, password, salt, out)
  }

  /// Hash `password` with `salt` into a fixed-size array.
  ///
  /// # Errors
  ///
  /// Returns [`ScryptError`] if `N != params.output_len`, parameters are
  /// out of range, or the working-set allocation fails.
  pub fn hash_array<const N: usize>(
    params: &ScryptParams,
    password: &[u8],
    salt: &[u8],
  ) -> Result<[u8; N], ScryptError> {
    let mut out = [0u8; N];
    Self::hash(params, password, salt, &mut out)?;
    Ok(out)
  }

  /// Verify `expected` against a freshly-computed hash in constant time.
  ///
  /// scrypt always runs to completion regardless of `expected.len()` — the
  /// length check is folded into the final boolean, not an early return,
  /// so wall-clock cost does not leak `params.output_len`. The dominant
  /// cost (ROMix at the configured `log_n / r / p`) is paid in every
  /// failure path.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] on any mismatch, malformed
  /// input, or parameter error.
  #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
  pub fn verify(params: &ScryptParams, password: &[u8], salt: &[u8], expected: &[u8]) -> Result<(), VerificationError> {
    let actual_len = params.output_len as usize;
    let mut actual = alloc::vec![0u8; actual_len];
    let hash_failed = Self::hash(params, password, salt, &mut actual).is_err();

    let bytes_match = ct::constant_time_eq(&actual, expected);
    ct::zeroize(&mut actual);

    let success = !hash_failed & bytes_match;
    if core::hint::black_box(success) {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }

  /// Hash `password` with `salt` and encode the result as a PHC string.
  ///
  /// Emits `$scrypt$ln=<log_n>,r=<r>,p=<p>$<salt>$<hash>` (RFC 4648
  /// base64, no padding). scrypt has no version segment per PHC
  /// convention.
  ///
  /// # Errors
  ///
  /// Propagates any [`ScryptError`] from parameter validation, input
  /// length checks, or working-set allocation.
  #[cfg(feature = "phc-strings")]
  pub fn hash_string_with_salt(
    params: &ScryptParams,
    password: &[u8],
    salt: &[u8],
  ) -> Result<alloc::string::String, ScryptError> {
    let mut hash = alloc::vec![0u8; params.output_len as usize];
    Self::hash(params, password, salt, &mut hash)?;
    let encoded = phc_integration::encode_string(params, salt, &hash);
    ct::zeroize(&mut hash);
    Ok(encoded)
  }

  /// Hash `password` with a fresh 16-byte salt from the operating system
  /// CSPRNG and encode the result as a PHC string.
  ///
  /// # Panics
  ///
  /// Panics if the platform entropy source fails.
  ///
  /// # Errors
  ///
  /// Propagates any [`ScryptError`] from parameter validation, input length
  /// checks, or working-set allocation.
  #[cfg(all(feature = "phc-strings", feature = "getrandom"))]
  pub fn hash_string(params: &ScryptParams, password: &[u8]) -> Result<alloc::string::String, ScryptError> {
    let mut salt = [0u8; 16];
    getrandom::fill(&mut salt).unwrap_or_else(|e| panic!("getrandom failed: {e}"));
    Self::hash_string_with_salt(params, password, &salt)
  }

  /// Verify `password` against a PHC-encoded hash in constant time.
  ///
  /// Parses the encoded string, rebuilds the cost parameters, re-hashes
  /// `password` with the embedded salt, and compares in constant time.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] on any mismatch, malformed string, or
  /// parameter error. Errors are intentionally opaque — callers needing
  /// to distinguish parse failures should use
  /// [`Scrypt::decode_string`].
  #[cfg(feature = "phc-strings")]
  #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
  pub fn verify_string(password: &[u8], encoded: &str) -> Result<(), VerificationError> {
    Self::verify_string_with_policy(
      password,
      encoded,
      &ScryptVerifyPolicy::new(u8::MAX, u32::MAX, u32::MAX, usize::MAX),
    )
  }

  /// Verify `password` against a PHC string after enforcing operational
  /// bounds on its encoded cost parameters.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] on any mismatch, malformed string,
  /// parameter error, or policy violation.
  #[cfg(feature = "phc-strings")]
  #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
  pub fn verify_string_with_policy(
    password: &[u8],
    encoded: &str,
    policy: &ScryptVerifyPolicy,
  ) -> Result<(), VerificationError> {
    let parsed = phc_integration::decode_string(encoded).map_err(|_| VerificationError::new())?;
    if !policy.allows(&parsed.params, parsed.hash.len()) {
      return Err(VerificationError::new());
    }
    let mut actual = alloc::vec![0u8; parsed.hash.len()];
    if Self::hash(&parsed.params, password, &parsed.salt, &mut actual).is_err() {
      ct::zeroize(&mut actual);
      return Err(VerificationError::new());
    }
    let ok = ct::constant_time_eq(&actual, &parsed.hash);
    ct::zeroize(&mut actual);
    if core::hint::black_box(ok) {
      Ok(())
    } else {
      Err(VerificationError::new())
    }
  }

  /// Decode a PHC string without re-hashing.
  ///
  /// Returns the parsed cost parameters, salt, and reference hash.
  ///
  /// # Errors
  ///
  /// Returns [`crate::auth::phc::PhcError`] on any parse failure, or if
  /// the encoded algorithm does not match `Self::ALGORITHM`.
  #[cfg(feature = "phc-strings")]
  pub fn decode_string(
    encoded: &str,
  ) -> Result<(ScryptParams, alloc::vec::Vec<u8>, alloc::vec::Vec<u8>), crate::auth::phc::PhcError> {
    let parsed = phc_integration::decode_string(encoded)?;
    Ok((parsed.params, parsed.salt, parsed.hash))
  }
}

// ─── PHC string integration (feature: phc-strings) ─────────────────────────

#[cfg(feature = "phc-strings")]
mod phc_integration {
  use alloc::{string::String, vec::Vec};

  use super::{MIN_OUTPUT_LEN, ScryptParams};
  use crate::auth::phc::{self, PhcError};

  /// Parsed PHC components reconstituted into rscrypto types.
  pub(super) struct ParsedPhc {
    pub params: ScryptParams,
    pub salt: Vec<u8>,
    pub hash: Vec<u8>,
  }

  /// Build `$scrypt$ln=<log_n>,r=<r>,p=<p>$<salt>$<hash>`.
  pub(super) fn encode_string(params: &ScryptParams, salt: &[u8], hash: &[u8]) -> String {
    let mut out = String::new();
    out.push('$');
    out.push_str(super::Scrypt::ALGORITHM);
    out.push_str("$ln=");
    phc::push_u32_decimal(&mut out, u32::from(params.get_log_n()));
    out.push_str(",r=");
    phc::push_u32_decimal(&mut out, params.get_r());
    out.push_str(",p=");
    phc::push_u32_decimal(&mut out, params.get_p());
    out.push('$');
    phc::base64_encode_into(salt, &mut out);
    out.push('$');
    phc::base64_encode_into(hash, &mut out);
    out
  }

  /// Parse a PHC string and reconstitute `(params, salt, hash)`.
  pub(super) fn decode_string(encoded: &str) -> Result<ParsedPhc, PhcError> {
    let parts = phc::parse(encoded)?;
    if parts.algorithm != super::Scrypt::ALGORITHM {
      return Err(PhcError::AlgorithmMismatch);
    }
    // scrypt has no version segment per PHC convention. A `v=` prefix
    // would have been parsed as the version slot; reject it.
    if parts.version.is_some() {
      return Err(PhcError::UnsupportedVersion);
    }

    let mut log_n: Option<u32> = None;
    let mut r: Option<u32> = None;
    let mut p: Option<u32> = None;

    for pair in phc::PhcParamIter::new(parts.parameters) {
      let (k, v) = pair?;
      let value = phc::parse_param_u32(v)?;
      match k {
        "ln" => {
          if log_n.replace(value).is_some() {
            return Err(PhcError::DuplicateParam);
          }
        }
        "r" => {
          if r.replace(value).is_some() {
            return Err(PhcError::DuplicateParam);
          }
        }
        "p" => {
          if p.replace(value).is_some() {
            return Err(PhcError::DuplicateParam);
          }
        }
        _ => return Err(PhcError::UnknownParam),
      }
    }

    let log_n_u32 = log_n.ok_or(PhcError::MissingParam)?;
    let r = r.ok_or(PhcError::MissingParam)?;
    let p = p.ok_or(PhcError::MissingParam)?;

    if log_n_u32 > u8::MAX as u32 {
      return Err(PhcError::ParamOutOfRange);
    }

    let salt = phc::decode_base64_to_vec(parts.salt_b64)?;
    let hash = phc::decode_base64_to_vec(parts.hash_b64)?;

    if hash.len() < MIN_OUTPUT_LEN {
      return Err(PhcError::InvalidLength);
    }

    let params = ScryptParams::new()
      .log_n(log_n_u32 as u8)
      .r(r)
      .p(p)
      .output_len(hash.len() as u32)
      .build()
      .map_err(|_| PhcError::ParamOutOfRange)?;

    Ok(ParsedPhc { params, salt, hash })
  }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use alloc::vec;

  use super::*;

  // ── RFC 7914 §11 vector 1 ─────────────────────────────────────────────
  // P="" S="" N=16 r=1 p=1 dkLen=64
  const RFC_V1_EXPECTED: [u8; 64] = [
    0x77, 0xd6, 0x57, 0x62, 0x38, 0x65, 0x7b, 0x20, 0x3b, 0x19, 0xca, 0x42, 0xc1, 0x8a, 0x04, 0x97, 0xf1, 0x6b, 0x48,
    0x44, 0xe3, 0x07, 0x4a, 0xe8, 0xdf, 0xdf, 0xfa, 0x3f, 0xed, 0xe2, 0x14, 0x42, 0xfc, 0xd0, 0x06, 0x9d, 0xed, 0x09,
    0x48, 0xf8, 0x32, 0x6a, 0x75, 0x3a, 0x0f, 0xc8, 0x1f, 0x17, 0xe8, 0xd3, 0xe0, 0xfb, 0x2e, 0x0d, 0x36, 0x28, 0xcf,
    0x35, 0xe2, 0x0c, 0x38, 0xd1, 0x89, 0x06,
  ];

  #[test]
  fn rfc7914_vector_1_empty_inputs() {
    let params = ScryptParams::new()
      .log_n(4) // N = 16
      .r(1)
      .p(1)
      .output_len(64)
      .build()
      .unwrap();
    let mut out = [0u8; 64];
    Scrypt::hash(&params, b"", b"", &mut out).unwrap();
    assert_eq!(out, RFC_V1_EXPECTED);
  }

  // ── Differential: rscrypto vs RustCrypto `scrypt` ─────────────────────

  fn oracle_scrypt(password: &[u8], salt: &[u8], log_n: u8, r: u32, p: u32, out_len: usize) -> alloc::vec::Vec<u8> {
    let params = scrypt::Params::new(log_n, r, p, out_len).unwrap();
    let mut out = vec![0u8; out_len];
    scrypt::scrypt(password, salt, &params, &mut out).unwrap();
    out
  }

  #[test]
  fn matches_oracle_small_params() {
    // Small enough to run under Miri too.
    let cases: &[(u8, u32, u32, usize)] = &[(4, 1, 1, 32), (5, 2, 1, 32), (6, 2, 2, 32)];
    for &(log_n, r, p, out_len) in cases {
      let params = ScryptParams::new()
        .log_n(log_n)
        .r(r)
        .p(p)
        .output_len(out_len as u32)
        .build()
        .unwrap();
      let mut actual = vec![0u8; out_len];
      Scrypt::hash(&params, b"password", b"salty-salty-salt", &mut actual).unwrap();
      let expected = oracle_scrypt(b"password", b"salty-salty-salt", log_n, r, p, out_len);
      assert_eq!(actual, expected, "mismatch log_n={log_n} r={r} p={p} T={out_len}");
    }
  }

  #[cfg(not(miri))]
  #[test]
  fn matches_oracle_owasp_shape() {
    // log_n=10 keeps the test fast; OWASP uses 17 in production.
    let log_n = 10;
    let r = 8;
    let p = 1;
    let out_len = 32;
    let params = ScryptParams::new()
      .log_n(log_n)
      .r(r)
      .p(p)
      .output_len(out_len as u32)
      .build()
      .unwrap();
    let password = b"correct horse battery staple";
    let salt = b"random-salt-1234";
    let mut actual = vec![0u8; out_len];
    Scrypt::hash(&params, password, salt, &mut actual).unwrap();
    let expected = oracle_scrypt(password, salt, log_n, r, p, out_len);
    assert_eq!(actual, expected);
  }

  // ── Verify ────────────────────────────────────────────────────────────

  #[test]
  fn verify_accepts_correct() {
    let params = ScryptParams::new().log_n(4).r(1).p(1).output_len(32).build().unwrap();
    let h = Scrypt::hash_array::<32>(&params, b"password", b"random-salt-1234").unwrap();
    assert!(Scrypt::verify(&params, b"password", b"random-salt-1234", &h).is_ok());
  }

  #[test]
  fn verify_rejects_wrong_password() {
    let params = ScryptParams::new().log_n(4).r(1).p(1).output_len(32).build().unwrap();
    let h = Scrypt::hash_array::<32>(&params, b"password", b"random-salt-1234").unwrap();
    assert!(Scrypt::verify(&params, b"wrong-password!!", b"random-salt-1234", &h).is_err());
  }

  #[test]
  fn verify_rejects_wrong_salt() {
    let params = ScryptParams::new().log_n(4).r(1).p(1).output_len(32).build().unwrap();
    let h = Scrypt::hash_array::<32>(&params, b"password", b"random-salt-1234").unwrap();
    assert!(Scrypt::verify(&params, b"password", b"other-salt-000000", &h).is_err());
  }

  #[test]
  fn verify_rejects_length_mismatch() {
    let params = ScryptParams::new().log_n(4).r(1).p(1).output_len(32).build().unwrap();
    let wrong_len = [0u8; 16];
    assert!(Scrypt::verify(&params, b"password", b"random-salt-1234", &wrong_len).is_err());
  }

  #[cfg(all(feature = "phc-strings", feature = "getrandom"))]
  #[test]
  fn hash_string_uses_random_salt_and_verifies() {
    let params = ScryptParams::new().log_n(4).r(1).p(1).output_len(32).build().unwrap();
    let encoded = Scrypt::hash_string(&params, b"password").unwrap();
    assert!(Scrypt::verify_string(b"password", &encoded).is_ok());
    assert!(Scrypt::verify_string(b"wrong-password", &encoded).is_err());
  }

  // ── Byte flips at every position ──────────────────────────────────────

  #[test]
  fn verify_rejects_byte_flip_at_every_position() {
    let params = ScryptParams::new().log_n(4).r(1).p(1).output_len(32).build().unwrap();
    let password = b"correct horse battery staple";
    let salt = b"random-salt-1234";
    let hash = Scrypt::hash_array::<32>(&params, password, salt).unwrap();
    for pos in 0..hash.len() {
      let mut tampered = hash;
      tampered[pos] ^= 0x01;
      assert!(
        Scrypt::verify(&params, password, salt, &tampered).is_err(),
        "verify must reject flip at byte {pos}",
      );
    }
  }

  // ── Parameter validation ──────────────────────────────────────────────

  #[test]
  fn validate_rejects_zero_log_n() {
    assert_eq!(
      ScryptParams::new().log_n(0).build().unwrap_err(),
      ScryptError::InvalidLogN,
    );
  }

  #[test]
  fn validate_rejects_log_n_too_large() {
    assert_eq!(
      ScryptParams::new().log_n(64).build().unwrap_err(),
      ScryptError::InvalidLogN,
    );
  }

  #[test]
  fn validate_rejects_zero_r() {
    assert_eq!(ScryptParams::new().r(0).build().unwrap_err(), ScryptError::InvalidR,);
  }

  #[test]
  fn validate_rejects_zero_p() {
    assert_eq!(ScryptParams::new().p(0).build().unwrap_err(), ScryptError::InvalidP,);
  }

  #[test]
  fn validate_rejects_r_times_p_over_limit() {
    // r*p = 2^30 → exceeds 2^30 - 1.
    assert_eq!(
      ScryptParams::new()
        .log_n(4)
        .r(1 << 15)
        .p(1 << 15)
        .output_len(32)
        .build()
        .unwrap_err(),
      ScryptError::InvalidP,
    );
  }

  #[test]
  fn validate_rejects_zero_output_len() {
    assert_eq!(
      ScryptParams::new().output_len(0).build().unwrap_err(),
      ScryptError::InvalidOutputLen,
    );
  }

  // ── Output length mismatch at hash-time ───────────────────────────────

  #[test]
  fn output_len_mismatch_rejected() {
    let params = ScryptParams::new().log_n(4).r(1).p(1).output_len(32).build().unwrap();
    let mut out = [0u8; 16];
    assert_eq!(
      Scrypt::hash(&params, b"pw", b"salty-salty-salt", &mut out).unwrap_err(),
      ScryptError::InvalidOutputLen,
    );
  }

  // ── Resource-overflow path on 64-bit hosts ────────────────────────────

  // `log_n = 63` with `r = 2^20` validates (r·p = 2^20 ≤ 2^30−1) but
  // `N · 2r = 2^63 · 2^21 = 2^84` does not fit in a 64-bit usize. The
  // `checked_mul` in `scrypt_hash` must surface this as `ResourceOverflow`
  // instead of panicking or silently allocating a truncated buffer.
  #[cfg(target_pointer_width = "64")]
  #[test]
  fn hash_rejects_impossible_memory_size_on_64bit() {
    let params = ScryptParams::new()
      .log_n(63)
      .r(1 << 20)
      .p(1)
      .output_len(32)
      .build()
      .unwrap();
    let mut out = [0u8; 32];
    assert_eq!(
      Scrypt::hash(&params, b"pw", b"salty-salty-salt", &mut out).unwrap_err(),
      ScryptError::ResourceOverflow,
    );
  }

  // ── Error trait plumbing ──────────────────────────────────────────────

  #[test]
  fn error_is_copy_and_implements_error_trait() {
    fn assert_copy<T: Copy>() {}
    fn assert_err<T: core::error::Error>() {}
    assert_copy::<ScryptError>();
    assert_err::<ScryptError>();
  }

  #[test]
  fn error_display_is_non_empty_for_every_variant() {
    let all = [
      ScryptError::InvalidLogN,
      ScryptError::InvalidR,
      ScryptError::InvalidP,
      ScryptError::InvalidOutputLen,
      ScryptError::ResourceOverflow,
      ScryptError::AllocationFailed,
    ];
    for e in all {
      let s = alloc::format!("{e}");
      assert!(!s.is_empty());
    }
  }

  // ── Kernel dispatch ──────────────────────────────────────────────────

  #[test]
  fn kernel_id_stringifies() {
    assert_eq!(KernelId::Portable.as_str(), "portable");
  }

  #[test]
  fn portable_kernel_has_no_required_caps() {
    assert!(required_caps(KernelId::Portable).is_empty());
  }

  #[test]
  fn active_kernel_is_portable() {
    assert_eq!(active_kernel(), KernelId::Portable);
  }

  // ── Salsa20/8 self-consistency ───────────────────────────────────────

  #[test]
  fn salsa20_8_is_deterministic() {
    let a = SalsaBlock([0x1234_5678; BLOCK_WORDS]);
    let mut a1 = a;
    let mut a2 = a;
    salsa20_8(&mut a1);
    salsa20_8(&mut a2);
    assert_eq!(a1.0, a2.0);
    assert_ne!(a1.0, a.0, "Salsa20/8 must not be the identity on a constant input");
  }

  // ── PHC integration ──────────────────────────────────────────────────

  #[cfg(feature = "phc-strings")]
  mod phc_tests {
    use alloc::vec;

    use super::*;
    use crate::auth::phc::PhcError;

    fn small_params() -> ScryptParams {
      ScryptParams::new().log_n(4).r(1).p(1).output_len(32).build().unwrap()
    }

    #[test]
    fn hash_string_with_salt_round_trip() {
      let params = small_params();
      let salt = [0xAAu8; 16];
      let encoded = Scrypt::hash_string_with_salt(&params, b"password", &salt).unwrap();
      assert!(encoded.starts_with("$scrypt$ln=4,r=1,p=1$"));
      assert!(Scrypt::verify_string(b"password", &encoded).is_ok());
      assert!(Scrypt::verify_string(b"wrongpassword", &encoded).is_err());
    }

    #[test]
    fn verify_string_with_policy_enforces_scrypt_bounds() {
      let params = small_params();
      let salt = [0xA1u8; 16];
      let encoded = Scrypt::hash_string_with_salt(&params, b"password", &salt).unwrap();

      let allowed = ScryptVerifyPolicy::new(4, 1, 1, 32);
      assert!(Scrypt::verify_string_with_policy(b"password", &encoded, &allowed).is_ok());

      let low_log_n = ScryptVerifyPolicy::new(3, 1, 1, 32);
      assert!(Scrypt::verify_string_with_policy(b"password", &encoded, &low_log_n).is_err());

      let short_output = ScryptVerifyPolicy::new(4, 1, 1, 31);
      assert!(Scrypt::verify_string_with_policy(b"password", &encoded, &short_output).is_err());
    }

    #[test]
    fn decode_string_extracts_params_salt_hash() {
      let params = small_params();
      let salt = vec![0xDDu8; 16];
      let encoded = Scrypt::hash_string_with_salt(&params, b"pw", &salt).unwrap();

      let (decoded_params, decoded_salt, decoded_hash) = Scrypt::decode_string(&encoded).unwrap();
      assert_eq!(decoded_params.get_log_n(), 4);
      assert_eq!(decoded_params.get_r(), 1);
      assert_eq!(decoded_params.get_p(), 1);
      assert_eq!(decoded_params.get_output_len(), 32);
      assert_eq!(decoded_salt, salt);
      assert_eq!(decoded_hash.len(), 32);

      let mut rehashed = [0u8; 32];
      Scrypt::hash(&decoded_params, b"pw", &decoded_salt, &mut rehashed).unwrap();
      assert_eq!(rehashed.as_slice(), decoded_hash.as_slice());
    }

    #[test]
    fn decode_string_rejects_duplicate_params() {
      let params = small_params();
      let encoded = Scrypt::hash_string_with_salt(&params, b"pw", &[0xFFu8; 16]).unwrap();
      let broken = encoded.replace("r=1", "ln=4");
      assert_eq!(Scrypt::decode_string(&broken).unwrap_err(), PhcError::DuplicateParam);
    }

    #[test]
    fn decode_string_rejects_unknown_param() {
      let params = small_params();
      let encoded = Scrypt::hash_string_with_salt(&params, b"pw", &[0xFFu8; 16]).unwrap();
      let broken = encoded.replace("ln=4", "bogus=1");
      assert_eq!(Scrypt::decode_string(&broken).unwrap_err(), PhcError::UnknownParam);
    }

    #[test]
    fn decode_string_rejects_algorithm_mismatch() {
      // Argon2id-shaped string with scrypt decoder.
      assert_eq!(
        Scrypt::decode_string("$argon2id$v=19$m=32,t=2,p=1$c29tZXNhbHQ$c29tZWhhc2g").unwrap_err(),
        PhcError::AlgorithmMismatch,
      );
    }

    #[test]
    fn decode_string_rejects_version_segment() {
      // scrypt PHC has no version segment; a `v=19` slot must be refused.
      assert_eq!(
        Scrypt::decode_string("$scrypt$v=1$ln=4,r=1,p=1$c29tZXNhbHQ$c29tZWhhc2g").unwrap_err(),
        PhcError::UnsupportedVersion,
      );
    }

    #[test]
    fn decode_string_distinguishes_version_segment_from_version_param() {
      // `v=1` in the version slot → UnsupportedVersion (no version segment
      // allowed). `version=1` in the params slot → UnknownParam. Pins both
      // paths so a future parser regression that conflates them is caught.
      assert_eq!(
        Scrypt::decode_string("$scrypt$v=1$ln=4,r=1,p=1$c29tZXNhbHQ$c29tZWhhc2g").unwrap_err(),
        PhcError::UnsupportedVersion,
      );
      assert_eq!(
        Scrypt::decode_string("$scrypt$version=1,ln=4,r=1,p=1$c29tZXNhbHQ$c29tZWhhc2g").unwrap_err(),
        PhcError::UnknownParam,
      );
    }

    #[test]
    fn decode_string_rejects_missing_required_param() {
      // Drop the `ln=` key from an otherwise-valid string.
      let params = small_params();
      let encoded = Scrypt::hash_string_with_salt(&params, b"pw", &[0x11u8; 16]).unwrap();
      let broken = encoded.replace("ln=4,", "");
      assert_eq!(Scrypt::decode_string(&broken).unwrap_err(), PhcError::MissingParam);
    }

    #[test]
    fn decode_string_rejects_out_of_range_log_n() {
      // `ln=999` exceeds `u8::MAX` / RFC 7914 log_n ≤ 63. The decoder rejects
      // at `log_n > u8::MAX` check, then at `ScryptParams::build()` if it
      // slipped past.
      let params = small_params();
      let encoded = Scrypt::hash_string_with_salt(&params, b"pw", &[0x22u8; 16]).unwrap();
      let broken = encoded.replace("ln=4", "ln=999");
      assert_eq!(Scrypt::decode_string(&broken).unwrap_err(), PhcError::ParamOutOfRange);
    }

    #[test]
    fn hash_array_and_hash_agree_byte_for_byte() {
      let params = small_params();
      let arr = Scrypt::hash_array::<32>(&params, b"pw", b"salty-salty-salt").unwrap();
      let mut via_hash = [0u8; 32];
      Scrypt::hash(&params, b"pw", b"salty-salty-salt", &mut via_hash).unwrap();
      assert_eq!(arr, via_hash);
    }

    #[test]
    fn hash_is_deterministic() {
      // Two independent `Scrypt::hash` calls on identical inputs must
      // produce byte-identical outputs. Guards against accidental reliance
      // on uninitialised working-set memory in ROMix.
      let params = small_params();
      let mut a = [0u8; 32];
      let mut b = [0u8; 32];
      Scrypt::hash(&params, b"pw", b"salty-salty-salt", &mut a).unwrap();
      Scrypt::hash(&params, b"pw", b"salty-salty-salt", &mut b).unwrap();
      assert_eq!(a, b);
    }

    #[test]
    fn encoded_format_exact_for_known_vector() {
      let params = small_params();
      let salt = b"exampleSALTvalue"; // 16 bytes
      let encoded = Scrypt::hash_string_with_salt(&params, b"password", salt).unwrap();
      let segments: alloc::vec::Vec<&str> = encoded.split('$').collect();
      assert_eq!(segments[0], "");
      assert_eq!(segments[1], "scrypt");
      assert_eq!(segments[2], "ln=4,r=1,p=1");
      assert_eq!(segments[3].len(), 22); // 16 bytes base64-nopad
      assert_eq!(segments[4].len(), 43); // 32 bytes base64-nopad
      assert_eq!(segments.len(), 5);
    }
  }
}
