//! Argon2 password hashing (RFC 9106).
//!
//! Ships all three variants:
//!
//! - [`Argon2d`] — data-dependent indexing, highest throughput, vulnerable to side-channel timing
//!   attacks. Useful for non-interactive, trusted- hardware settings (e.g. cryptocurrency
//!   proof-of-work).
//! - [`Argon2i`] — data-independent indexing: memory-access patterns do not depend on the password.
//!   Useful when the adversary can observe cache or memory-access timing.
//! - [`Argon2id`] — hybrid: first half of the first pass runs Argon2i, the rest runs Argon2d.
//!   **Recommended default for password hashing** per RFC 9106 §4 and the [OWASP Password Storage
//!   Cheat Sheet][owasp-passwords].
//!
//! The implementation uses the BlaMka compression function from RFC 9106
//! §3.6 layered on top of [`crate::Blake2b`]. A runtime-cached [`KernelId`]
//! dispatcher picks the highest-throughput BlaMka kernel for the host: a
//! 4-way NEON kernel on aarch64 ships today, with per-arch x86_64 / VSX /
//! s390x / RVV / simd128 kernels rolling in behind the same contract.
//!
//! # Examples
//!
//! ```rust
//! use rscrypto::{Argon2Params, Argon2id};
//!
//! let params = Argon2Params::new(19 * 1024, 2, 1).expect("valid params");
//!
//! let password = b"correct horse battery staple";
//! let salt = b"random-salt-1234";
//!
//! let mut hash = [0u8; 32];
//! Argon2id::derive(&params, password, salt, &mut hash).expect("derive");
//!
//! assert!(Argon2id::verify(&params, password, salt, &hash).is_ok());
//! assert!(Argon2id::verify(&params, b"wrong", salt, &hash).is_err());
//! ```
//!
//! # Security
//!
//! - Salt length must be ≥ 8 bytes (RFC 9106 §3.1). 16 bytes recommended.
//! - Memory cost must satisfy `m ≥ 8 · p` (KiB).
//! - [`Argon2Params::new`] rejects invalid cost profiles, so constructed parameters are always
//!   valid.
//! - The memory matrix is zeroized on drop.
//! - [`Argon2id::verify`] is constant-time with respect to the stored hash bytes.
//!
//! # Compliance
//!
//! Argon2 is **not FIPS 140-3 approved**. NIST SP 800-132 only covers
//! PBKDF2 for password-based key derivation; deployments under FIPS
//! policy should use [`crate::Pbkdf2Sha256`]. This module is suitable
//! for OWASP-aligned password hashing outside strict FIPS boundaries.
//!
//! Requires `alloc` — the memory matrix (`m_kib · 1024` bytes) cannot be
//! stack-allocated. Bare-metal / heap-less targets should select
//! [`crate::Pbkdf2Sha256`] (alloc-free).
//!
//! [owasp-passwords]: https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html

#![allow(clippy::indexing_slicing)]
#![allow(clippy::unwrap_used)]
// unwraps here are on slice→array conversions whose lengths are fixed by construction.

use alloc::vec::Vec;
use core::fmt;

use crate::{
  hashes::crypto::blake2b::{Blake2b, Blake2b512},
  traits::{Digest as _, VerificationError, ct},
};

#[cfg(target_arch = "aarch64")]
mod aarch64;
mod dispatch;
mod kernels;
#[cfg(target_arch = "powerpc64")]
mod power;
#[cfg(target_arch = "riscv64")]
mod riscv64;
#[cfg(target_arch = "s390x")]
mod s390x;
#[cfg(target_arch = "wasm32")]
mod wasm;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use dispatch::active_compress;
pub use dispatch::{ALL_KERNELS, KernelId, required_caps};
use kernels::CompressFn;

// ─── Constants ──────────────────────────────────────────────────────────────

/// Memory block size in bytes (RFC 9106 §3).
pub const BLOCK_SIZE: usize = 1024;

/// Memory block size in 64-bit words.
const BLOCK_WORDS: usize = BLOCK_SIZE / 8; // 128

/// Number of synchronisation points per pass (RFC 9106 §3.4).
const SYNC_POINTS: u32 = 4;

/// Minimum salt length in bytes (RFC 9106 §3.1).
pub const MIN_SALT_LEN: usize = 8;

/// Maximum encoded parameter byte length (bounds password/secret/AD/salt).
const MAX_VAR_BYTES: u64 = u32::MAX as u64;

/// Minimum output length in bytes (RFC 9106 §3.1).
pub const MIN_OUTPUT_LEN: usize = 4;

/// OWASP baseline parameters (see [`Argon2Params::new`]).
const DEFAULT_MEMORY_KIB: u32 = 19 * 1024;
const DEFAULT_TIME_COST: u32 = 2;
const DEFAULT_PARALLELISM: u32 = 1;
#[cfg(feature = "phc-strings")]
const DEFAULT_OUTPUT_LEN: usize = 32;
const ARGON2_VERSION: u32 = 0x13;

/// BlaMka P permutation operates on 8×8 matrices of 16-byte "registers",
/// i.e. on 16-u64 slabs.
const P_LANE_WORDS: usize = 16;

/// Minimum per-lane segment work before Rayon lane parallelism is worth its
/// fixed scheduling cost.
#[cfg(feature = "parallel")]
const MIN_PARALLEL_SEGMENT_BLOCKS: u32 = 32;

#[repr(align(64))]
#[derive(Clone, Copy)]
struct MemoryBlock([u64; BLOCK_WORDS]);

impl MemoryBlock {
  #[inline(always)]
  const fn zero() -> Self {
    Self([0u64; BLOCK_WORDS])
  }
}

/// Volatile zeroisation for a slice of u64 words. The workspace `ct`
/// module only exports byte-slice zeroization; Argon2 holds most scratch
/// as `[u64; BLOCK_WORDS]`, so a word-oriented helper is cleaner than
/// repeated byte-slice transmutes.
#[inline]
fn zeroize_u64_slice_no_fence(words: &mut [u64]) {
  let mut chunks = words.chunks_exact_mut(8);
  for chunk in &mut chunks {
    // SAFETY: chunk has exactly 8 initialized u64 values and [u64; 8] has
    // the same alignment requirement as u64.
    unsafe { core::ptr::write_volatile(chunk.as_mut_ptr().cast::<[u64; 8]>(), [0u64; 8]) };
  }
  for w in chunks.into_remainder() {
    // SAFETY: w is a valid, aligned, dereferenceable pointer to initialized u64.
    unsafe { core::ptr::write_volatile(w, 0) };
  }
}

#[inline]
fn zeroize_u64_slice(words: &mut [u64]) {
  zeroize_u64_slice_no_fence(words);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

// ─── Variant ────────────────────────────────────────────────────────────────

/// Argon2 variant selector.
///
/// The variant controls reference-block indexing:
///
/// - `Argon2d` — data-dependent (Argon2d mode throughout).
/// - `Argon2i` — data-independent (Argon2i mode throughout).
/// - `Argon2id` — data-independent for the first half of the first pass, data-dependent afterwards
///   (RFC 9106 §3.4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Argon2Variant {
  Argon2d,
  Argon2i,
  Argon2id,
}

impl Argon2Variant {
  /// Argon2 `y` constant encoded into `H0` per RFC 9106 §3.2.
  #[inline]
  const fn y(self) -> u32 {
    match self {
      Self::Argon2d => 0,
      Self::Argon2i => 1,
      Self::Argon2id => 2,
    }
  }
}

// ─── Error ──────────────────────────────────────────────────────────────────

/// Invalid Argon2 parameter or input.
///
/// Surfaced at construction or derivation time — never at `verify` time (a parameter
/// error during verification would leak whether the encoded hash was valid,
/// so verify collapses these into [`crate::VerificationError`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Argon2Error {
  /// `time_cost` must be ≥ 1.
  InvalidTimeCost,
  /// `memory_cost` must satisfy RFC 9106 bounds (`m ≥ 8 · p`, `m ≤ 2^32 − 1`).
  InvalidMemoryCost,
  /// `parallelism` must be in 1..=2^24-1.
  InvalidParallelism,
  /// `output_len` must be in 4..=2^32-1.
  InvalidOutputLen,
  /// Salt must be ≥ 8 bytes (RFC 9106 §3.1).
  SaltTooShort,
  /// Salt length exceeds 2^32-1 bytes.
  SaltTooLong,
  /// Password length exceeds 2^32-1 bytes.
  PasswordTooLong,
  /// Optional secret length exceeds 2^32-1 bytes.
  SecretTooLong,
  /// Optional associated-data length exceeds 2^32-1 bytes.
  AssociatedDataTooLong,
  /// The requested memory matrix exceeds the target's address space.
  ResourceOverflow,
  /// The allocator refused to provide the memory matrix.
  AllocationFailed,
  /// The platform entropy source failed while generating a PHC salt.
  #[cfg(all(feature = "phc-strings", feature = "getrandom"))]
  EntropyUnavailable,
  /// Password generation parameters exceed the verifier's resource limits.
  #[cfg(feature = "phc-strings")]
  VerificationLimitTooLow,
}

impl fmt::Display for Argon2Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::InvalidTimeCost => "Argon2 time_cost must be at least 1",
      Self::InvalidMemoryCost => "Argon2 memory_cost is out of range (m >= 8 * p, m <= 2^32 - 1)",
      Self::InvalidParallelism => "Argon2 parallelism must be in 1..=2^24-1",
      Self::InvalidOutputLen => "Argon2 output length must be in 4..=2^32-1",
      Self::SaltTooShort => "Argon2 salt must be at least 8 bytes",
      Self::SaltTooLong => "Argon2 salt exceeds 2^32-1 bytes",
      Self::PasswordTooLong => "Argon2 password exceeds 2^32-1 bytes",
      Self::SecretTooLong => "Argon2 secret exceeds 2^32-1 bytes",
      Self::AssociatedDataTooLong => "Argon2 associated data exceeds 2^32-1 bytes",
      Self::ResourceOverflow => "Argon2 memory matrix exceeds the target's address space",
      Self::AllocationFailed => "Argon2 memory-matrix allocation failed",
      #[cfg(all(feature = "phc-strings", feature = "getrandom"))]
      Self::EntropyUnavailable => "Argon2 entropy source unavailable",
      #[cfg(feature = "phc-strings")]
      Self::VerificationLimitTooLow => "Argon2 verification limits do not admit the generation parameters",
    })
  }
}

impl core::error::Error for Argon2Error {}

// ─── Parameters ─────────────────────────────────────────────────────────────

/// Validated Argon2 cost parameters.
///
/// # Examples
///
/// ```rust
/// use rscrypto::{Argon2Params, Argon2id};
///
/// let params = Argon2Params::new(65_536, 3, 4)?;
///
/// let mut out = [0u8; 32];
/// Argon2id::derive(&params, b"password", b"random-salt-1234", &mut out)?;
/// # Ok::<(), rscrypto::Argon2Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Argon2Params {
  time_cost: u32,
  memory_cost_kib: u32,
  parallelism: u32,
}

impl Default for Argon2Params {
  fn default() -> Self {
    Self {
      time_cost: DEFAULT_TIME_COST,
      memory_cost_kib: DEFAULT_MEMORY_KIB,
      parallelism: DEFAULT_PARALLELISM,
    }
  }
}

impl Argon2Params {
  /// Construct an Argon2 v1.3 parameter profile.
  pub const fn new(memory_cost_kib: u32, time_cost: u32, parallelism: u32) -> Result<Self, Argon2Error> {
    if time_cost < 1 {
      return Err(Argon2Error::InvalidTimeCost);
    }
    if parallelism < 1 || parallelism > (1 << 24) - 1 {
      return Err(Argon2Error::InvalidParallelism);
    }
    let Some(min_memory) = parallelism.checked_mul(8) else {
      return Err(Argon2Error::InvalidMemoryCost);
    };
    if memory_cost_kib < min_memory {
      return Err(Argon2Error::InvalidMemoryCost);
    }
    Ok(Self {
      time_cost,
      memory_cost_kib,
      parallelism,
    })
  }

  fn check_inputs(password: &[u8], salt: &[u8], context: Argon2Context<'_>) -> Result<(), Argon2Error> {
    if password.len() as u64 > MAX_VAR_BYTES {
      return Err(Argon2Error::PasswordTooLong);
    }
    if salt.len() < MIN_SALT_LEN {
      return Err(Argon2Error::SaltTooShort);
    }
    if salt.len() as u64 > MAX_VAR_BYTES {
      return Err(Argon2Error::SaltTooLong);
    }
    if context.secret.len() as u64 > MAX_VAR_BYTES {
      return Err(Argon2Error::SecretTooLong);
    }
    if context.associated_data.len() as u64 > MAX_VAR_BYTES {
      return Err(Argon2Error::AssociatedDataTooLong);
    }
    Ok(())
  }

  /// Time cost (iterations).
  #[must_use]
  pub const fn get_time_cost(&self) -> u32 {
    self.time_cost
  }

  /// Memory cost in KiB.
  #[must_use]
  pub const fn get_memory_cost_kib(&self) -> u32 {
    self.memory_cost_kib
  }

  /// Parallelism (lanes).
  #[must_use]
  pub const fn get_parallelism(&self) -> u32 {
    self.parallelism
  }
}

/// Borrowed optional Argon2 inputs that are not encoded in a PHC string.
#[derive(Clone, Copy, Default)]
pub struct Argon2Context<'a> {
  secret: &'a [u8],
  associated_data: &'a [u8],
}

impl fmt::Debug for Argon2Context<'_> {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("Argon2Context")
      .field("secret", &"[REDACTED]")
      .field("secret_len", &self.secret.len())
      .field("associated_data_len", &self.associated_data.len())
      .finish()
  }
}

impl<'a> Argon2Context<'a> {
  /// Borrow a pepper and associated data for one Argon2 operation.
  #[must_use]
  pub const fn new(secret: &'a [u8], associated_data: &'a [u8]) -> Self {
    Self {
      secret,
      associated_data,
    }
  }
}

/// Finite memory and work ceilings for Argon2id password verification.
#[cfg(feature = "phc-strings")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Argon2VerificationLimits {
  max_memory_bytes: u64,
  max_block_work: u64,
  max_parallelism: u32,
}

#[derive(Clone, Copy)]
struct Argon2Shape {
  blocks: u32,
  memory_bytes: u64,
  #[cfg(feature = "phc-strings")]
  block_work: u64,
}

const fn argon2_shape(params: Argon2Params) -> Argon2Shape {
  let lane_group = params.parallelism.strict_mul(SYNC_POINTS);
  let blocks = (params.memory_cost_kib / lane_group).strict_mul(lane_group);
  Argon2Shape {
    blocks,
    memory_bytes: (blocks as u64).strict_mul(BLOCK_SIZE as u64),
    #[cfg(feature = "phc-strings")]
    block_work: (blocks as u64).strict_mul(params.time_cost as u64),
  }
}

#[cfg(feature = "phc-strings")]
impl Argon2VerificationLimits {
  /// Derive ceilings from the largest deployment profile the verifier admits.
  #[must_use]
  pub const fn for_profile(params: Argon2Params) -> Self {
    let shape = argon2_shape(params);
    Self {
      max_memory_bytes: shape.memory_bytes,
      max_block_work: shape.block_work,
      max_parallelism: params.parallelism,
    }
  }

  const fn allows(&self, params: Argon2Params) -> bool {
    let usage = Self::for_profile(params);
    usage.max_memory_bytes <= self.max_memory_bytes
      && usage.max_block_work <= self.max_block_work
      && usage.max_parallelism <= self.max_parallelism
  }
}

#[cfg(feature = "phc-strings")]
impl Default for Argon2VerificationLimits {
  fn default() -> Self {
    Self::for_profile(Argon2Params::default())
  }
}

// ─── Kernel dispatch ────────────────────────────────────────────────────────
//
// [`KernelId`], [`ALL_KERNELS`], [`required_caps`], and the private
// `active_compress` selector live in the [`dispatch`] submodule; the
// per-arch kernels live in sibling files (`kernels.rs` for portable,
// `aarch64.rs` for NEON, etc.). This module only *calls through* the
// resolved [`CompressFn`] — no inline BlaMka code below this point.

// ─── Diagnostic entry points for per-kernel tests ──────────────────────────

/// The kernel selected by the runtime dispatcher on the current host.
///
/// Useful for reporting and for tests that cross-check the active kernel
/// against the runtime dispatcher.
#[cfg(feature = "diag")]
#[must_use]
pub fn diag_active_kernel() -> KernelId {
  dispatch::active_kernel()
}

/// Hash via the kernel selected by the runtime dispatcher (default path).
///
/// # Errors
///
/// Returns [`Argon2Error`] for invalid operation inputs or output length.
#[cfg(feature = "diag")]
pub fn diag_hash_active(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
) -> Result<(), Argon2Error> {
  argon2_hash(params, password, salt, variant, out)
}

/// Hash via the **portable** kernel, regardless of host capabilities.
///
/// # Errors
///
/// Returns [`Argon2Error`] for invalid operation inputs or output length.
#[cfg(feature = "diag")]
pub fn diag_hash_portable(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
) -> Result<(), Argon2Error> {
  argon2_hash_with_kernel_diag_blake2b(
    params,
    password,
    salt,
    variant,
    out,
    dispatch::compress_fn_for(KernelId::Portable),
  )
}

/// Hash via the aarch64 NEON kernel.
///
/// # Errors
///
/// Returns [`Argon2Error`] for invalid parameters.
#[cfg(all(feature = "diag", target_arch = "aarch64"))]
pub fn diag_hash_aarch64_neon(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
) -> Result<(), Argon2Error> {
  argon2_hash_with_kernel(
    params,
    password,
    salt,
    variant,
    out,
    dispatch::compress_fn_for(KernelId::Aarch64Neon),
  )
}

/// Hash via the x86_64 AVX2 kernel.
///
/// # Errors
///
/// Returns [`Argon2Error`] for invalid parameters.
///
/// # Panics
///
/// Panics if the host does not support AVX2. The per-kernel tests
/// gate this call on `crate::platform::caps()` having the kernel's
/// required caps before invoking it.
#[cfg(all(feature = "diag", target_arch = "x86_64"))]
pub fn diag_hash_x86_avx2(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
) -> Result<(), Argon2Error> {
  assert!(
    crate::platform::caps().has(dispatch::required_caps(KernelId::X86Avx2)),
    "AVX2 not available on host"
  );
  argon2_hash_with_kernel(
    params,
    password,
    salt,
    variant,
    out,
    dispatch::compress_fn_for(KernelId::X86Avx2),
  )
}

/// Hash via the x86_64 AVX-512 kernel.
///
/// # Errors
///
/// Returns [`Argon2Error`] for invalid parameters.
///
/// # Panics
///
/// Panics if the host does not support AVX-512F + AVX-512VL.
#[cfg(all(feature = "diag", target_arch = "x86_64"))]
pub fn diag_hash_x86_avx512(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
) -> Result<(), Argon2Error> {
  assert!(
    crate::platform::caps().has(dispatch::required_caps(KernelId::X86Avx512)),
    "AVX-512F + AVX-512VL not available on host"
  );
  argon2_hash_with_kernel(
    params,
    password,
    salt,
    variant,
    out,
    dispatch::compress_fn_for(KernelId::X86Avx512),
  )
}

/// Hash via the powerpc64 VSX kernel.
///
/// # Errors
///
/// Returns [`Argon2Error`] for invalid parameters.
///
/// # Panics
///
/// Panics if the host does not support VSX.
#[cfg(all(feature = "diag", target_arch = "powerpc64"))]
pub fn diag_hash_power_vsx(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
) -> Result<(), Argon2Error> {
  assert!(
    crate::platform::caps().has(dispatch::required_caps(KernelId::PowerVsx)),
    "POWER VSX not available on host"
  );
  argon2_hash_with_kernel(
    params,
    password,
    salt,
    variant,
    out,
    dispatch::compress_fn_for(KernelId::PowerVsx),
  )
}

/// Hash via the s390x z/Vector kernel.
///
/// # Errors
///
/// Returns [`Argon2Error`] for invalid parameters.
///
/// # Panics
///
/// Panics if the host does not support the z13+ vector facility.
#[cfg(all(feature = "diag", target_arch = "s390x"))]
pub fn diag_hash_s390x_vector(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
) -> Result<(), Argon2Error> {
  assert!(
    crate::platform::caps().has(dispatch::required_caps(KernelId::S390xVector)),
    "s390x vector facility not available on host"
  );
  argon2_hash_with_kernel(
    params,
    password,
    salt,
    variant,
    out,
    dispatch::compress_fn_for(KernelId::S390xVector),
  )
}

/// Hash via the riscv64 RVV kernel.
///
/// # Errors
///
/// Returns [`Argon2Error`] for invalid parameters.
///
/// # Panics
///
/// Panics if the host does not support the RISC-V V extension.
#[cfg(all(feature = "diag", target_arch = "riscv64"))]
pub fn diag_hash_riscv64_v(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
) -> Result<(), Argon2Error> {
  assert!(
    crate::platform::caps().has(dispatch::required_caps(KernelId::Riscv64V)),
    "RISC-V V extension not available on host"
  );
  argon2_hash_with_kernel(
    params,
    password,
    salt,
    variant,
    out,
    dispatch::compress_fn_for(KernelId::Riscv64V),
  )
}

/// Hash via the wasm32 simd128 kernel.
///
/// # Errors
///
/// Returns [`Argon2Error`] for invalid parameters.
///
/// # Panics
///
/// Panics if the host does not support wasm SIMD128.
#[cfg(all(feature = "diag", target_arch = "wasm32"))]
pub fn diag_hash_wasm_simd128(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
) -> Result<(), Argon2Error> {
  assert!(
    crate::platform::caps().has(dispatch::required_caps(KernelId::WasmSimd128)),
    "wasm simd128 not available on host"
  );
  argon2_hash_with_kernel(
    params,
    password,
    salt,
    variant,
    out,
    dispatch::compress_fn_for(KernelId::WasmSimd128),
  )
}

/// Single-block compress via the portable kernel (diagnostic).
///
/// Runs one 1 KiB BlaMka compression, bypassing the full hash pipeline.
/// Used by kernel microbenches and cross-kernel differential tests.
#[cfg(feature = "diag")]
pub fn diag_compress_portable(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  // SAFETY: portable kernel has no unsafe preconditions.
  unsafe { kernels::compress_portable(dst, x, y, xor_into) }
}

/// Single-block compress via the aarch64 NEON kernel (diagnostic).
#[cfg(all(feature = "diag", target_arch = "aarch64"))]
pub fn diag_compress_aarch64_neon(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  // SAFETY: NEON is baseline on aarch64 — the module gate enforces the
  // target_arch check at compile time.
  unsafe { aarch64::compress_neon(dst, x, y, xor_into) }
}

/// Single-block compress via the x86_64 AVX2 kernel (diagnostic).
///
/// # Panics
///
/// Panics if the host does not support AVX2.
#[cfg(all(feature = "diag", target_arch = "x86_64"))]
pub fn diag_compress_x86_avx2(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  assert!(
    crate::platform::caps().has(dispatch::required_caps(KernelId::X86Avx2)),
    "AVX2 not available on host"
  );
  // SAFETY: the assertion above witnesses the AVX2 cap on the host before
  // the call, satisfying `compress_avx2`'s `#[target_feature]` precondition.
  unsafe { x86_64::compress_avx2(dst, x, y, xor_into) }
}

/// Single-block compress via the x86_64 AVX-512 kernel (diagnostic).
///
/// # Panics
///
/// Panics if the host does not support AVX-512F + AVX-512VL.
#[cfg(all(feature = "diag", target_arch = "x86_64"))]
pub fn diag_compress_x86_avx512(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  assert!(
    crate::platform::caps().has(dispatch::required_caps(KernelId::X86Avx512)),
    "AVX-512F + AVX-512VL not available on host"
  );
  // SAFETY: the assertion above witnesses both AVX-512F and AVX-512VL on
  // the host, satisfying `compress_avx512`'s `#[target_feature]`.
  unsafe { x86_64::compress_avx512(dst, x, y, xor_into) }
}

/// Single-block compress via the powerpc64 VSX kernel (diagnostic).
///
/// # Panics
///
/// Panics if the host does not support VSX.
#[cfg(all(feature = "diag", target_arch = "powerpc64"))]
pub fn diag_compress_power_vsx(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  assert!(
    crate::platform::caps().has(dispatch::required_caps(KernelId::PowerVsx)),
    "POWER VSX not available on host"
  );
  // SAFETY: assertion witnesses VSX on the host.
  unsafe { power::compress_vsx(dst, x, y, xor_into) }
}

/// Single-block compress via the s390x z/Vector kernel (diagnostic).
///
/// # Panics
///
/// Panics if the host does not support the z13+ vector facility.
#[cfg(all(feature = "diag", target_arch = "s390x"))]
pub fn diag_compress_s390x_vector(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  assert!(
    crate::platform::caps().has(dispatch::required_caps(KernelId::S390xVector)),
    "s390x vector facility not available on host"
  );
  // SAFETY: assertion witnesses the vector facility on the host.
  unsafe { s390x::compress_vector(dst, x, y, xor_into) }
}

/// Single-block compress via the riscv64 RVV kernel (diagnostic).
///
/// # Panics
///
/// Panics if the host does not support the RISC-V V extension.
#[cfg(all(feature = "diag", target_arch = "riscv64"))]
pub fn diag_compress_riscv64_v(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  assert!(
    crate::platform::caps().has(dispatch::required_caps(KernelId::Riscv64V)),
    "RISC-V V extension not available on host"
  );
  // SAFETY: assertion witnesses the V extension on the host.
  unsafe { riscv64::compress_rvv(dst, x, y, xor_into) }
}

/// Single-block compress via the wasm32 simd128 kernel (diagnostic).
///
/// # Panics
///
/// Panics if the host does not support wasm SIMD128.
#[cfg(all(feature = "diag", target_arch = "wasm32"))]
pub fn diag_compress_wasm_simd128(
  dst: &mut [u64; BLOCK_WORDS],
  x: &[u64; BLOCK_WORDS],
  y: &[u64; BLOCK_WORDS],
  xor_into: bool,
) {
  assert!(
    crate::platform::caps().has(dispatch::required_caps(KernelId::WasmSimd128)),
    "wasm simd128 not available on host"
  );
  // SAFETY: assertion witnesses simd128 availability on the host.
  unsafe { wasm::compress_simd128(dst, x, y, xor_into) }
}

/// Block-word count (128) — exposed for diagnostic kernel tests.
#[cfg(feature = "diag")]
pub const DIAG_BLOCK_WORDS: usize = BLOCK_WORDS;

// ─── H' variable-length Blake2b helper (RFC 9106 §3.3) ──────────────────────

/// Compute `H'(input, out.len())` per RFC 9106 §3.3 into `out`.
///
/// `out.len()` may be any positive value; for `≤ 64` this is a single
/// Blake2b call, and for larger outputs it becomes a chained 64→32-byte
/// extraction with a tail Blake2b of the remaining bytes.
fn h_prime(input_parts: &[&[u8]], out: &mut [u8]) {
  let out_len = out.len();
  assert!(out_len > 0, "H' output length must be positive");
  // Feed LE32(out_len) then the input parts into Blake2b. For out_len <= 64
  // the single-block output is the answer directly.
  let len_le = u32::try_from(out_len)
    .unwrap_or_else(|_| unreachable!("Argon2 H' output length was checked before expansion"))
    .to_le_bytes();

  if out_len <= 64 {
    let mut hasher =
      Blake2b::new(u8::try_from(out_len).unwrap_or_else(|_| unreachable!("guarded by `out_len <= 64` above")));
    hasher.update(&len_le);
    for part in input_parts {
      hasher.update(part);
    }
    hasher.finalize_into(out);
    return;
  }

  // out_len > 64: chained 64-byte outputs, copying the first 32 bytes of
  // each V_i into `out`, with a final V_{r+1} of length (out_len − 32·r).
  // r = ceil(out_len/32) - 2
  let r = (out_len.strict_add(31) / 32).strict_sub(2);
  // V_1 = Blake2b(LE32(T) || X, 64)
  let mut v_prev: [u8; 64] = {
    let mut hasher = Blake2b512::new();
    hasher.update(&len_le);
    for part in input_parts {
      hasher.update(part);
    }
    hasher.finalize()
  };
  out[..32].copy_from_slice(&v_prev[..32]);

  // V_i = Blake2b(V_{i-1}, 64) for i = 2..=r
  for i in 1..r {
    let v_next = Blake2b512::digest(&v_prev);
    out[i.strict_mul(32)..i.strict_mul(32).strict_add(32)].copy_from_slice(&v_next[..32]);
    v_prev = v_next;
  }

  // V_{r+1} = Blake2b(V_r, out_len − 32·r)
  let tail_off = r.strict_mul(32);
  let tail_len = out_len.strict_sub(tail_off);
  let mut hasher = Blake2b::new(
    u8::try_from(tail_len)
      .unwrap_or_else(|_| unreachable!("tail_len ∈ (32, 64] by construction (out_len > 64, r = ⌈out_len/32⌉ - 2)")),
  );
  hasher.update(&v_prev);
  hasher.finalize_into(&mut out[tail_off..]);

  ct::zeroize(&mut v_prev);
}

#[cfg(feature = "diag")]
fn h_prime_diag_blake2b_portable(input_parts: &[&[u8]], out: &mut [u8]) {
  let out_len = out.len();
  assert!(out_len > 0, "H' output length must be positive");
  let len_le = u32::try_from(out_len)
    .unwrap_or_else(|_| unreachable!("Argon2 H' output length was checked before expansion"))
    .to_le_bytes();

  if out_len <= 64 {
    let parts = [&len_le[..], input_parts[0]];
    if input_parts.len() == 1 {
      crate::hashes::crypto::blake2b::diag_hash_parts_portable(out_len as u8, &parts, out);
    } else {
      let mut data = [0u8; BLOCK_SIZE + 16];
      let mut pos = 0usize;
      data[pos..pos.strict_add(4)].copy_from_slice(&len_le);
      pos = pos.strict_add(4);
      for part in input_parts {
        data[pos..pos.strict_add(part.len())].copy_from_slice(part);
        pos = pos.strict_add(part.len());
      }
      crate::hashes::crypto::blake2b::diag_hash_parts_portable(out_len as u8, &[&data[..pos]], out);
    }
    return;
  }

  let r = (out_len.strict_add(31) / 32).strict_sub(2);
  let mut data = [0u8; BLOCK_SIZE + 16];
  let mut pos = 0usize;
  data[pos..pos.strict_add(4)].copy_from_slice(&len_le);
  pos = pos.strict_add(4);
  for part in input_parts {
    data[pos..pos.strict_add(part.len())].copy_from_slice(part);
    pos = pos.strict_add(part.len());
  }

  let mut v_prev = [0u8; 64];
  crate::hashes::crypto::blake2b::diag_hash_parts_portable(64, &[&data[..pos]], &mut v_prev);
  out[..32].copy_from_slice(&v_prev[..32]);

  for i in 1..r {
    let mut v_next = [0u8; 64];
    crate::hashes::crypto::blake2b::diag_hash_parts_portable(64, &[&v_prev], &mut v_next);
    out[i.strict_mul(32)..i.strict_mul(32).strict_add(32)].copy_from_slice(&v_next[..32]);
    v_prev = v_next;
  }

  let tail_off = r.strict_mul(32);
  let tail_len = out_len.strict_sub(tail_off);
  crate::hashes::crypto::blake2b::diag_hash_parts_portable(tail_len as u8, &[&v_prev], &mut out[tail_off..]);
  ct::zeroize(&mut v_prev);
}

// ─── H0 initialisation (RFC 9106 §3.2) ──────────────────────────────────────

/// Compute the 64-byte `H0` seed.
fn compute_h0(
  params: &Argon2Params,
  context: Argon2Context<'_>,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  output_len: usize,
) -> [u8; 64] {
  // All input lengths have been bounded ≤ MAX_VAR_BYTES (= u32::MAX) by
  // `check_inputs` before reaching here, so the `try_from` calls below are
  // infallible. We use `try_from + expect` rather than `as u32` to keep the
  // invariant readable at the call site.
  let len_u32 = |label: &'static str, len: usize| -> [u8; 4] {
    u32::try_from(len)
      .unwrap_or_else(|_| panic!("Argon2 H0: {label} length exceeded MAX_VAR_BYTES; check_inputs should have rejected"))
      .to_le_bytes()
  };

  let mut hasher = Blake2b512::new();
  hasher.update(&params.parallelism.to_le_bytes());
  hasher.update(&len_u32("output", output_len));
  hasher.update(&params.memory_cost_kib.to_le_bytes());
  hasher.update(&params.time_cost.to_le_bytes());
  hasher.update(&ARGON2_VERSION.to_le_bytes());
  hasher.update(&variant.y().to_le_bytes());
  hasher.update(&len_u32("password", password.len()));
  hasher.update(password);
  hasher.update(&len_u32("salt", salt.len()));
  hasher.update(salt);
  hasher.update(&len_u32("secret", context.secret.len()));
  hasher.update(context.secret);
  hasher.update(&len_u32("associated_data", context.associated_data.len()));
  hasher.update(context.associated_data);
  hasher.finalize()
}

#[cfg(feature = "diag")]
fn compute_h0_diag_blake2b_portable(
  params: &Argon2Params,
  context: Argon2Context<'_>,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  output_len: usize,
) -> [u8; 64] {
  let len_u32 = |label: &'static str, len: usize| -> [u8; 4] {
    u32::try_from(len)
      .unwrap_or_else(|_| panic!("Argon2 H0: {label} length exceeded MAX_VAR_BYTES; check_inputs should have rejected"))
      .to_le_bytes()
  };

  let parallelism = params.parallelism.to_le_bytes();
  let output_len = len_u32("output", output_len);
  let memory_cost = params.memory_cost_kib.to_le_bytes();
  let time_cost = params.time_cost.to_le_bytes();
  let version = ARGON2_VERSION.to_le_bytes();
  let variant = variant.y().to_le_bytes();
  let password_len = len_u32("password", password.len());
  let salt_len = len_u32("salt", salt.len());
  let secret_len = len_u32("secret", context.secret.len());
  let associated_data_len = len_u32("associated_data", context.associated_data.len());
  let mut out = [0u8; 64];
  crate::hashes::crypto::blake2b::diag_hash_parts_portable(
    64,
    &[
      &parallelism,
      &output_len,
      &memory_cost,
      &time_cost,
      &version,
      &variant,
      &password_len,
      password,
      &salt_len,
      salt,
      &secret_len,
      context.secret,
      &associated_data_len,
      context.associated_data,
    ],
    &mut out,
  );
  out
}

// ─── Block conversion ───────────────────────────────────────────────────────

#[inline(always)]
fn block_from_bytes(bytes: &[u8; BLOCK_SIZE]) -> MemoryBlock {
  let mut out = MemoryBlock::zero();
  for i in 0..BLOCK_WORDS {
    out.0[i] = u64::from_le_bytes(
      bytes[i.strict_mul(8)..i.strict_mul(8).strict_add(8)]
        .try_into()
        .unwrap(),
    );
  }
  out
}

#[inline(always)]
fn block_to_bytes(block: &[u64; BLOCK_WORDS]) -> [u8; BLOCK_SIZE] {
  let mut out = [0u8; BLOCK_SIZE];
  for i in 0..BLOCK_WORDS {
    out[i.strict_mul(8)..i.strict_mul(8).strict_add(8)].copy_from_slice(&block[i].to_le_bytes());
  }
  out
}

// ─── Argon2i pseudo-random address stream (RFC 9106 §3.4.2) ────────────────

/// Buffer of 128 `J1||J2` word pairs used for a single segment of Argon2i
/// (or Argon2id's data-independent slices).
#[derive(Clone)]
struct AddressBlock {
  words: MemoryBlock,
}

impl AddressBlock {
  fn zeros() -> Self {
    Self {
      words: MemoryBlock::zero(),
    }
  }

  /// Generate a fresh address block keyed by
  /// `(pass, lane, slice, blocks, total_passes, variant_y, counter)`.
  #[allow(clippy::too_many_arguments)] // RFC 9106 §3.4.2 fixes this list; wrapping it in a struct is empty ceremony.
  fn refresh(
    &mut self,
    compress: CompressFn,
    pass: u32,
    lane: u32,
    slice: u32,
    blocks: u32,
    total_passes: u32,
    variant_y: u32,
    counter: u64,
  ) {
    let mut input = MemoryBlock::zero();
    input.0[0] = pass as u64;
    input.0[1] = lane as u64;
    input.0[2] = slice as u64;
    input.0[3] = blocks as u64;
    input.0[4] = total_passes as u64;
    input.0[5] = variant_y as u64;
    input.0[6] = counter;

    let zero = MemoryBlock::zero();
    let mut intermediate = MemoryBlock::zero();
    // SAFETY: the CompressFn came from `active_compress()` (or
    // `compress_fn_for` in per-kernel tests), which only returns a
    // kernel whose `required_caps` are a subset of the host's caps.
    unsafe {
      compress(&mut intermediate.0, &zero.0, &input.0, /* xor_into = */ false);
      compress(&mut self.words.0, &zero.0, &intermediate.0, /* xor_into = */ false);
    }
  }
}

// ─── Fill engine ───────────────────────────────────────────────────────────

/// Layout helper. A fresh Argon2 hash call allocates `m' × p` contiguous
/// u64 blocks; rows correspond to lanes, columns to positions within a lane.
struct Matrix {
  blocks: Vec<MemoryBlock>,
  lane_len: u32,
  lanes: u32,
  segment_len: u32,
}

impl Matrix {
  fn new(params: Argon2Params) -> Result<Self, Argon2Error> {
    let shape = argon2_shape(params);
    let lanes = params.parallelism;
    let m_prime = shape.blocks;
    let lane_len = m_prime / lanes;
    let segment_len = lane_len / SYNC_POINTS;
    if shape.memory_bytes > isize::MAX as u64 {
      return Err(Argon2Error::ResourceOverflow);
    }
    let total = m_prime as usize;
    let mut blocks = Vec::new();
    blocks
      .try_reserve_exact(total)
      .map_err(|_| Argon2Error::AllocationFailed)?;
    blocks.resize(total, MemoryBlock::zero());
    Ok(Self {
      blocks,
      lane_len,
      lanes,
      segment_len,
    })
  }

  #[inline(always)]
  fn len(&self) -> usize {
    self.blocks.len()
  }

  #[inline(always)]
  fn index(&self, lane: u32, col: u32) -> usize {
    (lane as usize)
      .strict_mul(self.lane_len as usize)
      .strict_add(col as usize)
  }

  #[inline(always)]
  fn get(&self, lane: u32, col: u32) -> &[u64; BLOCK_WORDS] {
    let idx = self.index(lane, col);
    self.get_index(idx)
  }

  #[inline(always)]
  fn get_index(&self, idx: usize) -> &[u64; BLOCK_WORDS] {
    &self.blocks[idx].0
  }

  #[inline(always)]
  fn set(&mut self, lane: u32, col: u32, block: MemoryBlock) {
    let idx = self.index(lane, col);
    self.blocks[idx] = block;
  }
}

impl Drop for Matrix {
  fn drop(&mut self) {
    for block in &mut self.blocks {
      zeroize_u64_slice_no_fence(&mut block.0);
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}

/// Pointer-based view of the memory matrix shared between sequential and
/// parallel fill paths.
///
/// Carries no Rust-level borrow of the underlying [`Matrix`]; ownership /
/// aliasing discipline is enforced by the callers:
///
/// - **Sequential path** (`fill_segment`): the view is constructed under an exclusive `&mut Matrix`
///   borrow, the matrix is single-threaded for the call's duration, and the view is dropped before
///   control returns to `argon2_hash`. Exclusivity is trivial.
/// - **Parallel path** (`fill_slice_parallel`): the view is shared across `rayon::scope` tasks.
///   Disjointness across tasks is upheld by the Argon2 reference-index function (RFC 9106 §3.4),
///   which guarantees that within a single slice every lane writes only to its own segment range
///   and reads only from blocks in already-completed slices (or strictly earlier columns of the
///   same segment, which are this task's own writes).
///
/// `Send + Sync` are required to cross rayon thread boundaries; both are
/// `unsafe impl`d on the assumption that callers honour the discipline
/// above. The methods on this view are themselves `unsafe` for the same
/// reason — the runtime, not the type system, owns the soundness proof.
#[derive(Clone, Copy)]
struct MatrixView {
  ptr: *mut MemoryBlock,
  total_len: usize,
}

#[cfg(feature = "parallel")]
// SAFETY: `MatrixView` is `Send + Sync` only because the parallel filler
// (`fill_slice_parallel`) honours the RFC 9106 §3.4 disjointness discipline:
//   1. Within one slice, lane `i` writes exclusively to its own segment and reads only from (a)
//      blocks in already-completed slices, or (b) earlier columns of the in-progress segment which
//      the same task wrote.
//   2. `rayon::scope` synchronises all per-lane tasks before advancing to the next slice — no
//      read-write race can cross slice boundaries.
//   3. Sequential callers (`fill_segment`) hold an exclusive `&mut Matrix` borrow for the duration
//      of the call and never split the view.
// All `block`/`block_mut` methods are `unsafe`; the type system does not
// enforce the proof, the runtime does.
unsafe impl Send for MatrixView {}
#[cfg(feature = "parallel")]
// SAFETY: see the Send impl above — same disjointness argument.
unsafe impl Sync for MatrixView {}

impl MatrixView {
  #[inline(always)]
  fn from_blocks(blocks: &mut [MemoryBlock]) -> Self {
    Self {
      ptr: blocks.as_mut_ptr(),
      total_len: blocks.len(),
    }
  }

  /// Borrow the block at flat index `idx` immutably.
  ///
  /// # Safety
  ///
  /// `idx < self.total_len`, and no concurrent task may be mutating the
  /// block at `idx` while the returned reference is in use. The lifetime
  /// `'a` is unconstrained at the type level — the caller is responsible
  /// for keeping the underlying `Matrix` alive and aliasing-free.
  #[inline(always)]
  unsafe fn block<'a>(self, idx: usize) -> &'a [u64; BLOCK_WORDS] {
    debug_assert!(idx < self.total_len);
    // SAFETY: caller guarantees `idx` is in bounds, no concurrent write,
    // and that the `Matrix` outlives the returned borrow.
    unsafe { &(*self.ptr.add(idx)).0 }
  }

  /// Borrow the block at flat index `idx` mutably.
  ///
  /// # Safety
  ///
  /// `idx < self.total_len`, and the block at `idx` must be exclusively
  /// owned by the calling task — no other task (or this task) may hold a
  /// concurrent immutable or mutable reference to the same index.
  #[inline(always)]
  unsafe fn block_mut<'a>(self, idx: usize) -> &'a mut [u64; BLOCK_WORDS] {
    debug_assert!(idx < self.total_len);
    // SAFETY: caller guarantees `idx` is in bounds, exclusive access, and
    // that the `Matrix` outlives the returned borrow.
    unsafe { &mut (*self.ptr.add(idx)).0 }
  }
}

/// Compute the reference block `(ref_lane, ref_index)` for a position
/// `(lane, col)` in pass `pass`, using `j1`/`j2` pseudo-random words.
#[allow(clippy::too_many_arguments)] // RFC 9106 §3.4 ties eight fields together; a ctx struct would just forward them all.
#[inline(always)]
fn reference_index(
  pass: u32,
  lane: u32,
  slice: u32,
  col: u32,
  j1: u32,
  j2: u32,
  lanes: u32,
  segment_len: u32,
  lane_len: u32,
) -> (u32, u32) {
  // Reference lane
  let ref_lane = if pass == 0 && slice == 0 { lane } else { j2 % lanes };

  // Reference area size
  let same_lane = ref_lane == lane;
  let position_in_segment = col.wrapping_sub(slice.wrapping_mul(segment_len));
  // We know position_in_segment < segment_len (by construction).
  let area_size: u32 = if pass == 0 {
    // First pass: previous slices in the current lane are available.
    if same_lane {
      // Blocks 0..col − 1 available (col excludes position itself).
      // col = slice·segment_len + position_in_segment; position > 0 or col > 0.
      // Reference area = col − 1
      col.wrapping_sub(1)
    } else {
      // Other lane: blocks in slices 0..slice completed, minus 1 if position
      // in current segment == 0 (prevents self-reference in racing lane).
      let completed_slices = slice.wrapping_mul(segment_len);
      if position_in_segment == 0 {
        completed_slices.wrapping_sub(1)
      } else {
        completed_slices
      }
    }
  } else {
    // Subsequent passes: wrap the full lane, excluding current segment's
    // already-computed portion for same-lane references.
    if same_lane {
      // lane_len − segment_len + position_in_segment − 1
      lane_len
        .wrapping_sub(segment_len)
        .wrapping_add(position_in_segment)
        .wrapping_sub(1)
    } else {
      // lane_len − segment_len, minus 1 if position_in_segment == 0
      let base = lane_len.wrapping_sub(segment_len);
      if position_in_segment == 0 {
        base.wrapping_sub(1)
      } else {
        base
      }
    }
  };

  // Map j1 to a reference slot using the φ function (RFC 9106 §3.4.1.2).
  let j1_u64 = j1 as u64;
  let relative_position = {
    let x = (j1_u64.wrapping_mul(j1_u64)) >> 32;
    let y = (area_size as u64).wrapping_mul(x) >> 32;
    (area_size as u64).wrapping_sub(1).wrapping_sub(y) as u32
  };

  // Absolute start position of the reference area (wraps across the lane).
  let start_position = if pass == 0 || slice == (SYNC_POINTS - 1) {
    0
  } else {
    (slice.wrapping_add(1)).wrapping_mul(segment_len)
  };

  let ref_index = (start_position.wrapping_add(relative_position)) % lane_len;

  (ref_lane, ref_index)
}

/// Fill a single segment (lane × slice) of the matrix.
///
/// Safe wrapper: takes an exclusive `&mut Matrix`, constructs a
/// [`MatrixView`] from the underlying block storage, and dispatches into
/// [`fill_segment_inner`]. The exclusive borrow makes the inner kernel's
/// safety contract trivially satisfied for single-threaded callers.
#[allow(clippy::too_many_arguments)] // RFC 9106 §3.4 fixes this list; struct wrapper is empty ceremony.
fn fill_segment(
  matrix: &mut Matrix,
  compress: CompressFn,
  pass: u32,
  lane: u32,
  slice: u32,
  variant: Argon2Variant,
  time_cost: u32,
) {
  let lanes = matrix.lanes;
  let segment_len = matrix.segment_len;
  let lane_len = matrix.lane_len;
  let total_blocks = matrix.len() as u32;
  let view = MatrixView::from_blocks(&mut matrix.blocks);
  // SAFETY: `&mut matrix` is held exclusively for the duration of the
  // call. The view is the only handle to the matrix's storage during
  // `fill_segment_inner`; aliasing and concurrency contracts are
  // trivially upheld.
  unsafe {
    fill_segment_inner(
      view,
      compress,
      pass,
      lane,
      slice,
      lanes,
      segment_len,
      lane_len,
      total_blocks,
      variant,
      time_cost,
    );
  }
}

/// Fill a single segment via a [`MatrixView`].
///
/// Common kernel for the sequential and parallel fill paths. The body is
/// identical to the legacy `fill_segment(&mut Matrix, ...)`; the only
/// change is that block reads/writes go through [`MatrixView::block`] /
/// [`MatrixView::block_mut`] rather than through `&mut Matrix`.
///
/// # Safety
///
/// Callers must guarantee that, for the indices touched by this call:
///
/// - The current segment range `[lane * lane_len + slice * segment_len .. lane * lane_len + (slice
///   + 1) * segment_len]` is exclusively writeable by this task (no other task may read or write
///   within this range while this call runs).
/// - All read indices are stable: blocks at any flat index outside this task's segment range must
///   not be mutated by any other task while this call runs.
///
/// Both conditions are upheld by:
/// - The sequential path's exclusive `&mut Matrix` borrow, OR
/// - The Argon2 reference-index function (RFC 9106 §3.4) when called from `fill_slice_parallel` —
///   see the doc-comment on [`MatrixView`] for the disjointness argument.
#[allow(clippy::too_many_arguments, clippy::doc_lazy_continuation)]
unsafe fn fill_segment_inner(
  view: MatrixView,
  compress: CompressFn,
  pass: u32,
  lane: u32,
  slice: u32,
  lanes: u32,
  segment_len: u32,
  lane_len: u32,
  total_blocks: u32,
  variant: Argon2Variant,
  time_cost: u32,
) {
  let variant_y = variant.y();
  let lane_len_usize = lane_len as usize;
  let lane_base = (lane as usize).strict_mul(lane_len_usize);

  // Determine if this segment is data-independent:
  //   Argon2i: always
  //   Argon2id: pass == 0 and slice in 0..2
  //   Argon2d: never
  let is_independent = match variant {
    Argon2Variant::Argon2i => true,
    Argon2Variant::Argon2id => pass == 0 && slice < 2,
    Argon2Variant::Argon2d => false,
  };

  let mut address_block = AddressBlock::zeros();
  let mut address_counter: u64 = 0;
  if is_independent {
    address_counter = 1; // RFC 9106: counter starts at 1 for first block.
    address_block.refresh(
      compress,
      pass,
      lane,
      slice,
      total_blocks,
      time_cost,
      variant_y,
      address_counter,
    );
  }

  // Starting column for this segment. Skip first two blocks of lane 0 on
  // pass 0 — they're computed during initialisation.
  let starting_col: u32 = if pass == 0 && slice == 0 { 2 } else { 0 };
  let segment_start = slice.strict_mul(segment_len);
  let first_col = segment_start.strict_add(starting_col);

  let mut prev_idx = if first_col == 0 {
    lane_base.strict_add(lane_len_usize.strict_sub(1))
  } else {
    lane_base.strict_add(first_col.strict_sub(1) as usize)
  };

  for seg_col in starting_col..segment_len {
    let col = segment_start.strict_add(seg_col);
    let cur_idx = lane_base.strict_add(col as usize);

    // Determine J1, J2
    let (j1, j2) = if is_independent {
      let addr_pos = (seg_col as usize) % BLOCK_WORDS;
      if addr_pos == 0 && seg_col != 0 {
        address_counter = address_counter.strict_add(1);
        address_block.refresh(
          compress,
          pass,
          lane,
          slice,
          total_blocks,
          time_cost,
          variant_y,
          address_counter,
        );
      }
      let word = address_block.words.0[addr_pos];
      ((word & 0xFFFF_FFFFu64) as u32, (word >> 32) as u32)
    } else {
      // Data-dependent: J1, J2 from the previous block.
      // SAFETY: `prev_idx` is either `lane_base + first_col - 1` or the
      // previous iteration's `cur_idx`. Both are within this task's lane
      // (so within either the segment range or this lane's already-
      // completed columns); no concurrent writer touches them.
      let prev_block = unsafe { view.block(prev_idx) };
      let word = prev_block[0];
      ((word & 0xFFFF_FFFFu64) as u32, (word >> 32) as u32)
    };

    let (ref_lane, ref_index) = reference_index(pass, lane, slice, col, j1, j2, lanes, segment_len, lane_len);

    // Compute new block: G(B[lane][prev], B[ref_lane][ref_index]), optionally
    // XOR-accumulated into existing block (v1.3, pass > 0).
    let xor_into = pass > 0;
    let ref_idx = (ref_lane as usize)
      .strict_mul(lane_len_usize)
      .strict_add(ref_index as usize);
    // SAFETY:
    // - `prev_idx` is either the previous iteration's `cur_idx` or (on the segment boundary) `lane_base
    //   + lane_len - 1` — both within this task's own lane, never concurrently written.
    // - `(ref_lane, ref_index)` per RFC 9106 §3.4 lives in a completed region, never within the current
    //   slice's still-being-written range. Not concurrently written.
    // - `cur_idx = lane_base + col` is in this task's exclusive segment write range per
    //   `fill_segment_inner`'s safety contract.
    // - `prev_idx` and `cur_idx` differ by design (`cur_idx` is the block being written this iteration,
    //   `prev_idx` is one position earlier or wrapped around to the lane's last block). `ref_idx` is in
    //   a completed region outside the current segment, never the write target. The three block
    //   locations are pairwise disjoint.
    // - `compress` was resolved by `active_compress`, so its `required_caps` are a subset of the host's
    //   caps.
    unsafe {
      let prev_block = view.block(prev_idx);
      let ref_block = view.block(ref_idx);
      let dst = view.block_mut(cur_idx);
      compress(dst, prev_block, ref_block, xor_into);
    }

    prev_idx = cur_idx;
  }
}

// ─── Slice driver (sequential / parallel) ──────────────────────────────────

/// Drive one slice's worth of segment fills.
///
/// Sequential: one segment per lane, in order. Always available.
#[inline]
fn fill_slice_sequential(
  matrix: &mut Matrix,
  compress: CompressFn,
  pass: u32,
  slice: u32,
  variant: Argon2Variant,
  time_cost: u32,
) {
  let lanes = matrix.lanes;
  for lane in 0..lanes {
    fill_segment(matrix, compress, pass, lane, slice, variant, time_cost);
  }
}

/// Drive one slice's segment fills in parallel: one rayon task per lane.
///
/// Within a slice, every lane writes only to its own segment range and
/// reads only from already-completed regions (RFC 9106 §3.4). The shared
/// [`MatrixView`] therefore carries no aliasing hazard across the
/// `rayon::scope` tasks — see the `MatrixView` and `fill_segment_inner`
/// doc-comments for the full safety argument.
///
/// `rayon::scope` joins all spawned tasks before returning, so the
/// `&mut Matrix` borrow is released only after every lane has finished
/// the slice. The next slice (or finalisation) sees a fully-completed
/// snapshot.
#[cfg(feature = "parallel")]
fn fill_slice_parallel(
  matrix: &mut Matrix,
  compress: CompressFn,
  pass: u32,
  slice: u32,
  variant: Argon2Variant,
  time_cost: u32,
) {
  let lanes = matrix.lanes;
  let segment_len = matrix.segment_len;
  let lane_len = matrix.lane_len;
  let total_blocks = matrix.len() as u32;
  let view = MatrixView::from_blocks(&mut matrix.blocks);

  // Spawn lanes 1..lanes onto rayon workers; run lane 0 inline on the
  // calling thread. This is the standard Blake3 / parallel-tree-reduction
  // pattern: it saves one task dispatch per slice (n=8 slices × t passes
  // × 1 task ≈ measurable when each fill_segment is short), and ensures
  // the calling thread participates as a worker rather than purely
  // blocking on `scope()`.
  rayon::scope(|s| {
    for lane in 1..lanes {
      s.spawn(move |_| {
        // SAFETY: `view` is shared across all spawned tasks. Each task
        // is responsible for exactly one lane's segment within the
        // current slice. Per RFC 9106 §3.4:
        //
        // - Lane `lane`'s write range is `[lane*lane_len + slice*segment_len .. lane*lane_len +
        //   (slice+1)*segment_len]`. This range is pairwise disjoint across lanes, so concurrent tasks
        //   never write the same index.
        // - Cross-lane reads (the `(ref_lane, ref_index)` fetch in `fill_segment_inner`) only target blocks
        //   in completed slices — never the current slice's still-being-written range. Same-lane reads of
        //   the previous column are within this task's own write range, accessed sequentially.
        //
        // No two tasks ever access the same block when at least one is
        // a writer. The Send/Sync impls on `MatrixView` are sound under
        // this discipline; see the `MatrixView` doc-comment.
        unsafe {
          fill_segment_inner(
            view,
            compress,
            pass,
            lane,
            slice,
            lanes,
            segment_len,
            lane_len,
            total_blocks,
            variant,
            time_cost,
          );
        }
      });
    }

    // SAFETY: lane 0's segment range is exclusively writeable by this
    // call (no other spawned task touches lane 0). The same disjointness
    // argument used for the spawned tasks applies — see above.
    unsafe {
      fill_segment_inner(
        view,
        compress,
        pass,
        0,
        slice,
        lanes,
        segment_len,
        lane_len,
        total_blocks,
        variant,
        time_cost,
      );
    }
  });
}

/// Dispatch one slice fill: parallel when `parallel` is on and `lanes > 1`,
/// and each lane segment has enough work to amortize Rayon scheduling;
/// sequential otherwise. Tiny segments are common in test / benchmark
/// cost shapes (`m=64,p=2` means only 8 blocks per spawned task), where
/// Rayon overhead dominates the actual Argon2 fill on Apple Silicon.
#[inline]
fn fill_slice(
  matrix: &mut Matrix,
  compress: CompressFn,
  pass: u32,
  slice: u32,
  variant: Argon2Variant,
  time_cost: u32,
) {
  #[cfg(feature = "parallel")]
  if matrix.lanes > 1 && matrix.segment_len >= MIN_PARALLEL_SEGMENT_BLOCKS {
    fill_slice_parallel(matrix, compress, pass, slice, variant, time_cost);
    return;
  }
  fill_slice_sequential(matrix, compress, pass, slice, variant, time_cost);
}

// ─── Full Argon2 hash function ─────────────────────────────────────────────

fn argon2_hash(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
) -> Result<(), Argon2Error> {
  argon2_hash_with_context(params, Argon2Context::default(), password, salt, variant, out)
}

fn argon2_hash_with_context(
  params: &Argon2Params,
  context: Argon2Context<'_>,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
) -> Result<(), Argon2Error> {
  argon2_hash_with_kernel_inner(params, context, password, salt, variant, out, active_compress(), false)
}

#[cfg(feature = "diag")]
fn argon2_hash_with_kernel(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
  compress: CompressFn,
) -> Result<(), Argon2Error> {
  argon2_hash_with_kernel_inner(
    params,
    Argon2Context::default(),
    password,
    salt,
    variant,
    out,
    compress,
    false,
  )
}

#[cfg(feature = "diag")]
fn argon2_hash_with_kernel_diag_blake2b(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
  compress: CompressFn,
) -> Result<(), Argon2Error> {
  argon2_hash_with_kernel_inner(
    params,
    Argon2Context::default(),
    password,
    salt,
    variant,
    out,
    compress,
    true,
  )
}

#[allow(clippy::too_many_arguments)] // Params, context, inputs, variant, output, and kernel are the real operation boundary.
fn argon2_hash_with_kernel_inner(
  params: &Argon2Params,
  context: Argon2Context<'_>,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
  compress: CompressFn,
  #[cfg_attr(not(feature = "diag"), allow(unused_variables))] diag_blake2b: bool,
) -> Result<(), Argon2Error> {
  Argon2Params::check_inputs(password, salt, context)?;
  if out.len() < MIN_OUTPUT_LEN || out.len() as u64 > MAX_VAR_BYTES {
    return Err(Argon2Error::InvalidOutputLen);
  }

  // Compute H0.
  let mut h0 = {
    #[cfg(feature = "diag")]
    {
      if diag_blake2b {
        compute_h0_diag_blake2b_portable(params, context, password, salt, variant, out.len())
      } else {
        compute_h0(params, context, password, salt, variant, out.len())
      }
    }
    #[cfg(not(feature = "diag"))]
    {
      compute_h0(params, context, password, salt, variant, out.len())
    }
  };

  // Allocate the memory matrix.
  let mut matrix = Matrix::new(*params)?;
  let lane_len = matrix.lane_len;
  let lanes = matrix.lanes;

  // Initialise first two blocks per lane.
  for lane in 0..lanes {
    let mut buf = [0u8; BLOCK_SIZE];
    // B[lane][0] = H'(H0 || LE32(0) || LE32(lane), BLOCK_SIZE)
    let lane_le = lane.to_le_bytes();
    #[cfg(feature = "diag")]
    if diag_blake2b {
      h_prime_diag_blake2b_portable(&[&h0, &0u32.to_le_bytes(), &lane_le], &mut buf);
    } else {
      h_prime(&[&h0, &0u32.to_le_bytes(), &lane_le], &mut buf);
    }
    #[cfg(not(feature = "diag"))]
    h_prime(&[&h0, &0u32.to_le_bytes(), &lane_le], &mut buf);
    matrix.set(lane, 0, block_from_bytes(&buf));

    // B[lane][1] = H'(H0 || LE32(1) || LE32(lane), BLOCK_SIZE)
    #[cfg(feature = "diag")]
    if diag_blake2b {
      h_prime_diag_blake2b_portable(&[&h0, &1u32.to_le_bytes(), &lane_le], &mut buf);
    } else {
      h_prime(&[&h0, &1u32.to_le_bytes(), &lane_le], &mut buf);
    }
    #[cfg(not(feature = "diag"))]
    h_prime(&[&h0, &1u32.to_le_bytes(), &lane_le], &mut buf);
    matrix.set(lane, 1, block_from_bytes(&buf));
    ct::zeroize(&mut buf);
  }

  // Main fill loop: pass × slice × (lanes synchronised at slice boundaries).
  // Lane parallelism within a slice is gated on the `parallel` feature and
  // skipped when `parallelism == 1` to avoid rayon overhead.
  for pass in 0..params.time_cost {
    for slice in 0..SYNC_POINTS {
      fill_slice(&mut matrix, compress, pass, slice, variant, params.time_cost);
    }
  }

  // Finalisation: C = XOR of last block of each lane; output = H'(C, T).
  let mut acc = *matrix.get(0, lane_len - 1);
  for lane in 1..lanes {
    let blk = matrix.get(lane, lane_len - 1);
    for i in 0..BLOCK_WORDS {
      acc[i] ^= blk[i];
    }
  }
  let mut acc_bytes = block_to_bytes(&acc);
  #[cfg(feature = "diag")]
  if diag_blake2b {
    h_prime_diag_blake2b_portable(&[&acc_bytes], out);
  } else {
    h_prime(&[&acc_bytes], out);
  }
  #[cfg(not(feature = "diag"))]
  h_prime(&[&acc_bytes], out);

  // Wipe scratch
  ct::zeroize(&mut h0);
  zeroize_u64_slice(&mut acc);
  ct::zeroize(&mut acc_bytes);
  // Matrix zeroized via Drop
  Ok(())
}

// ─── Public typed hashers ───────────────────────────────────────────────────

macro_rules! define_argon2_variant {
  (
    $(#[$meta:meta])*
    $name:ident { variant: $variant:expr, algorithm: $algorithm:literal }
  ) => {
    $(#[$meta])*
    #[derive(Debug, Clone, Copy, Default)]
    pub struct $name;

    impl $name {
      /// Algorithm identifier used in diagnostics.
      pub const ALGORITHM: &'static str = $algorithm;

      /// Derive bytes from `password` and `salt` into `out`.
      ///
      /// # Errors
      ///
      /// Returns [`Argon2Error`] if the salt or output length is invalid.
      pub fn derive(
        params: &Argon2Params,
        password: &[u8],
        salt: &[u8],
        out: &mut [u8],
      ) -> Result<(), Argon2Error> {
        argon2_hash(params, password, salt, $variant, out)
      }

      /// Derive bytes with borrowed pepper and associated data.
      ///
      /// # Errors
      ///
      /// Returns [`Argon2Error`] if any input length is invalid.
      pub fn derive_with_context(
        params: &Argon2Params,
        context: Argon2Context<'_>,
        password: &[u8],
        salt: &[u8],
        out: &mut [u8],
      ) -> Result<(), Argon2Error> {
        argon2_hash_with_context(params, context, password, salt, $variant, out)
      }

      /// Verify `expected` against a freshly-computed hash in constant time.
      ///
      /// # Errors
      ///
      /// Returns an opaque [`VerificationError`] on any mismatch, malformed
      /// input, or parameter error.
      #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
      pub fn verify(
        params: &Argon2Params,
        password: &[u8],
        salt: &[u8],
        expected: &[u8],
      ) -> Result<(), VerificationError> {
        if expected.len() < MIN_OUTPUT_LEN || expected.len() as u64 > MAX_VAR_BYTES {
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
  };
}

define_argon2_variant! {
  /// Argon2d — data-dependent indexing variant of Argon2 (RFC 9106).
  ///
  /// Fastest of the three variants but vulnerable to side-channel timing
  /// attacks if an adversary can observe memory accesses. Use only in
  /// trusted-hardware or non-interactive settings (e.g. cryptocurrency PoW).
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::{Argon2Params, Argon2d};
  /// let params = Argon2Params::new(32, 3, 4).unwrap();
  /// let mut h = [0u8; 32];
  /// Argon2d::derive(&params, &[0x01; 32], &[0x02; 16], &mut h).unwrap();
  /// ```
  Argon2d { variant: Argon2Variant::Argon2d, algorithm: "argon2d" }
}

define_argon2_variant! {
  /// Argon2i — data-independent indexing variant of Argon2 (RFC 9106).
  ///
  /// Memory-access patterns do not depend on the password, closing the
  /// cache-timing channel Argon2d accepts. Slower than Argon2d and Argon2id;
  /// prefer Argon2id for password hashing unless you specifically need
  /// data-independent access patterns throughout.
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::{Argon2Params, Argon2i};
  /// let params = Argon2Params::new(32, 3, 4).unwrap();
  /// let mut h = [0u8; 32];
  /// Argon2i::derive(&params, &[0x01; 32], &[0x02; 16], &mut h).unwrap();
  /// ```
  Argon2i { variant: Argon2Variant::Argon2i, algorithm: "argon2i" }
}

define_argon2_variant! {
  /// Argon2id — hybrid variant of Argon2 (RFC 9106). **Recommended default.**
  ///
  /// Runs Argon2i (data-independent) for the first half of the first pass
  /// and Argon2d (data-dependent) for the rest. OWASP recommends this
  /// variant for password hashing.
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::{Argon2Params, Argon2id};
  /// let params = Argon2Params::new(32, 3, 4).unwrap();
  /// let mut h = [0u8; 32];
  /// Argon2id::derive(&params, &[0x01; 32], &[0x02; 16], &mut h).unwrap();
  /// ```
  Argon2id { variant: Argon2Variant::Argon2id, algorithm: "argon2id" }
}

/// Argon2id password generation and bounded PHC verification.
#[cfg(feature = "phc-strings")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Argon2idPassword {
  generation: Argon2Params,
  limits: Argon2VerificationLimits,
}

#[cfg(feature = "phc-strings")]
impl Default for Argon2idPassword {
  fn default() -> Self {
    let generation = Argon2Params::default();
    Self {
      generation,
      limits: Argon2VerificationLimits::for_profile(generation),
    }
  }
}

#[cfg(feature = "phc-strings")]
impl Argon2idPassword {
  /// Use `generation` for new hashes and derive matching verification limits.
  ///
  /// # Errors
  ///
  /// Returns [`Argon2Error::ResourceOverflow`] if the profile cannot fit the
  /// target address space.
  pub fn new(generation: Argon2Params) -> Result<Self, Argon2Error> {
    if argon2_shape(generation).memory_bytes > isize::MAX as u64 {
      return Err(Argon2Error::ResourceOverflow);
    }
    Ok(Self {
      generation,
      limits: Argon2VerificationLimits::for_profile(generation),
    })
  }

  /// Use distinct generation parameters and finite verification limits.
  pub fn with_limits(generation: Argon2Params, limits: Argon2VerificationLimits) -> Result<Self, Argon2Error> {
    if argon2_shape(generation).memory_bytes > isize::MAX as u64 {
      return Err(Argon2Error::ResourceOverflow);
    }
    if !limits.allows(generation) {
      return Err(Argon2Error::VerificationLimitTooLow);
    }
    Ok(Self { generation, limits })
  }

  /// Hash a password with an OS-generated salt and canonical Argon2id PHC encoding.
  #[cfg(all(feature = "phc-strings", feature = "getrandom"))]
  pub fn hash_password(&self, password: &[u8]) -> Result<alloc::string::String, Argon2Error> {
    self.hash_password_with_context(password, Argon2Context::default())
  }

  /// Hash a password with borrowed pepper and associated data.
  #[cfg(all(feature = "phc-strings", feature = "getrandom"))]
  pub fn hash_password_with_context(
    &self,
    password: &[u8],
    context: Argon2Context<'_>,
  ) -> Result<alloc::string::String, Argon2Error> {
    let mut salt = [0u8; PASSWORD_SALT_LEN];
    getrandom::fill(&mut salt).map_err(|_| Argon2Error::EntropyUnavailable)?;
    let mut verifier = crate::secret::ZeroizingBytes::<PASSWORD_OUTPUT_LEN>::zeroed();
    Argon2id::derive_with_context(&self.generation, context, password, &salt, verifier.as_mut_array())?;
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
    self.verify_password_with_context(password, encoded, Argon2Context::default())
  }

  /// Verify a password with borrowed pepper and associated data.
  #[cfg(feature = "phc-strings")]
  #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
  pub fn verify_password_with_context(
    &self,
    password: &[u8],
    encoded: &str,
    context: Argon2Context<'_>,
  ) -> Result<crate::auth::PasswordStatus, VerificationError> {
    let approved = password_phc::approve(encoded, self.limits).map_err(|_| VerificationError::new())?;
    let mut actual = crate::secret::ZeroizingBytes::<PASSWORD_OUTPUT_LEN>::zeroed();
    Argon2id::derive_with_context(
      &approved.params,
      context,
      password,
      approved.salt(),
      actual.as_mut_array(),
    )
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
const MAX_PHC_SALT_LEN: usize = 48;
#[cfg(feature = "phc-strings")]
const PASSWORD_OUTPUT_LEN: usize = DEFAULT_OUTPUT_LEN;

#[cfg(feature = "phc-strings")]
mod password_phc {
  #[cfg(any(feature = "getrandom", test))]
  use alloc::string::String;

  #[cfg(any(feature = "getrandom", test))]
  use super::ARGON2_VERSION;
  use super::{
    Argon2Params, Argon2VerificationLimits, MAX_PHC_SALT_LEN, MIN_SALT_LEN, PASSWORD_OUTPUT_LEN, argon2_shape,
  };
  use crate::auth::phc::{self, PhcError};

  pub(super) struct ApprovedPhc {
    pub params: Argon2Params,
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
      return Err(if matches!(key, "m" | "t" | "p") {
        PhcError::DuplicateParam
      } else {
        PhcError::UnknownParam
      });
    }
    Ok(value)
  }

  pub(super) fn approve(encoded: &str, limits: Argon2VerificationLimits) -> Result<ApprovedPhc, PhcError> {
    let parts = phc::parse(encoded)?;
    if parts.algorithm != "argon2id" {
      return Err(PhcError::AlgorithmMismatch);
    }
    if parts.version != Some("19") {
      return Err(PhcError::UnsupportedVersion);
    }

    let mut values = phc::PhcParamIter::new(parts.parameters);
    let memory_cost_kib = phc::parse_param_u32(next_param(&mut values, "m")?)?;
    let time_cost = phc::parse_param_u32(next_param(&mut values, "t")?)?;
    let parallelism = phc::parse_param_u32(next_param(&mut values, "p")?)?;
    if let Some(extra) = values.next() {
      let (key, _) = extra?;
      return Err(if matches!(key, "m" | "t" | "p") {
        PhcError::DuplicateParam
      } else {
        PhcError::UnknownParam
      });
    }
    let params = Argon2Params::new(memory_cost_kib, time_cost, parallelism).map_err(|_| PhcError::ParamOutOfRange)?;
    let shape = argon2_shape(params);
    if shape.memory_bytes > isize::MAX as u64 || !limits.allows(params) {
      return Err(PhcError::ParamOutOfRange);
    }

    let salt_len = phc::base64_decoded_len(parts.salt_b64.len());
    let output_len = phc::base64_decoded_len(parts.hash_b64.len());
    if !(MIN_SALT_LEN..=MAX_PHC_SALT_LEN).contains(&salt_len) || output_len != PASSWORD_OUTPUT_LEN {
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
  pub(super) fn encode(params: Argon2Params, salt: &[u8], verifier: &[u8; PASSWORD_OUTPUT_LEN]) -> String {
    let mut out = String::with_capacity(128);
    out.push_str("$argon2id$v=");
    phc::push_u32_decimal(&mut out, ARGON2_VERSION);
    out.push_str("$m=");
    phc::push_u32_decimal(&mut out, params.get_memory_cost_kib());
    out.push_str(",t=");
    phc::push_u32_decimal(&mut out, params.get_time_cost());
    out.push_str(",p=");
    phc::push_u32_decimal(&mut out, params.get_parallelism());
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
  #[cfg(not(miri))]
  use alloc::vec;

  use super::*;

  #[cfg(not(miri))]
  const PASSWORD: &[u8] = &[0x01; 32];
  #[cfg(not(miri))]
  const SALT: &[u8] = &[0x02; 16];
  #[cfg(not(miri))]
  const SECRET: &[u8] = &[0x03; 8];
  #[cfg(not(miri))]
  const AD: &[u8] = &[0x04; 12];

  #[cfg(not(miri))]
  fn canon_params() -> Argon2Params {
    Argon2Params::new(32, 3, 4).unwrap()
  }

  #[cfg(not(miri))]
  fn canon_context() -> Argon2Context<'static> {
    Argon2Context::new(SECRET, AD)
  }

  #[test]
  #[cfg(not(miri))]
  fn rfc9106_appendix_a_vectors() {
    let expected_d: [u8; 32] = [
      0x51, 0x2b, 0x39, 0x1b, 0x6f, 0x11, 0x62, 0x97, 0x53, 0x71, 0xd3, 0x09, 0x19, 0x73, 0x42, 0x94, 0xf8, 0x68, 0xe3,
      0xbe, 0x39, 0x84, 0xf3, 0xc1, 0xa1, 0x3a, 0x4d, 0xb9, 0xfa, 0xbe, 0x4a, 0xcb,
    ];
    let expected_i: [u8; 32] = [
      0xc8, 0x14, 0xd9, 0xd1, 0xdc, 0x7f, 0x37, 0xaa, 0x13, 0xf0, 0xd7, 0x7f, 0x24, 0x94, 0xbd, 0xa1, 0xc8, 0xde, 0x6b,
      0x01, 0x6d, 0xd3, 0x88, 0xd2, 0x99, 0x52, 0xa4, 0xc4, 0x67, 0x2b, 0x6c, 0xe8,
    ];
    let expected_id: [u8; 32] = [
      0x0d, 0x64, 0x0d, 0xf5, 0x8d, 0x78, 0x76, 0x6c, 0x08, 0xc0, 0x37, 0xa3, 0x4a, 0x8b, 0x53, 0xc9, 0xd0, 0x1e, 0xf0,
      0x45, 0x2d, 0x75, 0xb6, 0x5e, 0xb5, 0x25, 0x20, 0xe9, 0x6b, 0x01, 0xe6, 0x59,
    ];

    let mut actual = [0u8; 32];
    Argon2d::derive_with_context(&canon_params(), canon_context(), PASSWORD, SALT, &mut actual).unwrap();
    assert_eq!(actual, expected_d);
    Argon2i::derive_with_context(&canon_params(), canon_context(), PASSWORD, SALT, &mut actual).unwrap();
    assert_eq!(actual, expected_i);
    Argon2id::derive_with_context(&canon_params(), canon_context(), PASSWORD, SALT, &mut actual).unwrap();
    assert_eq!(actual, expected_id);
  }

  #[test]
  #[cfg(not(miri))]
  fn raw_verify_accepts_only_the_exact_inputs() {
    let params = canon_params();
    let mut expected = [0u8; 32];
    Argon2id::derive(&params, PASSWORD, SALT, &mut expected).unwrap();

    assert!(Argon2id::verify(&params, PASSWORD, SALT, &expected).is_ok());
    assert!(Argon2id::verify(&params, b"wrong", SALT, &expected).is_err());
    assert!(Argon2id::verify(&params, PASSWORD, &[0xff; 16], &expected).is_err());
  }

  #[test]
  fn params_are_valid_by_construction() {
    assert_eq!(Argon2Params::new(8, 0, 1), Err(Argon2Error::InvalidTimeCost));
    assert_eq!(Argon2Params::new(8, 1, 0), Err(Argon2Error::InvalidParallelism));
    assert_eq!(Argon2Params::new(16, 1, 4), Err(Argon2Error::InvalidMemoryCost));
    assert!(Argon2Params::new(32, 1, 4).is_ok());
  }

  #[test]
  fn derive_rejects_invalid_operation_lengths() {
    let params = Argon2Params::new(32, 1, 4).unwrap();
    let mut out = [0u8; 32];
    assert_eq!(
      Argon2id::derive(&params, b"pw", &[0u8; 7], &mut out),
      Err(Argon2Error::SaltTooShort)
    );

    let mut short = [0u8; 3];
    assert_eq!(
      Argon2id::derive(&params, b"pw", &[0u8; 16], &mut short),
      Err(Argon2Error::InvalidOutputLen)
    );
  }

  #[test]
  fn error_traits() {
    fn assert_copy<T: Copy>() {}
    fn assert_err<T: core::error::Error>() {}
    assert_copy::<Argon2Error>();
    assert_err::<Argon2Error>();
  }

  #[test]
  fn context_debug_redacts_borrowed_inputs() {
    let debug = alloc::format!("{:?}", Argon2Context::new(b"pepper-value", b"tenant-value"));
    assert!(!debug.contains("pepper-value"));
    assert!(!debug.contains("tenant-value"));
    assert!(debug.contains("[REDACTED]"));
  }

  #[cfg(not(miri))]
  fn oracle_hash(
    algorithm: argon2::Algorithm,
    password: &[u8],
    salt: &[u8],
    memory_kib: u32,
    time: u32,
    parallelism: u32,
    output_len: usize,
  ) -> vec::Vec<u8> {
    let params = argon2::Params::new(memory_kib, time, parallelism, Some(output_len)).unwrap();
    let oracle = argon2::Argon2::new(algorithm, argon2::Version::V0x13, params);
    let mut output = alloc::vec![0u8; output_len];
    oracle.hash_password_into(password, salt, &mut output).unwrap();
    output
  }

  #[test]
  #[cfg(not(miri))]
  fn all_variants_match_the_oracle() {
    let cases: &[(u32, u32, u32, usize)] = &[(8, 1, 1, 16), (16, 2, 1, 32), (32, 3, 2, 64)];
    for &(memory, time, parallelism, output_len) in cases {
      let params = Argon2Params::new(memory, time, parallelism).unwrap();
      let mut actual = alloc::vec![0u8; output_len];

      Argon2d::derive(&params, b"password", &[0u8; 16], &mut actual).unwrap();
      assert_eq!(
        actual,
        oracle_hash(
          argon2::Algorithm::Argon2d,
          b"password",
          &[0u8; 16],
          memory,
          time,
          parallelism,
          output_len,
        )
      );

      Argon2i::derive(&params, b"password", &[0u8; 16], &mut actual).unwrap();
      assert_eq!(
        actual,
        oracle_hash(
          argon2::Algorithm::Argon2i,
          b"password",
          &[0u8; 16],
          memory,
          time,
          parallelism,
          output_len,
        )
      );

      Argon2id::derive(&params, b"password", &[0u8; 16], &mut actual).unwrap();
      assert_eq!(
        actual,
        oracle_hash(
          argon2::Algorithm::Argon2id,
          b"password",
          &[0u8; 16],
          memory,
          time,
          parallelism,
          output_len,
        )
      );
    }
  }

  #[test]
  fn kernel_contract_includes_the_portable_fallback() {
    assert!(ALL_KERNELS.contains(&KernelId::Portable));
    assert!(required_caps(KernelId::Portable).is_empty());
    assert_eq!(KernelId::Portable.as_str(), "portable");
  }

  #[cfg(all(feature = "phc-strings", not(miri)))]
  mod phc_tests {
    use alloc::format;

    use super::*;
    use crate::auth::{PasswordStatus, phc::PhcError};

    fn small_params() -> Argon2Params {
      Argon2Params::new(32, 2, 1).unwrap()
    }

    fn encode(params: Argon2Params, password: &[u8], salt: &[u8], context: Argon2Context<'_>) -> alloc::string::String {
      let mut verifier = [0u8; PASSWORD_OUTPUT_LEN];
      Argon2id::derive_with_context(&params, context, password, salt, &mut verifier).unwrap();
      password_phc::encode(params, salt, &verifier)
    }

    #[test]
    fn canonical_password_record_round_trips() {
      let params = small_params();
      let password = Argon2idPassword::new(params).unwrap();
      let encoded = encode(params, b"password", &[0xaa; 16], Argon2Context::default());

      assert!(encoded.starts_with("$argon2id$v=19$m=32,t=2,p=1$"));
      assert_eq!(
        password.verify_password(b"password", &encoded),
        Ok(PasswordStatus::Current)
      );
      assert!(password.verify_password(b"wrong", &encoded).is_err());
    }

    #[test]
    fn accepted_older_profile_requests_rehash() {
      let generation = Argon2Params::new(40, 2, 1).unwrap();
      let password = Argon2idPassword::new(generation).unwrap();
      let encoded = encode(small_params(), b"password", &[0xbb; 16], Argon2Context::default());

      assert_eq!(
        password.verify_password(b"password", &encoded),
        Ok(PasswordStatus::NeedsRehash)
      );
    }

    #[test]
    fn accepted_noncurrent_salt_length_requests_rehash() {
      let params = small_params();
      let password = Argon2idPassword::new(params).unwrap();
      let encoded = encode(params, b"password", &[0xbb; 8], Argon2Context::default());

      assert_eq!(
        password.verify_password(b"password", &encoded),
        Ok(PasswordStatus::NeedsRehash)
      );
    }

    #[test]
    fn borrowed_context_is_required_for_context_bound_records() {
      let params = small_params();
      let password = Argon2idPassword::new(params).unwrap();
      let context = Argon2Context::new(b"pepper", b"tenant");
      let encoded = encode(params, b"password", &[0xcc; 16], context);

      assert!(password.verify_password(b"password", &encoded).is_err());
      assert_eq!(
        password.verify_password_with_context(b"password", &encoded, context),
        Ok(PasswordStatus::Current)
      );
      assert!(
        password
          .verify_password_with_context(b"password", &encoded, Argon2Context::new(b"wrong", b"tenant"),)
          .is_err()
      );
    }

    #[test]
    fn approval_rejects_resource_requests_before_base64_decode() {
      let limits = Argon2VerificationLimits::for_profile(small_params());
      let expensive = "$argon2id$v=19$m=40,t=2,p=1$*$*";
      assert_eq!(
        password_phc::approve(expensive, limits).err(),
        Some(PhcError::ParamOutOfRange)
      );

      let admitted = "$argon2id$v=19$m=32,t=2,p=1$*$*";
      assert_eq!(
        password_phc::approve(admitted, limits).err(),
        Some(PhcError::InvalidLength)
      );
    }

    #[test]
    fn actual_argon2_shape_defines_the_limit() {
      let limits = Argon2VerificationLimits::for_profile(small_params());
      let rounded_equivalent = Argon2Params::new(35, 2, 1).unwrap();
      assert!(limits.allows(rounded_equivalent));
      let next_matrix = Argon2Params::new(36, 2, 1).unwrap();
      assert!(!limits.allows(next_matrix));
    }

    #[test]
    fn generator_and_parser_share_the_full_argon2_parallelism_domain() {
      let params = Argon2Params::new(2_048, 1, 256).unwrap();
      let encoded = password_phc::encode(params, &[0x44; 16], &[0u8; PASSWORD_OUTPUT_LEN]);
      let limits = Argon2VerificationLimits::for_profile(params);

      assert!(password_phc::approve(&encoded, limits).is_ok());
    }

    #[test]
    fn parser_accepts_only_the_canonical_argon2id_protocol() {
      let limits = Argon2VerificationLimits::for_profile(small_params());
      let salt = "AAAAAAAAAAAAAAAAAAAAAA";
      let hash = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
      let cases = [
        format!("$argon2i$v=19$m=32,t=2,p=1$${salt}$${hash}"),
        format!("$argon2id$m=32,t=2,p=1$${salt}$${hash}"),
        format!("$argon2id$v=16$m=32,t=2,p=1$${salt}$${hash}"),
        format!("$argon2id$v=19$t=2,m=32,p=1$${salt}$${hash}"),
        format!("$argon2id$v=19$m=32,m=32,p=1$${salt}$${hash}"),
        format!("$argon2id$v=19$m=32,t=2,x=1$${salt}$${hash}"),
      ];
      for encoded in cases {
        assert!(password_phc::approve(&encoded, limits).is_err(), "{encoded}");
      }
    }

    #[cfg(feature = "getrandom")]
    #[test]
    fn generated_records_use_fresh_salts() {
      let password = Argon2idPassword::new(small_params()).unwrap();
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
