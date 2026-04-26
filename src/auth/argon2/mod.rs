//! Argon2 password hashing (RFC 9106).
//!
//! Ships all three variants:
//!
//! - [`Argon2d`] — data-dependent indexing, highest throughput, vulnerable to side-channel timing
//!   attacks. Useful for non-interactive, trusted- hardware settings (e.g. cryptocurrency
//!   proof-of-work).
//! - [`Argon2i`] — data-independent indexing, side-channel resistant. Useful when the adversary can
//!   observe memory-access patterns.
//! - [`Argon2id`] — hybrid: first half of the first pass runs Argon2i, the rest runs Argon2d.
//!   **Recommended default for password hashing** per RFC 9106 §4 and OWASP 2024 guidance.
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
//! let params = Argon2Params::new()
//!   .memory_cost_kib(19 * 1024)
//!   .time_cost(2)
//!   .parallelism(1)
//!   .build()
//!   .expect("valid params");
//!
//! let password = b"correct horse battery staple";
//! let salt = b"random-salt-1234";
//!
//! let mut hash = [0u8; 32];
//! Argon2id::hash(&params, password, salt, &mut hash).expect("hash");
//!
//! assert!(Argon2id::verify(&params, password, salt, &hash).is_ok());
//! assert!(Argon2id::verify(&params, b"wrong", salt, &hash).is_err());
//! ```
//!
//! # Security
//!
//! - Salt length must be ≥ 8 bytes (RFC 9106 §3.1). 16 bytes recommended.
//! - Memory cost must satisfy `m ≥ 8 · p` (KiB).
//! - [`Argon2Params::validate`] enforces all RFC 9106 bounds; invalid inputs surface as
//!   [`Argon2Error`] at build time, not at hash time.
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

/// Default OWASP 2024 parameters (see [`Argon2Params::new`]).
const DEFAULT_MEMORY_KIB: u32 = 19 * 1024;
const DEFAULT_TIME_COST: u32 = 2;
const DEFAULT_PARALLELISM: u32 = 1;
const DEFAULT_OUTPUT_LEN: usize = 32;

/// BlaMka P permutation operates on 8×8 matrices of 16-byte "registers",
/// i.e. on 16-u64 slabs.
const P_LANE_WORDS: usize = 16;

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

// ─── Variant, Version ───────────────────────────────────────────────────────

/// Argon2 variant selector.
///
/// The variant controls reference-block indexing:
///
/// - `Argon2d` — data-dependent (Argon2d mode throughout).
/// - `Argon2i` — data-independent (Argon2i mode throughout).
/// - `Argon2id` — data-independent for the first half of the first pass, data-dependent afterwards
///   (RFC 9106 §3.4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
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

/// Argon2 algorithm version.
///
/// `V0x13` (1.3) is the RFC 9106 default and should be used for all new
/// deployments. `V0x10` (1.0) is supported only for decoding legacy PHC
/// strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Argon2Version {
  V0x10,
  #[default]
  V0x13,
}

impl Argon2Version {
  #[inline]
  const fn as_u32(self) -> u32 {
    match self {
      Self::V0x10 => 0x10,
      Self::V0x13 => 0x13,
    }
  }
}

// ─── Error ──────────────────────────────────────────────────────────────────

/// Invalid Argon2 parameter or input.
///
/// Surfaced at `build` or `hash` time — never at `verify` time (a parameter
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
    })
  }
}

impl core::error::Error for Argon2Error {}

// ─── Parameters ─────────────────────────────────────────────────────────────

/// Validated Argon2 cost-parameter set.
///
/// Constructed via [`Argon2Params::new`] and the setters; call
/// [`Argon2Params::build`] to validate and produce a `Result<Argon2Params,
/// Argon2Error>`. The output length is enforced by validators but is the
/// same for every hash produced with this parameter set.
///
/// # Examples
///
/// ```rust
/// use rscrypto::{Argon2Params, Argon2id};
///
/// let params = Argon2Params::new()
///   .time_cost(3)
///   .memory_cost_kib(65_536)
///   .parallelism(4)
///   .output_len(32)
///   .build()
///   .expect("valid params");
///
/// let mut out = [0u8; 32];
/// Argon2id::hash(&params, b"password", b"random-salt-1234", &mut out).unwrap();
/// ```
#[derive(Clone)]
pub struct Argon2Params {
  time_cost: u32,
  memory_cost_kib: u32,
  parallelism: u32,
  output_len: u32,
  version: Argon2Version,
  secret: Vec<u8>,
  associated_data: Vec<u8>,
}

impl fmt::Debug for Argon2Params {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Argon2Params")
      .field("time_cost", &self.time_cost)
      .field("memory_cost_kib", &self.memory_cost_kib)
      .field("parallelism", &self.parallelism)
      .field("output_len", &self.output_len)
      .field("version", &self.version)
      .field("secret_len", &self.secret.len())
      .field("associated_data_len", &self.associated_data.len())
      .finish()
  }
}

impl Drop for Argon2Params {
  fn drop(&mut self) {
    if !self.secret.is_empty() {
      ct::zeroize(&mut self.secret);
    }
    if !self.associated_data.is_empty() {
      ct::zeroize(&mut self.associated_data);
    }
  }
}

impl Default for Argon2Params {
  fn default() -> Self {
    Self::new()
  }
}

impl Argon2Params {
  /// Create a new parameter builder pre-populated with OWASP 2024 defaults
  /// (m = 19 MiB, t = 2, p = 1, output = 32 bytes, v = 1.3). Call the setters
  /// to override, then [`Argon2Params::build`] to validate.
  #[must_use]
  pub fn new() -> Self {
    Self {
      time_cost: DEFAULT_TIME_COST,
      memory_cost_kib: DEFAULT_MEMORY_KIB,
      parallelism: DEFAULT_PARALLELISM,
      output_len: DEFAULT_OUTPUT_LEN as u32,
      version: Argon2Version::V0x13,
      secret: Vec::new(),
      associated_data: Vec::new(),
    }
  }

  /// Set the time cost (iterations). Must be ≥ 1.
  #[must_use]
  pub const fn time_cost(mut self, t: u32) -> Self {
    self.time_cost = t;
    self
  }

  /// Set the memory cost in KiB. Must satisfy `m ≥ 8 · p`.
  #[must_use]
  pub const fn memory_cost_kib(mut self, m: u32) -> Self {
    self.memory_cost_kib = m;
    self
  }

  /// Set the parallelism (number of lanes). Must be 1..=2^24-1.
  #[must_use]
  pub const fn parallelism(mut self, p: u32) -> Self {
    self.parallelism = p;
    self
  }

  /// Set the output tag length in bytes. Must be 4..=2^32-1.
  #[must_use]
  pub const fn output_len(mut self, t: u32) -> Self {
    self.output_len = t;
    self
  }

  /// Set the algorithm version.
  #[must_use]
  pub const fn version(mut self, v: Argon2Version) -> Self {
    self.version = v;
    self
  }

  /// Set the optional secret (pepper). Empty disables pepper.
  #[must_use]
  pub fn secret(mut self, secret: &[u8]) -> Self {
    ct::zeroize(&mut self.secret);
    self.secret.clear();
    self.secret.extend_from_slice(secret);
    self
  }

  /// Set the optional associated data. Empty means "none".
  #[must_use]
  pub fn associated_data(mut self, data: &[u8]) -> Self {
    ct::zeroize(&mut self.associated_data);
    self.associated_data.clear();
    self.associated_data.extend_from_slice(data);
    self
  }

  /// Validate every field against RFC 9106 bounds and return the finalised
  /// parameter set.
  pub fn build(self) -> Result<Self, Argon2Error> {
    self.validate()?;
    Ok(self)
  }

  /// Run validation without consuming the builder — returns the first error.
  pub fn validate(&self) -> Result<(), Argon2Error> {
    if self.time_cost < 1 {
      return Err(Argon2Error::InvalidTimeCost);
    }
    if self.parallelism < 1 || self.parallelism > (1 << 24) - 1 {
      return Err(Argon2Error::InvalidParallelism);
    }
    // RFC 9106 §3.1: m >= 8 * p, m <= 2^32 - 1.
    let min_memory = self.parallelism.checked_mul(8).ok_or(Argon2Error::InvalidMemoryCost)?;
    if self.memory_cost_kib < min_memory {
      return Err(Argon2Error::InvalidMemoryCost);
    }
    if (self.output_len as usize) < MIN_OUTPUT_LEN {
      return Err(Argon2Error::InvalidOutputLen);
    }
    if self.secret.len() as u64 > MAX_VAR_BYTES {
      return Err(Argon2Error::SecretTooLong);
    }
    if self.associated_data.len() as u64 > MAX_VAR_BYTES {
      return Err(Argon2Error::AssociatedDataTooLong);
    }
    Ok(())
  }

  /// Common length checks for `password` / `salt` shared by all three variants.
  fn check_inputs(&self, password: &[u8], salt: &[u8]) -> Result<(), Argon2Error> {
    if password.len() as u64 > MAX_VAR_BYTES {
      return Err(Argon2Error::PasswordTooLong);
    }
    if salt.len() < MIN_SALT_LEN {
      return Err(Argon2Error::SaltTooShort);
    }
    if salt.len() as u64 > MAX_VAR_BYTES {
      return Err(Argon2Error::SaltTooLong);
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

  /// Output tag length in bytes.
  #[must_use]
  pub const fn get_output_len(&self) -> u32 {
    self.output_len
  }

  /// Algorithm version.
  #[must_use]
  pub const fn get_version(&self) -> Argon2Version {
    self.version
  }
}

/// Operational limits for verifying Argon2 PHC strings from untrusted storage.
///
/// PHC strings encode their own memory, iteration, parallelism, and output
/// lengths. Use `Argon2id::verify_string_with_policy` (or the matching
/// variant method) when those encoded parameters can be controlled by another
/// tenant, database row, network peer, or migration input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Argon2VerifyPolicy {
  /// Maximum encoded memory cost in KiB.
  pub max_memory_cost_kib: u32,
  /// Maximum encoded time cost.
  pub max_time_cost: u32,
  /// Maximum encoded parallelism.
  pub max_parallelism: u32,
  /// Maximum encoded output length in bytes.
  pub max_output_len: usize,
}

impl Argon2VerifyPolicy {
  /// Build a policy from explicit upper bounds.
  #[must_use]
  pub const fn new(max_memory_cost_kib: u32, max_time_cost: u32, max_parallelism: u32, max_output_len: usize) -> Self {
    Self {
      max_memory_cost_kib,
      max_time_cost,
      max_parallelism,
      max_output_len,
    }
  }

  /// Return `true` when `params` and `output_len` are within this policy.
  #[must_use]
  pub const fn allows(&self, params: &Argon2Params, output_len: usize) -> bool {
    params.memory_cost_kib <= self.max_memory_cost_kib
      && params.time_cost <= self.max_time_cost
      && params.parallelism <= self.max_parallelism
      && output_len <= self.max_output_len
  }
}

impl Default for Argon2VerifyPolicy {
  fn default() -> Self {
    Self::new(
      DEFAULT_MEMORY_KIB,
      DEFAULT_TIME_COST,
      DEFAULT_PARALLELISM,
      DEFAULT_OUTPUT_LEN,
    )
  }
}

// ─── Kernel dispatch ────────────────────────────────────────────────────────
//
// [`KernelId`], [`ALL_KERNELS`], [`required_caps`], and the private
// `active_compress` selector live in the [`dispatch`] submodule; the
// per-arch kernels live in sibling files (`kernels.rs` for portable,
// `aarch64.rs` for NEON, etc.). This module only *calls through* the
// resolved [`CompressFn`] — no inline BlaMka code below this point.

// ─── Diagnostic entry points for forced-kernel tests and benches ───────────

/// The kernel selected by the runtime dispatcher on the current host.
///
/// Useful for reporting in benchmarks and for forced-kernel tests that
/// want to cross-check the active kernel matches expectations.
#[cfg(feature = "diag")]
#[must_use]
pub fn diag_active_kernel() -> KernelId {
  dispatch::active_kernel()
}

/// Hash via the kernel selected by the runtime dispatcher (default path).
///
/// # Errors
///
/// Returns [`Argon2Error`] for invalid parameters per
/// [`Argon2Params::validate`].
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
/// Returns [`Argon2Error`] for invalid parameters per
/// [`Argon2Params::validate`].
#[cfg(feature = "diag")]
pub fn diag_hash_portable(
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
/// Panics if the host does not support AVX2. The forced-kernel tests
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

/// Block-word count (128) — exposed for diagnostic kernel benches.
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
    .unwrap_or_else(|_| unreachable!("Argon2 H' out_len <= u32::MAX, validated by Argon2Params::validate"))
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

// ─── H0 initialisation (RFC 9106 §3.2) ──────────────────────────────────────

/// Compute the 64-byte `H0` seed.
fn compute_h0(params: &Argon2Params, password: &[u8], salt: &[u8], variant: Argon2Variant) -> [u8; 64] {
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
  hasher.update(&params.output_len.to_le_bytes());
  hasher.update(&params.memory_cost_kib.to_le_bytes());
  hasher.update(&params.time_cost.to_le_bytes());
  hasher.update(&params.version.as_u32().to_le_bytes());
  hasher.update(&variant.y().to_le_bytes());
  hasher.update(&len_u32("password", password.len()));
  hasher.update(password);
  hasher.update(&len_u32("salt", salt.len()));
  hasher.update(salt);
  hasher.update(&len_u32("secret", params.secret.len()));
  hasher.update(&params.secret);
  hasher.update(&len_u32("associated_data", params.associated_data.len()));
  hasher.update(&params.associated_data);
  hasher.finalize()
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

    let mut zero = MemoryBlock::zero();
    let mut intermediate = MemoryBlock::zero();
    // SAFETY: the CompressFn came from `active_compress()` (or
    // `compress_fn_for` in forced-kernel tests), which only returns a
    // kernel whose `required_caps` are a subset of the host's caps.
    unsafe {
      compress(&mut intermediate.0, &zero.0, &input.0, /* xor_into = */ false);
      compress(&mut self.words.0, &zero.0, &intermediate.0, /* xor_into = */ false);
    }
    zeroize_u64_slice_no_fence(&mut zero.0);
    zeroize_u64_slice_no_fence(&mut intermediate.0);
    zeroize_u64_slice_no_fence(&mut input.0);
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
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
  fn new(mem_kib: u32, lanes: u32) -> Result<Self, Argon2Error> {
    // m' = 4 · p · ⌊m / (4p)⌋
    let four_p = lanes.strict_mul(SYNC_POINTS);
    let m_prime = mem_kib / four_p * four_p;
    let lane_len = m_prime / lanes;
    let segment_len = lane_len / SYNC_POINTS;
    let total = m_prime as usize;
    let mut blocks = Vec::new();
    blocks
      .try_reserve_exact(total)
      .map_err(|_| Argon2Error::InvalidMemoryCost)?;
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
#[cfg(feature = "parallel")]
unsafe impl Send for MatrixView {}
// SAFETY: see the Send impl above — same disjointness argument.
#[cfg(feature = "parallel")]
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
  version: Argon2Version,
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
      version,
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
  version: Argon2Version,
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
    let xor_into = pass > 0 && version == Argon2Version::V0x13;
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
  version: Argon2Version,
  time_cost: u32,
) {
  let lanes = matrix.lanes;
  for lane in 0..lanes {
    fill_segment(matrix, compress, pass, lane, slice, variant, version, time_cost);
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
  version: Argon2Version,
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
            version,
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
        version,
        time_cost,
      );
    }
  });
}

/// Dispatch one slice fill: parallel when `parallel` is on and `lanes > 1`,
/// sequential otherwise. The `lanes == 1` fast-path skips rayon entirely
/// — no thread dispatch, no `Send`/`Sync` machinery — which keeps the
/// single-lane (OWASP-default) configuration free of any parallel
/// overhead.
#[inline]
fn fill_slice(
  matrix: &mut Matrix,
  compress: CompressFn,
  pass: u32,
  slice: u32,
  variant: Argon2Variant,
  version: Argon2Version,
  time_cost: u32,
) {
  #[cfg(feature = "parallel")]
  if matrix.lanes > 1 {
    fill_slice_parallel(matrix, compress, pass, slice, variant, version, time_cost);
    return;
  }
  fill_slice_sequential(matrix, compress, pass, slice, variant, version, time_cost);
}

// ─── Full Argon2 hash function ─────────────────────────────────────────────

fn argon2_hash(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
) -> Result<(), Argon2Error> {
  argon2_hash_with_kernel(params, password, salt, variant, out, active_compress())
}

fn argon2_hash_with_kernel(
  params: &Argon2Params,
  password: &[u8],
  salt: &[u8],
  variant: Argon2Variant,
  out: &mut [u8],
  compress: CompressFn,
) -> Result<(), Argon2Error> {
  params.validate()?;
  params.check_inputs(password, salt)?;
  if out.len() < MIN_OUTPUT_LEN || out.len() as u64 > MAX_VAR_BYTES || out.len() != params.output_len as usize {
    return Err(Argon2Error::InvalidOutputLen);
  }

  // Compute H0.
  let mut h0 = compute_h0(params, password, salt, variant);

  // Allocate the memory matrix.
  let mut matrix = Matrix::new(params.memory_cost_kib, params.parallelism)?;
  let lane_len = matrix.lane_len;
  let lanes = matrix.lanes;

  // Initialise first two blocks per lane.
  for lane in 0..lanes {
    let mut buf = [0u8; BLOCK_SIZE];
    // B[lane][0] = H'(H0 || LE32(0) || LE32(lane), BLOCK_SIZE)
    let lane_le = lane.to_le_bytes();
    h_prime(&[&h0, &0u32.to_le_bytes(), &lane_le], &mut buf);
    matrix.set(lane, 0, block_from_bytes(&buf));

    // B[lane][1] = H'(H0 || LE32(1) || LE32(lane), BLOCK_SIZE)
    h_prime(&[&h0, &1u32.to_le_bytes(), &lane_le], &mut buf);
    matrix.set(lane, 1, block_from_bytes(&buf));
    ct::zeroize(&mut buf);
  }

  // Main fill loop: pass × slice × (lanes synchronised at slice boundaries).
  // Lane parallelism within a slice is gated on the `parallel` feature and
  // skipped when `parallelism == 1` to avoid rayon overhead.
  for pass in 0..params.time_cost {
    for slice in 0..SYNC_POINTS {
      fill_slice(
        &mut matrix,
        compress,
        pass,
        slice,
        variant,
        params.version,
        params.time_cost,
      );
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
    $name:ident { variant: $variant:expr, phc_algorithm: $phc_alg:literal }
  ) => {
    $(#[$meta])*
    #[derive(Debug, Clone, Copy, Default)]
    pub struct $name;

    impl $name {
      /// Algorithm identifier used in PHC strings and diagnostics (e.g.
      /// `"argon2d"`, `"argon2i"`, `"argon2id"`).
      pub const ALGORITHM: &'static str = $phc_alg;

      /// Hash `password` with `salt` into `out`.
      ///
      /// # Errors
      ///
      /// Returns [`Argon2Error`] if parameters are out of range, the salt is
      /// too short, or `out.len()` does not match `params.output_len`.
      pub fn hash(
        params: &Argon2Params,
        password: &[u8],
        salt: &[u8],
        out: &mut [u8],
      ) -> Result<(), Argon2Error> {
        argon2_hash(params, password, salt, $variant, out)
      }

      /// Hash `password` with `salt` into a fixed-size array.
      ///
      /// # Errors
      ///
      /// Returns [`Argon2Error`] if `N != params.output_len`.
      pub fn hash_array<const N: usize>(
        params: &Argon2Params,
        password: &[u8],
        salt: &[u8],
      ) -> Result<[u8; N], Argon2Error> {
        let mut out = [0u8; N];
        Self::hash(params, password, salt, &mut out)?;
        Ok(out)
      }

      /// Verify `expected` against a freshly-computed hash in constant time.
      ///
      /// The Argon2 hash always runs, even when `expected.len()` does not
      /// match `params.output_len`. The length check is folded into the
      /// final boolean, not an early return, so wall-clock cost does not
      /// depend on whether the supplied tag has the right length —
      /// `params.output_len` is not leaked through timing.
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
        // Always allocate and hash to `params.output_len` regardless of
        // `expected.len()`. The dominant cost (Argon2 fill) is constant; the
        // residual byte-compare cost difference (≤ tens of ns) is dwarfed by
        // the hash itself (≥ milliseconds at any realistic cost knobs).
        let actual_len = params.output_len as usize;
        let mut actual = alloc::vec![0u8; actual_len];
        let hash_failed = Self::hash(params, password, salt, &mut actual).is_err();

        // `constant_time_eq` returns false for length mismatches; combined
        // with the always-run hash above, length-vs-bytes cases collapse
        // into a single failure path.
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
      /// Emits `$argon2{d|i|id}$v=<ver>$m=<m>,t=<t>,p=<p>$<salt>$<hash>`
      /// (standard RFC 4648 base64, no padding). `params.secret` and
      /// `params.associated_data` are honoured during hashing but are **not**
      /// embedded in the PHC string, per the PHC spec — callers holding
      /// secrets must re-supply them when verifying.
      ///
      /// # Errors
      ///
      /// Propagates any [`Argon2Error`] from parameter validation or input
      /// length checks.
      #[cfg(feature = "phc-strings")]
      pub fn hash_string_with_salt(
        params: &Argon2Params,
        password: &[u8],
        salt: &[u8],
      ) -> Result<alloc::string::String, Argon2Error> {
        let mut hash = alloc::vec![0u8; params.output_len as usize];
        Self::hash(params, password, salt, &mut hash)?;
        let encoded = phc_integration::encode_string(Self::ALGORITHM, params, salt, &hash);
        ct::zeroize(&mut hash);
        Ok(encoded)
      }

      /// Hash `password` with a fresh 16-byte salt from the operating
      /// system CSPRNG and encode the result as a PHC string.
      ///
      /// # Panics
      ///
      /// Panics if the platform entropy source fails.
      ///
      /// # Errors
      ///
      /// Propagates any [`Argon2Error`] from parameter validation or input
      /// length checks.
      #[cfg(all(feature = "phc-strings", feature = "getrandom"))]
      pub fn hash_string(
        params: &Argon2Params,
        password: &[u8],
      ) -> Result<alloc::string::String, Argon2Error> {
        let mut salt = [0u8; 16];
        getrandom::fill(&mut salt).unwrap_or_else(|e| panic!("getrandom failed: {e}"));
        Self::hash_string_with_salt(params, password, &salt)
      }

      /// Verify `password` against a PHC-encoded hash in constant time.
      ///
      /// Parses the encoded string, rebuilds the cost parameters, re-hashes
      /// `password` with the embedded salt, and compares in constant time.
      /// Works only when the encoded string's algorithm matches `Self` — a
      /// PHC string emitted by another variant is rejected opaquely.
      ///
      /// # Errors
      ///
      /// Returns [`VerificationError`] on any mismatch, malformed string,
      /// variant mismatch, or parameter error. Errors are intentionally
      /// opaque — callers needing to distinguish parse failures should use
      /// the lower-level `decode_string` helper.
      #[cfg(feature = "phc-strings")]
      #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
      pub fn verify_string(
        password: &[u8],
        encoded: &str,
      ) -> Result<(), VerificationError> {
        Self::verify_string_with_context(password, encoded, &[], &[])
      }

      /// Verify `password` against a PHC string after enforcing operational
      /// bounds on its encoded cost parameters.
      ///
      /// # Errors
      ///
      /// Returns [`VerificationError`] on any mismatch, malformed string,
      /// variant mismatch, parameter error, or policy violation.
      #[cfg(feature = "phc-strings")]
      #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
      pub fn verify_string_with_policy(
        password: &[u8],
        encoded: &str,
        policy: &Argon2VerifyPolicy,
      ) -> Result<(), VerificationError> {
        Self::verify_string_with_context_and_policy(password, encoded, &[], &[], policy)
      }

      /// Verify `password` against a PHC-encoded hash using an external
      /// pepper and/or associated data.
      ///
      /// PHC strings carry the algorithm, cost parameters, salt, and hash.
      /// They do not carry secret pepper material. This helper is the
      /// verification counterpart for PHC strings produced from
      /// [`Argon2Params::secret`] or [`Argon2Params::associated_data`].
      ///
      /// # Errors
      ///
      /// Returns [`VerificationError`] on any mismatch, malformed string,
      /// variant mismatch, or parameter error.
      #[cfg(feature = "phc-strings")]
      #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
      pub fn verify_string_with_context(
        password: &[u8],
        encoded: &str,
        secret: &[u8],
        associated_data: &[u8],
      ) -> Result<(), VerificationError> {
        Self::verify_string_with_context_and_policy(
          password,
          encoded,
          secret,
          associated_data,
          &Argon2VerifyPolicy::new(u32::MAX, u32::MAX, u32::MAX, usize::MAX),
        )
      }

      /// Verify `password` against a PHC string with external context and
      /// explicit operational bounds.
      ///
      /// # Errors
      ///
      /// Returns [`VerificationError`] on any mismatch, malformed string,
      /// variant mismatch, parameter error, or policy violation.
      #[cfg(feature = "phc-strings")]
      #[must_use = "password verification must be checked; a dropped Result silently accepts the wrong password"]
      pub fn verify_string_with_context_and_policy(
        password: &[u8],
        encoded: &str,
        secret: &[u8],
        associated_data: &[u8],
        policy: &Argon2VerifyPolicy,
      ) -> Result<(), VerificationError> {
        let parsed = phc_integration::decode_string(encoded, Self::ALGORITHM)
          .map_err(|_| VerificationError::new())?;
        let mut params = parsed.params;
        if !secret.is_empty() {
          params = params.secret(secret);
        }
        if !associated_data.is_empty() {
          params = params.associated_data(associated_data);
        }
        params = params.build().map_err(|_| VerificationError::new())?;
        if !policy.allows(&params, parsed.hash.len()) {
          return Err(VerificationError::new());
        }
        let mut actual = alloc::vec![0u8; parsed.hash.len()];
        if Self::hash(&params, password, &parsed.salt, &mut actual).is_err() {
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
      /// Returns the parsed cost parameters, salt, and reference hash. Use
      /// when you need programmatic access to the encoded components —
      /// e.g. to re-hash with a new salt or compare multiple candidates.
      ///
      /// # Errors
      ///
      /// Returns [`crate::auth::phc::PhcError`] on any parse failure, or if
      /// the encoded algorithm does not match `Self::ALGORITHM`.
      #[cfg(feature = "phc-strings")]
      pub fn decode_string(
        encoded: &str,
      ) -> Result<(Argon2Params, alloc::vec::Vec<u8>, alloc::vec::Vec<u8>), crate::auth::phc::PhcError> {
        let parsed = phc_integration::decode_string(encoded, Self::ALGORITHM)?;
        Ok((parsed.params, parsed.salt, parsed.hash))
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
  /// let params = Argon2Params::new().memory_cost_kib(32).time_cost(3).parallelism(4).output_len(32).build().unwrap();
  /// let mut h = [0u8; 32];
  /// Argon2d::hash(&params, &[0x01; 32], &[0x02; 16], &mut h).unwrap();
  /// ```
  Argon2d { variant: Argon2Variant::Argon2d, phc_algorithm: "argon2d" }
}

define_argon2_variant! {
  /// Argon2i — data-independent indexing variant of Argon2 (RFC 9106).
  ///
  /// Side-channel resistant. Slower than Argon2d and Argon2id; prefer
  /// Argon2id for password hashing unless you specifically need data-
  /// independent access patterns throughout.
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::{Argon2Params, Argon2i};
  /// let params = Argon2Params::new().memory_cost_kib(32).time_cost(3).parallelism(4).output_len(32).build().unwrap();
  /// let mut h = [0u8; 32];
  /// Argon2i::hash(&params, &[0x01; 32], &[0x02; 16], &mut h).unwrap();
  /// ```
  Argon2i { variant: Argon2Variant::Argon2i, phc_algorithm: "argon2i" }
}

define_argon2_variant! {
  /// Argon2id — hybrid variant of Argon2 (RFC 9106). **Recommended default.**
  ///
  /// Runs Argon2i (data-independent) for the first half of the first pass
  /// and Argon2d (data-dependent) for the rest. OWASP 2024 recommends this
  /// variant for password hashing.
  ///
  /// # Examples
  ///
  /// ```rust
  /// use rscrypto::{Argon2Params, Argon2id};
  /// let params = Argon2Params::new().memory_cost_kib(32).time_cost(3).parallelism(4).output_len(32).build().unwrap();
  /// let mut h = [0u8; 32];
  /// Argon2id::hash(&params, &[0x01; 32], &[0x02; 16], &mut h).unwrap();
  /// ```
  Argon2id { variant: Argon2Variant::Argon2id, phc_algorithm: "argon2id" }
}

// ─── PHC string integration (feature: phc-strings) ─────────────────────────

#[cfg(feature = "phc-strings")]
mod phc_integration {
  use alloc::{string::String, vec::Vec};

  use super::{Argon2Params, Argon2Version};
  use crate::auth::phc::{self, PhcError};

  /// Parsed PHC components reconstituted into rscrypto types.
  pub(super) struct ParsedPhc {
    pub params: Argon2Params,
    pub salt: Vec<u8>,
    pub hash: Vec<u8>,
  }

  /// Build `$argon2{d|i|id}$v=<ver>$m=<m>,t=<t>,p=<p>$<salt>$<hash>`.
  pub(super) fn encode_string(algorithm: &str, params: &Argon2Params, salt: &[u8], hash: &[u8]) -> String {
    let mut out = String::new();
    out.push('$');
    out.push_str(algorithm);
    out.push_str("$v=");
    phc::push_u32_decimal(&mut out, params.get_version().as_u32());
    out.push_str("$m=");
    phc::push_u32_decimal(&mut out, params.get_memory_cost_kib());
    out.push_str(",t=");
    phc::push_u32_decimal(&mut out, params.get_time_cost());
    out.push_str(",p=");
    phc::push_u32_decimal(&mut out, params.get_parallelism());
    out.push('$');
    phc::base64_encode_into(salt, &mut out);
    out.push('$');
    phc::base64_encode_into(hash, &mut out);
    out
  }

  /// Parse a PHC string and reconstitute `(params, salt, hash)`.
  pub(super) fn decode_string(encoded: &str, expected_algorithm: &str) -> Result<ParsedPhc, PhcError> {
    let parts = phc::parse(encoded)?;
    if parts.algorithm != expected_algorithm {
      return Err(PhcError::AlgorithmMismatch);
    }

    // Parse version. Required for Argon2 (RFC 9106 recommends v=19).
    let version = match parts.version {
      Some(v) => match phc::parse_param_u32(v)? {
        0x10 => Argon2Version::V0x10,
        0x13 => Argon2Version::V0x13,
        _ => return Err(PhcError::UnsupportedVersion),
      },
      None => Argon2Version::V0x13, // Permissive default: some encoders omit v=.
    };

    // Parse the params segment: exactly {m, t, p}, each exactly once, in any order.
    let mut memory_kib: Option<u32> = None;
    let mut time_cost: Option<u32> = None;
    let mut parallelism: Option<u32> = None;
    for pair in phc::PhcParamIter::new(parts.parameters) {
      let (k, v) = pair?;
      let value = phc::parse_param_u32(v)?;
      match k {
        "m" => {
          if memory_kib.replace(value).is_some() {
            return Err(PhcError::DuplicateParam);
          }
        }
        "t" => {
          if time_cost.replace(value).is_some() {
            return Err(PhcError::DuplicateParam);
          }
        }
        "p" => {
          if parallelism.replace(value).is_some() {
            return Err(PhcError::DuplicateParam);
          }
        }
        _ => return Err(PhcError::UnknownParam),
      }
    }
    let m = memory_kib.ok_or(PhcError::MissingParam)?;
    let t = time_cost.ok_or(PhcError::MissingParam)?;
    let p = parallelism.ok_or(PhcError::MissingParam)?;

    // Decode salt and hash.
    let salt = phc::decode_base64_to_vec(parts.salt_b64)?;
    let hash = phc::decode_base64_to_vec(parts.hash_b64)?;

    if salt.len() < super::MIN_SALT_LEN || hash.len() < super::MIN_OUTPUT_LEN {
      return Err(PhcError::InvalidLength);
    }

    // Build params with the extracted cost knobs; output_len equals the
    // decoded hash length so verify's length check passes.
    let params = Argon2Params::new()
      .memory_cost_kib(m)
      .time_cost(t)
      .parallelism(p)
      .output_len(hash.len() as u32)
      .version(version);
    let params = params.build().map_err(|_| PhcError::ParamOutOfRange)?;

    Ok(ParsedPhc { params, salt, hash })
  }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use alloc::vec;

  use super::*;

  // RFC 9106 Appendix A test vectors (version 0x13, m=32, t=3, p=4, T=32).
  const PASSWORD: &[u8] = &[0x01u8; 32];
  const SALT: &[u8] = &[0x02u8; 16];
  const SECRET: &[u8] = &[0x03u8; 8];
  const AD: &[u8] = &[0x04u8; 12];

  fn canon_params() -> Argon2Params {
    Argon2Params::new()
      .memory_cost_kib(32)
      .time_cost(3)
      .parallelism(4)
      .output_len(32)
      .version(Argon2Version::V0x13)
      .secret(SECRET)
      .associated_data(AD)
      .build()
      .unwrap()
  }

  // ── RFC 9106 Appendix A ────────────────────────────────────────────────

  #[test]
  #[cfg(not(miri))]
  fn argon2d_rfc_appendix_a_vector() {
    let expected: [u8; 32] = [
      0x51, 0x2b, 0x39, 0x1b, 0x6f, 0x11, 0x62, 0x97, 0x53, 0x71, 0xd3, 0x09, 0x19, 0x73, 0x42, 0x94, 0xf8, 0x68, 0xe3,
      0xbe, 0x39, 0x84, 0xf3, 0xc1, 0xa1, 0x3a, 0x4d, 0xb9, 0xfa, 0xbe, 0x4a, 0xcb,
    ];
    let mut out = [0u8; 32];
    Argon2d::hash(&canon_params(), PASSWORD, SALT, &mut out).unwrap();
    assert_eq!(out, expected);
  }

  #[test]
  #[cfg(not(miri))]
  fn argon2i_rfc_appendix_a_vector() {
    let expected: [u8; 32] = [
      0xc8, 0x14, 0xd9, 0xd1, 0xdc, 0x7f, 0x37, 0xaa, 0x13, 0xf0, 0xd7, 0x7f, 0x24, 0x94, 0xbd, 0xa1, 0xc8, 0xde, 0x6b,
      0x01, 0x6d, 0xd3, 0x88, 0xd2, 0x99, 0x52, 0xa4, 0xc4, 0x67, 0x2b, 0x6c, 0xe8,
    ];
    let mut out = [0u8; 32];
    Argon2i::hash(&canon_params(), PASSWORD, SALT, &mut out).unwrap();
    assert_eq!(out, expected);
  }

  #[test]
  #[cfg(not(miri))]
  fn argon2id_rfc_appendix_a_vector() {
    let expected: [u8; 32] = [
      0x0d, 0x64, 0x0d, 0xf5, 0x8d, 0x78, 0x76, 0x6c, 0x08, 0xc0, 0x37, 0xa3, 0x4a, 0x8b, 0x53, 0xc9, 0xd0, 0x1e, 0xf0,
      0x45, 0x2d, 0x75, 0xb6, 0x5e, 0xb5, 0x25, 0x20, 0xe9, 0x6b, 0x01, 0xe6, 0x59,
    ];
    let mut out = [0u8; 32];
    Argon2id::hash(&canon_params(), PASSWORD, SALT, &mut out).unwrap();
    assert_eq!(out, expected);
  }

  // ── Verify ─────────────────────────────────────────────────────────────

  #[test]
  #[cfg(not(miri))]
  fn argon2id_verify_accepts_correct() {
    let params = canon_params();
    let h = Argon2id::hash_array::<32>(&params, PASSWORD, SALT).unwrap();
    assert!(Argon2id::verify(&params, PASSWORD, SALT, &h).is_ok());
  }

  #[test]
  #[cfg(not(miri))]
  fn argon2id_verify_rejects_wrong_password() {
    let params = canon_params();
    let h = Argon2id::hash_array::<32>(&params, PASSWORD, SALT).unwrap();
    assert!(Argon2id::verify(&params, b"wrong_password_wrong_wrong!!wrong", SALT, &h).is_err());
  }

  #[test]
  #[cfg(not(miri))]
  fn argon2id_verify_rejects_wrong_salt() {
    let params = canon_params();
    let h = Argon2id::hash_array::<32>(&params, PASSWORD, SALT).unwrap();
    let wrong_salt = [0xffu8; 16];
    assert!(Argon2id::verify(&params, PASSWORD, &wrong_salt, &h).is_err());
  }

  #[cfg(all(feature = "phc-strings", feature = "getrandom"))]
  #[test]
  #[cfg(not(miri))]
  fn hash_string_uses_random_salt_and_verifies() {
    let params = Argon2Params::new()
      .memory_cost_kib(32)
      .time_cost(1)
      .parallelism(1)
      .output_len(32)
      .build()
      .unwrap();

    let encoded = Argon2id::hash_string(&params, b"password").unwrap();
    assert!(Argon2id::verify_string(b"password", &encoded).is_ok());
    assert!(Argon2id::verify_string(b"wrong-password", &encoded).is_err());
  }

  // ── Errors ─────────────────────────────────────────────────────────────

  #[test]
  fn invalid_params_fail_build() {
    assert_eq!(
      Argon2Params::new().time_cost(0).build().unwrap_err(),
      Argon2Error::InvalidTimeCost
    );
    assert_eq!(
      Argon2Params::new().parallelism(0).build().unwrap_err(),
      Argon2Error::InvalidParallelism
    );
    assert_eq!(
      Argon2Params::new()
        .parallelism(4)
        .memory_cost_kib(16) // < 8 * p
        .build()
        .unwrap_err(),
      Argon2Error::InvalidMemoryCost
    );
    assert_eq!(
      Argon2Params::new().output_len(3).build().unwrap_err(),
      Argon2Error::InvalidOutputLen
    );
  }

  #[test]
  fn short_salt_rejected() {
    let params = Argon2Params::new().memory_cost_kib(32).parallelism(4).build().unwrap();
    let mut out = [0u8; 32];
    assert_eq!(
      Argon2id::hash(&params, b"pw", &[0u8; 7], &mut out).unwrap_err(),
      Argon2Error::SaltTooShort
    );
  }

  #[test]
  fn output_len_mismatch_rejected() {
    let params = Argon2Params::new().memory_cost_kib(32).parallelism(4).build().unwrap();
    let mut out = [0u8; 16]; // params.output_len is 32
    assert_eq!(
      Argon2id::hash(&params, b"pw", &[0u8; 16], &mut out).unwrap_err(),
      Argon2Error::InvalidOutputLen
    );
  }

  #[test]
  fn error_traits() {
    fn assert_copy<T: Copy>() {}
    fn assert_err<T: core::error::Error>() {}
    assert_copy::<Argon2Error>();
    assert_err::<Argon2Error>();
  }

  // ── Differential: rscrypto vs rustcrypto `argon2` oracle ──────────────

  fn oracle_hash(
    algo: argon2::Algorithm,
    password: &[u8],
    salt: &[u8],
    m_kib: u32,
    t: u32,
    p: u32,
    out_len: usize,
  ) -> vec::Vec<u8> {
    let params = argon2::Params::new(m_kib, t, p, Some(out_len)).unwrap();
    let ctx = argon2::Argon2::new(algo, argon2::Version::V0x13, params);
    let mut out = alloc::vec![0u8; out_len];
    ctx.hash_password_into(password, salt, &mut out).unwrap();
    out
  }

  #[test]
  #[cfg(not(miri))]
  fn argon2d_matches_oracle_small_params() {
    let cases: &[(u32, u32, u32, usize)] = &[(8, 1, 1, 16), (16, 2, 1, 32), (32, 3, 2, 32)];
    for &(m, t, p, out_len) in cases {
      let params = Argon2Params::new()
        .memory_cost_kib(m)
        .time_cost(t)
        .parallelism(p)
        .output_len(out_len as u32)
        .build()
        .unwrap();
      let mut actual = alloc::vec![0u8; out_len];
      Argon2d::hash(&params, b"password", &[0u8; 16], &mut actual).unwrap();
      let expected = oracle_hash(argon2::Algorithm::Argon2d, b"password", &[0u8; 16], m, t, p, out_len);
      assert_eq!(actual, expected, "argon2d mismatch m={m} t={t} p={p} T={out_len}");
    }
  }

  #[test]
  #[cfg(not(miri))]
  fn argon2i_matches_oracle_small_params() {
    let cases: &[(u32, u32, u32, usize)] = &[(8, 1, 1, 16), (16, 2, 1, 32), (32, 3, 2, 32)];
    for &(m, t, p, out_len) in cases {
      let params = Argon2Params::new()
        .memory_cost_kib(m)
        .time_cost(t)
        .parallelism(p)
        .output_len(out_len as u32)
        .build()
        .unwrap();
      let mut actual = alloc::vec![0u8; out_len];
      Argon2i::hash(&params, b"password", &[0u8; 16], &mut actual).unwrap();
      let expected = oracle_hash(argon2::Algorithm::Argon2i, b"password", &[0u8; 16], m, t, p, out_len);
      assert_eq!(actual, expected, "argon2i mismatch m={m} t={t} p={p} T={out_len}");
    }
  }

  #[test]
  #[cfg(not(miri))]
  fn argon2id_matches_oracle_small_params() {
    let cases: &[(u32, u32, u32, usize)] = &[(8, 1, 1, 16), (16, 2, 1, 32), (32, 3, 2, 32), (32, 3, 4, 64)];
    for &(m, t, p, out_len) in cases {
      let params = Argon2Params::new()
        .memory_cost_kib(m)
        .time_cost(t)
        .parallelism(p)
        .output_len(out_len as u32)
        .build()
        .unwrap();
      let mut actual = alloc::vec![0u8; out_len];
      Argon2id::hash(&params, b"password", &[0u8; 16], &mut actual).unwrap();
      let expected = oracle_hash(argon2::Algorithm::Argon2id, b"password", &[0u8; 16], m, t, p, out_len);
      assert_eq!(actual, expected, "argon2id mismatch m={m} t={t} p={p} T={out_len}");
    }
  }

  // ── Kernel dispatch plumbing ─────────────────────────────────────────

  #[test]
  fn kernel_id_stringifies() {
    assert_eq!(KernelId::Portable.as_str(), "portable");
  }

  #[test]
  fn portable_kernel_has_no_required_caps() {
    assert!(required_caps(KernelId::Portable).is_empty());
  }

  // ── PHC string integration ───────────────────────────────────────────

  #[cfg(all(feature = "phc-strings", not(miri)))]
  mod phc_tests {
    use alloc::{string::String, vec};

    use super::*;
    use crate::auth::phc::PhcError;

    fn small_params() -> Argon2Params {
      Argon2Params::new()
        .memory_cost_kib(32)
        .time_cost(2)
        .parallelism(1)
        .output_len(32)
        .build()
        .unwrap()
    }

    #[test]
    fn hash_string_with_salt_round_trip_id() {
      let params = small_params();
      let salt = [0xAAu8; 16];
      let encoded = Argon2id::hash_string_with_salt(&params, b"password", &salt).unwrap();
      assert!(encoded.starts_with("$argon2id$v=19$m=32,t=2,p=1$"));
      assert!(Argon2id::verify_string(b"password", &encoded).is_ok());
      assert!(Argon2id::verify_string(b"wrongpassword", &encoded).is_err());
    }

    #[test]
    fn verify_string_with_policy_enforces_argon2_bounds() {
      let params = small_params();
      let salt = [0xA1u8; 16];
      let encoded = Argon2id::hash_string_with_salt(&params, b"password", &salt).unwrap();

      let allowed = Argon2VerifyPolicy::new(32, 2, 1, 32);
      assert!(Argon2id::verify_string_with_policy(b"password", &encoded, &allowed).is_ok());

      let low_memory = Argon2VerifyPolicy::new(31, 2, 1, 32);
      assert!(Argon2id::verify_string_with_policy(b"password", &encoded, &low_memory).is_err());

      let short_output = Argon2VerifyPolicy::new(32, 2, 1, 31);
      assert!(Argon2id::verify_string_with_policy(b"password", &encoded, &short_output).is_err());
    }

    #[test]
    fn verify_string_with_context_and_policy_enforces_argon2_bounds() {
      let params = small_params().secret(b"pepper").build().unwrap();
      let salt = [0xA2u8; 16];
      let encoded = Argon2id::hash_string_with_salt(&params, b"password", &salt).unwrap();

      let allowed = Argon2VerifyPolicy::new(32, 2, 1, 32);
      assert!(Argon2id::verify_string_with_context_and_policy(b"password", &encoded, b"pepper", &[], &allowed).is_ok());

      let low_time = Argon2VerifyPolicy::new(32, 1, 1, 32);
      assert!(
        Argon2id::verify_string_with_context_and_policy(b"password", &encoded, b"pepper", &[], &low_time).is_err()
      );
    }

    #[test]
    fn hash_string_round_trip_d_and_i() {
      let params = small_params();
      let salt = [0xBBu8; 16];
      let encoded_d = Argon2d::hash_string_with_salt(&params, b"pw", &salt).unwrap();
      let encoded_i = Argon2i::hash_string_with_salt(&params, b"pw", &salt).unwrap();
      assert!(encoded_d.starts_with("$argon2d$"));
      assert!(encoded_i.starts_with("$argon2i$"));
      assert!(Argon2d::verify_string(b"pw", &encoded_d).is_ok());
      assert!(Argon2i::verify_string(b"pw", &encoded_i).is_ok());
    }

    #[test]
    fn verify_string_rejects_variant_mismatch() {
      let params = small_params();
      let salt = [0xCCu8; 16];
      let encoded_d = Argon2d::hash_string_with_salt(&params, b"pw", &salt).unwrap();
      // Cross-feed: Argon2id expects its own algorithm label.
      assert!(Argon2id::verify_string(b"pw", &encoded_d).is_err());
      assert!(Argon2i::verify_string(b"pw", &encoded_d).is_err());
    }

    #[test]
    fn decode_string_extracts_params_salt_hash() {
      let params = small_params();
      let salt = vec![0xDDu8; 16];
      let encoded = Argon2id::hash_string_with_salt(&params, b"pw", &salt).unwrap();

      let (decoded_params, decoded_salt, decoded_hash) = Argon2id::decode_string(&encoded).unwrap();
      assert_eq!(decoded_params.get_memory_cost_kib(), 32);
      assert_eq!(decoded_params.get_time_cost(), 2);
      assert_eq!(decoded_params.get_parallelism(), 1);
      assert_eq!(decoded_params.get_output_len(), 32);
      assert_eq!(decoded_params.get_version(), Argon2Version::V0x13);
      assert_eq!(decoded_salt, salt);
      assert_eq!(decoded_hash.len(), 32);

      // Re-hashing with decoded params should give the same hash.
      let mut rehashed = [0u8; 32];
      Argon2id::hash(&decoded_params, b"pw", &decoded_salt, &mut rehashed).unwrap();
      assert_eq!(rehashed.as_slice(), decoded_hash.as_slice());
    }

    #[test]
    fn decode_string_rejects_malformed() {
      assert_eq!(
        Argon2id::decode_string("not a phc string").unwrap_err(),
        PhcError::MalformedInput
      );
      // Short salt (2 bytes of "aa") should fail InvalidLength (< MIN_SALT_LEN = 8).
      assert_eq!(
        Argon2id::decode_string("$argon2id$m=32,t=2,p=1$YWE$aGFzaA").unwrap_err(),
        PhcError::InvalidLength
      );
      // Hash segment with 2 bytes ("aa" → base64 "YWE") is below MIN_OUTPUT_LEN.
      assert_eq!(
        Argon2id::decode_string("$argon2id$m=32,t=2,p=1$QUFBQUFBQUFBQUFBQUFBQQ$YWE").unwrap_err(),
        PhcError::InvalidLength
      );
    }

    #[test]
    fn verify_string_rejects_tampered_hash() {
      let params = small_params();
      let salt = [0xEEu8; 16];
      let encoded = Argon2id::hash_string_with_salt(&params, b"pw", &salt).unwrap();
      // Flip a byte in the hash segment (last character).
      let mut tampered: String = encoded.chars().collect();
      let len = tampered.len();
      tampered.replace_range(len - 1..len, "A"); // the original value was "A" or not — either way the test verifies rejection on corruption
      // The tampering may coincidentally match; try multiple flips.
      let mut matched_tamper = encoded.clone();
      let variants = ["A", "B", "C"];
      let last_char = encoded.chars().last().unwrap();
      for v in variants {
        if v.chars().next().unwrap() != last_char {
          let mut t = encoded.clone();
          let len = t.len();
          t.replace_range(len - 1..len, v);
          matched_tamper = t;
          break;
        }
      }
      assert!(Argon2id::verify_string(b"pw", &matched_tamper).is_err());
    }

    #[test]
    fn decode_string_rejects_duplicate_params() {
      // Encode a hash and munge the params segment to include a duplicate `m=`.
      let params = small_params();
      let encoded = Argon2id::hash_string_with_salt(&params, b"pw", &[0xFFu8; 16]).unwrap();
      let broken = encoded.replace("t=2", "m=99");
      assert_eq!(Argon2id::decode_string(&broken).unwrap_err(), PhcError::DuplicateParam);
    }

    #[test]
    fn decode_string_accepts_missing_version() {
      // Omitting the `$v=19` segment is permissive and defaults to V0x13.
      let params = small_params();
      let salt = [0x77u8; 16];
      let encoded = Argon2id::hash_string_with_salt(&params, b"pw", &salt).unwrap();
      let no_v = encoded.replace("$v=19", "");
      let (p, s, h) = Argon2id::decode_string(&no_v).unwrap();
      assert_eq!(p.get_version(), Argon2Version::V0x13);
      assert_eq!(s, salt);
      assert_eq!(h.len(), 32);
    }

    #[test]
    fn verify_string_with_context_handles_secret_and_associated_data() {
      let params = small_params()
        .secret(b"pepper")
        .associated_data(b"context")
        .build()
        .unwrap();
      let salt = [0x99u8; 16];
      let encoded = Argon2id::hash_string_with_salt(&params, b"pw", &salt).unwrap();

      assert!(Argon2id::verify_string(b"pw", &encoded).is_err());
      assert!(Argon2id::verify_string_with_context(b"pw", &encoded, b"pepper", b"context").is_ok());
      assert!(Argon2id::verify_string_with_context(b"pw", &encoded, b"wrong", b"context").is_err());
      assert!(Argon2id::verify_string_with_context(b"pw", &encoded, b"pepper", b"wrong").is_err());
    }

    #[test]
    fn encoded_format_exact_for_known_vector() {
      // Verify exact string shape with fixed params + salt.
      let params = Argon2Params::new()
        .memory_cost_kib(8)
        .time_cost(1)
        .parallelism(1)
        .output_len(16)
        .version(Argon2Version::V0x13)
        .build()
        .unwrap();
      let salt = b"exampleSALTvalue"; // 16 bytes
      let encoded = Argon2id::hash_string_with_salt(&params, b"password", salt).unwrap();
      // Shape check: exactly 5 segments after leading $.
      let segments: alloc::vec::Vec<&str> = encoded.split('$').collect();
      assert_eq!(segments[0], "");
      assert_eq!(segments[1], "argon2id");
      assert_eq!(segments[2], "v=19");
      assert_eq!(segments[3], "m=8,t=1,p=1");
      // segments[4] is base64 salt (16 bytes → 22 chars)
      assert_eq!(segments[4].len(), 22);
      // segments[5] is base64 hash (16 bytes → 22 chars)
      assert_eq!(segments[5].len(), 22);
      assert_eq!(segments.len(), 6);
    }
  }
}
