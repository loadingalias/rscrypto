#![allow(clippy::indexing_slicing)]

//! AEGIS-256 authenticated encryption (draft-irtf-cfrg-aegis-aead).
//!
//! High-performance AES-based AEAD with a 256-bit key, 256-bit nonce,
//! and 128-bit authentication tag. Uses raw AES round functions (not full
//! AES encryption), achieving ~2-3x the throughput of AES-256-GCM on
//! hardware with AES-NI or AES-CE.

use core::fmt;

#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  all(target_arch = "powerpc64", target_endian = "little"),
  target_arch = "riscv64",
))]
use super::targets::AeadBackend;
#[cfg(any(
  target_arch = "aarch64",
  all(target_arch = "powerpc64", target_endian = "little"),
  target_arch = "riscv64",
))]
use super::targets::{AeadPrimitive, select_backend};
use super::{AeadBufferError, Nonce256, OpenError, SealError};
use crate::traits::{Aead, ct};

const KEY_SIZE: usize = 32;
const NONCE_SIZE: usize = Nonce256::LENGTH;
const TAG_SIZE: usize = 16;
const BLOCK_SIZE: usize = 16;

/// Fibonacci-derived constant C0.
const C0: [u8; 16] = [
  0x00, 0x01, 0x01, 0x02, 0x03, 0x05, 0x08, 0x0d, 0x15, 0x22, 0x37, 0x59, 0x90, 0xe9, 0x79, 0x62,
];

/// Fibonacci-derived constant C1.
const C1: [u8; 16] = [
  0xdb, 0x3d, 0x18, 0x55, 0x6d, 0xc2, 0x2f, 0xf1, 0x20, 0x11, 0x31, 0x42, 0x73, 0xb5, 0x28, 0xdd,
];

// ---------------------------------------------------------------------------
// Block helpers
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "s390x"))]
type Block = [u8; BLOCK_SIZE];

/// Split a 32-byte array into two 16-byte halves.
#[inline(always)]
fn split_halves(bytes: &[u8; 32]) -> (&[u8; 16], &[u8; 16]) {
  // Infallible: split_first_chunk on [u8; 32] always yields a [u8; 16] prefix.
  let (lo, hi) = bytes.split_first_chunk::<16>().unwrap_or((&[0; 16], &[]));
  // `hi` is &[u8; 16] since 32 - 16 = 16. Use first_chunk for the second half.
  let hi: &[u8; 16] = hi.first_chunk().unwrap_or(&[0; 16]);
  (lo, hi)
}

#[cfg(not(target_arch = "s390x"))]
#[inline(always)]
fn xor_block(a: &Block, b: &Block) -> Block {
  let mut out = [0u8; BLOCK_SIZE];
  for i in 0..BLOCK_SIZE {
    out[i] = a[i] ^ b[i];
  }
  out
}

#[cfg(not(target_arch = "s390x"))]
#[inline(always)]
fn and_block(a: &Block, b: &Block) -> Block {
  let mut out = [0u8; BLOCK_SIZE];
  for i in 0..BLOCK_SIZE {
    out[i] = a[i] & b[i];
  }
  out
}

#[cfg(not(target_arch = "s390x"))]
#[inline(always)]
fn zero_block() -> Block {
  [0u8; BLOCK_SIZE]
}

// ---------------------------------------------------------------------------
// Portable backend
// ---------------------------------------------------------------------------
//
// On s390x, the `s390x_vperm` module provides a Hamburg vperm backend.
// Gate this section to suppress dead-code warnings.

#[cfg(not(target_arch = "s390x"))]
type State = [Block; 6];

#[cfg(not(any(target_arch = "s390x", target_arch = "riscv64")))]
#[inline(always)]
fn aes_round(block: &Block, round_key: &Block) -> Block {
  super::aes_round::aes_enc_round_portable(block, round_key)
}

#[cfg(any(target_arch = "riscv64", test))]
#[inline]
fn update_riscv_fixslice(s: &mut State, m: &Block) {
  let tmp = s[5];

  let mut first = [s[4], s[3], s[2], s[1]];
  let first_keys = [s[5], s[4], s[3], s[2]];
  super::aes::aes_enc_round_4_fixslice(&mut first, &first_keys);

  let mut second = [s[0], tmp, zero_block(), zero_block()];
  let second_keys = [s[1], s[0], zero_block(), zero_block()];
  super::aes::aes_enc_round_4_fixslice(&mut second, &second_keys);

  s[5] = first[0];
  s[4] = first[1];
  s[3] = first[2];
  s[2] = first[3];
  s[1] = second[0];
  s[0] = xor_block(&second[1], m);
}

/// AEGIS-256 Update function: absorb one 128-bit message block into the state.
///
/// Each step applies a single AES round to rotate the state pipeline.
/// The message block is XORed into S0 after the round.
#[cfg(all(not(target_arch = "s390x"), not(target_arch = "riscv64")))]
#[inline]
fn update(s: &mut State, m: &Block) {
  let tmp = s[5];
  s[5] = aes_round(&s[4], &s[5]);
  s[4] = aes_round(&s[3], &s[4]);
  s[3] = aes_round(&s[2], &s[3]);
  s[2] = aes_round(&s[1], &s[2]);
  s[1] = aes_round(&s[0], &s[1]);
  s[0] = xor_block(&aes_round(&tmp, &s[0]), m);
}

#[cfg(target_arch = "riscv64")]
#[inline]
fn update(s: &mut State, m: &Block) {
  update_riscv_fixslice(s, m);
}

/// Initialize AEGIS-256 state from key and nonce.
///
/// Splits key and nonce into 128-bit halves, seeds the 6-block state,
/// then runs 16 Update calls (4 iterations of 4 Updates).
#[cfg(not(target_arch = "s390x"))]
fn init(key: &[u8; KEY_SIZE], nonce: &[u8; NONCE_SIZE]) -> State {
  let (k0_ref, k1_ref) = split_halves(key);
  let (n0_ref, n1_ref) = split_halves(nonce);
  let k0 = *k0_ref;
  let k1 = *k1_ref;
  let n0 = *n0_ref;
  let n1 = *n1_ref;

  let k0_xor_n0 = xor_block(&k0, &n0);
  let k1_xor_n1 = xor_block(&k1, &n1);

  let mut s: State = [k0_xor_n0, k1_xor_n1, C1, C0, xor_block(&k0, &C0), xor_block(&k1, &C1)];

  for _ in 0..4 {
    update(&mut s, &k0);
    update(&mut s, &k1);
    update(&mut s, &k0_xor_n0);
    update(&mut s, &k1_xor_n1);
  }

  s
}

/// Absorb associated data into the state.
#[cfg(not(target_arch = "s390x"))]
fn process_aad(s: &mut State, aad: &[u8]) {
  let mut offset = 0usize;

  // Full 16-byte blocks.
  while offset.strict_add(BLOCK_SIZE) <= aad.len() {
    let mut block = [0u8; BLOCK_SIZE];
    block.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
    update(s, &block);
    offset = offset.strict_add(BLOCK_SIZE);
  }

  // Last partial block (zero-padded).
  if offset < aad.len() {
    let mut block = zero_block();
    block[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
    update(s, &block);
  }
}

/// Compute the AEGIS-256 keystream word from the current state.
#[cfg(not(target_arch = "s390x"))]
#[inline(always)]
fn keystream(s: &State) -> Block {
  // z = S1 ^ S4 ^ S5 ^ (S2 & S3)
  let s2_and_s3 = and_block(&s[2], &s[3]);
  let mut z = xor_block(&s[1], &s[4]);
  z = xor_block(&z, &s[5]);
  z = xor_block(&z, &s2_and_s3);
  z
}

/// Finalize the state and extract the 128-bit authentication tag.
///
/// XORs the bit-lengths of AAD and message into S3, then runs 7 Update
/// rounds and XORs all six state blocks together.
#[cfg(not(target_arch = "s390x"))]
fn finalize(s: &mut State, ad_len: usize, msg_len: usize) -> [u8; TAG_SIZE] {
  // t = S3 ^ (LE64(ad_len_bits) || LE64(msg_len_bits))
  let ad_bits = (ad_len as u64).strict_mul(8);
  let msg_bits = (msg_len as u64).strict_mul(8);
  let mut t = s[3];
  let ad_bytes = ad_bits.to_le_bytes();
  let msg_bytes = msg_bits.to_le_bytes();
  for i in 0..8 {
    t[i] ^= ad_bytes[i];
    t[i.strict_add(8)] ^= msg_bytes[i];
  }

  for _ in 0..7 {
    update(s, &t);
  }

  // tag = S0 ^ S1 ^ S2 ^ S3 ^ S4 ^ S5
  let mut tag = xor_block(&s[0], &s[1]);
  tag = xor_block(&tag, &s[2]);
  tag = xor_block(&tag, &s[3]);
  tag = xor_block(&tag, &s[4]);
  tag = xor_block(&tag, &s[5]);
  tag
}

// ---------------------------------------------------------------------------
// riscv64 scalar AES backend (Zkne)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[path = "aegis256/aarch64_ce.rs"]
mod ce;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[path = "aegis256/x86_64_ni.rs"]
mod ni;
#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[path = "aegis256/powerpc64_ppc.rs"]
mod ppc;
#[cfg(target_arch = "riscv64")]
#[allow(unsafe_code)]
#[path = "aegis256/riscv64_vperm.rs"]
mod rv_vperm;
#[cfg(target_arch = "riscv64")]
#[allow(unsafe_code)]
#[path = "aegis256/riscv64_zkne.rs"]
mod rv_zkne;
#[cfg(target_arch = "riscv64")]
#[allow(unsafe_code)]
#[path = "aegis256/riscv64_zvkned.rs"]
mod rv_zvkned;
#[cfg(target_arch = "s390x")]
#[allow(unsafe_code)]
#[path = "aegis256/s390x_vperm.rs"]
mod s390x_vperm;
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  all(target_arch = "powerpc64", target_endian = "little"),
  target_arch = "riscv64",
))]
#[inline]
fn resolve_backend() -> AeadBackend {
  let caps = crate::platform::caps();

  #[cfg(target_arch = "x86_64")]
  {
    use crate::platform::caps::x86;
    if caps.has(x86::AESNI) && caps.has(x86::AVX) {
      return AeadBackend::X86Aesni;
    }
    AeadBackend::Portable
  }

  #[cfg(not(target_arch = "x86_64"))]
  {
    select_backend(AeadPrimitive::Aegis256, crate::platform::arch(), caps)
  }
}

// ---------------------------------------------------------------------------
// Key
// ---------------------------------------------------------------------------

define_aead_key_type!(Aegis256Key, KEY_SIZE, "AEGIS-256 256-bit secret key.");

// ---------------------------------------------------------------------------
// Tag
// ---------------------------------------------------------------------------

define_aead_tag_type!(Aegis256Tag, TAG_SIZE, "AEGIS-256 128-bit authentication tag.");

// ---------------------------------------------------------------------------
// AEAD
// ---------------------------------------------------------------------------

/// AEGIS-256 authenticated encryption with associated data.
///
/// High-performance AES-based AEAD with a 256-bit key, 256-bit nonce,
/// and 128-bit authentication tag. On hardware with AES round instructions
/// (AES-NI, AES-CE, POWER8 vcipher), AEGIS-256 achieves
/// ~2-3x the throughput of AES-256-GCM.
///
/// # Security
///
/// On x86_64 (AES-NI), aarch64 (AES-CE), and POWER (vcipher), all AES
/// round operations use constant-time hardware instructions. On RISC-V
/// without hardware AES extensions (Zkne / Zvkned), the implementation
/// falls back to the constant-time portable round function instead of
/// secret-indexed lookup tables. That fallback is much slower, but it
/// avoids the cache-timing side channel.
///
/// # Examples
///
/// ```
/// use rscrypto::{Aead, Aegis256, Aegis256Key, aead::Nonce256};
///
/// let key = Aegis256Key::from_bytes([0u8; 32]);
/// let nonce = Nonce256::from_bytes([0u8; 32]);
/// let aead = Aegis256::new(&key);
///
/// let mut buf = *b"hello";
/// let tag = aead.encrypt_in_place(&nonce, b"", &mut buf)?;
/// aead.decrypt_in_place(&nonce, b"", &mut buf, &tag)?;
/// assert_eq!(&buf, b"hello");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Tampering is reported as an opaque verification failure.
///
/// ```
/// use rscrypto::{
///   Aead, Aegis256, Aegis256Key,
///   aead::{Nonce256, OpenError},
/// };
///
/// let key = Aegis256Key::from_bytes([0u8; 32]);
/// let nonce = Nonce256::from_bytes([0u8; 32]);
/// let aead = Aegis256::new(&key);
///
/// let mut sealed = [0u8; 5 + Aegis256::TAG_SIZE];
/// aead.encrypt(&nonce, b"", b"hello", &mut sealed)?;
/// sealed[0] ^= 1;
///
/// let mut opened = [0u8; 5];
/// assert_eq!(
///   aead.decrypt(&nonce, b"", &sealed, &mut opened),
///   Err(OpenError::verification())
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct Aegis256 {
  key: Aegis256Key,
  #[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "powerpc64", target_endian = "little"),
    target_arch = "riscv64",
  ))]
  backend: AeadBackend,
}

impl fmt::Debug for Aegis256 {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("Aegis256").finish_non_exhaustive()
  }
}

impl Aegis256 {
  /// Key length in bytes.
  pub const KEY_SIZE: usize = KEY_SIZE;

  /// Nonce length in bytes.
  pub const NONCE_SIZE: usize = NONCE_SIZE;

  /// Tag length in bytes.
  pub const TAG_SIZE: usize = TAG_SIZE;

  /// Construct a new AEGIS-256 instance from `key`.
  #[inline]
  #[must_use]
  pub fn new(key: &Aegis256Key) -> Self {
    <Self as Aead>::new(key)
  }

  /// Rebuild a typed tag from raw tag bytes.
  #[inline]
  pub fn tag_from_slice(bytes: &[u8]) -> Result<Aegis256Tag, AeadBufferError> {
    <Self as Aead>::tag_from_slice(bytes)
  }

  /// Encrypt `buffer` in place and return the detached authentication tag.
  #[inline]
  pub fn encrypt_in_place(&self, nonce: &Nonce256, aad: &[u8], buffer: &mut [u8]) -> Result<Aegis256Tag, SealError> {
    <Self as Aead>::encrypt_in_place(self, nonce, aad, buffer)
  }

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  #[inline]
  pub fn decrypt_in_place(
    &self,
    nonce: &Nonce256,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Aegis256Tag,
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt_in_place(self, nonce, aad, buffer, tag)
  }

  /// Encrypt `plaintext` into `out` as `ciphertext || tag`.
  #[inline]
  pub fn encrypt(&self, nonce: &Nonce256, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), SealError> {
    <Self as Aead>::encrypt(self, nonce, aad, plaintext, out)
  }

  /// Decrypt a combined `ciphertext || tag` into `out`.
  #[inline]
  pub fn decrypt(
    &self,
    nonce: &Nonce256,
    aad: &[u8],
    ciphertext_and_tag: &[u8],
    out: &mut [u8],
  ) -> Result<(), OpenError> {
    <Self as Aead>::decrypt(self, nonce, aad, ciphertext_and_tag, out)
  }
}

// ---------------------------------------------------------------------------
// Portable encrypt/decrypt helpers
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "s390x"))]
fn encrypt_portable(key: &[u8; KEY_SIZE], nonce: &[u8; NONCE_SIZE], aad: &[u8], buffer: &mut [u8]) -> [u8; TAG_SIZE] {
  let mut s = init(key, nonce);
  process_aad(&mut s, aad);
  let msg_len = buffer.len();
  let mut offset = 0usize;

  // Full blocks.
  while offset.strict_add(BLOCK_SIZE) <= buffer.len() {
    let z = keystream(&s);
    let mut xi = [0u8; BLOCK_SIZE];
    xi.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
    update(&mut s, &xi);
    buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xor_block(&xi, &z));
    offset = offset.strict_add(BLOCK_SIZE);
  }

  // Partial tail.
  if offset < buffer.len() {
    let z = keystream(&s);
    let tail_len = buffer.len().strict_sub(offset);
    let mut pad = zero_block();
    pad[..tail_len].copy_from_slice(&buffer[offset..]);
    update(&mut s, &pad);
    let ct = xor_block(&pad, &z);
    buffer[offset..].copy_from_slice(&ct[..tail_len]);
  }

  finalize(&mut s, aad.len(), msg_len)
}

#[cfg(not(target_arch = "s390x"))]
fn decrypt_portable(key: &[u8; KEY_SIZE], nonce: &[u8; NONCE_SIZE], aad: &[u8], buffer: &mut [u8]) -> [u8; TAG_SIZE] {
  let mut s = init(key, nonce);
  process_aad(&mut s, aad);
  let ct_len = buffer.len();
  let mut offset = 0usize;

  // Full blocks.
  while offset.strict_add(BLOCK_SIZE) <= buffer.len() {
    let z = keystream(&s);
    let mut ci = [0u8; BLOCK_SIZE];
    ci.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
    let xi = xor_block(&ci, &z);
    update(&mut s, &xi);
    buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xi);
    offset = offset.strict_add(BLOCK_SIZE);
  }

  // Partial tail.
  if offset < buffer.len() {
    let z = keystream(&s);
    let tail_len = buffer.len().strict_sub(offset);
    let mut pad = zero_block();
    pad[..tail_len].copy_from_slice(&buffer[offset..]);
    // Decrypt only valid bytes; rest stays zero for Update.
    let mut pt_pad = zero_block();
    for i in 0..tail_len {
      pt_pad[i] = pad[i] ^ z[i];
    }
    update(&mut s, &pt_pad);
    buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
  }

  finalize(&mut s, aad.len(), ct_len)
}

// ---------------------------------------------------------------------------
// Aead trait implementation
// ---------------------------------------------------------------------------

impl Aead for Aegis256 {
  const KEY_SIZE: usize = KEY_SIZE;
  const NONCE_SIZE: usize = NONCE_SIZE;
  const TAG_SIZE: usize = TAG_SIZE;

  type Key = Aegis256Key;
  type Nonce = Nonce256;
  type Tag = Aegis256Tag;

  fn new(key: &Self::Key) -> Self {
    Self {
      key: key.clone(),
      #[cfg(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "powerpc64", target_endian = "little"),
        target_arch = "riscv64",
      ))]
      backend: resolve_backend(),
    }
  }

  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError> {
    if bytes.len() != TAG_SIZE {
      return Err(AeadBufferError::new());
    }
    let mut tag = [0u8; TAG_SIZE];
    tag.copy_from_slice(bytes);
    Ok(Aegis256Tag::from_bytes(tag))
  }

  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Result<Self::Tag, SealError> {
    let key = self.key.as_bytes();
    let nonce = nonce.as_bytes();

    // NOTE: VAES-256 (`ni_wide`) is intentionally NOT dispatched here.
    // AEGIS-256's update is a serial chain of 6 AES rounds reading old state.
    // VAES-256 packs into 3 YMM registers but requires 3 cross-lane shuffles
    // (`vperm2i128`, 3-cycle latency) before each set of 3 VAESENC, adding
    // ~3 cycles to the critical path. On Zen4 with 2 AES ports: AES-NI steady-
    // state is ~5 cyc/block; VAES-256 is ~8 cyc/block. AES-NI wins for serial
    // update chains (unlike AES-GCM where blocks are independent).
    #[cfg(target_arch = "x86_64")]
    if self.backend == AeadBackend::X86Aesni {
      // SAFETY: backend resolution confirmed AES-NI + AVX are available.
      // VEX encoding gives 3-operand VAESENC, eliminating register copies.
      let tag = unsafe { ni::encrypt_fused(key, nonce, aad, buffer) };
      return Ok(Aegis256Tag::from_bytes(tag));
    }

    #[cfg(target_arch = "aarch64")]
    if self.backend == AeadBackend::Aarch64Aes {
      // SAFETY: backend resolution confirmed AES-CE is available.
      let tag = unsafe { ce::encrypt_fused(key, nonce, aad, buffer) };
      return Ok(Aegis256Tag::from_bytes(tag));
    }

    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    if self.backend == AeadBackend::Power8Crypto {
      // SAFETY: backend resolution confirmed POWER8 crypto is available.
      let tag = unsafe { ppc::encrypt_fused(key, nonce, aad, buffer) };
      return Ok(Aegis256Tag::from_bytes(tag));
    }

    #[cfg(target_arch = "riscv64")]
    if self.backend == AeadBackend::Riscv64VectorCrypto {
      // SAFETY: backend resolution confirmed vector AES (`zvkned`) is available.
      let tag = unsafe { rv_zvkned::encrypt_fused(key, nonce, aad, buffer) };
      return Ok(Aegis256Tag::from_bytes(tag));
    }

    #[cfg(target_arch = "riscv64")]
    if self.backend == AeadBackend::Riscv64ScalarCrypto {
      // SAFETY: backend resolution confirmed scalar AES (`zkne`) is available.
      let tag = unsafe { rv_zkne::encrypt_fused(key, nonce, aad, buffer) };
      return Ok(Aegis256Tag::from_bytes(tag));
    }

    #[cfg(target_arch = "riscv64")]
    if self.backend == AeadBackend::Riscv64Vperm {
      // SAFETY: backend resolution confirmed the RISC-V V extension is available.
      let tag = unsafe { rv_vperm::encrypt_fused(key, nonce, aad, buffer) };
      return Ok(Aegis256Tag::from_bytes(tag));
    }

    #[cfg(target_arch = "s390x")]
    {
      // Hamburg vperm AES rounds — constant-time via z/Vector VPERM.
      // SAFETY: z13+ vector facility is available on all supported s390x.
      #[allow(clippy::needless_return)]
      return Ok(Aegis256Tag::from_bytes(unsafe {
        s390x_vperm::encrypt_fused(key, nonce, aad, buffer)
      }));
    }

    #[cfg(not(target_arch = "s390x"))]
    Ok(Aegis256Tag::from_bytes(encrypt_portable(key, nonce, aad, buffer)))
  }

  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), OpenError> {
    let key = self.key.as_bytes();
    let nonce = nonce.as_bytes();

    #[cfg(target_arch = "x86_64")]
    let computed = if self.backend == AeadBackend::X86Aesni {
      // SAFETY: backend resolution confirmed AES-NI + AVX are available.
      // VEX encoding gives 3-operand VAESENC, eliminating register copies.
      unsafe { ni::decrypt_fused(key, nonce, aad, buffer) }
    } else {
      decrypt_portable(key, nonce, aad, buffer)
    };

    #[cfg(target_arch = "aarch64")]
    let computed = if self.backend == AeadBackend::Aarch64Aes {
      // SAFETY: backend resolution confirmed AES-CE is available.
      unsafe { ce::decrypt_fused(key, nonce, aad, buffer) }
    } else {
      decrypt_portable(key, nonce, aad, buffer)
    };

    #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
    let computed = if self.backend == AeadBackend::Power8Crypto {
      // SAFETY: backend resolution confirmed POWER8 crypto is available.
      unsafe { ppc::decrypt_fused(key, nonce, aad, buffer) }
    } else {
      decrypt_portable(key, nonce, aad, buffer)
    };

    #[cfg(target_arch = "riscv64")]
    let computed = if self.backend == AeadBackend::Riscv64VectorCrypto {
      // SAFETY: backend resolution confirmed vector AES (`zvkned`) is available.
      unsafe { rv_zvkned::decrypt_fused(key, nonce, aad, buffer) }
    } else if self.backend == AeadBackend::Riscv64ScalarCrypto {
      // SAFETY: backend resolution confirmed scalar AES (`zkne`) is available.
      unsafe { rv_zkne::decrypt_fused(key, nonce, aad, buffer) }
    } else if self.backend == AeadBackend::Riscv64Vperm {
      // SAFETY: backend resolution confirmed the RISC-V V extension is available.
      unsafe { rv_vperm::decrypt_fused(key, nonce, aad, buffer) }
    } else {
      decrypt_portable(key, nonce, aad, buffer)
    };

    #[cfg(target_arch = "s390x")]
    // SAFETY: z13+ vector facility is available on all supported s390x.
    let computed = unsafe { s390x_vperm::decrypt_fused(key, nonce, aad, buffer) };

    #[cfg(not(any(
      target_arch = "x86_64",
      target_arch = "aarch64",
      all(target_arch = "powerpc64", target_endian = "little"),
      target_arch = "riscv64",
      target_arch = "s390x",
    )))]
    let computed = decrypt_portable(key, nonce, aad, buffer);

    if !ct::constant_time_eq(&computed, tag.as_bytes()) {
      ct::zeroize(buffer);
      return Err(OpenError::verification());
    }

    Ok(())
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use alloc::{vec, vec::Vec};
  use std::eprintln;

  use super::*;

  #[cfg(not(target_arch = "s390x"))]
  #[inline(always)]
  fn portable_aes_round(block: &[u8; 16], round_key: &[u8; 16]) -> [u8; 16] {
    super::super::aes_round::aes_enc_round_portable(block, round_key)
  }

  fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
      .step_by(2)
      .map(|i| u8::from_str_radix(&s[i..i.strict_add(2)], 16).unwrap())
      .collect()
  }

  fn hex_block(s: &str) -> [u8; 16] {
    let v = hex(s);
    let mut out = [0u8; 16];
    out.copy_from_slice(&v);
    out
  }

  // -- AESRound test vector (Appendix A.1) --

  #[cfg(not(target_arch = "s390x"))]
  #[test]
  fn aes_round_matches_spec_vector() {
    let input = hex_block("000102030405060708090a0b0c0d0e0f");
    let rk = hex_block("101112131415161718191a1b1c1d1e1f");
    let expected = hex_block("7a7b4e5638782546a8c0477a3b813f43");

    assert_eq!(portable_aes_round(&input, &rk), expected);
  }

  // -- Update test vector (Appendix A.2) --

  #[cfg(not(target_arch = "s390x"))]
  #[test]
  fn update_matches_spec_vector() {
    let mut s: State = [
      hex_block("1fa1207ed76c86f2c4bb40e8b395b43e"),
      hex_block("b44c375e6c1e1978db64bcd12e9e332f"),
      hex_block("0dab84bfa9f0226432ff630f233d4e5b"),
      hex_block("d7ef65c9b93e8ee60c75161407b066e7"),
      hex_block("a760bb3da073fbd92bdc24734b1f56fb"),
      hex_block("a828a18d6a964497ac6e7e53c5f55c73"),
    ];
    let m = hex_block("b165617ed04ab738afb2612c6d18a1ec");

    update(&mut s, &m);

    assert_eq!(s[0], hex_block("e6bc643bae82dfa3d991b1b323839dcd"));
    assert_eq!(s[1], hex_block("648578232ba0f2f0a3677f617dc052c3"));
    assert_eq!(s[2], hex_block("ea788e0e572044a46059212dd007a789"));
    assert_eq!(s[3], hex_block("2f1498ae19b80da13fba698f088a8590"));
    assert_eq!(s[4], hex_block("a54c2ee95e8c2a2c3dae2ec743ae6b86"));
    assert_eq!(s[5], hex_block("a3240fceb68e32d5d114df1b5363ab67"));
  }

  #[cfg(not(target_arch = "s390x"))]
  #[test]
  fn riscv_fixslice_update_matches_portable_update() {
    let mut portable: State = [
      hex_block("1fa1207ed76c86f2c4bb40e8b395b43e"),
      hex_block("b44c375e6c1e1978db64bcd12e9e332f"),
      hex_block("0dab84bfa9f0226432ff630f233d4e5b"),
      hex_block("d7ef65c9b93e8ee60c75161407b066e7"),
      hex_block("a760bb3da073fbd92bdc24734b1f56fb"),
      hex_block("a828a18d6a964497ac6e7e53c5f55c73"),
    ];
    let mut fixslice = portable;
    let m = hex_block("b165617ed04ab738afb2612c6d18a1ec");

    let tmp = portable[5];
    portable[5] = portable_aes_round(&portable[4], &portable[5]);
    portable[4] = portable_aes_round(&portable[3], &portable[4]);
    portable[3] = portable_aes_round(&portable[2], &portable[3]);
    portable[2] = portable_aes_round(&portable[1], &portable[2]);
    portable[1] = portable_aes_round(&portable[0], &portable[1]);
    portable[0] = xor_block(&portable_aes_round(&tmp, &portable[0]), &m);

    update_riscv_fixslice(&mut fixslice, &m);
    assert_eq!(fixslice, portable);
  }

  // -- Spec test vectors (Appendix A.3) --

  fn spec_key() -> Aegis256Key {
    Aegis256Key::from_bytes(
      hex_block("10010000000000000000000000000000")
        .iter()
        .chain(hex_block("00000000000000000000000000000000").iter())
        .copied()
        .collect::<Vec<u8>>()
        .try_into()
        .unwrap(),
    )
  }

  fn spec_nonce() -> Nonce256 {
    let mut nonce_bytes = [0u8; 32];
    nonce_bytes[..16].copy_from_slice(&hex_block("10000200000000000000000000000000"));
    nonce_bytes[16..].copy_from_slice(&hex_block("00000000000000000000000000000000"));
    Nonce256::from_bytes(nonce_bytes)
  }

  fn verify_encrypt(msg: &[u8], aad: &[u8], expected_ct_hex: &str, expected_tag128_hex: &str) {
    let aead = Aegis256::new(&spec_key());
    let nonce = spec_nonce();
    let expected_ct = hex(expected_ct_hex);
    let expected_tag = hex(expected_tag128_hex);

    // Encrypt.
    let mut buf = msg.to_vec();
    let tag = aead.encrypt_in_place(&nonce, aad, &mut buf).unwrap();
    assert_eq!(&buf, &expected_ct, "ciphertext mismatch");
    assert_eq!(tag.as_bytes(), expected_tag.as_slice(), "tag mismatch");

    // Decrypt round-trip.
    aead.decrypt_in_place(&nonce, aad, &mut buf, &tag).unwrap();
    assert_eq!(&buf, msg, "plaintext recovery mismatch");
  }

  /// Test vector 1: 16 bytes msg, no AAD.
  #[test]
  fn spec_vector_1() {
    verify_encrypt(
      &[0u8; 16],
      b"",
      "754fc3d8c973246dcc6d741412a4b236",
      "3fe91994768b332ed7f570a19ec5896e",
    );
  }

  /// Test vector 2: empty msg, no AAD (tag-only).
  #[test]
  fn spec_vector_2() {
    verify_encrypt(b"", b"", "", "e3def978a0f054afd1e761d7553afba3");
  }

  /// Test vector 3: 32 bytes msg + 8 bytes AAD.
  #[test]
  fn spec_vector_3() {
    verify_encrypt(
      &hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"),
      &hex("0001020304050607"),
      "f373079ed84b2709faee373584585d60accd191db310ef5d8b11833df9dec711",
      "8d86f91ee606e9ff26a01b64ccbdd91d",
    );
  }

  /// Test vector 4: 13 bytes msg + 8 bytes AAD (partial block).
  #[test]
  fn spec_vector_4() {
    verify_encrypt(
      &hex("000102030405060708090a0b0c0d"),
      &hex("0001020304050607"),
      "f373079ed84b2709faee37358458",
      "c60b9c2d33ceb058f96e6dd03c215652",
    );
  }

  /// Test vector 5: 40 bytes msg + 42 bytes AAD (multiple blocks + partial).
  #[test]
  fn spec_vector_5() {
    verify_encrypt(
      &hex("101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f3031323334353637"),
      &hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20212223242526272829"),
      "57754a7d09963e7c787583a2e7b859bb24fa1e04d49fd550b2511a358e3bca252a9b1b8b30cc4a67",
      "ab8a7d53fd0e98d727accca94925e128",
    );
  }

  // -- Functional tests --

  #[test]
  fn round_trip_empty() {
    let key = Aegis256Key::from_bytes([0u8; 32]);
    let nonce = Nonce256::from_bytes([0u8; 32]);
    let aead = Aegis256::new(&key);

    let mut buf = [];
    let tag = aead.encrypt_in_place(&nonce, b"", &mut buf).unwrap();
    aead.decrypt_in_place(&nonce, b"", &mut buf, &tag).unwrap();
  }

  #[test]
  fn round_trip_with_data() {
    let key = Aegis256Key::from_bytes([0x42; 32]);
    let nonce = Nonce256::from_bytes([0x13; 32]);
    let aead = Aegis256::new(&key);
    let plaintext = b"the quick brown fox jumps over the lazy dog";

    let mut buf = *plaintext;
    let tag = aead.encrypt_in_place(&nonce, b"header", &mut buf).unwrap();
    assert_ne!(&buf[..], &plaintext[..]);

    aead.decrypt_in_place(&nonce, b"header", &mut buf, &tag).unwrap();
    assert_eq!(&buf[..], &plaintext[..]);
  }

  #[test]
  fn round_trip_with_aad_only() {
    let key = Aegis256Key::from_bytes([0xFF; 32]);
    let nonce = Nonce256::from_bytes([0xAA; 32]);
    let aead = Aegis256::new(&key);

    let mut buf = [];
    let tag = aead
      .encrypt_in_place(&nonce, b"associated data only", &mut buf)
      .unwrap();
    aead
      .decrypt_in_place(&nonce, b"associated data only", &mut buf, &tag)
      .unwrap();
  }

  #[test]
  fn buffer_zeroed_on_auth_failure() {
    let key = Aegis256Key::from_bytes([0x42; 32]);
    let nonce = Nonce256::from_bytes([0x13; 32]);
    let aead = Aegis256::new(&key);

    let mut buf = *b"zero me on failure";
    let tag = aead.encrypt_in_place(&nonce, b"aad", &mut buf).unwrap();

    let mut bad_tag = tag.to_bytes();
    bad_tag[0] ^= 0xFF;
    let bad_tag = Aegis256Tag::from_bytes(bad_tag);

    let result = aead.decrypt_in_place(&nonce, b"aad", &mut buf, &bad_tag);
    assert!(result.is_err());
    assert!(buf.iter().all(|&b| b == 0), "buffer not zeroed on auth failure");
  }

  #[test]
  fn tampered_ciphertext_fails() {
    let key = Aegis256Key::from_bytes([1; 32]);
    let nonce = Nonce256::from_bytes([2; 32]);
    let aead = Aegis256::new(&key);

    let mut buf = *b"secret";
    let tag = aead.encrypt_in_place(&nonce, b"", &mut buf).unwrap();

    buf[0] ^= 1;
    let result = aead.decrypt_in_place(&nonce, b"", &mut buf, &tag);
    assert!(result.is_err());
    assert_eq!(&buf, &[0u8; 6]);
  }

  #[test]
  fn tampered_tag_fails() {
    let key = Aegis256Key::from_bytes([3; 32]);
    let nonce = Nonce256::from_bytes([4; 32]);
    let aead = Aegis256::new(&key);

    let mut buf = *b"data";
    let tag = aead.encrypt_in_place(&nonce, b"aad", &mut buf).unwrap();

    let mut bad_tag_bytes = tag.to_bytes();
    bad_tag_bytes[15] ^= 1;
    let bad_tag = Aegis256Tag::from_bytes(bad_tag_bytes);

    let result = aead.decrypt_in_place(&nonce, b"aad", &mut buf, &bad_tag);
    assert!(result.is_err());
    assert_eq!(&buf, &[0u8; 4]);
  }

  #[test]
  fn wrong_aad_fails() {
    let key = Aegis256Key::from_bytes([5; 32]);
    let nonce = Nonce256::from_bytes([6; 32]);
    let aead = Aegis256::new(&key);

    let mut buf = *b"msg";
    let tag = aead.encrypt_in_place(&nonce, b"correct", &mut buf).unwrap();

    let result = aead.decrypt_in_place(&nonce, b"wrong", &mut buf, &tag);
    assert!(result.is_err());
  }

  #[test]
  fn wrong_nonce_fails() {
    let key = Aegis256Key::from_bytes([9; 32]);
    let nonce = Nonce256::from_bytes([10; 32]);
    let aead = Aegis256::new(&key);

    let mut buf = *b"nonce test";
    let tag = aead.encrypt_in_place(&nonce, b"aad", &mut buf).unwrap();

    let wrong_nonce = Nonce256::from_bytes([11; 32]);
    let result = aead.decrypt_in_place(&wrong_nonce, b"aad", &mut buf, &tag);
    assert!(result.is_err());
  }

  #[test]
  fn combined_encrypt_decrypt_round_trip() {
    let key = Aegis256Key::from_bytes([7; 32]);
    let nonce = Nonce256::from_bytes([8; 32]);
    let aead = Aegis256::new(&key);
    let pt = b"combined mode";

    let mut sealed = vec![0u8; pt.len().strict_add(TAG_SIZE)];
    aead.encrypt(&nonce, b"h", pt.as_slice(), &mut sealed).unwrap();

    let mut opened = vec![0u8; pt.len()];
    aead.decrypt(&nonce, b"h", &sealed, &mut opened).unwrap();
    assert_eq!(&opened, &pt[..]);
  }

  #[test]
  fn tag_from_slice_rejects_wrong_length() {
    assert!(Aegis256::tag_from_slice(&[0u8; 15]).is_err());
    assert!(Aegis256::tag_from_slice(&[0u8; 17]).is_err());
    assert!(Aegis256::tag_from_slice(&[0u8; 16]).is_ok());
  }

  #[test]
  fn multi_block_round_trip() {
    let key = Aegis256Key::from_bytes([0xAB; 32]);
    let nonce = Nonce256::from_bytes([0xCD; 32]);
    let aead = Aegis256::new(&key);

    // 100 bytes = 6 full blocks + 4-byte tail.
    let plaintext = [0x77u8; 100];
    let mut buf = plaintext;
    let tag = aead
      .encrypt_in_place(&nonce, b"multi-block aad that is longer than one rate block", &mut buf)
      .unwrap();
    aead
      .decrypt_in_place(
        &nonce,
        b"multi-block aad that is longer than one rate block",
        &mut buf,
        &tag,
      )
      .unwrap();
    assert_eq!(buf, plaintext);
  }

  #[test]
  fn exact_block_boundary() {
    let key = Aegis256Key::from_bytes([0x10; 32]);
    let nonce = Nonce256::from_bytes([0x20; 32]);
    let aead = Aegis256::new(&key);

    // Exactly 16 bytes = 1 full block, 0-byte tail.
    let plaintext = [0x55u8; 16];
    let mut buf = plaintext;
    let tag = aead.encrypt_in_place(&nonce, b"", &mut buf).unwrap();
    aead.decrypt_in_place(&nonce, b"", &mut buf, &tag).unwrap();
    assert_eq!(buf, plaintext);

    // Exactly 32 bytes = 2 full blocks, 0-byte tail.
    let plaintext32 = [0x66u8; 32];
    let mut buf32 = plaintext32;
    let tag32 = aead.encrypt_in_place(&nonce, b"", &mut buf32).unwrap();
    aead.decrypt_in_place(&nonce, b"", &mut buf32, &tag32).unwrap();
    assert_eq!(buf32, plaintext32);
  }

  /// Round-trip at 4-block boundary sizes to exercise the 4x unrolled loop.
  #[test]
  fn four_block_boundaries() {
    let key = Aegis256Key::from_bytes([0xF4; 32]);
    let nonce = Nonce256::from_bytes([0xB0; 32]);
    let aead = Aegis256::new(&key);
    let aad = b"four-block-test";

    for &size in &[48, 64, 80, 96, 112, 128, 256, 1024, 4096] {
      let plaintext: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
      let mut buf = plaintext.clone();
      let tag = aead.encrypt_in_place(&nonce, aad, &mut buf).unwrap();
      assert_ne!(&buf, &plaintext, "size {size}: ciphertext must differ");
      aead.decrypt_in_place(&nonce, aad, &mut buf, &tag).unwrap();
      assert_eq!(&buf, &plaintext, "size {size}: round-trip failed");
    }
  }

  // -- vperm S-box table validation --
  //
  // Verifies the Hamburg tower-field lookup tables produce the correct AES
  // S-box output for all 256 input values. This runs on all platforms using
  // pure scalar simulation of the VPERM operations, catching transcription
  // errors in the constant tables without needing s390x hardware.

  /// Scalar simulation of vperm: table[index & 0x0F].
  fn vperm_scalar(table: &[u8; 16], index: u8) -> u8 {
    table[(index & 0x0F) as usize]
  }

  /// Scalar simulation of vperm with PSHUFB zeroing: returns 0 when bit 7 set.
  fn vperm_z_scalar(table: &[u8; 16], index: u8) -> u8 {
    if index & 0x80 != 0 {
      0
    } else {
      table[(index & 0x0F) as usize]
    }
  }

  // Import the Hamburg vperm tables from the shared constants in aes_round.rs.
  use super::super::aes_round::{
    VPERM_INV_HI as T_INV_HI, VPERM_INV_LO as T_INV_LO, VPERM_IPT_HI as T_IPT_HI, VPERM_IPT_LO as T_IPT_LO,
    VPERM_SBOT as T_SBOT, VPERM_SBOU as T_SBOU,
  };

  /// Compute AES S-box for one byte using the vperm tower-field tables.
  ///
  /// This mirrors the `aes_round` Phase 2-4 logic (input transform,
  /// GF(2^4) inverse, output transform) but operates on a single byte
  /// without SIMD.
  fn vperm_sbox_scalar(input: u8) -> u8 {
    let lo_nib = input & 0x0F;
    let hi_nib = input >> 4;

    // Phase 2: Input transform (AES basis → tower field)
    let ipt_l = vperm_scalar(&T_IPT_LO, lo_nib);
    let ipt_h = vperm_scalar(&T_IPT_HI, hi_nib);
    let x = ipt_l ^ ipt_h;

    // Phase 3: Nibble extraction of transformed value
    let t_lo = x & 0x0F;
    let t_hi = x >> 4;

    // Phase 4: GF(2^4) inverse (5 vperm + 5 XOR)
    // vperm_z_scalar emulates PSHUFB zeroing for indices with bit 7 set.
    let ak = vperm_scalar(&T_INV_HI, t_lo);
    let j = t_hi ^ t_lo;
    let inv_i = vperm_scalar(&T_INV_LO, t_hi);
    let iak = inv_i ^ ak;
    let inv_j = vperm_scalar(&T_INV_LO, j);
    let jak = inv_j ^ ak;
    let inv_iak = vperm_z_scalar(&T_INV_LO, iak); // zeroing for 0x80 sentinel
    let io = inv_iak ^ j; // output high nibble
    let inv_jak = vperm_z_scalar(&T_INV_LO, jak); // zeroing for 0x80 sentinel
    let jo = inv_jak ^ t_hi; // output low nibble

    // Phase 5: Output transform (SubBytes only, no MixColumns)
    // sbo tables encode the combined inverse-isomorphism + AES affine
    // WITHOUT MixColumns — giving the pure S-box output.
    // io/jo can have bit 7 set from the 0x80 sentinel, so use vperm_z.
    let su = vperm_z_scalar(&T_SBOU, io);
    let st = vperm_z_scalar(&T_SBOT, jo);
    su ^ st
  }

  #[test]
  fn vperm_sbox_tables_match_aes_sbox() {
    // The canonical AES S-box (FIPS 197, Table 4).
    #[rustfmt::skip]
    const AES_SBOX: [u8; 256] = [
      0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
      0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
      0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
      0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
      0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
      0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
      0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
      0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
      0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
      0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
      0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
      0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
      0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
      0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
      0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
      0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
    ];

    let mut failures = 0u32;
    for input in 0u16..256 {
      let got = vperm_sbox_scalar(input as u8);
      // The Hamburg vperm S-box omits the AES affine constant.
      // vpaes_sbox(x) = AES_sbox(x) ^ AES_AFFINE for all x.
      use super::super::aes_round::AES_AFFINE;
      let expected = AES_SBOX[input as usize] ^ AES_AFFINE;
      if got != expected {
        if failures < 16 {
          eprintln!(
            "vperm S-box mismatch at input 0x{:02X}: got 0x{:02X}, expected 0x{:02X}",
            input, got, expected,
          );
        }
        failures = failures.strict_add(1);
      }
    }
    assert_eq!(failures, 0, "{failures} vperm S-box mismatches out of 256");
  }

  // -- Full vperm AES round validation --
  //
  // Simulates the complete vperm AES round (SubBytes + ShiftRows +
  // MixColumns + AddRoundKey) using scalar operations and verifies
  // it matches the portable aes_enc_round_portable() for multiple
  // test vectors.

  /// Full vperm AES round simulation (scalar): SubBytes → ShiftRows → MixColumns → AddRoundKey.
  #[cfg(not(target_arch = "s390x"))]
  fn vperm_aes_round_scalar(block: &[u8; 16], round_key: &[u8; 16]) -> [u8; 16] {
    // SubBytes via vperm tower field (includes affine constant compensation)
    use super::super::aes_round::{AES_AFFINE, VPERM_SR as SR};
    let mut sb = [0u8; 16];
    for i in 0..16 {
      sb[i] = vperm_sbox_scalar(block[i]) ^ AES_AFFINE;
    }

    // ShiftRows
    let mut sr = [0u8; 16];
    for i in 0..16 {
      sr[i] = sb[SR[i] as usize];
    }

    // MixColumns via xtime decomposition
    fn xtime(b: u8) -> u8 {
      let r = (b as u16) << 1;
      (r ^ (if r & 0x100 != 0 { 0x1B } else { 0 })) as u8
    }
    let mut mc = [0u8; 16];
    for col in 0..4 {
      let c = col * 4;
      let (b0, b1, b2, b3) = (sr[c], sr[c + 1], sr[c + 2], sr[c + 3]);
      mc[c] = xtime(b0) ^ xtime(b1) ^ b1 ^ b2 ^ b3;
      mc[c + 1] = b0 ^ xtime(b1) ^ xtime(b2) ^ b2 ^ b3;
      mc[c + 2] = b0 ^ b1 ^ xtime(b2) ^ xtime(b3) ^ b3;
      mc[c + 3] = xtime(b0) ^ b0 ^ b1 ^ b2 ^ xtime(b3);
    }

    // AddRoundKey
    let mut result = [0u8; 16];
    for i in 0..16 {
      result[i] = mc[i] ^ round_key[i];
    }
    result
  }

  #[cfg(not(target_arch = "s390x"))]
  #[test]
  fn vperm_full_round_matches_portable() {
    let input = hex_block("000102030405060708090a0b0c0d0e0f");
    let rk = hex_block("101112131415161718191a1b1c1d1e1f");
    let expected = portable_aes_round(&input, &rk);

    let got = vperm_aes_round_scalar(&input, &rk);
    assert_eq!(got, expected, "vperm round mismatch for spec vector");

    // Test with all-zero
    let zero = [0u8; 16];
    assert_eq!(vperm_aes_round_scalar(&zero, &zero), portable_aes_round(&zero, &zero));

    // Test with all-0xFF
    let ff = [0xFFu8; 16];
    assert_eq!(vperm_aes_round_scalar(&ff, &ff), portable_aes_round(&ff, &ff));

    // Test with random-looking patterns
    let a = hex_block("deadbeefcafebabe0123456789abcdef");
    let b = hex_block("fedcba9876543210aabbccddeeff0011");
    assert_eq!(vperm_aes_round_scalar(&a, &b), portable_aes_round(&a, &b));

    // Exhaustive: test all single-byte patterns in position 0
    for val in 0u16..256 {
      let mut block = [0u8; 16];
      block[0] = val as u8;
      let key = [0u8; 16];
      let got = vperm_aes_round_scalar(&block, &key);
      let expected = portable_aes_round(&block, &key);
      assert_eq!(got, expected, "vperm round mismatch for block[0]=0x{:02X}", val,);
    }
  }
}
