//! Secret-handling utilities for cryptographic operations.

/// Opaque result of a content-independent cryptographic comparison.
///
/// Safe code can obtain a decision only from a semantic owner such as a key,
/// shared secret, or authentication tag. The private representation prevents
/// implicit conversion or inspection; [`declassify`](Self::declassify) is the
/// single explicit boundary that exposes the equality result as a `bool`.
///
/// `CtDecision` preserves the source-level comparison structure. Constant-time
/// claims still depend on the exact compiler, target, features, and generated
/// binary covered by the release evidence in `ct.toml`.
#[must_use = "a cryptographic comparison decision must be composed or explicitly declassified"]
pub struct CtDecision {
  mask: u8,
}

impl core::fmt::Debug for CtDecision {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str("CtDecision(..)")
  }
}

impl CtDecision {
  const TRUE_MASK: u8 = u8::MAX;

  #[cfg(any(
    test,
    feature = "aegis256",
    feature = "aes-gcm",
    feature = "aes-gcm-siv",
    feature = "aes-siv",
    feature = "argon2",
    feature = "ascon-aead",
    feature = "blake3",
    feature = "chacha20poly1305",
    feature = "ecdsa-p256",
    feature = "ecdsa-p384",
    feature = "ed25519",
    feature = "hmac",
    feature = "hmac-sha3",
    feature = "kmac",
    feature = "ml-dsa",
    feature = "ml-kem",
    feature = "p256-ecdh",
    feature = "p384-ecdh",
    feature = "poly1305",
    feature = "rsa",
    feature = "scrypt",
    feature = "x25519",
    feature = "xchacha20poly1305"
  ))]
  #[inline(always)]
  const fn from_difference(difference: u64) -> Self {
    let nonzero = ((difference | difference.wrapping_neg()) >> 63) as u8;
    Self {
      mask: nonzero.wrapping_sub(1),
    }
  }

  /// Expose the comparison result as a branchable boolean.
  ///
  /// Call this only at the semantic boundary where revealing equality is
  /// intended. Declassification consumes the decision so the opaque value
  /// cannot accidentally be reused after that boundary.
  #[inline(always)]
  #[must_use]
  pub const fn declassify(self) -> bool {
    self.mask == Self::TRUE_MASK
  }

  #[cfg(any(feature = "ed25519", feature = "kmac", feature = "rsa"))]
  #[inline(always)]
  pub(crate) const fn into_u8(self) -> u8 {
    self.mask & 1
  }

  #[cfg(feature = "ml-kem")]
  #[inline(always)]
  pub(crate) const fn into_mask(self) -> u8 {
    self.mask
  }
}

impl core::ops::BitAnd for CtDecision {
  type Output = Self;

  #[inline(always)]
  fn bitand(self, rhs: Self) -> Self::Output {
    Self {
      mask: self.mask & rhs.mask,
    }
  }
}

impl core::ops::BitOr for CtDecision {
  type Output = Self;

  #[inline(always)]
  fn bitor(self, rhs: Self) -> Self::Output {
    Self {
      mask: self.mask | rhs.mask,
    }
  }
}

impl core::ops::Not for CtDecision {
  type Output = Self;

  #[inline(always)]
  fn not(self) -> Self::Output {
    Self {
      mask: self.mask ^ Self::TRUE_MASK,
    }
  }
}

#[cfg(any(
  test,
  feature = "aegis256",
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "aes-siv",
  feature = "argon2",
  feature = "ascon-aead",
  feature = "blake3",
  feature = "chacha20poly1305",
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "hmac",
  feature = "hmac-sha3",
  feature = "kmac",
  feature = "ml-dsa",
  feature = "ml-kem",
  feature = "p256-ecdh",
  feature = "p384-ecdh",
  feature = "poly1305",
  feature = "rsa",
  feature = "scrypt",
  feature = "x25519",
  feature = "xchacha20poly1305"
))]
#[inline(always)]
fn byte_difference(left: &[u8], right: &[u8]) -> u64 {
  let mut difference = 0u64;
  let (left_chunks, left_remainder) = left.as_chunks::<8>();
  let (right_chunks, right_remainder) = right.as_chunks::<8>();

  for (left_chunk, right_chunk) in left_chunks.iter().zip(right_chunks) {
    difference |= u64::from_ne_bytes(*left_chunk) ^ u64::from_ne_bytes(*right_chunk);
  }

  let mut remainder = 0u8;
  for (left_byte, right_byte) in left_remainder.iter().zip(right_remainder) {
    remainder |= left_byte ^ right_byte;
  }
  difference | u64::from(remainder)
}

/// Compare two fixed-size byte arrays with work independent of their contents.
///
/// This is crate-private by design. Public comparison policy belongs to the
/// concrete cryptographic type that owns the bytes. Optimized machine-code
/// evidence is tracked separately in `ct.toml`; source structure is not a
/// universal constant-time guarantee.
#[cfg(any(
  test,
  feature = "aegis256",
  feature = "aes-gcm",
  feature = "aes-gcm-siv",
  feature = "aes-siv",
  all(feature = "argon2", feature = "phc-strings"),
  feature = "ascon-aead",
  feature = "blake3",
  feature = "chacha20poly1305",
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "hmac",
  feature = "hmac-sha3",
  feature = "ml-dsa",
  feature = "ml-kem",
  feature = "p256-ecdh",
  feature = "p384-ecdh",
  feature = "poly1305",
  all(feature = "scrypt", feature = "phc-strings"),
  feature = "x25519",
  feature = "xchacha20poly1305"
))]
#[inline(always)]
pub(crate) fn fixed_eq<const N: usize>(left: &[u8; N], right: &[u8; N]) -> CtDecision {
  // SECURITY: Keep the accumulated word opaque before declassification. LLVM can otherwise fold
  // equality into target-specific vector reductions; exact binary evidence still owns the
  // constant-time claim.
  CtDecision::from_difference(core::hint::black_box(byte_difference(left, right)))
}

/// Compare two byte slices whose lengths are public protocol inputs.
///
/// Length mismatch is intentionally observable. Equal-length contents are
/// traversed without content-dependent exits. Keep callers individually
/// classified in `ct.toml`; fixed-shape owner types must use [`fixed_eq`].
#[cfg(any(
  test,
  feature = "argon2",
  feature = "kmac",
  feature = "ml-kem",
  feature = "rsa",
  feature = "scrypt"
))]
#[inline]
pub(crate) fn public_len_eq(left: &[u8], right: &[u8]) -> CtDecision {
  if left.len() != right.len() {
    return CtDecision::from_difference(1);
  }
  CtDecision::from_difference(byte_difference(left, right))
}

/// Volatile-zero a byte slice without emitting a compiler fence.
///
/// Use when batching multiple zeroizations under a single fence.
/// Call `compiler_fence(SeqCst)` once after all buffers are zeroed.
#[inline(always)]
pub(crate) fn zeroize_no_fence(buf: &mut [u8]) {
  // SAFETY: align_to_mut returns valid prefix/words/suffix over the same allocation.
  let (prefix, words, suffix) = unsafe { buf.align_to_mut::<u64>() };
  for byte in prefix.iter_mut() {
    // SAFETY: byte is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(byte, 0) };
  }
  for word in words.iter_mut() {
    // SAFETY: word is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(word, 0) };
  }
  for byte in suffix.iter_mut() {
    // SAFETY: byte is a valid, aligned, dereferenceable pointer to initialized memory.
    unsafe { core::ptr::write_volatile(byte, 0) };
  }
}

/// Overwrite a byte slice with zeros using volatile writes.
///
/// The compiler cannot elide these writes, ensuring sensitive data
/// is cleared from memory even if the buffer is not read afterward.
///
/// # Examples
///
/// ```
/// use rscrypto::traits::ct::zeroize;
///
/// let mut buf = [0xFFu8; 16];
/// zeroize(&mut buf);
/// assert_eq!(buf, [0u8; 16]);
/// ```
#[inline(always)]
pub fn zeroize(buf: &mut [u8]) {
  zeroize_no_fence(buf);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

#[cfg(any(
  feature = "aes-gcm",
  feature = "ascon-aead",
  feature = "blake2b",
  feature = "blake2s",
  feature = "blake3",
  feature = "chacha20poly1305",
  feature = "p256-ecdh",
  feature = "p384-ecdh",
  feature = "poly1305",
  feature = "sha2",
  feature = "sha3",
  feature = "xchacha20poly1305"
))]
mod word_zero_sealed {
  /// Marker for primitive integer types whose zero representation is
  /// `0` and whose `write_volatile` of zero is a sound clear.
  ///
  /// Sealed: only the integer types listed here are accepted as scratch
  /// types for [`zeroize_words_no_fence`] / [`zeroize_words`]. New
  /// implementors must be reviewed for soundness (no padding, no Drop).
  pub(crate) trait WordZero: Copy {
    const ZERO: Self;
  }

  impl WordZero for u8 {
    const ZERO: Self = 0;
  }
  impl WordZero for u16 {
    const ZERO: Self = 0;
  }
  impl WordZero for u32 {
    const ZERO: Self = 0;
  }
  impl WordZero for u64 {
    const ZERO: Self = 0;
  }
  impl WordZero for u128 {
    const ZERO: Self = 0;
  }
  impl WordZero for usize {
    const ZERO: Self = 0;
  }
}

#[cfg(any(
  feature = "aes-gcm",
  feature = "ascon-aead",
  feature = "blake2b",
  feature = "blake2s",
  feature = "blake3",
  feature = "chacha20poly1305",
  feature = "p256-ecdh",
  feature = "p384-ecdh",
  feature = "poly1305",
  feature = "sha2",
  feature = "sha3",
  feature = "xchacha20poly1305"
))]
pub(crate) use word_zero_sealed::WordZero;

/// Volatile-zero a slice of `WordZero` integers without a compiler fence.
///
/// Use for word-shaped scratch buffers (compression states, Argon2 working
/// blocks, HMAC accumulators) that hand-rolled `for word in words { ... }`
/// loops over `core::ptr::write_volatile` patterns. Caller is responsible
/// for emitting a single `compiler_fence(SeqCst)` after all related
/// zeroizations.
#[cfg(any(
  feature = "aes-gcm",
  feature = "ascon-aead",
  feature = "blake2b",
  feature = "blake2s",
  feature = "blake3",
  feature = "chacha20poly1305",
  feature = "p256-ecdh",
  feature = "p384-ecdh",
  feature = "poly1305",
  feature = "sha2",
  feature = "sha3",
  feature = "xchacha20poly1305"
))]
#[inline(always)]
pub(crate) fn zeroize_words_no_fence<T: WordZero>(words: &mut [T]) {
  for word in words {
    // SAFETY: `word` is a valid, aligned, dereferenceable pointer to `T`.
    // `T: WordZero` guarantees `T` is a primitive integer with no padding
    // or Drop, so `write_volatile(word, T::ZERO)` is a sound clear.
    unsafe { core::ptr::write_volatile(word, T::ZERO) };
  }
}

/// Prevent the compiler from moving memory operations across a cleanup boundary.
#[cfg(feature = "blake3")]
#[inline(always)]
pub(crate) fn zeroize_fence() {
  #[cfg(target_arch = "powerpc64")]
  // SAFETY: the empty template has no operands, instructions, registers, stack
  // use, or unwind path. Omitting `nomem` gives it the compiler memory clobber
  // required here; the empty template deliberately emits no hardware fence.
  unsafe {
    core::arch::asm!("", options(nostack, preserves_flags));
  }

  #[cfg(not(target_arch = "powerpc64"))]
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

/// Volatile-zero a slice of `WordZero` integers and emit a compiler fence.
#[cfg(any(
  feature = "aes-gcm",
  feature = "ascon-aead",
  feature = "blake3",
  feature = "p256-ecdh",
  feature = "p384-ecdh",
  feature = "sha2",
  feature = "sha3"
))]
#[inline(always)]
pub(crate) fn zeroize_words<T: WordZero>(words: &mut [T]) {
  zeroize_words_no_fence(words);
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

/// Run `f` with Arm data-independent timing (`PSTATE.DIT`) set for the calling thread.
///
/// On AArch64 cores with `FEAT_DIT`, the architecture guarantees data-independent
/// timing for its listed instructions only while `PSTATE.DIT` is set. rscrypto sets
/// it inside its asymmetric, post-quantum, and password operations, where the toggle
/// costs well under 1% at realistic parameters. Short symmetric operations such as MACs, AEADs, and
/// fixed-size comparisons do not toggle it per call, because a toggle costs about
/// 30 ns on Apple Silicon. Wrap that work, or a whole worker loop, in this function
/// to cover it for one toggle per scope.
///
/// The previous state is restored when `f` returns or unwinds. On other
/// architectures, on cores without `FEAT_DIT`, and under Miri, this calls `f`
/// unchanged. It adds no constant-time claim beyond `ct.toml`.
///
/// # Examples
///
/// ```
/// use rscrypto::traits::ct::with_data_independent_timing;
///
/// let sum = with_data_independent_timing(|| 2 + 2);
/// assert_eq!(sum, 4);
/// ```
#[inline]
pub fn with_data_independent_timing<R>(f: impl FnOnce() -> R) -> R {
  let _guard = DataIndependentTiming::enter();
  f()
}

/// Holds `PSTATE.DIT` set on AArch64 and restores the previous state when dropped.
#[must_use = "the guard must stay alive for the whole secret-dependent operation"]
pub(crate) struct DataIndependentTiming {
  #[cfg(all(target_arch = "aarch64", not(miri)))]
  restore_disabled: bool,
}

impl DataIndependentTiming {
  /// Set `PSTATE.DIT` where `FEAT_DIT` exists; a no-op elsewhere.
  #[inline]
  pub(crate) fn enter() -> Self {
    #[cfg(test)]
    tests::DIT_ENTRIES.with(|entries| entries.set(entries.get().strict_add(1)));
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
      let Some(previous) = Self::state_if_supported() else {
        return Self {
          restore_disabled: false,
        };
      };
      let restore_disabled = previous == 0;
      if restore_disabled {
        // SAFETY: `state_if_supported` established FEAT_DIT. `.inst 0xd503415f`
        // encodes `msr DIT, #1`; it has no register operands. MSR DIT is available
        // at EL0, writes only PSTATE.DIT, and `Drop` restores the prior disabled
        // state on every normal or unwinding exit.
        unsafe {
          core::arch::asm!(".inst 0xd503415f", options(nostack, preserves_flags));
        }
      }
      Self { restore_disabled }
    }
    #[cfg(not(all(target_arch = "aarch64", not(miri))))]
    {
      Self {}
    }
  }

  #[cfg(all(target_arch = "aarch64", not(miri)))]
  #[inline]
  fn supported() -> bool {
    #[cfg(feature = "std")]
    {
      std::arch::is_aarch64_feature_detected!("dit")
    }
    #[cfg(not(feature = "std"))]
    {
      cfg!(target_feature = "dit")
    }
  }

  /// Read `PSTATE.DIT` when `FEAT_DIT` exists.
  #[cfg(all(target_arch = "aarch64", not(miri)))]
  #[inline]
  pub(crate) fn state_if_supported() -> Option<u64> {
    if !Self::supported() {
      return None;
    }
    let state: u64;
    // SAFETY: `supported` establishes FEAT_DIT before the DIT system register is
    // accessed. `.inst 0xd53b42a8` encodes `mrs x8, DIT`; the explicit late
    // output declares the complete register effect. MRS DIT is available at
    // EL0, touches no memory or stack, and the conservative asm options keep it
    // ordered with the guarded arithmetic.
    unsafe {
      core::arch::asm!(
        ".inst 0xd53b42a8",
        lateout("x8") state,
        options(nostack, preserves_flags)
      );
    }
    Some(state)
  }
}

impl Drop for DataIndependentTiming {
  #[inline]
  fn drop(&mut self) {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    if self.restore_disabled {
      // SAFETY: `restore_disabled` can be true only after FEAT_DIT was
      // established and `enter` enabled PSTATE.DIT. `.inst 0xd503405f` encodes
      // `msr DIT, #0`; it restores the prior state, has no register operands,
      // and touches no memory.
      unsafe {
        core::arch::asm!(".inst 0xd503405f", options(nostack, preserves_flags));
      }
    }
  }
}

impl core::fmt::Debug for DataIndependentTiming {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str("DataIndependentTiming")
  }
}

#[cfg(test)]
pub(crate) mod tests {
  use super::*;

  std::thread_local! {
    /// Guard entries on this thread; lets tests prove an operation entered DIT.
    pub(crate) static DIT_ENTRIES: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
  }

  /// Guard entries on this thread made while running `f`.
  pub(crate) fn dit_entries_during(f: impl FnOnce()) -> u64 {
    let before = DIT_ENTRIES.with(core::cell::Cell::get);
    f();
    DIT_ENTRIES.with(core::cell::Cell::get).strict_sub(before)
  }

  /// Every asymmetric, post-quantum, and password operation must run under the guard.
  #[cfg(any(
    feature = "x25519",
    feature = "ed25519",
    feature = "ecdsa-p256",
    feature = "ecdsa-p384",
    feature = "p256-ecdh",
    feature = "p384-ecdh",
    feature = "ml-kem",
    feature = "ml-dsa",
    feature = "slh-dsa",
    feature = "argon2",
    feature = "scrypt",
    feature = "pbkdf2"
  ))]
  #[test]
  fn slow_secret_operations_enter_data_independent_timing() {
    fn covered<R>(name: &str, operation: impl FnOnce() -> R) {
      let entries = dit_entries_during(|| {
        core::hint::black_box(operation());
      });
      assert!(entries >= 1, "{name} must enter data-independent timing");
    }

    #[cfg(feature = "x25519")]
    {
      let secret = crate::X25519SecretKey::from_bytes([0x53; 32]);
      let peer = crate::X25519SecretKey::from_bytes([0x35; 32]).public_key();
      covered("X25519 public key", || core::hint::black_box(secret.public_key()));
      covered("X25519 agreement", || {
        core::hint::black_box(secret.diffie_hellman(&peer))
      });
    }
    #[cfg(feature = "ed25519")]
    {
      let secret = crate::Ed25519SecretKey::from_bytes([0x53; 32]);
      covered("Ed25519 public key", || core::hint::black_box(secret.public_key()));
      covered("Ed25519 signing", || core::hint::black_box(secret.sign(b"dit")));
      let keypair = crate::Ed25519Keypair::from_secret_key(crate::Ed25519SecretKey::from_bytes([0x35; 32]));
      covered("Ed25519 keypair signing", || {
        core::hint::black_box(keypair.sign(b"dit"))
      });
      covered("Ed25519 keypair derivation", || {
        core::hint::black_box(crate::Ed25519Keypair::from_secret_key(
          crate::Ed25519SecretKey::from_bytes([7; 32]),
        ))
      });
      let mut der = [0; crate::Ed25519SecretKey::PKCS8_DER_LENGTH];
      secret.to_pkcs8_der_into(&mut der);
      covered("Ed25519 PKCS #8 import", || {
        core::hint::black_box(crate::Ed25519SecretKey::from_pkcs8_der(&der).expect("PKCS #8 import"))
      });
    }
    #[cfg(feature = "ecdsa-p256")]
    {
      let secret = crate::EcdsaP256SecretKey::from_bytes([1; 32]).expect("valid scalar");
      covered("P-256 ECDSA public key", || core::hint::black_box(secret.public_key()));
      covered("P-256 ECDSA blinded public key", || {
        core::hint::black_box(secret.try_public_key_blinded_with(|blind| {
          blind.fill(3);
          Ok::<(), ()>(())
        }))
      });
      covered("P-256 ECDSA signing", || core::hint::black_box(secret.try_sign(b"dit")));
      covered("P-256 ECDSA blinded signing", || {
        core::hint::black_box(secret.try_sign_blinded_with(b"dit", |blind| {
          blind.fill(3);
          Ok::<(), ()>(())
        }))
      });
      let mut der = [0; crate::EcdsaP256SecretKey::PKCS8_DER_LENGTH];
      covered("P-256 ECDSA PKCS #8 export", || secret.to_pkcs8_der_into(&mut der));
      covered("P-256 ECDSA PKCS #8 import", || {
        core::hint::black_box(crate::EcdsaP256SecretKey::from_pkcs8_der(&der).expect("PKCS #8 import"))
      });
      // The export's privateKey contents are a SEC1 ECPrivateKey.
      covered("P-256 ECDSA SEC1 import", || {
        core::hint::black_box(crate::EcdsaP256SecretKey::from_sec1_der(&der[29..]).expect("SEC1 import"))
      });
    }
    #[cfg(feature = "ecdsa-p384")]
    {
      let secret = crate::EcdsaP384SecretKey::from_bytes([1; 48]).expect("valid scalar");
      covered("P-384 ECDSA public key", || core::hint::black_box(secret.public_key()));
      covered("P-384 ECDSA blinded public key", || {
        core::hint::black_box(secret.try_public_key_blinded_with(|blind| {
          blind.fill(3);
          Ok::<(), ()>(())
        }))
      });
      covered("P-384 ECDSA signing", || core::hint::black_box(secret.try_sign(b"dit")));
      covered("P-384 ECDSA blinded signing", || {
        core::hint::black_box(secret.try_sign_blinded_with(b"dit", |blind| {
          blind.fill(3);
          Ok::<(), ()>(())
        }))
      });
      let mut der = [0; crate::EcdsaP384SecretKey::PKCS8_DER_LENGTH];
      covered("P-384 ECDSA PKCS #8 export", || secret.to_pkcs8_der_into(&mut der));
      covered("P-384 ECDSA PKCS #8 import", || {
        core::hint::black_box(crate::EcdsaP384SecretKey::from_pkcs8_der(&der).expect("PKCS #8 import"))
      });
      // The export's privateKey contents are a SEC1 ECPrivateKey.
      covered("P-384 ECDSA SEC1 import", || {
        core::hint::black_box(crate::EcdsaP384SecretKey::from_sec1_der(&der[27..]).expect("SEC1 import"))
      });
    }
    #[cfg(feature = "p256-ecdh")]
    {
      let generate = |byte: u8| {
        crate::P256EphemeralSecret::try_generate_with(|candidate| {
          candidate.fill(byte);
          Ok::<(), ()>(())
        })
        .expect("valid scalar")
      };
      let peer = generate(0x35).public_key();
      let secret = generate(0x53);
      covered("P-256 ECDH public key", || core::hint::black_box(secret.public_key()));
      covered("P-256 ECDH agreement", || {
        core::hint::black_box(secret.diffie_hellman(&peer))
      });
    }
    #[cfg(feature = "p384-ecdh")]
    {
      let generate = |byte: u8| {
        crate::P384EphemeralSecret::try_generate_with(|candidate| {
          candidate.fill(byte);
          Ok::<(), ()>(())
        })
        .expect("valid scalar")
      };
      let peer = generate(0x35).public_key();
      let secret = generate(0x53);
      covered("P-384 ECDH public key", || core::hint::black_box(secret.public_key()));
      covered("P-384 ECDH agreement", || {
        core::hint::black_box(secret.diffie_hellman(&peer))
      });
    }
    #[cfg(feature = "ml-kem")]
    {
      use crate::traits::Kem as _;
      let fill = |byte: u8| {
        move |out: &mut [u8]| {
          out.fill(byte);
          Ok::<(), crate::MlKemError>(())
        }
      };
      let mut keys = None;
      covered("ML-KEM key generation", || {
        keys = Some(crate::MlKem768::generate_keypair(fill(1)).expect("key generation"))
      });
      let (ek, dk) = keys.expect("generated keys");
      let mut encapsulated = None;
      covered("ML-KEM encapsulation", || {
        encapsulated = Some(crate::MlKem768::encapsulate(&ek, fill(2)).expect("encapsulation"))
      });
      let (ciphertext, _) = encapsulated.expect("encapsulated");
      covered("ML-KEM decapsulation", || {
        core::hint::black_box(crate::MlKem768::decapsulate(&dk, &ciphertext))
      });
      let mut prepared = None;
      covered("ML-KEM key preparation", || {
        prepared = Some(dk.prepare().expect("prepare"))
      });
      let prepared = prepared.expect("prepared key");
      covered("ML-KEM prepared decapsulation", || {
        core::hint::black_box(prepared.decapsulate(&ciphertext))
      });
      let seed = crate::MlKem768Seed::from_bytes([0x31; 64]);
      let mut der = [0; crate::MlKem768Seed::PKCS8_DER_LENGTH];
      seed.to_pkcs8_der_into(&mut der);
      covered("ML-KEM PKCS #8 import", || {
        core::hint::black_box(crate::MlKem768DecapsulationKey::from_pkcs8_der(&der))
      });
      covered("ML-KEM PKCS #8 seed import", || {
        core::hint::black_box(crate::MlKem768Seed::from_pkcs8_der(&der))
      });
      let mut expanded = [0; crate::MlKem768DecapsulationKey::PKCS8_DER_LENGTH];
      dk.to_pkcs8_der_into(&mut expanded);
      covered("ML-KEM PKCS #8 expanded-key import", || {
        core::hint::black_box(crate::MlKem768DecapsulationKey::from_pkcs8_der(&expanded))
      });
    }
    #[cfg(feature = "ml-dsa")]
    {
      let mut keys = None;
      covered("ML-DSA key generation", || {
        keys = Some(crate::MlDsa65::keypair_from_seed(&[0x31; 32]).expect("key generation"))
      });
      let (_, secret) = keys.expect("generated keys");
      covered("ML-DSA signing", || {
        core::hint::black_box(secret.sign_deterministic(b"dit", &[]))
      });
      let mut storage = Default::default();
      let mut prepared = None;
      covered("ML-DSA key preparation", || {
        prepared = Some(secret.prepare(&mut storage).expect("prepare"))
      });
      let prepared = prepared.expect("prepared key");
      covered("ML-DSA prepared signing", || {
        core::hint::black_box(prepared.sign_deterministic(b"dit", &[]))
      });
      let seed = crate::MlDsa65Seed::from_bytes([0x31; 32]);
      let mut der = [0; crate::MlDsa65Seed::PKCS8_DER_LENGTH];
      seed.to_pkcs8_der_into(&mut der);
      covered("ML-DSA PKCS #8 import", || {
        core::hint::black_box(crate::MlDsa65SecretKey::from_pkcs8_der(&der))
      });
      covered("ML-DSA PKCS #8 seed import", || {
        core::hint::black_box(crate::MlDsa65Seed::from_pkcs8_der(&der))
      });
    }
    #[cfg(feature = "slh-dsa")]
    {
      let fill = |out: &mut [u8]| {
        out.fill(0x31);
        Ok::<(), crate::SlhDsaError>(())
      };
      let mut keys = None;
      covered("SLH-DSA key generation", || {
        keys = Some(crate::SlhDsaShake128f::generate_keypair(fill).expect("key generation"))
      });
      let (_, secret) = keys.expect("generated keys");
      let mut signature = [0; crate::SlhDsaShake128f::SIGNATURE_LENGTH];
      covered("SLH-DSA signing", || {
        core::hint::black_box(secret.sign_deterministic(b"dit", &[], &mut signature))
      });
      let encoded = secret.expose_secret();
      covered("SLH-DSA secret-key import", || {
        core::hint::black_box(crate::SlhDsaShake128fSecretKey::try_from_slice(encoded.as_bytes()))
      });
      let mut der = [0; crate::SlhDsaShake128fSecretKey::PKCS8_DER_LENGTH];
      secret.to_pkcs8_der_into(&mut der);
      covered("SLH-DSA PKCS #8 import", || {
        core::hint::black_box(crate::SlhDsaShake128fSecretKey::from_pkcs8_der(&der))
      });
      let prehash = crate::HashSlhDsaShake128fWithShake128SecretKey::try_from_slice(encoded.as_bytes())
        .expect("HashSLH-DSA secret key");
      covered("HashSLH-DSA signing", || {
        core::hint::black_box(prehash.sign_prehash_with(&[0x42; 32], &[], fill, &mut signature))
      });
    }
    #[cfg(feature = "argon2")]
    {
      let params = crate::Argon2Params::new(8, 1, 1).expect("minimal params");
      let mut out = [0u8; 16];
      covered("Argon2id", || {
        core::hint::black_box(crate::Argon2id::derive(&params, b"pw", b"abcdefgh", &mut out))
      });
      #[cfg(feature = "phc-strings")]
      {
        let policy = crate::Argon2idPassword::new(params).expect("bounded PHC profile");
        let mut memory = [const { crate::Argon2Block::ZERO }; 8];
        // Rejection never enters derivation, so these calls prove the PHC
        // entry point enters the guard before approval can return.
        covered("Argon2id PHC caller memory", || {
          core::hint::black_box(policy.verify_password_with_memory(b"pw", "invalid", &mut memory))
        });
        covered("Argon2id PHC context and caller memory", || {
          core::hint::black_box(policy.verify_password_with_context_and_memory(
            b"pw",
            "invalid",
            crate::Argon2Context::new(b"pepper", b"tenant"),
            &mut memory,
          ))
        });
      }
    }
    #[cfg(feature = "scrypt")]
    {
      let params = crate::ScryptParams::new(1, 1, 1).expect("minimal params");
      let mut out = [0u8; 16];
      covered("scrypt", || {
        core::hint::black_box(crate::Scrypt::derive(&params, b"pw", b"salt", &mut out))
      });
      #[cfg(feature = "phc-strings")]
      {
        let policy = crate::ScryptPassword::new(params).expect("bounded PHC profile");
        let mut memory = [const { crate::ScryptBlock::ZERO }; 10];
        covered("scrypt PHC caller memory", || {
          core::hint::black_box(policy.verify_password_with_memory(b"pw", "invalid", &mut memory))
        });
      }
    }
    #[cfg(feature = "pbkdf2")]
    {
      let mut out = [0u8; 32];
      covered("PBKDF2 derivation", || {
        crate::Pbkdf2Sha256::derive_key_primitive(b"pw", b"salt", 1, &mut out)
      });
      covered("PBKDF2 verification", || {
        crate::Pbkdf2Sha256::verify_password_primitive(b"pw", b"salt", 1, &[0; 32])
      });
      covered("PBKDF2 instance verification", || {
        crate::Pbkdf2Sha256::new(b"pw").verify_primitive(b"salt", 1, &[0; 32])
      });
    }
  }

  /// BINSEC proves arithmetic leaves; the DIT register write and its feature
  /// detection must stay out of them, at the public entry points instead.
  #[cfg(all(rscrypto_internal, feature = "diag"))]
  #[test]
  fn proof_leaves_stay_outside_data_independent_timing() {
    #[cfg(feature = "ml-dsa")]
    assert_eq!(
      dit_entries_during(|| {
        core::hint::black_box(crate::auth::diag_mldsa_montgomery(3, 5));
      }),
      0
    );
    #[cfg(feature = "pbkdf2")]
    assert_eq!(
      dit_entries_during(|| {
        core::hint::black_box(crate::auth::diag_pbkdf2_sha256_verify_portable(&[1; 32], &[2; 32]));
      }),
      0
    );
  }

  #[test]
  fn data_independent_timing_is_scoped_and_restores_the_previous_state() {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    let before = DataIndependentTiming::state_if_supported();
    assert_eq!(
      dit_entries_during(|| assert_eq!(with_data_independent_timing(|| 7), 7)),
      1
    );
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    if let Some(before) = before {
      let _outer = DataIndependentTiming::enter();
      assert_ne!(DataIndependentTiming::state_if_supported(), Some(0));
      // A nested scope must not clear the state its caller set.
      with_data_independent_timing(|| ());
      assert_ne!(DataIndependentTiming::state_if_supported(), Some(0));
      drop(_outer);
      assert_eq!(DataIndependentTiming::state_if_supported(), Some(before));
    }
  }

  #[test]
  fn fixed_eq_checks_every_position() {
    let value = [0x5a; 64];
    assert!(fixed_eq(&value, &value).declassify());
    for index in [0, value.len() / 2, value.len() - 1] {
      let mut different = value;
      different[index] ^= 1;
      assert!(!fixed_eq(&value, &different).declassify());
    }
  }

  #[test]
  fn public_len_eq_exposes_only_length_and_result() {
    assert!(public_len_eq(b"abcdef", b"abcdef").declassify());
    assert!(!public_len_eq(b"abcdef", b"abcdeg").declassify());
    assert!(!public_len_eq(b"abcdef", b"abcde").declassify());
  }

  #[test]
  fn decisions_compose_before_declassification() {
    let equal = fixed_eq(b"equal", b"equal");
    let different = fixed_eq(b"equal", b"other");
    assert!(!(equal & different).declassify());

    let equal = fixed_eq(b"equal", b"equal");
    let different = fixed_eq(b"equal", b"other");
    assert!((equal | different).declassify());

    assert!((!fixed_eq(b"equal", b"other")).declassify());
  }

  // ── zeroize ─────────────────────────────────────────────────────────────

  #[test]
  fn zeroize_clears_buffer() {
    let mut buf = [0xFFu8; 37]; // odd size to exercise prefix/suffix
    zeroize(&mut buf);
    assert!(buf.iter().all(|&b| b == 0));
  }

  #[test]
  fn zeroize_empty_is_noop() {
    let mut buf = [];
    zeroize(&mut buf); // must not panic
  }
}
