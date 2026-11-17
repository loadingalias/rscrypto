//! HPKE: hybrid public key encryption (RFC 9180), Base mode.
//!
//! This module implements the Base mode of
//! [RFC 9180](https://www.rfc-editor.org/rfc/rfc9180.html) with its verified
//! errata except 7121, including erratum 7937 on the `DeriveKeyPair` suite
//! identifier.
//! [`draft-ietf-hpke-hpke-06`](https://datatracker.ietf.org/doc/draft-ietf-hpke-hpke/06/)
//! will replace RFC 9180; its Base mode is byte-for-byte the same.
//!
//! # X25519 private keys and erratum 7121
//!
//! Erratum [7121](https://www.rfc-editor.org/errata/eid7121) clamps a
//! serialized X25519 private key. rscrypto does not apply it: an X25519
//! private key is the 32 bytes that `DeriveKeyPair` returns, without
//! clamping. `draft-ietf-hpke-hpke-06` section 7.1.2 removes the clamp
//! ([hpkewg/hpke#42](https://github.com/hpkewg/hpke/pull/42)), and the RFC 9180
//! and Wycheproof vectors keep the bytes unclamped. X25519 clamps the scalar
//! when it uses it, so both forms give the same public key and shared secret.
//! No byte on the wire changes.
//!
//! Only qualified suites exist, each a type that implements [`HpkeSuite`]:
//!
//! | Suite | KEM | KDF | AEAD | Features |
//! | --- | --- | --- | --- | --- |
//! | [`HpkeX25519HkdfSha256Aes128Gcm`] | DHKEM(X25519, HKDF-SHA256), `0x0020` | HKDF-SHA256 | AES-128-GCM | `hpke`, `x25519`, `aes-gcm` |
//! | [`HpkeXWingHkdfSha256ChaCha20Poly1305`] | X-Wing, `0x647A` | HKDF-SHA256 | ChaCha20Poly1305 | `hpke`, `x-wing`, `chacha20poly1305` |
//!
//! The X-Wing suite follows `draft-ietf-hpke-pq-05` for its KEM identifier and
//! `DeriveKeyPair`. The `hpke` feature alone gives no suite.
//!
//! A sender runs [`HpkeSender::setup`] with the recipient's public key and
//! sends the encapsulation with its ciphertexts. The recipient runs
//! [`HpkeReceiver::setup`] with the encapsulation and its private key. Each
//! context seals or opens messages in order, and both can export secrets.
//! [`HpkeSuite::seal`] and [`HpkeSuite::open`] handle one message.
//!
//! The setup and `DeriveKeyPair` calls return their context or key by value,
//! which can leave a moved-from copy of its secrets in the caller's frame.
//! With `alloc`, [`HpkeSender::setup_in`], [`HpkeReceiver::setup_in`], and
//! [`HpkeSuite::derive_keypair_in`] build it directly in a box.
//!
//! ```
//! # #[cfg(all(feature = "x25519", feature = "aes-gcm"))] {
//! use rscrypto::{HpkeReceiver, HpkeSender, HpkeSuite, HpkeX25519HkdfSha256Aes128Gcm as Suite};
//!
//! // A fixed filler keeps the example reproducible. Use fresh randomness
//! // for real keys.
//! let fill = |out: &mut [u8]| {
//!   out.fill(7);
//!   Ok(())
//! };
//! let recipient = Suite::derive_keypair(b"recipient input keying material");
//! let (encapsulation, mut sender) = HpkeSender::<Suite>::setup(&recipient.public_key(), b"info", fill)?;
//! let mut ciphertext = [0u8; 5 + Suite::TAG_SIZE];
//! sender.seal(b"aad", b"hello", &mut ciphertext)?;
//!
//! let mut receiver = HpkeReceiver::<Suite>::setup(&encapsulation, &recipient, b"info")?;
//! let mut plaintext = [0u8; 5];
//! receiver.open(b"aad", &ciphertext, &mut plaintext)?;
//! assert_eq!(&plaintext, b"hello");
//! # }
//! # Ok::<(), rscrypto::HpkeError>(())
//! ```

#[cfg(feature = "alloc")]
use alloc::{boxed::Box, vec::Vec};
#[cfg(feature = "alloc")]
use core::alloc::Allocator;
use core::{error::Error, fmt};

use crate::{
  HkdfSha256,
  aead::{Nonce96, expert::AeadWithNonce},
  platform::ProcessOrigin,
  secret::ZeroizingBytes,
  traits::{
    Aead,
    ct::{self, DataIndependentTiming},
  },
};

/// HPKE operation error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HpkeError {
  /// The caller-provided random source failed.
  RandomGenerationFailed,
  /// The KEM rejected the recipient public key, such as an X25519 key of small order.
  EncapsulationFailed,
  /// The KEM rejected the encapsulation, such as an X25519 part of small order.
  DecapsulationFailed,
  /// A ciphertext failed authentication. The context does not change.
  OpenFailed,
  /// The context seals or opens no more messages.
  ///
  /// It reached its last sequence number, or a process fork copied a sender
  /// context from the process that set it up.
  MessageLimitReached,
  /// An output buffer has the wrong length, or an input exceeds the AEAD's length bound.
  InvalidLength,
  /// An export asked for more than 255 times the KDF output size.
  ExportTooLong,
}

impl fmt::Display for HpkeError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::RandomGenerationFailed => "HPKE random generation failed",
      Self::EncapsulationFailed => "HPKE encapsulation failed",
      Self::DecapsulationFailed => "HPKE decapsulation failed",
      Self::OpenFailed => "HPKE open failed",
      Self::MessageLimitReached => "HPKE context seals or opens no more messages",
      Self::InvalidLength => "HPKE buffer or input length is invalid",
      Self::ExportTooLong => "HPKE export length is too long",
    })
  }
}

impl Error for HpkeError {}

const VERSION_LABEL: &[u8] = b"HPKE-v1";
const BASE_MODE: u8 = 0x00;
const KDF_HKDF_SHA256: u16 = 0x0001;
/// Nh of HKDF-SHA256.
const HASH_SIZE: usize = 32;
/// Nn of every suite's AEAD.
const NONCE_SIZE: usize = 12;
/// Nsecret of every suite's KEM.
const SHARED_SECRET_SIZE: usize = 32;
const MAX_EXPORT_SIZE: usize = 255 * HASH_SIZE;

/// `concat("HPKE", I2OSP(kem_id, 2), I2OSP(kdf_id, 2), I2OSP(aead_id, 2))`.
const fn hpke_suite_id(kem_id: u16, kdf_id: u16, aead_id: u16) -> [u8; 10] {
  let [k0, k1] = kem_id.to_be_bytes();
  let [f0, f1] = kdf_id.to_be_bytes();
  let [a0, a1] = aead_id.to_be_bytes();
  [b'H', b'P', b'K', b'E', k0, k1, f0, f1, a0, a1]
}

/// `concat("KEM", I2OSP(kem_id, 2))`, the suite identifier of KEM-internal labels.
const fn kem_suite_id(kem_id: u16) -> [u8; 5] {
  let [k0, k1] = kem_id.to_be_bytes();
  [b'K', b'E', b'M', k0, k1]
}

/// `LabeledExtract(salt, label, ikm)` with HKDF-SHA256.
fn labeled_extract(suite_id: &[u8], salt: &[u8], label: &[u8], ikm: &[u8]) -> HkdfSha256 {
  HkdfSha256::extract_vectored(salt, &[VERSION_LABEL, suite_id, label, ikm])
}

/// `LabeledExpand(prk, label, info, L)` with HKDF-SHA256, where `info` is the
/// concatenation of its two parts and `L` is `out.len()`.
fn labeled_expand(prk: &HkdfSha256, suite_id: &[u8], label: &[u8], info: [&[u8]; 2], out: &mut [u8]) {
  let length = u16::try_from(out.len())
    .expect("every HPKE expansion is at most 255 * Nh bytes")
    .to_be_bytes();
  prk
    .expand_vectored(&[&length, VERSION_LABEL, suite_id, label, info[0], info[1]], out)
    .expect("every HPKE expansion is at most 255 * Nh bytes");
}

mod sealed {
  /// Only this module implements [`super::HpkeSuite`].
  pub trait Sealed {}
}

/// Zero-sized capability that only this module constructs, so the hidden
/// suite operations stay crate-internal.
#[doc(hidden)]
#[derive(Debug)]
pub struct HpkeSuiteToken(());

/// One qualified HPKE suite: a KEM, HKDF-SHA256, and an AEAD.
///
/// Sealed: rscrypto implements it for each suite in the [module](self)
/// documentation, and for no other type.
pub trait HpkeSuite: sealed::Sealed + Sized {
  /// The HPKE KEM identifier.
  const KEM_ID: u16;
  /// The HPKE KDF identifier.
  const KDF_ID: u16 = KDF_HKDF_SHA256;
  /// The HPKE AEAD identifier.
  const AEAD_ID: u16;
  /// Nenc: the encapsulation size in bytes.
  const ENCAPSULATION_SIZE: usize;
  /// Random bytes that a sender setup takes from `fill_random`.
  const ENCAPSULATION_RANDOM_SIZE: usize;
  /// Nt: the AEAD tag size in bytes.
  const TAG_SIZE: usize = <Self::Aead as Aead>::TAG_SIZE;
  /// `psk_id_hash` of Base mode, `LabeledExtract("", "psk_id_hash", "")`: a
  /// constant of the suite. A unit test derives it.
  #[doc(hidden)]
  const __PSK_ID_HASH: [u8; HASH_SIZE];

  /// The recipient public key that [`HpkeSender::setup`] takes.
  type PublicKey;
  /// The recipient private key that [`HpkeReceiver::setup`] takes.
  type PrivateKey;
  /// The encapsulation that the sender sends to the recipient.
  type Encapsulation;
  /// The suite's AEAD. Its tag type is the detached tag of
  /// [`HpkeSender::seal_in_place_detached`].
  type Aead: AeadWithNonce<Nonce = Nonce96>;

  /// `DeriveKeyPair(ikm)`: the recipient key pair that `ikm` determines.
  ///
  /// `ikm` must hold at least 32 bytes of entropy.
  #[must_use]
  fn derive_keypair(ikm: &[u8]) -> Self::PrivateKey;

  /// Like [`Self::derive_keypair`], with the key derived in place in memory
  /// from `alloc`. Moving the box moves only a pointer, so no by-value copy of
  /// the key remains.
  #[cfg(feature = "alloc")]
  #[must_use]
  fn derive_keypair_in<A: Allocator>(ikm: &[u8], alloc: A) -> Box<Self::PrivateKey, A>;

  /// `Encap(pkR)` with the randomness from `fill_random`, writing the
  /// shared secret into `shared_secret`.
  #[doc(hidden)]
  fn __encap(
    public_key: &Self::PublicKey,
    fill_random: &mut impl FnMut(&mut [u8]) -> Result<(), HpkeError>,
    shared_secret: &mut [u8; SHARED_SECRET_SIZE],
    token: HpkeSuiteToken,
  ) -> Result<Self::Encapsulation, HpkeError>;

  /// `Decap(enc, skR)`, writing the shared secret into `shared_secret`.
  #[doc(hidden)]
  fn __decap(
    encapsulation: &Self::Encapsulation,
    private_key: &Self::PrivateKey,
    shared_secret: &mut [u8; SHARED_SECRET_SIZE],
    token: HpkeSuiteToken,
  ) -> Result<(), HpkeError>;

  /// The AEAD context for the `Nk`-byte key that `fill` writes in place.
  #[doc(hidden)]
  fn __aead(fill: impl FnOnce(&mut [u8]), token: HpkeSuiteToken) -> Self::Aead;

  /// Single-shot `Seal`: set up a sender, seal one message into `out` as
  /// `ciphertext || tag`, and return the encapsulation.
  ///
  /// # Errors
  ///
  /// As [`HpkeSender::setup`] and [`HpkeSender::seal`].
  fn seal(
    public_key: &Self::PublicKey,
    info: &[u8],
    aad: &[u8],
    plaintext: &[u8],
    out: &mut [u8],
    fill_random: impl FnMut(&mut [u8]) -> Result<(), HpkeError>,
  ) -> Result<Self::Encapsulation, HpkeError> {
    let (encapsulation, mut sender) = HpkeSender::<Self>::setup(public_key, info, fill_random)?;
    sender.seal(aad, plaintext, out)?;
    Ok(encapsulation)
  }

  /// Single-shot `Open`: set up a receiver and open one message into `out`.
  ///
  /// # Errors
  ///
  /// As [`HpkeReceiver::setup`] and [`HpkeReceiver::open`].
  fn open(
    encapsulation: &Self::Encapsulation,
    private_key: &Self::PrivateKey,
    info: &[u8],
    aad: &[u8],
    ciphertext: &[u8],
    out: &mut [u8],
  ) -> Result<(), HpkeError> {
    HpkeReceiver::<Self>::setup(encapsulation, private_key, info)?.open(aad, ciphertext, out)
  }
}

/// The key schedule output shared by both roles.
struct Context<S: HpkeSuite> {
  aead: S::Aead,
  base_nonce: [u8; NONCE_SIZE],
  exporter: HkdfSha256,
  sequence: u64,
  /// The process that ran the key schedule. Only that process seals, because a
  /// copy that a fork made would repeat the sequence numbers.
  origin: ProcessOrigin,
}

/// `KeySchedule<ROLE>(mode_base, shared_secret, info, "", "")` up to its
/// `secret`, which `expand` receives with the suite identifier and the
/// `key_schedule_context`.
fn key_schedule<S: HpkeSuite, R>(
  shared_secret: &[u8; SHARED_SECRET_SIZE],
  info: &[u8],
  expand: impl FnOnce(&HkdfSha256, &[u8; 10], [&[u8]; 2]) -> R,
) -> R {
  let suite_id = hpke_suite_id(S::KEM_ID, S::KDF_ID, S::AEAD_ID);
  let mut key_schedule_context = [0u8; 1 + 2 * HASH_SIZE];
  key_schedule_context[0] = BASE_MODE;
  key_schedule_context[1..=HASH_SIZE].copy_from_slice(&S::__PSK_ID_HASH);
  key_schedule_context[1 + HASH_SIZE..].copy_from_slice(labeled_extract(&suite_id, b"", b"info_hash", info).prk());
  let secret = labeled_extract(&suite_id, shared_secret, b"secret", b"");
  expand(&secret, &suite_id, [key_schedule_context.as_slice(), b""])
}

impl<S: HpkeSuite> Context<S> {
  /// The key schedule. The base nonce is written in place in the context.
  fn new(shared_secret: &[u8; SHARED_SECRET_SIZE], info: &[u8]) -> Self {
    key_schedule::<S, _>(shared_secret, info, |secret, suite_id, context| {
      let mut exporter_secret = ZeroizingBytes::<HASH_SIZE>::zeroed();
      labeled_expand(secret, suite_id, b"exp", context, exporter_secret.as_mut_array());
      let mut schedule = Self {
        aead: S::__aead(
          |key| labeled_expand(secret, suite_id, b"key", context, key),
          HpkeSuiteToken(()),
        ),
        base_nonce: [0; NONCE_SIZE],
        exporter: HkdfSha256::from_prk(exporter_secret.as_array()),
        sequence: 0,
        origin: ProcessOrigin::current(),
      };
      labeled_expand(secret, suite_id, b"base_nonce", context, &mut schedule.base_nonce);
      schedule
    })
  }

  /// A context with an all-zero AEAD key, which [`Self::schedule`] replaces.
  #[cfg(feature = "alloc")]
  fn unkeyed() -> Self {
    Self {
      aead: S::__aead(|key| key.fill(0), HpkeSuiteToken(())),
      base_nonce: [0; NONCE_SIZE],
      exporter: HkdfSha256::zeroed(),
      sequence: 0,
      origin: ProcessOrigin::current(),
    }
  }

  /// The key schedule, written into this context where it is.
  #[cfg(feature = "alloc")]
  fn schedule(&mut self, shared_secret: &[u8; SHARED_SECRET_SIZE], info: &[u8]) {
    key_schedule::<S, _>(shared_secret, info, |secret, suite_id, context| {
      self.aead = S::__aead(
        |key| labeled_expand(secret, suite_id, b"key", context, key),
        HpkeSuiteToken(()),
      );
      labeled_expand(secret, suite_id, b"base_nonce", context, &mut self.base_nonce);
      let mut exporter_secret = ZeroizingBytes::<HASH_SIZE>::zeroed();
      labeled_expand(secret, suite_id, b"exp", context, exporter_secret.as_mut_array());
      self.exporter.set_prk(exporter_secret.as_array());
    });
    self.origin = ProcessOrigin::current();
  }

  /// `ComputeNonce(seq)` for the current sequence number. The last `u64`
  /// sequence number is never used, so the next one always exists.
  fn nonce(&self) -> Result<Nonce96, HpkeError> {
    if self.sequence == u64::MAX {
      return Err(HpkeError::MessageLimitReached);
    }
    let mut nonce = self.base_nonce;
    for (byte, sequence) in nonce[NONCE_SIZE - 8..].iter_mut().zip(self.sequence.to_be_bytes()) {
      *byte ^= sequence;
    }
    Ok(Nonce96::from_bytes(nonce))
  }

  /// [`Self::nonce`] for a seal, which only the process that ran the key schedule makes.
  fn seal_nonce(&self) -> Result<Nonce96, HpkeError> {
    if !self.origin.is_current() {
      return Err(HpkeError::MessageLimitReached);
    }
    self.nonce()
  }

  fn seal(&mut self, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), HpkeError> {
    let nonce = self.seal_nonce()?;
    self
      .aead
      .encrypt(&nonce, aad, plaintext, out)
      .map_err(|_| HpkeError::InvalidLength)?;
    self.sequence = self.sequence.strict_add(1);
    Ok(())
  }

  fn seal_in_place_detached(&mut self, aad: &[u8], buffer: &mut [u8]) -> Result<<S::Aead as Aead>::Tag, HpkeError> {
    let nonce = self.seal_nonce()?;
    let tag = self
      .aead
      .encrypt_in_place(&nonce, aad, buffer)
      .map_err(|_| HpkeError::InvalidLength)?;
    self.sequence = self.sequence.strict_add(1);
    Ok(tag)
  }

  fn open(&mut self, aad: &[u8], ciphertext: &[u8], out: &mut [u8]) -> Result<(), HpkeError> {
    let nonce = self.nonce()?;
    let expected = S::Aead::plaintext_len(ciphertext.len()).map_err(|_| HpkeError::InvalidLength)?;
    if out.len() != expected {
      return Err(HpkeError::InvalidLength);
    }
    self
      .aead
      .decrypt(&nonce, aad, ciphertext, out)
      .map_err(|_| HpkeError::OpenFailed)?;
    self.sequence = self.sequence.strict_add(1);
    Ok(())
  }

  fn open_in_place_detached(
    &mut self,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &<S::Aead as Aead>::Tag,
  ) -> Result<(), HpkeError> {
    let nonce = self.nonce()?;
    self
      .aead
      .decrypt_in_place(&nonce, aad, buffer, tag)
      .map_err(|_| HpkeError::OpenFailed)?;
    self.sequence = self.sequence.strict_add(1);
    Ok(())
  }

  /// `Context.Export(exporter_context, L)` with `L = out.len()`.
  fn export(&self, exporter_context: &[u8], out: &mut [u8]) -> Result<(), HpkeError> {
    if out.len() > MAX_EXPORT_SIZE {
      return Err(HpkeError::ExportTooLong);
    }
    let suite_id = hpke_suite_id(S::KEM_ID, S::KDF_ID, S::AEAD_ID);
    labeled_expand(&self.exporter, &suite_id, b"sec", [exporter_context, b""], out);
    Ok(())
  }
}

impl<S: HpkeSuite> Drop for Context<S> {
  fn drop(&mut self) {
    ct::zeroize(&mut self.base_nonce);
  }
}

macro_rules! define_context {
  ($name:ident, $doc:literal) => {
    #[doc = $doc]
    ///
    /// Not `Clone` or `Copy`: a copy would reuse sequence numbers. A process
    /// fork also copies a context, so a sender seals only in the process that
    /// set it up. The AEAD key, base nonce, and exporter secret clear on drop;
    /// `Debug` is redacted.
    pub struct $name<S: HpkeSuite>(Context<S>);

    impl<S: HpkeSuite> $name<S> {
      /// Export `out.len()` bytes for `exporter_context` (`Context.Export`).
      ///
      /// # Errors
      ///
      /// Returns [`HpkeError::ExportTooLong`] when `out` is longer than
      /// 255 times the KDF output size, and writes nothing.
      pub fn export(&self, exporter_context: &[u8], out: &mut [u8]) -> Result<(), HpkeError> {
        self.0.export(exporter_context, out)
      }

      /// The sequence number of the next message.
      #[must_use]
      pub const fn sequence(&self) -> u64 {
        self.0.sequence
      }
    }

    impl<S: HpkeSuite> fmt::Debug for $name<S> {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(concat!(stringify!($name), "(****)"))
      }
    }
  };
}

define_context!(HpkeSender, "HPKE sender context: seals messages to one recipient.");
define_context!(HpkeReceiver, "HPKE receiver context: opens messages from one sender.");

impl<S: HpkeSuite> HpkeSender<S> {
  /// `SetupBaseS(pkR, info)`: encapsulate to `public_key` and return the
  /// encapsulation and the sender context.
  ///
  /// Takes [`HpkeSuite::ENCAPSULATION_RANDOM_SIZE`] bytes from `fill_random`.
  /// The callback must fill the entire buffer with fresh cryptographic
  /// randomness.
  ///
  /// # Errors
  ///
  /// Returns the error from `fill_random`, or [`HpkeError::EncapsulationFailed`]
  /// when the KEM rejects `public_key`.
  pub fn setup(
    public_key: &S::PublicKey,
    info: &[u8],
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), HpkeError>,
  ) -> Result<(S::Encapsulation, Self), HpkeError> {
    let _dit = DataIndependentTiming::enter();
    let mut shared_secret = ZeroizingBytes::<SHARED_SECRET_SIZE>::zeroed();
    let encapsulation = S::__encap(
      public_key,
      &mut fill_random,
      shared_secret.as_mut_array(),
      HpkeSuiteToken(()),
    )?;
    Ok((encapsulation, Self(Context::new(shared_secret.as_array(), info))))
  }

  /// Like [`Self::setup`], with the context built in place in memory from
  /// `alloc`.
  ///
  /// Moving the box moves only a pointer, so no by-value copy of the context
  /// remains. A failure returns before the allocation.
  ///
  /// # Errors
  ///
  /// As [`Self::setup`].
  #[cfg(feature = "alloc")]
  pub fn setup_in<A: Allocator>(
    public_key: &S::PublicKey,
    info: &[u8],
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), HpkeError>,
    alloc: A,
  ) -> Result<(S::Encapsulation, Box<Self, A>), HpkeError> {
    let _dit = DataIndependentTiming::enter();
    let mut shared_secret = ZeroizingBytes::<SHARED_SECRET_SIZE>::zeroed();
    let encapsulation = S::__encap(
      public_key,
      &mut fill_random,
      shared_secret.as_mut_array(),
      HpkeSuiteToken(()),
    )?;
    let mut sender = Box::new_in(Self(Context::unkeyed()), alloc);
    sender.0.schedule(shared_secret.as_array(), info);
    Ok((encapsulation, sender))
  }

  /// `Context.Seal(aad, pt)`: seal `plaintext` into `out` as `ciphertext || tag`.
  ///
  /// # Errors
  ///
  /// Returns [`HpkeError::InvalidLength`] when `out` is not
  /// `plaintext.len() + TAG_SIZE` bytes or the input exceeds the AEAD bound,
  /// and [`HpkeError::MessageLimitReached`] after the last sequence number or
  /// in a process fork child of the process that set up this sender. None of
  /// them changes the sequence number.
  pub fn seal(&mut self, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), HpkeError> {
    self.0.seal(aad, plaintext, out)
  }

  /// Like [`Self::seal`], encrypting `buffer` in place and returning the tag.
  ///
  /// # Errors
  ///
  /// As [`Self::seal`].
  pub fn seal_in_place_detached(&mut self, aad: &[u8], buffer: &mut [u8]) -> Result<<S::Aead as Aead>::Tag, HpkeError> {
    self.0.seal_in_place_detached(aad, buffer)
  }

  /// Like [`Self::seal`], returning `ciphertext || tag` in a new vector.
  ///
  /// # Errors
  ///
  /// As [`Self::seal`].
  #[cfg(feature = "alloc")]
  pub fn seal_to_vec(&mut self, aad: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, HpkeError> {
    let length = S::Aead::ciphertext_len(plaintext.len()).map_err(|_| HpkeError::InvalidLength)?;
    let mut out = alloc::vec![0u8; length];
    self.seal(aad, plaintext, &mut out)?;
    Ok(out)
  }
}

impl<S: HpkeSuite> HpkeReceiver<S> {
  /// `SetupBaseR(enc, skR, info)`: decapsulate `encapsulation` with
  /// `private_key` and return the receiver context.
  ///
  /// # Errors
  ///
  /// Returns [`HpkeError::DecapsulationFailed`] when the KEM rejects
  /// `encapsulation`.
  pub fn setup(encapsulation: &S::Encapsulation, private_key: &S::PrivateKey, info: &[u8]) -> Result<Self, HpkeError> {
    let _dit = DataIndependentTiming::enter();
    let mut shared_secret = ZeroizingBytes::<SHARED_SECRET_SIZE>::zeroed();
    S::__decap(
      encapsulation,
      private_key,
      shared_secret.as_mut_array(),
      HpkeSuiteToken(()),
    )?;
    Ok(Self(Context::new(shared_secret.as_array(), info)))
  }

  /// Like [`Self::setup`], with the context built in place in memory from
  /// `alloc`.
  ///
  /// Moving the box moves only a pointer, so no by-value copy of the context
  /// remains. A failure returns before the allocation.
  ///
  /// # Errors
  ///
  /// As [`Self::setup`].
  #[cfg(feature = "alloc")]
  pub fn setup_in<A: Allocator>(
    encapsulation: &S::Encapsulation,
    private_key: &S::PrivateKey,
    info: &[u8],
    alloc: A,
  ) -> Result<Box<Self, A>, HpkeError> {
    let _dit = DataIndependentTiming::enter();
    let mut shared_secret = ZeroizingBytes::<SHARED_SECRET_SIZE>::zeroed();
    S::__decap(
      encapsulation,
      private_key,
      shared_secret.as_mut_array(),
      HpkeSuiteToken(()),
    )?;
    let mut receiver = Box::new_in(Self(Context::unkeyed()), alloc);
    receiver.0.schedule(shared_secret.as_array(), info);
    Ok(receiver)
  }

  /// `Context.Open(aad, ct)`: open `ciphertext || tag` into `out`.
  ///
  /// A failure leaves the sequence number unchanged, so the context still
  /// opens the next valid message.
  ///
  /// # Errors
  ///
  /// Returns [`HpkeError::OpenFailed`] when authentication fails,
  /// [`HpkeError::InvalidLength`] when `ciphertext` is shorter than the tag or
  /// `out` is not `ciphertext.len() - TAG_SIZE` bytes, and
  /// [`HpkeError::MessageLimitReached`] after the last sequence number.
  pub fn open(&mut self, aad: &[u8], ciphertext: &[u8], out: &mut [u8]) -> Result<(), HpkeError> {
    self.0.open(aad, ciphertext, out)
  }

  /// Like [`Self::open`], decrypting `buffer` in place with the detached `tag`.
  ///
  /// # Errors
  ///
  /// As [`Self::open`].
  pub fn open_in_place_detached(
    &mut self,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &<S::Aead as Aead>::Tag,
  ) -> Result<(), HpkeError> {
    self.0.open_in_place_detached(aad, buffer, tag)
  }

  /// Like [`Self::open`], returning the plaintext in a new vector.
  ///
  /// # Errors
  ///
  /// As [`Self::open`].
  #[cfg(feature = "alloc")]
  pub fn open_to_vec(&mut self, aad: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, HpkeError> {
    let length = S::Aead::plaintext_len(ciphertext.len()).map_err(|_| HpkeError::InvalidLength)?;
    let mut out = alloc::vec![0u8; length];
    self.open(aad, ciphertext, &mut out)?;
    Ok(out)
  }
}

#[cfg(all(feature = "x25519", feature = "aes-gcm"))]
mod x25519_aes128gcm {
  use core::fmt;

  #[cfg(feature = "alloc")]
  use alloc::boxed::Box;
  #[cfg(feature = "alloc")]
  use core::alloc::Allocator;

  use super::{
    HpkeError, HpkeSuite, HpkeSuiteToken, SHARED_SECRET_SIZE, kem_suite_id, labeled_expand, labeled_extract, sealed,
  };
  use crate::{
    Aes128Gcm, Aes128GcmKey, X25519PublicKey, X25519SecretKey, X25519SharedSecret, secret::ZeroizingBytes,
    traits::ct::DataIndependentTiming,
  };

  const KEM_ID: u16 = 0x0020;
  const KEM_SUITE_ID: [u8; 5] = kem_suite_id(KEM_ID);

  /// HPKE suite DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-128-GCM (RFC 9180).
  ///
  /// Client ECH uses it. A recipient key is an [`HpkeX25519PrivateKey`], and
  /// the encapsulation is the sender's ephemeral [`X25519PublicKey`].
  #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
  pub struct HpkeX25519HkdfSha256Aes128Gcm;

  /// HPKE DHKEM(X25519) recipient key: an X25519 secret key and its public key.
  ///
  /// The serialized private key is the unclamped secret key bytes; the
  /// [module](super#x25519-private-keys-and-erratum-7121) documentation
  /// gives the reason.
  ///
  /// The public key is computed one time, so each receiver setup costs one
  /// X25519 function. Not `Clone` or `Copy`; the secret key clears on drop;
  /// `Debug` is redacted.
  pub struct HpkeX25519PrivateKey {
    secret: X25519SecretKey,
    public: X25519PublicKey,
  }

  impl HpkeX25519PrivateKey {
    /// Wrap `secret` and compute its public key. The serialized private key
    /// (`SerializePrivateKey`) is the secret key's bytes, as given.
    #[must_use]
    pub fn new(secret: X25519SecretKey) -> Self {
      let public = secret.public_key();
      Self { secret, public }
    }

    /// The public key `pkR`.
    #[must_use]
    pub const fn public_key(&self) -> X25519PublicKey {
      self.public
    }

    /// The secret key `skR`.
    #[must_use]
    pub const fn secret_key(&self) -> &X25519SecretKey {
      &self.secret
    }

    /// Explicitly duplicate this key.
    #[must_use]
    pub const fn duplicate_secret(&self) -> Self {
      Self {
        secret: self.secret.duplicate_secret(),
        public: self.public,
      }
    }
  }

  impl HpkeX25519PrivateKey {
    /// All-zero owner that [`Self::derive`] fills in place.
    fn zeroed() -> Self {
      Self {
        secret: X25519SecretKey::from_bytes([0; X25519SecretKey::LENGTH]),
        public: X25519PublicKey::from_bytes([0; X25519PublicKey::LENGTH]),
      }
    }

    /// `DeriveKeyPair(ikm)`, written in place.
    fn derive(&mut self, ikm: &[u8]) {
      let _dit = DataIndependentTiming::enter();
      let prk = labeled_extract(&KEM_SUITE_ID, b"", b"dkp_prk", ikm);
      labeled_expand(&prk, &KEM_SUITE_ID, b"sk", [b"", b""], self.secret.as_mut_bytes());
      self.public = self.secret.public_key();
    }
  }

  impl fmt::Debug for HpkeX25519PrivateKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      f.write_str("HpkeX25519PrivateKey(****)")
    }
  }

  /// `ExtractAndExpand(dh, kem_context)` with `kem_context = enc || pkRm`,
  /// written into `shared_secret`.
  fn extract_and_expand(
    dh: &[u8; X25519SharedSecret::LENGTH],
    encapsulation: &X25519PublicKey,
    public_key: &X25519PublicKey,
    shared_secret: &mut [u8; SHARED_SECRET_SIZE],
  ) {
    let prk = labeled_extract(&KEM_SUITE_ID, b"", b"eae_prk", dh);
    labeled_expand(
      &prk,
      &KEM_SUITE_ID,
      b"shared_secret",
      [encapsulation.as_bytes(), public_key.as_bytes()],
      shared_secret,
    );
  }

  impl sealed::Sealed for HpkeX25519HkdfSha256Aes128Gcm {}

  impl HpkeSuite for HpkeX25519HkdfSha256Aes128Gcm {
    const KEM_ID: u16 = KEM_ID;
    const AEAD_ID: u16 = 0x0001;
    const __PSK_ID_HASH: [u8; 32] = [
      0x72, 0x56, 0x11, 0xc9, 0xd9, 0x8c, 0x07, 0xc0, 0x3f, 0x60, 0x09, 0x5c, 0xd3, 0x2d, 0x40, 0x0d, 0x83, 0x47, 0xd4,
      0x5e, 0xd6, 0x70, 0x97, 0xbb, 0xad, 0x50, 0xfc, 0x56, 0xda, 0x74, 0x2d, 0x07,
    ];
    const ENCAPSULATION_SIZE: usize = X25519PublicKey::LENGTH;
    const ENCAPSULATION_RANDOM_SIZE: usize = X25519SecretKey::LENGTH;

    type PublicKey = X25519PublicKey;
    type PrivateKey = HpkeX25519PrivateKey;
    type Encapsulation = X25519PublicKey;
    type Aead = Aes128Gcm;

    fn derive_keypair(ikm: &[u8]) -> HpkeX25519PrivateKey {
      let mut key = HpkeX25519PrivateKey::zeroed();
      key.derive(ikm);
      key
    }

    #[cfg(feature = "alloc")]
    fn derive_keypair_in<A: Allocator>(ikm: &[u8], alloc: A) -> Box<HpkeX25519PrivateKey, A> {
      let mut key = Box::new_in(HpkeX25519PrivateKey::zeroed(), alloc);
      key.derive(ikm);
      key
    }

    /// The ephemeral key is `DeriveKeyPair` of 32 random bytes, as RFC 9180
    /// section 7.1.3 allows, so test vectors run through this path.
    fn __encap(
      public_key: &X25519PublicKey,
      fill_random: &mut impl FnMut(&mut [u8]) -> Result<(), HpkeError>,
      shared_secret: &mut [u8; SHARED_SECRET_SIZE],
      _: HpkeSuiteToken,
    ) -> Result<X25519PublicKey, HpkeError> {
      let mut ikm = ZeroizingBytes::<{ X25519SecretKey::LENGTH }>::zeroed();
      fill_random(ikm.as_mut_array())?;
      let ephemeral = Self::derive_keypair(ikm.as_array());
      let mut dh = ZeroizingBytes::<{ X25519SharedSecret::LENGTH }>::zeroed();
      ephemeral
        .secret
        .diffie_hellman_into(public_key, dh.as_mut_array())
        .map_err(|_| HpkeError::EncapsulationFailed)?;
      extract_and_expand(dh.as_array(), &ephemeral.public, public_key, shared_secret);
      Ok(ephemeral.public)
    }

    fn __decap(
      encapsulation: &X25519PublicKey,
      private_key: &HpkeX25519PrivateKey,
      shared_secret: &mut [u8; SHARED_SECRET_SIZE],
      _: HpkeSuiteToken,
    ) -> Result<(), HpkeError> {
      let mut dh = ZeroizingBytes::<{ X25519SharedSecret::LENGTH }>::zeroed();
      private_key
        .secret
        .diffie_hellman_into(encapsulation, dh.as_mut_array())
        .map_err(|_| HpkeError::DecapsulationFailed)?;
      extract_and_expand(dh.as_array(), encapsulation, &private_key.public, shared_secret);
      Ok(())
    }

    fn __aead(fill: impl FnOnce(&mut [u8]), _: HpkeSuiteToken) -> Aes128Gcm {
      Aes128Gcm::new(&Aes128GcmKey::generate(|key| fill(key)))
    }
  }
}

#[cfg(all(feature = "x25519", feature = "aes-gcm"))]
pub use x25519_aes128gcm::{HpkeX25519HkdfSha256Aes128Gcm, HpkeX25519PrivateKey};

#[cfg(all(feature = "x-wing", feature = "chacha20poly1305"))]
mod xwing_chacha20poly1305 {
  #[cfg(feature = "alloc")]
  use alloc::boxed::Box;
  #[cfg(feature = "alloc")]
  use core::alloc::Allocator;

  use super::{HpkeError, HpkeSuite, HpkeSuiteToken, SHARED_SECRET_SIZE, VERSION_LABEL, kem_suite_id, sealed};
  use crate::{
    ChaCha20Poly1305, ChaCha20Poly1305Key, XWingCiphertext, XWingDecapsulationKey, XWingError,
    XWingPreparedDecapsulationKey, XWingPreparedEncapsulationKey, hashes::crypto::keccak::secret,
    traits::ct::DataIndependentTiming,
  };

  const KEM_ID: u16 = 0x647A;
  const KEM_SUITE_ID: [u8; 5] = kem_suite_id(KEM_ID);
  const DERIVE_LABEL: &[u8] = b"DeriveKeyPair";

  /// `DeriveKeyPair(ikm)`, expanded in place into `key`.
  fn derive(ikm: &[u8], key: &mut XWingPreparedDecapsulationKey) {
    let _dit = DataIndependentTiming::enter();
    let label_length = u16::try_from(DERIVE_LABEL.len())
      .expect("the label is short")
      .to_be_bytes();
    let output_length = u16::try_from(XWingDecapsulationKey::LENGTH)
      .expect("the seed is short")
      .to_be_bytes();
    let mut seed = XWingDecapsulationKey::from_bytes([0; XWingDecapsulationKey::LENGTH]);
    secret::shake256(
      &[
        ikm,
        VERSION_LABEL,
        &KEM_SUITE_ID,
        &label_length,
        DERIVE_LABEL,
        &output_length,
      ],
      seed.as_mut_bytes(),
    );
    key.expand(&seed);
  }

  /// HPKE suite X-Wing, HKDF-SHA256, ChaCha20Poly1305.
  ///
  /// The KEM is X-Wing (`0x647A`), as `draft-ietf-hpke-pq-05` maps it to HPKE.
  /// Keys are the prepared X-Wing keys: prepare a recipient's encapsulation
  /// key or a decapsulation key one time for repeated use.
  #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
  pub struct HpkeXWingHkdfSha256ChaCha20Poly1305;

  impl sealed::Sealed for HpkeXWingHkdfSha256ChaCha20Poly1305 {}

  impl HpkeSuite for HpkeXWingHkdfSha256ChaCha20Poly1305 {
    const KEM_ID: u16 = KEM_ID;
    const AEAD_ID: u16 = 0x0003;
    const __PSK_ID_HASH: [u8; 32] = [
      0x36, 0xb5, 0xff, 0x55, 0x87, 0xa0, 0x30, 0x0f, 0xc5, 0x55, 0x73, 0x47, 0xee, 0x59, 0x90, 0x66, 0xaf, 0x3e, 0xd8,
      0x97, 0x74, 0x5f, 0x8c, 0xa8, 0xca, 0x19, 0xb4, 0x5f, 0x13, 0xab, 0xd9, 0x77,
    ];
    const ENCAPSULATION_SIZE: usize = XWingCiphertext::LENGTH;
    const ENCAPSULATION_RANDOM_SIZE: usize = 64;

    type PublicKey = XWingPreparedEncapsulationKey;
    type PrivateKey = XWingPreparedDecapsulationKey;
    type Encapsulation = XWingCiphertext;
    type Aead = ChaCha20Poly1305;

    /// The seed is `SHAKE256.LabeledDerive(ikm, "DeriveKeyPair", "", 32)`
    /// with the KEM suite identifier, as `draft-ietf-hpke-pq-05` section 4
    /// specifies. The prepared key's `decapsulation_key` is the serialized
    /// private key.
    fn derive_keypair(ikm: &[u8]) -> XWingPreparedDecapsulationKey {
      let mut key = XWingPreparedDecapsulationKey::unexpanded();
      derive(ikm, &mut key);
      key
    }

    #[cfg(feature = "alloc")]
    fn derive_keypair_in<A: Allocator>(ikm: &[u8], alloc: A) -> Box<XWingPreparedDecapsulationKey, A> {
      let mut key = Box::new_in(XWingPreparedDecapsulationKey::unexpanded(), alloc);
      derive(ikm, &mut key);
      key
    }

    fn __encap(
      public_key: &XWingPreparedEncapsulationKey,
      fill_random: &mut impl FnMut(&mut [u8]) -> Result<(), HpkeError>,
      shared_secret: &mut [u8; SHARED_SECRET_SIZE],
      _: HpkeSuiteToken,
    ) -> Result<XWingCiphertext, HpkeError> {
      let mut fill_error = None;
      public_key
        .encapsulate_into(
          |out| {
            fill_random(out).map_err(|error| {
              fill_error = Some(error);
              XWingError::RandomGenerationFailed
            })
          },
          shared_secret,
        )
        .map_err(|error| match (error, fill_error) {
          (XWingError::RandomGenerationFailed, Some(error)) => error,
          (XWingError::RandomGenerationFailed, None) => HpkeError::RandomGenerationFailed,
          _ => HpkeError::EncapsulationFailed,
        })
    }

    fn __decap(
      encapsulation: &XWingCiphertext,
      private_key: &XWingPreparedDecapsulationKey,
      shared_secret: &mut [u8; SHARED_SECRET_SIZE],
      _: HpkeSuiteToken,
    ) -> Result<(), HpkeError> {
      private_key
        .decapsulate_into(encapsulation, shared_secret)
        .map_err(|_| HpkeError::DecapsulationFailed)
    }

    fn __aead(fill: impl FnOnce(&mut [u8]), _: HpkeSuiteToken) -> ChaCha20Poly1305 {
      ChaCha20Poly1305::new(&ChaCha20Poly1305Key::generate(|key| fill(key)))
    }
  }
}

#[cfg(all(feature = "x-wing", feature = "chacha20poly1305"))]
pub use xwing_chacha20poly1305::HpkeXWingHkdfSha256ChaCha20Poly1305;

#[cfg(test)]
mod psk_id_hash_tests {
  use super::{HpkeSuite, hpke_suite_id, labeled_extract};

  fn assert_derived<S: HpkeSuite>() {
    let suite_id = hpke_suite_id(S::KEM_ID, S::KDF_ID, S::AEAD_ID);
    assert_eq!(
      labeled_extract(&suite_id, b"", b"psk_id_hash", b"").prk(),
      &S::__PSK_ID_HASH
    );
  }

  /// The stored Base-mode `psk_id_hash` equals its RFC 9180 definition; a
  /// wrong constant would change every key schedule of the suite.
  #[test]
  fn stored_psk_id_hash_is_the_labeled_extract() {
    #[cfg(all(feature = "x25519", feature = "aes-gcm"))]
    assert_derived::<super::HpkeX25519HkdfSha256Aes128Gcm>();
    #[cfg(all(feature = "x-wing", feature = "chacha20poly1305"))]
    assert_derived::<super::HpkeXWingHkdfSha256ChaCha20Poly1305>();
  }
}

#[cfg(test)]
#[cfg(all(feature = "x25519", feature = "aes-gcm"))]
mod tests {
  use super::{HpkeError, HpkeReceiver, HpkeSender, HpkeSuite, HpkeX25519HkdfSha256Aes128Gcm as Suite};

  /// RFC 9180 section 5.2: no sequence number repeats, so the context stops
  /// at its last one. A `u64` reaches its limit before the 96-bit nonce does.
  #[test]
  fn last_sequence_number_stops_both_roles() {
    let recipient = Suite::derive_keypair(&[0x42; 32]);
    let (encapsulation, mut sender) = HpkeSender::<Suite>::setup(&recipient.public_key(), b"", |out| {
      out.fill(0x24);
      Ok(())
    })
    .expect("setup");
    let mut receiver = HpkeReceiver::<Suite>::setup(&encapsulation, &recipient, b"").expect("setup");
    sender.0.sequence = u64::MAX - 1;
    receiver.0.sequence = u64::MAX - 1;

    let mut ciphertext = [0u8; 16];
    sender
      .seal(b"", b"", &mut ciphertext)
      .expect("the last sequence number seals");
    receiver
      .open(b"", &ciphertext, &mut [])
      .expect("the last sequence number opens");
    assert_eq!(
      sender.seal(b"", b"", &mut ciphertext).err(),
      Some(HpkeError::MessageLimitReached)
    );
    assert_eq!(
      receiver.open(b"", &ciphertext, &mut []).err(),
      Some(HpkeError::MessageLimitReached)
    );
    assert_eq!((sender.sequence(), receiver.sequence()), (u64::MAX, u64::MAX));
  }
}
