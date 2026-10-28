//! Transcript hashes, HMAC, and HKDF backed by rscrypto.

use alloc::boxed::Box;

use rscrypto::{Digest, Sha256, Sha384};
use rustls::crypto::hash;

/// SHA-256 for TLS transcripts.
pub static SHA256: &dyn hash::Hash = &Algorithm::Sha256;
/// SHA-384 for TLS transcripts.
pub static SHA384: &dyn hash::Hash = &Algorithm::Sha384;

#[derive(Clone, Copy)]
enum Algorithm {
  Sha256,
  Sha384,
}

impl hash::Hash for Algorithm {
  fn start(&self) -> Box<dyn hash::Context> {
    match self {
      Self::Sha256 => Box::new(Context(Sha256::new())),
      Self::Sha384 => Box::new(Context(Sha384::new())),
    }
  }

  fn hash(&self, data: &[u8]) -> hash::Output {
    match self {
      Self::Sha256 => hash::Output::new(&Sha256::digest(data)),
      Self::Sha384 => hash::Output::new(&Sha384::digest(data)),
    }
  }

  fn output_len(&self) -> usize {
    match self {
      Self::Sha256 => Sha256::OUTPUT_SIZE,
      Self::Sha384 => Sha384::OUTPUT_SIZE,
    }
  }

  fn algorithm(&self) -> rustls::crypto::HashAlgorithm {
    match self {
      Self::Sha256 => rustls::crypto::HashAlgorithm::SHA256,
      Self::Sha384 => rustls::crypto::HashAlgorithm::SHA384,
    }
  }
}

struct Context<D>(D);

impl<D: Digest + Send + Sync + 'static> hash::Context for Context<D> {
  fn fork_finish(&self) -> hash::Output {
    hash::Output::new(self.0.finalize().as_ref())
  }

  fn fork(&self) -> Box<dyn hash::Context> {
    Box::new(Self(self.0.clone()))
  }

  fn finish(self: Box<Self>) -> hash::Output {
    self.fork_finish()
  }

  fn update(&mut self, data: &[u8]) {
    self.0.update(data);
  }
}

#[cfg(any(feature = "hmac", feature = "hkdf"))]
mod hmac_adapter {
  use rscrypto::{HmacSha256, HmacSha384, Mac, SecretVec};
  use rustls::crypto::hmac;

  use super::{Algorithm, Box, hash};

  impl hmac::Hmac for Algorithm {
    fn with_key(&self, key: &[u8]) -> Box<dyn hmac::Key> {
      Box::new(Key {
        algorithm: *self,
        key: SecretVec::from_vec(key.to_vec()),
      })
    }

    fn hash_output_len(&self) -> usize {
      hash::Hash::output_len(self)
    }
  }

  struct Key {
    algorithm: Algorithm,
    key: SecretVec,
  }

  impl hmac::Key for Key {
    fn sign_concat(&self, first: &[u8], middle: &[&[u8]], last: &[u8]) -> hmac::Tag {
      match self.algorithm {
        Algorithm::Sha256 => sign::<HmacSha256>(self.key.as_bytes(), first, middle, last),
        Algorithm::Sha384 => sign::<HmacSha384>(self.key.as_bytes(), first, middle, last),
      }
    }

    fn tag_len(&self) -> usize {
      hash::Hash::output_len(&self.algorithm)
    }
  }

  pub(super) fn sign<M: Mac>(key: &[u8], first: &[u8], middle: &[&[u8]], last: &[u8]) -> hmac::Tag {
    let mut state = M::new(key);
    state.update(first);
    state.update_vectored(middle);
    state.update(last);
    hmac::Tag::new(state.finalize().as_ref())
  }
}

/// HMAC-SHA-256 with clearing key storage.
#[cfg(any(feature = "hmac", feature = "hkdf"))]
pub static HMAC_SHA256: &dyn rustls::crypto::hmac::Hmac = &Algorithm::Sha256;
/// HMAC-SHA-384 with clearing key storage.
#[cfg(any(feature = "hmac", feature = "hkdf"))]
pub static HMAC_SHA384: &dyn rustls::crypto::hmac::Hmac = &Algorithm::Sha384;

// The TLS 1.2 PRF (RFC 5246 §5): rustls's P_hash over this crate's HMAC.
// rustls derives the master secret and key block through it.
#[cfg(all(
  any(feature = "hmac", feature = "hkdf"),
  any(feature = "aes-gcm", feature = "chacha20poly1305")
))]
pub(crate) static PRF_SHA256: &dyn rustls::crypto::tls12::Prf =
  &tls12_prf::ExtendedMasterSecretOnly(rustls::crypto::tls12::PrfUsingHmac(&Algorithm::Sha256));
#[cfg(all(any(feature = "hmac", feature = "hkdf"), feature = "aes-gcm"))]
pub(crate) static PRF_SHA384: &dyn rustls::crypto::tls12::Prf =
  &tls12_prf::ExtendedMasterSecretOnly(rustls::crypto::tls12::PrfUsingHmac(&Algorithm::Sha384));

#[cfg(all(
  any(feature = "hmac", feature = "hkdf"),
  any(feature = "aes-gcm", feature = "chacha20poly1305")
))]
mod tls12_prf {
  use rustls::Error;
  use rustls::crypto::kx::ActiveKeyExchange;
  use rustls::crypto::tls12::{Prf, PrfSecret, PrfUsingHmac};
  use rustls::error::PeerIncompatible;

  use super::Box;

  /// RFC 7627's label for the session-hash master secret.
  const EXTENDED_MASTER_SECRET: &[u8] = b"extended master secret";

  /// A TLS 1.2 PRF that derives a master secret only with the extended master secret.
  ///
  /// rustls passes the legacy label `"master secret"` when a peer does not use the
  /// extension; refusing it here means no session on this provider uses the legacy
  /// derivation, whatever `require_ems` says. The refusal raises the error that
  /// `require_ems` raises, which aborts with `handshake_failure` (RFC 7627 §5.2).
  pub(crate) struct ExtendedMasterSecretOnly(pub(crate) PrfUsingHmac<'static>);

  impl Prf for ExtendedMasterSecretOnly {
    fn for_key_exchange(
      &self,
      output: &mut [u8; 48],
      kx: Box<dyn ActiveKeyExchange>,
      peer_pub_key: &[u8],
      label: &[u8],
      seed: &[u8],
    ) -> Result<(), Error> {
      if label != EXTENDED_MASTER_SECRET {
        // Dropping `kx` clears the ephemeral secret without completing the exchange.
        return Err(PeerIncompatible::ExtendedMainSecretExtensionRequired.into());
      }
      self.0.for_key_exchange(output, kx, peer_pub_key, label, seed)
    }

    fn new_secret(&self, master_secret: &[u8; 48]) -> Box<dyn PrfSecret> {
      self.0.new_secret(master_secret)
    }
  }
}

#[cfg(feature = "hkdf")]
mod hkdf_adapter {
  use rscrypto::{HkdfSha256, HkdfSha384, HmacSha256, HmacSha384, SecretBytes};
  use rustls::crypto::{hmac, tls13};

  use super::{Algorithm, Box, hash, hmac_adapter};

  impl tls13::Hkdf for Algorithm {
    fn extract_from_zero_ikm(&self, salt: Option<&[u8]>) -> Box<dyn tls13::HkdfExpander> {
      self.extract_from_secret(salt, &[0; 48][..hash::Hash::output_len(self)])
    }

    fn extract_from_secret(&self, salt: Option<&[u8]>, secret: &[u8]) -> Box<dyn tls13::HkdfExpander> {
      let salt = salt.unwrap_or(&[]);
      Box::new(match self {
        Self::Sha256 => Expander::Sha256(HkdfSha256::extract(salt, secret)),
        Self::Sha384 => Expander::Sha384(HkdfSha384::extract(salt, secret)),
      })
    }

    fn expander_for_okm(&self, okm: &tls13::OkmBlock) -> Box<dyn tls13::HkdfExpander> {
      Box::new(match self {
        Self::Sha256 => Expander::Sha256(HkdfSha256::from_prk(
          okm.as_ref().try_into().expect("SHA-256 PRK has 32 bytes"),
        )),
        Self::Sha384 => Expander::Sha384(HkdfSha384::from_prk(
          okm.as_ref().try_into().expect("SHA-384 PRK has 48 bytes"),
        )),
      })
    }

    fn hmac_sign(&self, key: &tls13::OkmBlock, message: &[u8]) -> hmac::Tag {
      match self {
        Self::Sha256 => hmac_adapter::sign::<HmacSha256>(key.as_ref(), message, &[], &[]),
        Self::Sha384 => hmac_adapter::sign::<HmacSha384>(key.as_ref(), message, &[], &[]),
      }
    }
  }

  enum Expander {
    Sha256(HkdfSha256),
    Sha384(HkdfSha384),
  }

  impl tls13::HkdfExpander for Expander {
    fn expand_slice(&self, info: &[&[u8]], output: &mut [u8]) -> Result<(), tls13::OutputLengthError> {
      match self {
        Self::Sha256(state) => state.expand_vectored(info, output),
        Self::Sha384(state) => state.expand_vectored(info, output),
      }
      .map_err(|_| tls13::OutputLengthError)
    }

    fn expand_block(&self, info: &[&[u8]]) -> tls13::OkmBlock {
      // The scratch owner clears the copy after rustls takes its clearing OkmBlock.
      match self {
        Self::Sha256(state) => tls13::OkmBlock::new(
          SecretBytes::<32>::try_fill_with(|output| state.expand_vectored(info, output))
            .expect("one SHA-256 block fits the HKDF bound")
            .as_ref(),
        ),
        Self::Sha384(state) => tls13::OkmBlock::new(
          SecretBytes::<48>::try_fill_with(|output| state.expand_vectored(info, output))
            .expect("one SHA-384 block fits the HKDF bound")
            .as_ref(),
        ),
      }
    }

    fn hash_len(&self) -> usize {
      match self {
        Self::Sha256(_) => HkdfSha256::OUTPUT_SIZE,
        Self::Sha384(_) => HkdfSha384::OUTPUT_SIZE,
      }
    }
  }
}

/// HKDF-SHA-256; both extraction and expansion run in rscrypto.
#[cfg(feature = "hkdf")]
pub static HKDF_SHA256: &dyn rustls::crypto::tls13::Hkdf = &Algorithm::Sha256;
/// HKDF-SHA-384; both extraction and expansion run in rscrypto.
#[cfg(feature = "hkdf")]
pub static HKDF_SHA384: &dyn rustls::crypto::tls13::Hkdf = &Algorithm::Sha384;
