//! TLS 1.3 key exchange groups backed by rscrypto.
//!
//! Each group requires its corresponding primitive features: the hybrids
//! `X25519MLKEM768`, `SECP256R1MLKEM768`, and `SECP384R1MLKEM1024` need their
//! classical curve feature and `ml-kem`; the pure ML-KEM groups need `ml-kem`.
//! Construct a group with `new` to use a caller-supplied
//! [`rustls::crypto::SecureRandom`] for all its entropy requests. The constants
//! and [`ALL_KX_GROUPS`] use OS entropy and also require `getrandom`.
//!
//! [`ALL_KX_GROUPS`] offers `X25519MLKEM768` first, then the classical curves.
//! The other hybrids and the pure ML-KEM groups are opt-in: add them to a
//! provider's `kx_groups` explicitly. Hybrid share layouts follow RFC 10024;
//! pure ML-KEM follows draft-ietf-tls-mlkem-11.

/// All enabled groups with OS entropy: hybrid first, then X25519, P-256, and P-384.
///
/// `SECP256R1MLKEM768`, `SECP384R1MLKEM1024`, `MLKEM768`, and `MLKEM1024` are
/// not included; add them explicitly.
#[cfg(feature = "getrandom")]
pub static ALL_KX_GROUPS: &[&dyn rustls::crypto::kx::SupportedKxGroup] = &[
  #[cfg(all(feature = "x25519", feature = "ml-kem"))]
  X25519MLKEM768,
  #[cfg(feature = "x25519")]
  X25519,
  #[cfg(feature = "p256-ecdh")]
  SECP256R1,
  #[cfg(feature = "p384-ecdh")]
  SECP384R1,
];

#[cfg(feature = "p256-ecdh")]
pub use implementation::P256Group;
#[cfg(all(feature = "p256-ecdh", feature = "ml-kem"))]
pub use implementation::P256MlKem768Group;
#[cfg(feature = "p384-ecdh")]
pub use implementation::P384Group;
#[cfg(all(feature = "p384-ecdh", feature = "ml-kem"))]
pub use implementation::P384MlKem1024Group;
#[cfg(all(feature = "getrandom", feature = "p256-ecdh"))]
pub use implementation::SECP256R1;
#[cfg(all(feature = "getrandom", feature = "p256-ecdh", feature = "ml-kem"))]
pub use implementation::SECP256R1MLKEM768;
#[cfg(all(feature = "getrandom", feature = "p384-ecdh"))]
pub use implementation::SECP384R1;
#[cfg(all(feature = "getrandom", feature = "p384-ecdh", feature = "ml-kem"))]
pub use implementation::SECP384R1MLKEM1024;
#[cfg(all(feature = "getrandom", feature = "x25519"))]
pub use implementation::X25519;
#[cfg(feature = "x25519")]
pub use implementation::X25519Group;
#[cfg(all(feature = "getrandom", feature = "x25519", feature = "ml-kem"))]
pub use implementation::X25519MLKEM768;
#[cfg(all(feature = "x25519", feature = "ml-kem"))]
pub use implementation::X25519MlKem768Group;
#[cfg(all(test, feature = "quinn", feature = "x25519", feature = "ml-kem"))]
pub(crate) use implementation::decapsulation_failed;
#[cfg(all(feature = "getrandom", feature = "ml-kem"))]
pub use implementation::{MLKEM768, MLKEM1024};
#[cfg(feature = "ml-kem")]
pub use implementation::{MlKem768Group, MlKem1024Group};

#[cfg(any(feature = "x25519", feature = "p256-ecdh", feature = "p384-ecdh", feature = "ml-kem"))]
mod implementation {
  use alloc::boxed::Box;
  #[cfg(feature = "p256-ecdh")]
  use rscrypto::{P256EphemeralSecret, P256PublicKey, P256SharedSecret};
  #[cfg(feature = "p384-ecdh")]
  use rscrypto::{P384EphemeralSecret, P384PublicKey, P384SharedSecret};
  #[cfg(feature = "x25519")]
  use rscrypto::{X25519PublicKey, X25519SecretKey, X25519SharedSecret};
  use rustls::Error;
  use rustls::crypto::SecureRandom;
  use rustls::crypto::kx::NamedGroup;
  use rustls::crypto::kx::ffdhe::FfdheGroup;
  use rustls::crypto::kx::{ActiveKeyExchange, SharedSecret, StartedKeyExchange, SupportedKxGroup};
  use rustls::error::PeerMisbehaved;
  #[cfg(all(feature = "getrandom", feature = "x25519"))]
  /// X25519 key agreement for TLS 1.3.
  pub static X25519: &dyn SupportedKxGroup = &X25519Group {
    random: crate::random::SECURE_RANDOM,
  };

  #[cfg(feature = "x25519")]
  /// X25519 with a caller-selected entropy source.
  #[derive(Debug)]
  pub struct X25519Group {
    random: &'static dyn SecureRandom,
  }

  #[cfg(feature = "x25519")]
  impl X25519Group {
    /// Use `random` for key generation, propagating its failures without fallback.
    ///
    /// The source must provide cryptographically secure random bytes.
    #[must_use]
    pub const fn new(random: &'static dyn SecureRandom) -> Self {
      Self { random }
    }
  }

  #[cfg(feature = "x25519")]
  impl SupportedKxGroup for X25519Group {
    fn start(&self) -> Result<StartedKeyExchange, Error> {
      Ok(StartedKeyExchange::Single(Box::new(X25519Exchange::start(
        self.random,
      )?)))
    }

    fn name(&self) -> NamedGroup {
      NamedGroup::X25519
    }
    fn ffdhe_group(&self) -> Option<FfdheGroup<'static>> {
      None
    }
  }

  #[cfg(feature = "x25519")]
  struct X25519Exchange {
    secret: X25519SecretKey,
    public: X25519PublicKey,
  }

  #[cfg(feature = "x25519")]
  impl X25519Exchange {
    fn start(random: &dyn SecureRandom) -> Result<Self, Error> {
      let secret = X25519SecretKey::try_generate_with(|output| random.fill(output))?;
      let public = secret.public_key();
      Ok(Self { secret, public })
    }

    fn agree(self, peer_pub_key: &[u8]) -> Result<X25519SharedSecret, Error> {
      let peer: [u8; 32] = peer_pub_key.try_into().map_err(|_| PeerMisbehaved::InvalidKeyShare)?;
      self
        .secret
        .diffie_hellman(&X25519PublicKey::from_bytes(peer))
        .map_err(|_| PeerMisbehaved::InvalidKeyShare.into())
    }
  }

  #[cfg(feature = "x25519")]
  impl ActiveKeyExchange for X25519Exchange {
    fn complete(self: Box<Self>, peer_pub_key: &[u8]) -> Result<SharedSecret, Error> {
      Ok(SharedSecret::from(self.agree(peer_pub_key)?.as_bytes().as_slice()))
    }

    fn pub_key(&self) -> &[u8] {
      self.public.as_bytes()
    }

    fn ffdhe_group(&self) -> Option<FfdheGroup<'static>> {
      None
    }

    fn group(&self) -> NamedGroup {
      NamedGroup::X25519
    }
  }

  #[cfg(any(feature = "p256-ecdh", feature = "p384-ecdh"))]
  macro_rules! nist_curve {
    ($group:ident, $static:ident, $exchange:ident, $secret:ident, $public:ident, $shared:ident, $name:ident) => {
      #[doc = concat!(stringify!($name), " key agreement for TLS 1.3.")]
      #[cfg(feature = "getrandom")]
      pub static $static: &dyn SupportedKxGroup = &$group {
        random: crate::random::SECURE_RANDOM,
      };

      #[doc = concat!(stringify!($name), " with a caller-selected entropy source.")]
      #[derive(Debug)]
      pub struct $group {
        random: &'static dyn SecureRandom,
      }

      impl $group {
        /// Use `random` for key generation, propagating its failures without fallback.
        ///
        /// The source must provide cryptographically secure random bytes.
        #[must_use]
        pub const fn new(random: &'static dyn SecureRandom) -> Self {
          Self { random }
        }
      }

      impl SupportedKxGroup for $group {
        fn start(&self) -> Result<StartedKeyExchange, Error> {
          Ok(StartedKeyExchange::Single(Box::new($exchange::start(
            self.random,
          )?)))
        }

        fn name(&self) -> NamedGroup {
          NamedGroup::$name
        }

        fn ffdhe_group(&self) -> Option<FfdheGroup<'static>> {
          None
        }
      }

      struct $exchange {
        secret: $secret,
        public: $public,
      }

      impl $exchange {
        fn start(random: &dyn SecureRandom) -> Result<Self, Error> {
          let secret =
            $secret::try_generate_with(|output| random.fill(output)).map_err(|_| Error::FailedToGetRandomBytes)?;
          let public = secret.public_key();
          Ok(Self { secret, public })
        }

        /// Validate the peer's canonical uncompressed point, then agree.
        fn agree(self, peer_pub_key: &[u8]) -> Result<$shared, Error> {
          let peer = $public::from_sec1_bytes(peer_pub_key).map_err(|_| PeerMisbehaved::InvalidKeyShare)?;
          Ok(self.secret.diffie_hellman(&peer))
        }
      }

      impl ActiveKeyExchange for $exchange {
        fn complete(self: Box<Self>, peer_pub_key: &[u8]) -> Result<SharedSecret, Error> {
          Ok(SharedSecret::from(
            self.agree(peer_pub_key)?.as_bytes().as_slice(),
          ))
        }

        fn pub_key(&self) -> &[u8] {
          self.public.as_sec1_bytes()
        }

        fn ffdhe_group(&self) -> Option<FfdheGroup<'static>> {
          None
        }

        fn group(&self) -> NamedGroup {
          NamedGroup::$name
        }
      }
    };
  }

  #[cfg(feature = "p256-ecdh")]
  nist_curve!(
    P256Group,
    SECP256R1,
    P256Exchange,
    P256EphemeralSecret,
    P256PublicKey,
    P256SharedSecret,
    secp256r1
  );
  #[cfg(feature = "p384-ecdh")]
  nist_curve!(
    P384Group,
    SECP384R1,
    P384Exchange,
    P384EphemeralSecret,
    P384PublicKey,
    P384SharedSecret,
    secp384r1
  );

  /// Report an ML-KEM decapsulation failure after the share length checks.
  ///
  /// RFC 10024, section 4.2, and draft-ietf-tls-mlkem-11 require an
  /// `internal_error` alert. rustls sends that alert over TCP, and the QUIC
  /// adapter sends the corresponding CRYPTO_ERROR. A typed ciphertext cannot
  /// fail decapsulation (ML-KEM rejects implicitly), so this path is unreachable.
  #[cfg(feature = "ml-kem")]
  pub(crate) fn decapsulation_failed(error: rscrypto::MlKemError) -> Error {
    Error::Other(rustls::error::OtherError::new(error))
  }

  /// Map encapsulation failure: entropy failure is local; anything else is
  /// an invalid peer encapsulation key.
  #[cfg(feature = "ml-kem")]
  fn encapsulation_failed(error: rscrypto::MlKemError) -> Error {
    match error {
      rscrypto::MlKemError::RandomGenerationFailed => Error::FailedToGetRandomBytes,
      _ => Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare),
    }
  }

  /// One RFC 10024 hybrid: an ML-KEM profile and a classical exchange.
  ///
  /// `post_quantum_first` selects the component order in both key shares and
  /// in the shared secret: true for X25519MLKEM768, false for the NIST curves.
  /// Each received share has one exact length. The server validates the
  /// encapsulation key (FIPS 203 section 7.2) before using entropy, and both
  /// roles validate the classical share; either failure is `InvalidKeyShare`,
  /// which rustls reports as `illegal_parameter`.
  #[cfg(all(
    feature = "ml-kem",
    any(feature = "x25519", feature = "p256-ecdh", feature = "p384-ecdh")
  ))]
  macro_rules! hybrid_group {
    (
      module: $module:ident,
      static: $static:ident,
      group: $group:ident,
      name: $name:ident,
      doc: $doc:literal,
      classical: $classical:ident / $classical_name:ident / $classical_len:expr,
      kem: $kem:ident / $encapsulation_key:ident / $ciphertext:ident / $decapsulation_key:ident,
      post_quantum_first: $post_quantum_first:expr $(,)?
    ) => {
      mod $module {
        use super::*;
        use alloc::alloc::Global;
        use alloc::vec::Vec;
        use rscrypto::{Kem, MlKemError, $ciphertext, $decapsulation_key, $encapsulation_key, $kem};
        use rustls::crypto::kx::{CompletedKeyExchange, HybridKeyExchange};

        const POST_QUANTUM_FIRST: bool = $post_quantum_first;
        const CLASSICAL_LEN: usize = $classical_len;
        const CLIENT_SHARE_LEN: usize = $kem::ENCAPSULATION_KEY_SIZE + CLASSICAL_LEN;
        const SERVER_SHARE_LEN: usize = $kem::CIPHERTEXT_SIZE + CLASSICAL_LEN;

        /// Return `(post_quantum, classical)` from a share of the exact length.
        fn split(share: &[u8], length: usize) -> Option<(&[u8], &[u8])> {
          if share.len() != length {
            return None;
          }
          Some(if POST_QUANTUM_FIRST {
            share.split_at(length.strict_sub(CLASSICAL_LEN))
          } else {
            let (classical, post_quantum) = share.split_at(CLASSICAL_LEN);
            (post_quantum, classical)
          })
        }

        fn concat(post_quantum: &[u8], classical: &[u8]) -> Vec<u8> {
          if POST_QUANTUM_FIRST {
            [post_quantum, classical].concat()
          } else {
            [classical, post_quantum].concat()
          }
        }

        #[doc = concat!($doc, ", as specified by RFC 10024.")]
        #[cfg(feature = "getrandom")]
        pub static $static: &dyn SupportedKxGroup = &$group {
          random: crate::random::SECURE_RANDOM,
        };

        #[doc = concat!($doc, " with a caller-selected entropy source.")]
        #[derive(Debug)]
        pub struct $group {
          random: &'static dyn SecureRandom,
        }

        impl $group {
          /// Use `random` for both components, propagating its failures without fallback.
          ///
          /// The source must provide cryptographically secure random bytes.
          #[must_use]
          pub const fn new(random: &'static dyn SecureRandom) -> Self {
            Self { random }
          }
        }

        impl SupportedKxGroup for $group {
          fn start(&self) -> Result<StartedKeyExchange, Error> {
            let classical = $classical::start(self.random)?;
            // The key share decapsulates once, so keep the prepared form keygen already computed.
            let (public_key, secret) = $kem::generate_prepared_keypair_in(
              |output| {
                self
                  .random
                  .fill(output)
                  .map_err(|_| MlKemError::RandomGenerationFailed)
              },
              Global,
            )
            .map_err(|_| Error::FailedToGetRandomBytes)?;
            let mut public = [0; CLIENT_SHARE_LEN];
            let (post_quantum_part, classical_part) = if POST_QUANTUM_FIRST {
              public.split_at_mut($kem::ENCAPSULATION_KEY_SIZE)
            } else {
              let (classical_part, post_quantum_part) = public.split_at_mut(CLASSICAL_LEN);
              (post_quantum_part, classical_part)
            };
            post_quantum_part.copy_from_slice(public_key.as_bytes());
            classical_part.copy_from_slice(classical.pub_key());
            Ok(StartedKeyExchange::Hybrid(Box::new(Exchange {
              classical,
              secret,
              public,
            })))
          }

          fn start_and_complete(&self, client_share: &[u8]) -> Result<CompletedKeyExchange, Error> {
            let (kem_share, classical_share) =
              split(client_share, CLIENT_SHARE_LEN).ok_or(PeerMisbehaved::InvalidKeyShare)?;
            let peer = $encapsulation_key::try_from_slice(kem_share).map_err(|_| PeerMisbehaved::InvalidKeyShare)?;
            let classical = $classical::start(self.random)?;
            let mut classical_public = [0; CLASSICAL_LEN];
            classical_public.copy_from_slice(classical.pub_key());
            let classical_secret = classical.agree(classical_share)?;
            let (ciphertext, kem_secret) = $kem::encapsulate(&peer, |output| {
              self
                .random
                .fill(output)
                .map_err(|_| MlKemError::RandomGenerationFailed)
            })
            .map_err(encapsulation_failed)?;
            Ok(CompletedKeyExchange {
              group: self.name(),
              pub_key: concat(ciphertext.as_bytes(), &classical_public),
              secret: SharedSecret::from(concat(kem_secret.as_bytes(), classical_secret.as_bytes())),
            })
          }

          fn name(&self) -> NamedGroup {
            NamedGroup::$name
          }

          fn ffdhe_group(&self) -> Option<FfdheGroup<'static>> {
            None
          }
        }

        struct Exchange {
          classical: $classical,
          secret: Box<$decapsulation_key>,
          public: [u8; CLIENT_SHARE_LEN],
        }

        impl ActiveKeyExchange for Exchange {
          fn complete(self: Box<Self>, server_share: &[u8]) -> Result<SharedSecret, Error> {
            let (kem_share, classical_share) =
              split(server_share, SERVER_SHARE_LEN).ok_or(PeerMisbehaved::InvalidKeyShare)?;
            let ciphertext = $ciphertext::try_from_slice(kem_share).map_err(|_| PeerMisbehaved::InvalidKeyShare)?;
            let classical_secret = self.classical.agree(classical_share)?;
            let kem_secret = self
              .secret
              .decapsulate(&ciphertext)
              .map_err(decapsulation_failed)?;
            Ok(SharedSecret::from(concat(
              kem_secret.as_bytes(),
              classical_secret.as_bytes(),
            )))
          }

          fn pub_key(&self) -> &[u8] {
            &self.public
          }

          fn ffdhe_group(&self) -> Option<FfdheGroup<'static>> {
            None
          }

          fn group(&self) -> NamedGroup {
            NamedGroup::$name
          }
        }

        impl HybridKeyExchange for Exchange {
          fn component(&self) -> (NamedGroup, &[u8]) {
            (NamedGroup::$classical_name, self.classical.pub_key())
          }

          fn complete_component(self: Box<Self>, peer_pub_key: &[u8]) -> Result<SharedSecret, Error> {
            Ok(SharedSecret::from(
              self.classical.agree(peer_pub_key)?.as_bytes().as_slice(),
            ))
          }

          fn into_key_exchange(self: Box<Self>) -> Box<dyn ActiveKeyExchange> {
            self
          }

          fn as_key_exchange(&self) -> &(dyn ActiveKeyExchange + 'static) {
            self
          }
        }
      }
    };
  }

  /// One pure ML-KEM group (draft-ietf-tls-mlkem-11): the client share is the
  /// encapsulation key, the server share is the ciphertext, and the shared
  /// secret is the 32-byte ML-KEM secret.
  #[cfg(feature = "ml-kem")]
  macro_rules! mlkem_group {
    (
      module: $module:ident,
      static: $static:ident,
      group: $group:ident,
      name: $name:ident,
      doc: $doc:literal,
      kem: $kem:ident / $encapsulation_key:ident / $ciphertext:ident / $decapsulation_key:ident $(,)?
    ) => {
      mod $module {
        use super::*;
        use alloc::alloc::Global;
        use rscrypto::{Kem, MlKemError, $ciphertext, $decapsulation_key, $encapsulation_key, $kem};
        use rustls::crypto::kx::CompletedKeyExchange;

        #[doc = concat!($doc, ", as specified by draft-ietf-tls-mlkem-11. Opt-in only.")]
        #[cfg(feature = "getrandom")]
        pub static $static: &dyn SupportedKxGroup = &$group {
          random: crate::random::SECURE_RANDOM,
        };

        #[doc = concat!($doc, " with a caller-selected entropy source.")]
        #[derive(Debug)]
        pub struct $group {
          random: &'static dyn SecureRandom,
        }

        impl $group {
          /// Use `random` for key generation and encapsulation, propagating its failures without fallback.
          ///
          /// The source must provide cryptographically secure random bytes.
          #[must_use]
          pub const fn new(random: &'static dyn SecureRandom) -> Self {
            Self { random }
          }
        }

        impl SupportedKxGroup for $group {
          fn start(&self) -> Result<StartedKeyExchange, Error> {
            let (public, secret) = $kem::generate_prepared_keypair_in(
              |output| {
                self
                  .random
                  .fill(output)
                  .map_err(|_| MlKemError::RandomGenerationFailed)
              },
              Global,
            )
            .map_err(|_| Error::FailedToGetRandomBytes)?;
            Ok(StartedKeyExchange::Single(Box::new(Exchange { secret, public })))
          }

          fn start_and_complete(&self, client_share: &[u8]) -> Result<CompletedKeyExchange, Error> {
            let peer = $encapsulation_key::try_from_slice(client_share).map_err(|_| PeerMisbehaved::InvalidKeyShare)?;
            let (ciphertext, secret) = $kem::encapsulate(&peer, |output| {
              self
                .random
                .fill(output)
                .map_err(|_| MlKemError::RandomGenerationFailed)
            })
            .map_err(encapsulation_failed)?;
            Ok(CompletedKeyExchange {
              group: self.name(),
              pub_key: ciphertext.as_bytes().to_vec(),
              secret: SharedSecret::from(secret.as_bytes().as_slice()),
            })
          }

          fn name(&self) -> NamedGroup {
            NamedGroup::$name
          }

          fn ffdhe_group(&self) -> Option<FfdheGroup<'static>> {
            None
          }
        }

        struct Exchange {
          secret: Box<$decapsulation_key>,
          public: $encapsulation_key,
        }

        impl ActiveKeyExchange for Exchange {
          fn complete(self: Box<Self>, server_share: &[u8]) -> Result<SharedSecret, Error> {
            let ciphertext = $ciphertext::try_from_slice(server_share).map_err(|_| PeerMisbehaved::InvalidKeyShare)?;
            let secret = self
              .secret
              .decapsulate(&ciphertext)
              .map_err(decapsulation_failed)?;
            Ok(SharedSecret::from(secret.as_bytes().as_slice()))
          }

          fn pub_key(&self) -> &[u8] {
            self.public.as_bytes()
          }

          fn ffdhe_group(&self) -> Option<FfdheGroup<'static>> {
            None
          }

          fn group(&self) -> NamedGroup {
            NamedGroup::$name
          }
        }
      }
    };
  }

  #[cfg(all(feature = "ml-kem", feature = "x25519"))]
  hybrid_group!(
    module: x25519_mlkem768,
    static: X25519MLKEM768,
    group: X25519MlKem768Group,
    name: X25519MLKEM768,
    doc: "ML-KEM-768 followed by X25519",
    classical: X25519Exchange / X25519 / 32,
    kem: MlKem768 / MlKem768EncapsulationKey / MlKem768Ciphertext / MlKem768PreparedDecapsulationKey,
    post_quantum_first: true,
  );
  #[cfg(all(feature = "getrandom", feature = "ml-kem", feature = "x25519"))]
  pub use x25519_mlkem768::X25519MLKEM768;
  #[cfg(all(feature = "ml-kem", feature = "x25519"))]
  pub use x25519_mlkem768::X25519MlKem768Group;

  #[cfg(all(feature = "ml-kem", feature = "p256-ecdh"))]
  hybrid_group!(
    module: secp256r1_mlkem768,
    static: SECP256R1MLKEM768,
    group: P256MlKem768Group,
    name: secp256r1MLKEM768,
    doc: "P-256 followed by ML-KEM-768",
    classical: P256Exchange / secp256r1 / P256PublicKey::SEC1_LENGTH,
    kem: MlKem768 / MlKem768EncapsulationKey / MlKem768Ciphertext / MlKem768PreparedDecapsulationKey,
    post_quantum_first: false,
  );
  #[cfg(all(feature = "ml-kem", feature = "p256-ecdh"))]
  pub use secp256r1_mlkem768::P256MlKem768Group;
  #[cfg(all(feature = "getrandom", feature = "ml-kem", feature = "p256-ecdh"))]
  pub use secp256r1_mlkem768::SECP256R1MLKEM768;

  #[cfg(all(feature = "ml-kem", feature = "p384-ecdh"))]
  hybrid_group!(
    module: secp384r1_mlkem1024,
    static: SECP384R1MLKEM1024,
    group: P384MlKem1024Group,
    name: secp384r1MLKEM1024,
    doc: "P-384 followed by ML-KEM-1024",
    classical: P384Exchange / secp384r1 / P384PublicKey::SEC1_LENGTH,
    kem: MlKem1024 / MlKem1024EncapsulationKey / MlKem1024Ciphertext / MlKem1024PreparedDecapsulationKey,
    post_quantum_first: false,
  );
  #[cfg(all(feature = "ml-kem", feature = "p384-ecdh"))]
  pub use secp384r1_mlkem1024::P384MlKem1024Group;
  #[cfg(all(feature = "getrandom", feature = "ml-kem", feature = "p384-ecdh"))]
  pub use secp384r1_mlkem1024::SECP384R1MLKEM1024;

  #[cfg(feature = "ml-kem")]
  mlkem_group!(
    module: mlkem768,
    static: MLKEM768,
    group: MlKem768Group,
    name: MLKEM768,
    doc: "Pure ML-KEM-768",
    kem: MlKem768 / MlKem768EncapsulationKey / MlKem768Ciphertext / MlKem768PreparedDecapsulationKey,
  );
  #[cfg(all(feature = "getrandom", feature = "ml-kem"))]
  pub use mlkem768::MLKEM768;
  #[cfg(feature = "ml-kem")]
  pub use mlkem768::MlKem768Group;

  #[cfg(feature = "ml-kem")]
  mlkem_group!(
    module: mlkem1024,
    static: MLKEM1024,
    group: MlKem1024Group,
    name: MLKEM1024,
    doc: "Pure ML-KEM-1024",
    kem: MlKem1024 / MlKem1024EncapsulationKey / MlKem1024Ciphertext / MlKem1024PreparedDecapsulationKey,
  );
  #[cfg(all(feature = "getrandom", feature = "ml-kem"))]
  pub use mlkem1024::MLKEM1024;
  #[cfg(feature = "ml-kem")]
  pub use mlkem1024::MlKem1024Group;

  #[cfg(all(test, feature = "ml-kem", feature = "x25519"))]
  mod hybrid_tests {
    use super::*;
    use rscrypto::MlKemError;

    #[test]
    fn decapsulation_failure_is_local_and_maps_to_internal_error() {
      let error = decapsulation_failed(MlKemError::InvalidCiphertext);
      assert!(matches!(error, Error::Other(_)));
      assert_eq!(
        rustls::error::AlertDescription::try_from(&error),
        Ok(rustls::error::AlertDescription::InternalError)
      );
    }

    #[cfg(all(
      feature = "hkdf",
      any(feature = "aes-gcm", feature = "chacha20poly1305"),
      feature = "ecdsa-p256"
    ))]
    #[test]
    fn hybrid_decapsulation_failure_sends_tcp_internal_error() {
      use crate::tests::tls;
      use alloc::sync::Arc;

      #[derive(Debug)]
      struct FailingGroup;
      struct FailingExchange(Box<dyn ActiveKeyExchange>);

      impl SupportedKxGroup for FailingGroup {
        fn start(&self) -> Result<StartedKeyExchange, Error> {
          let exchange = X25519MlKem768Group::new(&crate::tests::RANDOM).start()?.into_single();
          Ok(StartedKeyExchange::Single(Box::new(FailingExchange(exchange))))
        }

        fn name(&self) -> NamedGroup {
          NamedGroup::X25519MLKEM768
        }

        fn ffdhe_group(&self) -> Option<FfdheGroup<'static>> {
          None
        }
      }

      impl ActiveKeyExchange for FailingExchange {
        fn complete(self: Box<Self>, peer_pub_key: &[u8]) -> Result<SharedSecret, Error> {
          // Complete the real exchange before injecting the otherwise unreachable
          // local failure at rustls's key-exchange boundary.
          self.0.complete(peer_pub_key)?;
          Err(decapsulation_failed(MlKemError::InvalidCiphertext))
        }

        fn pub_key(&self) -> &[u8] {
          self.0.pub_key()
        }

        fn group(&self) -> NamedGroup {
          self.0.group()
        }

        fn ffdhe_group(&self) -> Option<FfdheGroup<'static>> {
          None
        }
      }

      let mut client_provider = tls::provider();
      client_provider.kx_groups = vec![&FailingGroup as &dyn SupportedKxGroup].into();
      let (client, server) = tls::configs(client_provider, tls::provider(), tls::Credential::P256, false);
      let (mut client, mut server) = tls::connections_from_configs(Arc::new(client), Arc::new(server));
      tls::transfer(&mut client, &mut server, |_, _| Ok(())).expect("ClientHello");
      assert!(matches!(
        tls::transfer(&mut server, &mut client, |_, _| Ok(())),
        Err(Error::Other(_))
      ));
      // The failure precedes traffic keys, so the alert is plaintext:
      // fatal (2), internal_error (80).
      assert!(client.output.ends_with(&[2, 80]));
      assert!(matches!(
        tls::transfer(&mut client, &mut server, |_, _| Ok(())),
        Err(Error::AlertReceived(rustls::error::AlertDescription::InternalError))
      ));
    }

    #[test]
    fn hybrid_entropy_failure_and_key_validation_precedence() {
      use core::sync::atomic::{AtomicUsize, Ordering};
      use rustls::crypto::GetRandomFailed;

      #[derive(Debug)]
      struct Failure {
        calls: AtomicUsize,
        fail_at: usize,
      }
      impl SecureRandom for Failure {
        fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
          output.fill(0x42);
          if self.calls.fetch_add(1, Ordering::Relaxed) == self.fail_at {
            Err(GetRandomFailed)
          } else {
            Ok(())
          }
        }
      }
      static FAILURES: [Failure; 4] = [
        Failure {
          calls: AtomicUsize::new(0),
          fail_at: 0,
        },
        Failure {
          calls: AtomicUsize::new(0),
          fail_at: 1,
        },
        Failure {
          calls: AtomicUsize::new(0),
          fail_at: 0,
        },
        Failure {
          calls: AtomicUsize::new(0),
          fail_at: 1,
        },
      ];
      for random in &FAILURES[..2] {
        assert!(matches!(
          X25519MlKem768Group::new(random).start(),
          Err(Error::FailedToGetRandomBytes)
        ));
        assert_eq!(random.calls.load(Ordering::Relaxed), random.fail_at.strict_add(1));
      }
      let client = X25519MlKem768Group::new(&crate::tests::RANDOM)
        .start()
        .expect("key exchange");
      for random in &FAILURES[2..] {
        let group = X25519MlKem768Group::new(random);
        let mut malformed = client.pub_key().to_vec();
        malformed[..1184].fill(0xff);
        assert!(matches!(
          group.start_and_complete(&malformed),
          Err(Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare))
        ));
        assert_eq!(random.calls.load(Ordering::Relaxed), 0);
        assert!(matches!(
          group.start_and_complete(client.pub_key()),
          Err(Error::FailedToGetRandomBytes)
        ));
        assert_eq!(random.calls.load(Ordering::Relaxed), random.fail_at.strict_add(1));
      }
    }
  }

  #[cfg(all(test, feature = "x25519", feature = "p256-ecdh", feature = "p384-ecdh"))]
  #[test]
  fn classical_entropy_failure_never_returns_a_share() {
    #[derive(Debug)]
    struct Unavailable;
    impl SecureRandom for Unavailable {
      fn fill(&self, output: &mut [u8]) -> Result<(), rustls::crypto::GetRandomFailed> {
        output.fill(0x42);
        Err(rustls::crypto::GetRandomFailed)
      }
    }
    for group in [
      &X25519Group::new(&Unavailable) as &dyn SupportedKxGroup,
      &P256Group::new(&Unavailable),
      &P384Group::new(&Unavailable),
    ] {
      assert!(matches!(group.start(), Err(Error::FailedToGetRandomBytes)));
    }
  }
}

#[cfg(all(
  test,
  feature = "x25519",
  feature = "p256-ecdh",
  feature = "p384-ecdh",
  feature = "ml-kem"
))]
#[path = "tests/key_exchange.rs"]
mod tests;
