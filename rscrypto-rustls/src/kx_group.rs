//! TLS 1.3 key exchange groups backed by rscrypto.
//!
//! Each group requires its corresponding primitive feature.
//! `X25519MLKEM768` requires both `x25519` and `ml-kem`.
//! Construct a group with `new` to use a caller-supplied
//! [`rustls::crypto::SecureRandom`] for all its entropy requests. The constants
//! and [`ALL_KX_GROUPS`] use OS entropy and also require `getrandom`.

/// All enabled groups with OS entropy: hybrid first, then X25519, P-256, and P-384.
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
#[cfg(feature = "p384-ecdh")]
pub use implementation::P384Group;
#[cfg(all(feature = "getrandom", feature = "p256-ecdh"))]
pub use implementation::SECP256R1;
#[cfg(all(feature = "getrandom", feature = "p384-ecdh"))]
pub use implementation::SECP384R1;
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

#[cfg(any(feature = "x25519", feature = "p256-ecdh", feature = "p384-ecdh"))]
mod implementation {
  use alloc::boxed::Box;
  #[cfg(feature = "p256-ecdh")]
  use rscrypto::{P256EphemeralSecret, P256PublicKey};
  #[cfg(feature = "p384-ecdh")]
  use rscrypto::{P384EphemeralSecret, P384PublicKey};
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
    ($group:ident, $static:ident, $exchange:ident, $secret:ident, $public:ident, $name:ident) => {
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
          let secret =
            $secret::try_generate_with(|output| self.random.fill(output)).map_err(|_| Error::FailedToGetRandomBytes)?;
          let public = secret.public_key();
          Ok(StartedKeyExchange::Single(Box::new($exchange { secret, public })))
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

      impl ActiveKeyExchange for $exchange {
        fn complete(self: Box<Self>, peer_pub_key: &[u8]) -> Result<SharedSecret, Error> {
          let peer = $public::from_sec1_bytes(peer_pub_key).map_err(|_| PeerMisbehaved::InvalidKeyShare)?;
          let secret = self.secret.diffie_hellman(&peer);
          Ok(SharedSecret::from(secret.as_bytes().as_slice()))
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
    secp256r1
  );
  #[cfg(feature = "p384-ecdh")]
  nist_curve!(
    P384Group,
    SECP384R1,
    P384Exchange,
    P384EphemeralSecret,
    P384PublicKey,
    secp384r1
  );

  #[cfg(all(feature = "ml-kem", feature = "x25519"))]
  mod hybrid {
    use super::*;
    use alloc::alloc::Global;
    use rscrypto::{
      Kem, MlKem768, MlKem768Ciphertext, MlKem768EncapsulationKey, MlKem768PreparedDecapsulationKey, MlKemError,
    };
    use rustls::crypto::kx::{CompletedKeyExchange, HybridKeyExchange};
    // RFC 10024, sections 4.1–4.3: the ML-KEM component
    // precedes X25519 in both shares and the input to the TLS key schedule.
    const HYBRID_CLIENT_SHARE_LEN: usize = MlKem768::ENCAPSULATION_KEY_SIZE + 32;
    const HYBRID_SERVER_SHARE_LEN: usize = MlKem768::CIPHERTEXT_SIZE + 32;

    /// ML-KEM-768 followed by X25519, as specified by RFC 10024.
    #[cfg(feature = "getrandom")]
    pub static X25519MLKEM768: &dyn SupportedKxGroup = &X25519MlKem768Group {
      random: crate::random::SECURE_RANDOM,
    };

    /// ML-KEM-768 and X25519 with a caller-selected entropy source.
    #[derive(Debug)]
    pub struct X25519MlKem768Group {
      random: &'static dyn SecureRandom,
    }

    impl X25519MlKem768Group {
      /// Use `random` for both components, propagating its failures without fallback.
      ///
      /// The source must provide cryptographically secure random bytes.
      #[must_use]
      pub const fn new(random: &'static dyn SecureRandom) -> Self {
        Self { random }
      }
    }

    impl SupportedKxGroup for X25519MlKem768Group {
      fn start(&self) -> Result<StartedKeyExchange, Error> {
        let classical = X25519Exchange::start(self.random)?;
        // The key share decapsulates once, so keep the prepared form keygen already computed.
        let (public_key, secret) = MlKem768::generate_prepared_keypair_in(
          |output| self.random.fill(output).map_err(|_| MlKemError::RandomGenerationFailed),
          Global,
        )
        .map_err(|_| Error::FailedToGetRandomBytes)?;
        let mut public = [0; HYBRID_CLIENT_SHARE_LEN];
        public[..MlKem768::ENCAPSULATION_KEY_SIZE].copy_from_slice(public_key.as_bytes());
        public[MlKem768::ENCAPSULATION_KEY_SIZE..].copy_from_slice(classical.pub_key());
        Ok(StartedKeyExchange::Hybrid(Box::new(HybridExchange {
          classical,
          secret,
          public,
        })))
      }

      fn start_and_complete(&self, client_share: &[u8]) -> Result<CompletedKeyExchange, Error> {
        if client_share.len() != HYBRID_CLIENT_SHARE_LEN {
          return Err(PeerMisbehaved::InvalidKeyShare.into());
        }
        let (kem_share, classical_share) = client_share.split_at(MlKem768::ENCAPSULATION_KEY_SIZE);
        let peer = MlKem768EncapsulationKey::try_from_slice(kem_share).map_err(|_| PeerMisbehaved::InvalidKeyShare)?;
        let classical = X25519Exchange::start(self.random)?;
        let classical_public = *classical.public.as_bytes();
        let classical_secret = classical.agree(classical_share)?;
        let (ciphertext, kem_secret) = MlKem768::encapsulate(&peer, |output| {
          self.random.fill(output).map_err(|_| MlKemError::RandomGenerationFailed)
        })
        .map_err(|error| match error {
          MlKemError::RandomGenerationFailed => Error::FailedToGetRandomBytes,
          _ => Error::PeerMisbehaved(PeerMisbehaved::InvalidKeyShare),
        })?;
        Ok(CompletedKeyExchange {
          group: self.name(),
          pub_key: [ciphertext.as_bytes().as_slice(), &classical_public].concat(),
          secret: SharedSecret::from([kem_secret.as_bytes().as_slice(), classical_secret.as_bytes()].concat()),
        })
      }

      fn name(&self) -> NamedGroup {
        NamedGroup::X25519MLKEM768
      }

      fn ffdhe_group(&self) -> Option<FfdheGroup<'static>> {
        None
      }
    }

    /// Report an ML-KEM decapsulation failure after the share length checks.
    ///
    /// RFC 10024, section 4.2, requires an `internal_error` alert. The QUIC adapter
    /// sends this error as CRYPTO_ERROR `internal_error`. rustls maps no error to
    /// that alert over TCP and sends no alert for `Error::Other`. A typed ciphertext
    /// cannot fail decapsulation (ML-KEM rejects implicitly), so this path is unreachable.
    pub(crate) fn decapsulation_failed(error: MlKemError) -> Error {
      Error::Other(rustls::error::OtherError::new(error))
    }

    struct HybridExchange {
      classical: X25519Exchange,
      secret: Box<MlKem768PreparedDecapsulationKey>,
      public: [u8; HYBRID_CLIENT_SHARE_LEN],
    }

    impl ActiveKeyExchange for HybridExchange {
      fn complete(self: Box<Self>, server_share: &[u8]) -> Result<SharedSecret, Error> {
        if server_share.len() != HYBRID_SERVER_SHARE_LEN {
          return Err(PeerMisbehaved::InvalidKeyShare.into());
        }
        let (kem_share, classical_share) = server_share.split_at(MlKem768::CIPHERTEXT_SIZE);
        let ciphertext = MlKem768Ciphertext::try_from_slice(kem_share).map_err(|_| PeerMisbehaved::InvalidKeyShare)?;
        let classical_secret = self.classical.agree(classical_share)?;
        let kem_secret = self.secret.decapsulate(&ciphertext).map_err(decapsulation_failed)?;
        Ok(SharedSecret::from(
          [kem_secret.as_bytes().as_slice(), classical_secret.as_bytes()].concat(),
        ))
      }

      fn pub_key(&self) -> &[u8] {
        &self.public
      }

      fn ffdhe_group(&self) -> Option<FfdheGroup<'static>> {
        None
      }

      fn group(&self) -> NamedGroup {
        NamedGroup::X25519MLKEM768
      }
    }

    impl HybridKeyExchange for HybridExchange {
      fn component(&self) -> (NamedGroup, &[u8]) {
        (NamedGroup::X25519, self.classical.pub_key())
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

    #[cfg(test)]
    #[test]
    fn decapsulation_failure_is_local_and_sends_no_tcp_alert() {
      let error = decapsulation_failed(MlKemError::InvalidCiphertext);
      assert!(matches!(error, Error::Other(_)));
      // Pinned rustls has no internal_error mapping. `Error::General` would send general_error;
      // this error sends no alert over TCP, and the QUIC adapter maps it to internal_error.
      rustls::error::AlertDescription::try_from(&error).expect_err("no TCP alert");
    }

    #[cfg(test)]
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
  #[cfg(all(feature = "getrandom", feature = "ml-kem", feature = "x25519"))]
  pub use hybrid::X25519MLKEM768;
  #[cfg(all(feature = "ml-kem", feature = "x25519"))]
  pub use hybrid::X25519MlKem768Group;
  #[cfg(all(test, feature = "quinn", feature = "ml-kem", feature = "x25519"))]
  pub(crate) use hybrid::decapsulation_failed;

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
