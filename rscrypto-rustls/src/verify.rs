//! Certificate and TLS signature verification through rscrypto.

/// Enabled certificate algorithms and TLS signature-scheme mappings.
#[cfg(any(
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "rsa",
  feature = "ml-dsa"
))]
pub const SUPPORTED_ALGORITHMS: rustls::crypto::WebPkiSupportedAlgorithms = implementation::ALGORITHMS;

/// Enabled certificate algorithms, including encodings used only in certificate chains.
///
/// Use this list with [`rustls::crypto::WebPkiSupportedAlgorithms::new`] when selecting a subset
/// of the TLS signature schemes in [`SUPPORTED_ALGORITHMS`].
#[cfg(any(
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "rsa",
  feature = "ml-dsa"
))]
pub const ALL_ALGORITHMS: &[&dyn rustls::pki_types::SignatureVerificationAlgorithm] = implementation::ALL;
#[cfg(any(
  feature = "ecdsa-p256",
  feature = "ecdsa-p384",
  feature = "ed25519",
  feature = "rsa",
  feature = "ml-dsa"
))]
mod implementation {

  #[cfg(feature = "ecdsa-p256")]
  use rscrypto::{EcdsaP256PublicKey, EcdsaP256Signature};
  #[cfg(feature = "ecdsa-p384")]
  use rscrypto::{EcdsaP384PublicKey, EcdsaP384Signature};
  #[cfg(feature = "ed25519")]
  use rscrypto::{Ed25519PublicKey, Ed25519Signature};
  #[cfg(feature = "rsa")]
  use rscrypto::{RsaPublicKey, RsaPublicKeyPolicy, RsaSignatureProfile};
  use rustls::crypto::SignatureScheme;
  use rustls::crypto::WebPkiSupportedAlgorithms;
  use rustls::pki_types::{AlgorithmIdentifier, InvalidSignature, SignatureVerificationAlgorithm, alg_id};
  #[derive(Clone, Copy, Debug)]
  enum Operation {
    #[cfg(feature = "ecdsa-p256")]
    P256Sha256,
    #[cfg(feature = "ecdsa-p256")]
    P256Sha384,
    #[cfg(feature = "ecdsa-p384")]
    P384Sha256,
    #[cfg(feature = "ecdsa-p384")]
    P384Sha384,
    #[cfg(feature = "ed25519")]
    Ed25519,
    #[cfg(feature = "rsa")]
    Rsa(SignatureScheme),
    #[cfg(feature = "ml-dsa")]
    MlDsa(SignatureScheme),
  }

  #[derive(Clone, Copy, Debug)]
  struct Algorithm {
    public_key: AlgorithmIdentifier,
    signature: AlgorithmIdentifier,
    operation: Operation,
  }

  impl SignatureVerificationAlgorithm for Algorithm {
    fn public_key_alg_id(&self) -> AlgorithmIdentifier {
      self.public_key
    }

    fn signature_alg_id(&self) -> AlgorithmIdentifier {
      self.signature
    }

    fn verify_signature(&self, key: &[u8], message: &[u8], signature: &[u8]) -> Result<(), InvalidSignature> {
      match self.operation {
        #[cfg(feature = "ecdsa-p256")]
        Operation::P256Sha256 | Operation::P256Sha384 => {
          let signature = EcdsaP256Signature::from_der(signature).map_err(|_| InvalidSignature)?;
          let key = EcdsaP256PublicKey::from_sec1_bytes(key).map_err(|_| InvalidSignature)?;
          match self.operation {
            Operation::P256Sha256 => key.verify(message, &signature),
            _ => key.verify_sha384(message, &signature),
          }
        }
        #[cfg(feature = "ecdsa-p384")]
        Operation::P384Sha256 | Operation::P384Sha384 => {
          let signature = EcdsaP384Signature::from_der(signature).map_err(|_| InvalidSignature)?;
          let key = EcdsaP384PublicKey::from_sec1_bytes(key).map_err(|_| InvalidSignature)?;
          match self.operation {
            Operation::P384Sha384 => key.verify(message, &signature),
            _ => key.verify_sha256(message, &signature),
          }
        }
        #[cfg(feature = "ed25519")]
        Operation::Ed25519 => {
          let signature = Ed25519Signature::from_bytes(signature.try_into().map_err(|_| InvalidSignature)?);
          let key = Ed25519PublicKey::from_bytes(key.try_into().map_err(|_| InvalidSignature)?);
          key.verify(message, &signature)
        }
        #[cfg(feature = "ml-dsa")]
        Operation::MlDsa(scheme) => {
          macro_rules! verify {
            ($public:ident, $signature:ident) => {{
              let signature = rscrypto::$signature::try_from_slice(signature).map_err(|_| InvalidSignature)?;
              let key = rscrypto::$public::try_from_slice(key).map_err(|_| InvalidSignature)?;
              key.verify(message, &signature)
            }};
          }
          match scheme {
            SignatureScheme::ML_DSA_44 => verify!(MlDsa44PublicKey, MlDsa44Signature),
            SignatureScheme::ML_DSA_65 => verify!(MlDsa65PublicKey, MlDsa65Signature),
            _ => verify!(MlDsa87PublicKey, MlDsa87Signature),
          }
        }
        #[cfg(feature = "rsa")]
        Operation::Rsa(scheme) => {
          // Preserve the deployed WebPKI policy, independently of the stricter
          // policy for new TLS signing keys. Bound the public exponent's work.
          let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_odd_exponents();
          let signature_lengths = policy.min_modulus_bits().div_ceil(8)..=policy.max_modulus_bits().div_ceil(8);
          if !signature_lengths.contains(&signature.len()) {
            return Err(InvalidSignature);
          }
          let key = RsaPublicKey::from_pkcs1_der_with_policy(key, &policy).map_err(|_| InvalidSignature)?;
          if key.public_exponent().as_u64() >= 1u64 << 33 {
            return Err(InvalidSignature);
          }
          let profile =
            RsaSignatureProfile::from_tls_certificate_signature_scheme(u16::from_be_bytes(scheme.to_array()))
              .map_err(|_| InvalidSignature)?;
          key.verify_signature(profile, message, signature)
        }
      }
      .map_err(|_| InvalidSignature)
    }
  }

  #[cfg(feature = "ecdsa-p256")]
  static P256_SHA256: Algorithm = Algorithm {
    public_key: alg_id::ECDSA_P256,
    signature: alg_id::ECDSA_SHA256,
    operation: Operation::P256Sha256,
  };
  #[cfg(feature = "ecdsa-p256")]
  static P256_SHA384: Algorithm = Algorithm {
    public_key: alg_id::ECDSA_P256,
    signature: alg_id::ECDSA_SHA384,
    operation: Operation::P256Sha384,
  };
  #[cfg(feature = "ecdsa-p384")]
  static P384_SHA256: Algorithm = Algorithm {
    public_key: alg_id::ECDSA_P384,
    signature: alg_id::ECDSA_SHA256,
    operation: Operation::P384Sha256,
  };
  #[cfg(feature = "ecdsa-p384")]
  static P384_SHA384: Algorithm = Algorithm {
    public_key: alg_id::ECDSA_P384,
    signature: alg_id::ECDSA_SHA384,
    operation: Operation::P384Sha384,
  };
  #[cfg(feature = "ed25519")]
  static ED25519: Algorithm = Algorithm {
    public_key: alg_id::ED25519,
    signature: alg_id::ED25519,
    operation: Operation::Ed25519,
  };
  #[cfg(feature = "rsa")]
  static RSA_PSS_SHA256: Algorithm = Algorithm {
    public_key: alg_id::RSA_ENCRYPTION,
    signature: alg_id::RSA_PSS_SHA256,
    operation: Operation::Rsa(SignatureScheme::RSA_PSS_SHA256),
  };
  #[cfg(feature = "rsa")]
  static RSA_PSS_SHA384: Algorithm = Algorithm {
    public_key: alg_id::RSA_ENCRYPTION,
    signature: alg_id::RSA_PSS_SHA384,
    operation: Operation::Rsa(SignatureScheme::RSA_PSS_SHA384),
  };
  #[cfg(feature = "rsa")]
  static RSA_PSS_SHA512: Algorithm = Algorithm {
    public_key: alg_id::RSA_ENCRYPTION,
    signature: alg_id::RSA_PSS_SHA512,
    operation: Operation::Rsa(SignatureScheme::RSA_PSS_SHA512),
  };
  #[cfg(feature = "rsa")]
  static RSA_PKCS1_SHA256: Algorithm = Algorithm {
    public_key: alg_id::RSA_ENCRYPTION,
    signature: alg_id::RSA_PKCS1_SHA256,
    operation: Operation::Rsa(SignatureScheme::RSA_PKCS1_SHA256),
  };
  #[cfg(feature = "rsa")]
  static RSA_PKCS1_SHA384: Algorithm = Algorithm {
    public_key: alg_id::RSA_ENCRYPTION,
    signature: alg_id::RSA_PKCS1_SHA384,
    operation: Operation::Rsa(SignatureScheme::RSA_PKCS1_SHA384),
  };
  #[cfg(feature = "rsa")]
  static RSA_PKCS1_SHA512: Algorithm = Algorithm {
    public_key: alg_id::RSA_ENCRYPTION,
    signature: alg_id::RSA_PKCS1_SHA512,
    operation: Operation::Rsa(SignatureScheme::RSA_PKCS1_SHA512),
  };

  // RFC 4055 requires verifiers to also accept absent PKCS#1 signature parameters.
  #[cfg(feature = "rsa")]
  static RSA_PKCS1_SHA256_ABSENT_PARAMS: Algorithm = Algorithm {
    signature: AlgorithmIdentifier::from_slice(&[0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b]),
    ..RSA_PKCS1_SHA256
  };
  #[cfg(feature = "rsa")]
  static RSA_PKCS1_SHA384_ABSENT_PARAMS: Algorithm = Algorithm {
    signature: AlgorithmIdentifier::from_slice(&[0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0c]),
    ..RSA_PKCS1_SHA384
  };
  #[cfg(feature = "rsa")]
  static RSA_PKCS1_SHA512_ABSENT_PARAMS: Algorithm = Algorithm {
    signature: AlgorithmIdentifier::from_slice(&[0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0d]),
    ..RSA_PKCS1_SHA512
  };

  #[cfg(feature = "ml-dsa")]
  static ML_DSA_44: Algorithm = Algorithm {
    public_key: alg_id::ML_DSA_44,
    signature: alg_id::ML_DSA_44,
    operation: Operation::MlDsa(SignatureScheme::ML_DSA_44),
  };
  #[cfg(feature = "ml-dsa")]
  static ML_DSA_65: Algorithm = Algorithm {
    public_key: alg_id::ML_DSA_65,
    signature: alg_id::ML_DSA_65,
    operation: Operation::MlDsa(SignatureScheme::ML_DSA_65),
  };
  #[cfg(feature = "ml-dsa")]
  static ML_DSA_87: Algorithm = Algorithm {
    public_key: alg_id::ML_DSA_87,
    signature: alg_id::ML_DSA_87,
    operation: Operation::MlDsa(SignatureScheme::ML_DSA_87),
  };

  pub(super) const ALL: &[&dyn SignatureVerificationAlgorithm] = &[
    #[cfg(feature = "ml-dsa")]
    &ML_DSA_44,
    #[cfg(feature = "ml-dsa")]
    &ML_DSA_65,
    #[cfg(feature = "ml-dsa")]
    &ML_DSA_87,
    #[cfg(feature = "ecdsa-p256")]
    &P256_SHA256,
    #[cfg(feature = "ecdsa-p256")]
    &P256_SHA384,
    #[cfg(feature = "ecdsa-p384")]
    &P384_SHA256,
    #[cfg(feature = "ecdsa-p384")]
    &P384_SHA384,
    #[cfg(feature = "ed25519")]
    &ED25519,
    #[cfg(feature = "rsa")]
    &RSA_PSS_SHA256,
    #[cfg(feature = "rsa")]
    &RSA_PSS_SHA384,
    #[cfg(feature = "rsa")]
    &RSA_PSS_SHA512,
    #[cfg(feature = "rsa")]
    &RSA_PKCS1_SHA256,
    #[cfg(feature = "rsa")]
    &RSA_PKCS1_SHA384,
    #[cfg(feature = "rsa")]
    &RSA_PKCS1_SHA512,
    #[cfg(feature = "rsa")]
    &RSA_PKCS1_SHA256_ABSENT_PARAMS,
    #[cfg(feature = "rsa")]
    &RSA_PKCS1_SHA384_ABSENT_PARAMS,
    #[cfg(feature = "rsa")]
    &RSA_PKCS1_SHA512_ABSENT_PARAMS,
  ];
  pub(super) const ALGORITHMS: WebPkiSupportedAlgorithms = {
    const MAPPING: &[(SignatureScheme, &[&dyn SignatureVerificationAlgorithm])] = &[
      #[cfg(feature = "ml-dsa")]
      (SignatureScheme::ML_DSA_87, &[&ML_DSA_87]),
      #[cfg(feature = "ml-dsa")]
      (SignatureScheme::ML_DSA_65, &[&ML_DSA_65]),
      #[cfg(feature = "ml-dsa")]
      (SignatureScheme::ML_DSA_44, &[&ML_DSA_44]),
      // TLS 1.3 binds the curve to its hash. Certificate paths use `all`,
      // including the cross-hash combinations, independently of this mapping.
      #[cfg(feature = "ecdsa-p384")]
      (SignatureScheme::ECDSA_NISTP384_SHA384, &[&P384_SHA384]),
      #[cfg(feature = "ecdsa-p256")]
      (SignatureScheme::ECDSA_NISTP256_SHA256, &[&P256_SHA256]),
      #[cfg(feature = "ed25519")]
      (SignatureScheme::ED25519, &[&ED25519]),
      #[cfg(feature = "rsa")]
      (SignatureScheme::RSA_PSS_SHA512, &[&RSA_PSS_SHA512]),
      #[cfg(feature = "rsa")]
      (SignatureScheme::RSA_PSS_SHA384, &[&RSA_PSS_SHA384]),
      #[cfg(feature = "rsa")]
      (SignatureScheme::RSA_PSS_SHA256, &[&RSA_PSS_SHA256]),
      #[cfg(feature = "rsa")]
      (SignatureScheme::RSA_PKCS1_SHA512, &[&RSA_PKCS1_SHA512]),
      #[cfg(feature = "rsa")]
      (SignatureScheme::RSA_PKCS1_SHA384, &[&RSA_PKCS1_SHA384]),
      #[cfg(feature = "rsa")]
      (SignatureScheme::RSA_PKCS1_SHA256, &[&RSA_PKCS1_SHA256]),
    ];
    match WebPkiSupportedAlgorithms::new(ALL, MAPPING) {
      Ok(algorithms) => algorithms,
      Err(_) => panic!("enabled signature mappings are nonempty"),
    }
  };
}
