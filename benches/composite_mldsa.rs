//! Composite ML-DSA production workloads (draft-ietf-lamps-pq-composite-sigs-19).
//! Fixtures use public, deterministic benchmark keys and a 32-byte message
//! with the empty context.
//!
//! Comparators: `darkbio-crypto` (`xdsa`) implements `id-MLDSA65-Ed25519-SHA512`
//! on RustCrypto ML-DSA and ed25519-dalek; `kryptering` implements
//! `id-MLDSA65-ECDSA-P256-SHA512` on RustCrypto ML-DSA and p256, with JOSE
//! encodings of the P-256 component. No other library implements these
//! composites. Before timing, every row checks its output against another
//! implementation.

#[path = "common/criterion.rs"]
mod bench_config;

use core::hint::black_box;

use bench_config::checked;
use criterion::Criterion;
use kryptering::traits::{Signer as _, Verifier as _};
use kryptering::{CompositeMlDsaVariant, SignatureAlgorithm, SoftwareKey, SoftwareSigner, SoftwareVerifier};
use rscrypto::{
  CompositeMlDsaError, EcdsaP256Signature, MlDsa65EcdsaP256, MlDsa65EcdsaP256PublicKey, MlDsa65EcdsaP256SecretKey,
  MlDsa65EcdsaP256Signature, MlDsa65Ed25519, MlDsa65Ed25519PublicKey, MlDsa65Ed25519SecretKey, MlDsa65Ed25519Signature,
  MlDsa65PreparedSecretKeyStorage, MlDsa65Signature,
};

const MESSAGE: [u8; 32] = [0xa5; 32];

fn seeded(byte: u8) -> impl FnMut(&mut [u8]) -> Result<(), CompositeMlDsaError> {
  move |out| {
    out.fill(byte);
    Ok(())
  }
}

/// Fixed hedging randomness: the rows time signing, not the random source.
fn hedge(out: &mut [u8]) -> Result<(), CompositeMlDsaError> {
  out.fill(0x5c);
  Ok(())
}

fn ed25519(c: &mut Criterion) {
  if !bench_config::selected("mldsa65-ed25519/") {
    return;
  }
  let (public, secret) = MlDsa65Ed25519::generate_keypair(seeded(0x42)).expect("benchmark key");
  let raw = secret.expose_secret();
  let oracle = darkbio_crypto::xdsa::SecretKey::from_bytes(raw.as_bytes());
  let oracle_public = darkbio_crypto::xdsa::PublicKey::from_bytes(&public.to_bytes()).expect("oracle public key");
  let public_bytes = public.to_bytes();
  // Both sign ML-DSA deterministically, so the oracle signature is the
  // expected output of the deterministic rows.
  let expected = oracle.sign(&MESSAGE).to_bytes();
  let mut tampered = expected;
  tampered[0] ^= 1;
  let mut storage = Box::new(MlDsa65PreparedSecretKeyStorage::new());
  let prepared = secret.prepare(&mut storage).expect("prepare");

  // Seeds to the encoded public key: expand both components and encode.
  // rscrypto also keeps the expanded ML-DSA key for signing; darkbio keeps
  // RustCrypto's signing key.
  let mut group = c.benchmark_group("mldsa65-ed25519/keygen/seed-encoded");
  group.bench_function(
    "rscrypto",
    checked(
      || {
        MlDsa65Ed25519SecretKey::from_bytes(black_box(raw.as_bytes()))
          .expect("import")
          .public_key()
          .to_bytes()
      },
      |pk| assert_eq!(pk, oracle.public_key().to_bytes()),
    ),
  );
  group.bench_function(
    "darkbio",
    checked(
      || {
        darkbio_crypto::xdsa::SecretKey::from_bytes(black_box(raw.as_bytes()))
          .public_key()
          .to_bytes()
      },
      |pk| assert_eq!(pk, public_bytes),
    ),
  );
  group.finish();

  // Deterministic signing with each library's best reused key: rscrypto's
  // prepared key, and darkbio's expanded RustCrypto key.
  let mut group = c.benchmark_group("mldsa65-ed25519/sign/reuse-key-deterministic");
  group.bench_function(
    "rscrypto",
    checked(
      || prepared.sign_deterministic(black_box(&MESSAGE), b"").expect("sign"),
      |signature| assert_eq!(signature.as_bytes(), &expected),
    ),
  );
  group.bench_function(
    "rscrypto-compact",
    checked(
      || secret.sign_deterministic(black_box(&MESSAGE), b"").expect("sign"),
      |signature| assert_eq!(signature.as_bytes(), &expected),
    ),
  );
  group.bench_function(
    "darkbio",
    checked(
      || oracle.sign(black_box(&MESSAGE)).to_bytes(),
      |signature| {
        let signature = MlDsa65Ed25519Signature::try_from_slice(&signature).expect("signature");
        public
          .verify(&MESSAGE, &signature)
          .expect("rscrypto accepts darkbio's signature");
      },
    ),
  );
  group.finish();

  // Hedged signing: darkbio has no hedged API and stays out of this group.
  let mut group = c.benchmark_group("mldsa65-ed25519/sign/reuse-key-randomized");
  group.bench_function(
    "rscrypto",
    checked(
      || prepared.sign_with(black_box(&MESSAGE), b"", hedge).expect("sign"),
      |signature| {
        oracle_public
          .verify(
            &MESSAGE,
            &darkbio_crypto::xdsa::Signature::from_bytes(signature.as_bytes()),
          )
          .expect("darkbio accepts rscrypto's signature");
      },
    ),
  );
  group.finish();

  // Parse and verify one signature with a parsed public key.
  let mut group = c.benchmark_group("mldsa65-ed25519/verify/reuse-key");
  let verify_rscrypto = |bytes: &[u8]| {
    MlDsa65Ed25519Signature::try_from_slice(bytes).is_ok_and(|signature| public.verify(&MESSAGE, &signature).is_ok())
  };
  let verify_darkbio = |bytes: &[u8; MlDsa65Ed25519Signature::LENGTH]| {
    oracle_public
      .verify(&MESSAGE, &darkbio_crypto::xdsa::Signature::from_bytes(bytes))
      .is_ok()
  };
  group.bench_function(
    "rscrypto",
    checked(
      || verify_rscrypto(black_box(&expected)),
      |accepted| {
        assert!(accepted);
        assert!(!verify_rscrypto(&tampered));
      },
    ),
  );
  group.bench_function(
    "darkbio",
    checked(
      || verify_darkbio(black_box(&expected)),
      |accepted| {
        assert!(accepted);
        assert!(!verify_darkbio(&tampered));
      },
    ),
  );
  group.finish();

  // Import the public-key bytes, then parse and verify one signature.
  let mut group = c.benchmark_group("mldsa65-ed25519/verify/import-key");
  let import_rscrypto = |bytes: &[u8]| {
    let public = MlDsa65Ed25519PublicKey::try_from_slice(&public_bytes).expect("public key");
    MlDsa65Ed25519Signature::try_from_slice(bytes).is_ok_and(|signature| public.verify(&MESSAGE, &signature).is_ok())
  };
  let import_darkbio = |bytes: &[u8; MlDsa65Ed25519Signature::LENGTH]| {
    darkbio_crypto::xdsa::PublicKey::from_bytes(&public_bytes)
      .expect("public key")
      .verify(&MESSAGE, &darkbio_crypto::xdsa::Signature::from_bytes(bytes))
      .is_ok()
  };
  group.bench_function(
    "rscrypto",
    checked(
      || import_rscrypto(black_box(&expected)),
      |accepted| {
        assert!(accepted);
        assert!(!import_rscrypto(&tampered));
      },
    ),
  );
  group.bench_function(
    "darkbio",
    checked(
      || import_darkbio(black_box(&expected)),
      |accepted| {
        assert!(accepted);
        assert!(!import_darkbio(&tampered));
      },
    ),
  );
  group.finish();
}

/// kryptering's raw private key: the ML-DSA seed and the P-256 scalar.
fn kryptering_private(secret: &MlDsa65EcdsaP256SecretKey) -> Vec<u8> {
  let raw = secret.expose_secret();
  [&raw.as_bytes()[..32], &raw.as_bytes()[39..71]].concat()
}

/// kryptering's raw public key: the uncompressed point without `04`.
fn kryptering_public(public: &MlDsa65EcdsaP256PublicKey) -> Vec<u8> {
  let bytes = public.to_bytes();
  [&bytes[..1952], &bytes[1953..]].concat()
}

fn raw_ecdsa(signature: &[u8]) -> Vec<u8> {
  let (ml_dsa, der) = signature.split_at(MlDsa65Signature::LENGTH);
  [ml_dsa, EcdsaP256Signature::from_der(der).expect("DER").as_bytes()].concat()
}

fn ecdsa_p256(c: &mut Criterion) {
  if !bench_config::selected("mldsa65-ecdsa-p256/") {
    return;
  }
  let variant = CompositeMlDsaVariant::MlDsa65Es256;
  let algorithm = SignatureAlgorithm::CompositeMlDsa(variant);
  let (public, secret) = MlDsa65EcdsaP256::generate_keypair(seeded(0x43)).expect("benchmark key");
  let public_bytes = public.to_bytes();
  let oracle_private = kryptering_private(&secret);
  let oracle_public = kryptering_public(&public);
  let key = SoftwareKey::from_composite_ml_dsa(variant, Some(&oracle_private), &oracle_public).expect("oracle key");
  let signer = SoftwareSigner::new(algorithm, key.clone()).expect("oracle signer");
  let verifier = SoftwareVerifier::new(algorithm, key).expect("oracle verifier");
  let signature = secret.sign_with(&MESSAGE, b"", hedge).expect("signature");
  let encoded = signature.as_bytes().to_vec();
  let raw = raw_ecdsa(&encoded);
  let mut tampered = encoded.clone();
  tampered[0] ^= 1;
  let mut tampered_raw = raw.clone();
  tampered_raw[0] ^= 1;
  let mut storage = Box::new(MlDsa65PreparedSecretKeyStorage::new());
  let prepared = secret.prepare(&mut storage).expect("prepare");

  // Hedged signing with each library's best reused key. kryptering keeps the
  // ML-DSA seed and expands it in each call; it has no prepared key. Its
  // randomness comes from the OS.
  let mut group = c.benchmark_group("mldsa65-ecdsa-p256/sign/reuse-key-randomized");
  group.bench_function(
    "rscrypto",
    checked(
      || prepared.sign_with(black_box(&MESSAGE), b"", hedge).expect("sign"),
      |signature| {
        assert!(
          verifier
            .verify(&MESSAGE, &raw_ecdsa(signature.as_bytes()))
            .expect("oracle")
        )
      },
    ),
  );
  group.bench_function(
    "rscrypto-compact",
    checked(
      || secret.sign_with(black_box(&MESSAGE), b"", hedge).expect("sign"),
      |signature| {
        assert!(
          verifier
            .verify(&MESSAGE, &raw_ecdsa(signature.as_bytes()))
            .expect("oracle")
        )
      },
    ),
  );
  group.bench_function(
    "kryptering",
    checked(
      || signer.sign(black_box(&MESSAGE)).expect("sign"),
      |signature| {
        let (ml_dsa, rs) = signature.split_at(MlDsa65Signature::LENGTH);
        let ecdsa = EcdsaP256Signature::from_bytes(rs.try_into().expect("r || s")).expect("scalars");
        let mut der = [0; EcdsaP256Signature::DER_MAX_LENGTH];
        let composite = MlDsa65EcdsaP256Signature::try_from_slice(&[ml_dsa, ecdsa.to_der_into(&mut der)].concat())
          .expect("signature");
        public
          .verify(&MESSAGE, &composite)
          .expect("rscrypto accepts kryptering's signature");
      },
    ),
  );
  group.finish();

  // Deterministic signing: kryptering has no deterministic API and stays out.
  let mut group = c.benchmark_group("mldsa65-ecdsa-p256/sign/reuse-key-deterministic");
  group.bench_function(
    "rscrypto",
    checked(
      || prepared.sign_deterministic(black_box(&MESSAGE), b"").expect("sign"),
      |signature| {
        assert!(
          verifier
            .verify(&MESSAGE, &raw_ecdsa(signature.as_bytes()))
            .expect("oracle")
        )
      },
    ),
  );
  group.finish();

  // Parse and verify one signature with a parsed public key. kryptering
  // takes the raw `r || s` encoding, converted outside timing.
  let mut group = c.benchmark_group("mldsa65-ecdsa-p256/verify/reuse-key");
  let verify_rscrypto = |bytes: &[u8]| {
    MlDsa65EcdsaP256Signature::try_from_slice(bytes).is_ok_and(|signature| public.verify(&MESSAGE, &signature).is_ok())
  };
  group.bench_function(
    "rscrypto",
    checked(
      || verify_rscrypto(black_box(&encoded)),
      |accepted| {
        assert!(accepted);
        assert!(!verify_rscrypto(&tampered));
      },
    ),
  );
  group.bench_function(
    "kryptering",
    checked(
      || verifier.verify(&MESSAGE, black_box(&raw)).unwrap_or(false),
      |accepted| {
        assert!(accepted);
        assert!(!verifier.verify(&MESSAGE, &tampered_raw).unwrap_or(false));
      },
    ),
  );
  group.finish();

  // Import the public-key bytes, then parse and verify one signature.
  let mut group = c.benchmark_group("mldsa65-ecdsa-p256/verify/import-key");
  let import_rscrypto = |bytes: &[u8]| {
    let public = MlDsa65EcdsaP256PublicKey::try_from_slice(&public_bytes).expect("public key");
    MlDsa65EcdsaP256Signature::try_from_slice(bytes).is_ok_and(|signature| public.verify(&MESSAGE, &signature).is_ok())
  };
  let import_kryptering = |bytes: &[u8]| {
    let key = SoftwareKey::from_composite_ml_dsa(variant, None, &oracle_public).expect("public key");
    SoftwareVerifier::new(algorithm, key)
      .expect("verifier")
      .verify(&MESSAGE, bytes)
      .unwrap_or(false)
  };
  group.bench_function(
    "rscrypto",
    checked(
      || import_rscrypto(black_box(&encoded)),
      |accepted| {
        assert!(accepted);
        assert!(!import_rscrypto(&tampered));
      },
    ),
  );
  group.bench_function(
    "kryptering",
    checked(
      || import_kryptering(black_box(&raw)),
      |accepted| {
        assert!(accepted);
        assert!(!import_kryptering(&tampered_raw));
      },
    ),
  );
  group.finish();
}

fn main() {
  // ML-DSA-65 fixtures and the comparators' key expansion exceed the 1 MiB
  // Windows main-thread stack; run on a thread with an explicit stack.
  std::thread::Builder::new()
    .stack_size(8 << 20)
    .spawn(|| bench_config::run(&[ed25519, ecdsa_p256]))
    .expect("spawn benchmark thread")
    .join()
    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}
