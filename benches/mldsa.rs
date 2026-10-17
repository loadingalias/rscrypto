//! ML-DSA production workloads. Fixtures use public, deterministic benchmark keys.
//! HashML-DSA rows cover SHAKE128, the prehash libcrux shares, and SHA-512.
//!
//! Before timing, every row checks its output against another implementation:
//! RustCrypto, libcrux, or AWS-LC.

#[path = "common/criterion.rs"]
mod bench_config;

use core::hint::black_box;

use bench_config::{checked, checked_routine};
use criterion::{BenchmarkId, Criterion};
use rscrypto::{MlDsa44, MlDsa65, MlDsa87};
use rustcrypto_ml_dsa as oracle;

/// A well-formed signature that must not verify: `signature` with its
/// commitment hash, the leading bytes, changed.
fn tampered<const N: usize, S>(
  mut signature: [u8; N],
  parse: impl FnOnce(&[u8]) -> Result<S, rscrypto::MlDsaError>,
) -> S {
  signature[0] ^= 1;
  parse(&signature).expect("a changed commitment hash keeps the signature well formed")
}

macro_rules! profile {
  (
    $function:ident,
    $corpus:ident,
    $name:literal,
    $profile:ident,
    $secret:ident,
    $signature:ident,
    $oracle:ident,
    $libcrux:ident
  ) => {
    fn $function(c: &mut Criterion) {
      if !bench_config::selected(concat!($name, "/")) {
        return;
      }
      let seed = [0x42; 32];
      let (public, secret) = $profile::keypair_from_seed(&seed).expect("benchmark key generation");
      let external = oracle::ExpandedSigningKey::<oracle::$oracle>::from_seed(&seed.into());
      let external_public = external.verifying_key();
      // Retained memory: borrowed key, caller-owned storage, and the handle.
      let mut secret_storage = Default::default();
      let mut public_storage = Default::default();
      let secret_storage_len = core::mem::size_of_val(&secret_storage);
      let public_storage_len = core::mem::size_of_val(&public_storage);
      let prepared_secret = secret.prepare(&mut secret_storage).expect("prepare secret");
      let prepared_public = public.prepare(&mut public_storage).expect("prepare public");
      eprintln!(
        "{} retained bytes: compact secret={}, prepared secret including borrowed key={} vs oracle={}, prepared public including borrowed key={} vs oracle={}",
        $name,
        core::mem::size_of_val(&secret),
        core::mem::size_of_val(&secret)
          .strict_add(secret_storage_len)
          .strict_add(core::mem::size_of_val(&prepared_secret)),
        core::mem::size_of_val(&external),
        core::mem::size_of_val(&public)
          .strict_add(public_storage_len)
          .strict_add(core::mem::size_of_val(&prepared_public)),
        core::mem::size_of_val(&external_public),
      );
      let message = [0xa5; 32];
      let context = b"rscrypto benchmark";
      #[expect(deprecated, reason = "the standard expanded wire encoding is the key-generation oracle")]
      let expanded = external.to_expanded();
      // RustCrypto's deterministic signature is the oracle of the signing rows
      // and the input of the rscrypto verification rows.
      let expected = external
        .sign_deterministic(&message, context)
        .expect("oracle signature");
      let signature = rscrypto::$signature::try_from_slice(expected.encode().as_slice()).expect("oracle signature");
      let tampered = tampered(signature.to_bytes(), rscrypto::$signature::try_from_slice);
      // libcrux takes the FIPS 204 key-generation seed and signs deterministically
      // with all-zero randomness.
      let libcrux = libcrux_ml_dsa::$libcrux::generate_key_pair(seed);
      let libcrux_signature =
        libcrux_ml_dsa::$libcrux::sign(&libcrux.signing_key, &message, context, [0; 32]).expect("libcrux signature");
      // Seed expansion, output encoding and destruction are timed. No OS entropy.
      let mut group = c.benchmark_group(concat!($name, "/keygen/seed-encoded"));
      group.bench_function(
        "rscrypto",
        checked(
          || {
            let (pk, sk) = $profile::keypair_from_seed(black_box(&seed)).expect("key generation");
            (pk.to_bytes(), sk.expose_secret())
          },
          |(pk, sk)| {
            assert_eq!(pk.as_slice(), external_public.encode().as_slice());
            assert_eq!(sk.as_bytes().as_slice(), expanded.as_slice());
          },
        ),
      );
      group.bench_function(
        "rustcrypto",
        checked(
          || {
            let sk = oracle::ExpandedSigningKey::<oracle::$oracle>::from_seed(black_box(&seed.into()));
            #[expect(deprecated, reason = "the measured output is the standard expanded wire encoding")]
            let encoded = sk.to_expanded();
            let encoded = rscrypto::SecretBytes::<{ rscrypto::$secret::LENGTH }>::new(encoded.into());
            (sk.verifying_key().encode(), encoded)
          },
          |(pk, sk)| {
            assert_eq!(pk.as_slice(), libcrux.verification_key.as_ref().as_slice());
            assert_eq!(sk.as_bytes().as_slice(), libcrux.signing_key.as_ref().as_slice());
          },
        ),
      );
      group.bench_function(
        "libcrux",
        checked(
          || black_box(libcrux_ml_dsa::$libcrux::generate_key_pair(black_box(seed))),
          |pair| {
            assert_eq!(
              pair.verification_key.as_ref().as_slice(),
              external_public.encode().as_slice()
            );
            assert_eq!(pair.signing_key.as_ref().as_slice(), expanded.as_slice());
          },
        ),
      );
      group.finish();
      // Preparation fills reused caller-owned storage; the secret handle's drop
      // clears it. Keep the encoded key outside timing, as in the signing and
      // verification rows. Each check signs or verifies with a prepared handle.
      let mut group = c.benchmark_group(concat!($name, "/prepare/encoded-secret"));
      group.bench_function(
        "rscrypto",
        checked_routine(
          Default::default(),
          |storage| {
            secret.prepare(&mut *storage).expect("prepare secret");
            let prepared = secret.prepare(storage).expect("prepare secret");
            let output = prepared
              .sign_deterministic(&message, context)
              .expect("prepared signature");
            assert_eq!(output.as_bytes(), signature.as_bytes());
          },
          |b, storage| {
            b.iter(|| {
              black_box(secret.prepare(&mut *storage).expect("prepare secret"));
            })
          },
        ),
      );
      group.finish();
      let mut group = c.benchmark_group(concat!($name, "/prepare/encoded-public"));
      group.bench_function(
        "rscrypto",
        checked_routine(
          Default::default(),
          |storage| {
            public.prepare(&mut *storage).expect("prepare public");
            let prepared = public.prepare(storage).expect("prepare public");
            prepared
              .verify_with_context(&message, context, &signature)
              .expect("prepared verification");
            prepared
              .verify_with_context(&message, context, &tampered)
              .expect_err("tampered signature must not verify");
          },
          |b, storage| {
            b.iter(|| {
              black_box(public.prepare(&mut *storage).expect("prepare public"));
            })
          },
        ),
      );
      group.finish();
      // Compact keys and explicit prepared keys have distinct memory contracts.
      // Prepared rows retain transformed polynomials and a complete matrix;
      // neither library's preparation is inside those timings.
      let mut group = c.benchmark_group(concat!($name, "/sign/encoded-key-deterministic"));
      group.bench_function(
        "rscrypto",
        checked(
          || {
            secret
              .sign_deterministic(black_box(&message), black_box(context))
              .expect("sign")
          },
          |output| assert_eq!(output.as_bytes(), signature.as_bytes()),
        ),
      );
      // libcrux signs from the encoded key on every call, like this row.
      group.bench_function(
        "libcrux",
        checked(
          || {
            libcrux_ml_dsa::$libcrux::sign(
              black_box(&libcrux.signing_key),
              black_box(&message),
              black_box(context),
              [0; 32],
            )
            .expect("sign")
          },
          |output| assert_eq!(output.as_ref().as_slice(), signature.as_bytes().as_slice()),
        ),
      );
      group.finish();
      let mut group = c.benchmark_group(concat!($name, "/sign/prepared-key-deterministic"));
      group.bench_function(
        "rscrypto",
        checked(
          || {
            prepared_secret
              .sign_deterministic(black_box(&message), black_box(context))
              .expect("prepared sign")
          },
          |output| assert_eq!(output.as_bytes(), signature.as_bytes()),
        ),
      );
      group.bench_function(
        "rustcrypto",
        checked(
          || {
            external
              .sign_deterministic(black_box(&message), black_box(context))
              .expect("sign")
              .encode()
          },
          |output| assert_eq!(output.as_slice(), libcrux_signature.as_ref().as_slice()),
        ),
      );
      group.finish();
      // Signature parsing is outside timing; message hashing and matrix expansion
      // are charged to the production verifier when its API performs them.
      let mut group = c.benchmark_group(concat!($name, "/verify/encoded-key"));
      group.bench_function(
        "rscrypto",
        checked(
          || black_box(public.verify_with_context(black_box(&message), black_box(context), black_box(&signature))),
          |result| {
            result.expect("verify");
            public
              .verify_with_context(&message, context, &tampered)
              .expect_err("tampered signature must not verify");
          },
        ),
      );
      group.bench_function(
        "libcrux",
        checked(
          || {
            black_box(libcrux_ml_dsa::$libcrux::verify(
              black_box(&libcrux.verification_key),
              black_box(&message),
              black_box(context),
              black_box(&libcrux_signature),
            ))
          },
          |result| {
            result.expect("verify");
            assert_eq!(libcrux_signature.as_ref().as_slice(), signature.as_bytes().as_slice());
          },
        ),
      );
      group.finish();
      let mut group = c.benchmark_group(concat!($name, "/verify/prepared-key"));
      group.bench_function(
        "rscrypto",
        checked(
          || {
            black_box(prepared_public.verify_with_context(
              black_box(&message),
              black_box(context),
              black_box(&signature),
            ))
          },
          |result| {
            result.expect("prepared verify");
            prepared_public
              .verify_with_context(&message, context, &tampered)
              .expect_err("tampered signature must not verify");
          },
        ),
      );
      group.bench_function(
        "rustcrypto",
        checked(
          || {
            black_box(external_public.verify_with_context(
              black_box(&message),
              black_box(context),
              black_box(&expected),
            ))
          },
          |accepted| assert!(accepted),
        ),
      );
      group.finish();
    }

    fn $corpus(c: &mut Criterion) {
      if !bench_config::selected(concat!($name, "/corpus/")) {
        return;
      }
      // Each row measures one fixture so rejection tails cannot disappear in
      // an aggregate mean. Both libraries receive identical seeds and inputs.
      // Checks compare rscrypto with RustCrypto, and RustCrypto with libcrux.
      for (index, (message_len, context_len)) in [
        (0, 0), (32, 1), (135, 254), (136, 255),
        (137, 17), (1024, 0), (4096, 17), (65536, 255),
      ].into_iter().enumerate() {
        let seed_byte = u8::try_from(index.strict_add(1)).expect("eight corpus seeds");
        let seed = [seed_byte; 32];
        let (_, secret) = $profile::keypair_from_seed(&seed).expect("corpus key generation");
        let mut storage = Default::default();
        let prepared = secret.prepare(&mut storage).expect("corpus preparation");
        let external = oracle::ExpandedSigningKey::<oracle::$oracle>::from_seed(&seed.into());
        let message = vec![seed_byte ^ 0xa5; message_len];
        let context = vec![0x55; context_len];
        let random = [seed_byte ^ 0x3c; 32];
        let prefix = [0, u8::try_from(context_len).expect("bounded context")];
        // libcrux's signing randomness is the FIPS 204 `rnd` input.
        let libcrux_sign = |rnd: [u8; 32]| {
          let pair = libcrux_ml_dsa::$libcrux::generate_key_pair(seed);
          libcrux_ml_dsa::$libcrux::sign(&pair.signing_key, &message, &context, rnd).expect("libcrux signature")
        };
        let case = format!("seed{seed_byte}-m{message_len}-c{context_len}");
        let mut group = c.benchmark_group(concat!($name, "/corpus/prepared-deterministic"));
        group.bench_function(
          BenchmarkId::new("rscrypto", &case),
          checked(
            || black_box(prepared.sign_deterministic(black_box(&message), black_box(&context)).expect("sign")),
            |output| {
              let expected = external.sign_deterministic(&message, &context).expect("oracle signature");
              assert_eq!(output.as_bytes().as_slice(), expected.encode().as_slice());
            },
          ),
        );
        group.bench_function(
          BenchmarkId::new("rustcrypto", &case),
          checked(
            || black_box(external.sign_deterministic(black_box(&message), black_box(&context)).expect("sign").encode()),
            |output| assert_eq!(output.as_slice(), libcrux_sign([0; 32]).as_ref().as_slice()),
          ),
        );
        group.finish();
        // Entropy is supplied identically from fixed fixture bytes; OS RNG
        // latency is outside both rows. Signature encoding is inside both.
        let mut group = c.benchmark_group(concat!($name, "/corpus/prepared-hedged"));
        group.bench_function(
          BenchmarkId::new("rscrypto", &case),
          checked(
            || {
              black_box(prepared.sign_with(black_box(&message), black_box(&context), |out| {
                out.copy_from_slice(black_box(&random));
                Ok(())
              }).expect("sign"))
            },
            |output| {
              let expected = external.sign_internal(&[&prefix, &context, &message], &random.into());
              assert_eq!(output.as_bytes().as_slice(), expected.encode().as_slice());
            },
          ),
        );
        group.bench_function(
          BenchmarkId::new("rustcrypto", &case),
          checked(
            || {
              black_box(external.sign_internal(
                &[black_box(&prefix), black_box(&context), black_box(&message)],
                black_box(&random.into()),
              ).encode())
            },
            |output| assert_eq!(output.as_slice(), libcrux_sign(random).as_ref().as_slice()),
          ),
        );
        group.finish();
      }
    }
  };
}

profile!(
  mldsa44,
  corpus44,
  "mldsa44",
  MlDsa44,
  MlDsa44SecretKey,
  MlDsa44Signature,
  MlDsa44,
  ml_dsa_44
);
profile!(
  mldsa65,
  corpus65,
  "mldsa65",
  MlDsa65,
  MlDsa65SecretKey,
  MlDsa65Signature,
  MlDsa65,
  ml_dsa_65
);
profile!(
  mldsa87,
  corpus87,
  "mldsa87",
  MlDsa87,
  MlDsa87SecretKey,
  MlDsa87Signature,
  MlDsa87,
  ml_dsa_87
);

/// DER OID of SHA-512, 2.16.840.1.101.3.4.2.3, as FIPS 204 Algorithm 4 binds it.
const SHA512_OID: [u8; 11] = [0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03];
/// DER OID of SHAKE128, 2.16.840.1.101.3.4.2.11, as FIPS 204 Algorithm 4 binds it.
const SHAKE128_OID: [u8; 11] = [0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x0b];

// HashML-DSA (FIPS 204 Algorithm 4) with the pure rows' key, 32-byte message,
// and context. rscrypto signs a caller-computed digest.
macro_rules! prehash_profile {
  ($function:ident, $name:literal, $profile:ident, $signature:ident, $libcrux:ident) => {
    fn $function(c: &mut Criterion) {
      use rscrypto::{MlDsaPrehash, MlDsaPrehashAlgorithm, Shake128};
      use sha2::Digest as _;

      if !bench_config::selected(concat!($name, "/hash-")) {
        return;
      }
      let seed = [0x42; 32];
      let message = [0xa5; 32];
      let context = b"rscrypto benchmark";
      let (public, secret) = $profile::keypair_from_seed(&seed).expect("benchmark key generation");
      let mut secret_storage = Default::default();
      let mut public_storage = Default::default();
      let prepared_secret = secret.prepare(&mut secret_storage).expect("prepare secret");
      let prepared_public = public.prepare(&mut public_storage).expect("prepare public");
      let external = oracle::ExpandedSigningKey::<oracle::$profile>::from_seed(&seed.into());
      let prefix = [1, u8::try_from(context.len()).expect("bounded context")];

      // SHAKE128 with a 32-byte digest is the only prehash libcrux exposes.
      // libcrux is an independent HashML-DSA: its deterministic signature is
      // the oracle of the rscrypto signing row and the rscrypto verifier's input.
      let libcrux = libcrux_ml_dsa::$libcrux::generate_key_pair(seed);
      let libcrux_signature =
        libcrux_ml_dsa::$libcrux::sign_pre_hashed_shake128(&libcrux.signing_key, &message, context, [0; 32])
          .expect("libcrux signature");
      let shake128_signature =
        rscrypto::$signature::try_from_slice(libcrux_signature.as_ref().as_slice()).expect("libcrux signature");
      let shake128_tampered = tampered(shake128_signature.to_bytes(), rscrypto::$signature::try_from_slice);
      let mut shake128_digest = [0; 32];
      Shake128::hash_into(&message, &mut shake128_digest);
      let shake128 = MlDsaPrehash::new(MlDsaPrehashAlgorithm::Shake128, &shake128_digest).expect("digest length");

      // NIST registers SHA-512 as the HashML-DSA prehash for all three sets.
      // No comparison library signs HashML-DSA with SHA-512, so RustCrypto's
      // Sign_internal over the FIPS 204 Algorithm 4 message M' is the oracle.
      // rscrypto has no SHA-512 in this feature set; the digest is a fixture.
      let sha512_digest: [u8; 64] = sha2::Sha512::digest(message).into();
      let sha512 = MlDsaPrehash::new(MlDsaPrehashAlgorithm::Sha512, &sha512_digest).expect("digest length");
      let formatted = [&prefix[..], context, &SHA512_OID, &sha512_digest].concat();
      let sha512_signature = rscrypto::$signature::try_from_slice(
        external
          .sign_internal(&[formatted.as_slice()], &[0; 32].into())
          .encode()
          .as_slice(),
      )
      .expect("oracle signature");
      let sha512_tampered = tampered(sha512_signature.to_bytes(), rscrypto::$signature::try_from_slice);

      // libcrux takes the message, so the rscrypto row adds its public
      // SHAKE128 before digest signing. Both sign from the encoded key.
      let mut group = c.benchmark_group(concat!($name, "/hash-sign/shake128/encoded-key-deterministic"));
      group.bench_function(
        "rscrypto",
        checked(
          || {
            let mut digest = [0; 32];
            Shake128::hash_into(black_box(&message), &mut digest);
            let prehash = MlDsaPrehash::new(MlDsaPrehashAlgorithm::Shake128, &digest).expect("digest length");
            black_box(
              secret
                .sign_prehash_deterministic(prehash, black_box(context))
                .expect("sign"),
            )
          },
          |output| assert_eq!(output.as_bytes(), shake128_signature.as_bytes()),
        ),
      );
      group.bench_function(
        "libcrux",
        checked(
          || {
            black_box(
              libcrux_ml_dsa::$libcrux::sign_pre_hashed_shake128(
                black_box(&libcrux.signing_key),
                black_box(&message),
                black_box(context),
                [0; 32],
              )
              .expect("sign"),
            )
          },
          |output| {
            use tiny_keccak::{Hasher as _, Xof as _};

            // RustCrypto's Sign_internal over M', with tiny-keccak's SHAKE128,
            // is independent of libcrux.
            let mut digest = [0; 32];
            let mut shake = tiny_keccak::Shake::v128();
            shake.update(&message);
            shake.squeeze(&mut digest);
            let formatted = [&prefix[..], context, &SHAKE128_OID, &digest].concat();
            let expected = external.sign_internal(&[formatted.as_slice()], &[0; 32].into());
            assert_eq!(output.as_ref().as_slice(), expected.encode().as_slice());
          },
        ),
      );
      group.finish();
      let mut group = c.benchmark_group(concat!($name, "/hash-verify/shake128/encoded-key"));
      group.bench_function(
        "rscrypto",
        checked(
          || {
            let mut digest = [0; 32];
            Shake128::hash_into(black_box(&message), &mut digest);
            let prehash = MlDsaPrehash::new(MlDsaPrehashAlgorithm::Shake128, &digest).expect("digest length");
            black_box(public.verify_prehash(prehash, black_box(context), black_box(&shake128_signature)))
          },
          |result| {
            result.expect("verify");
            public
              .verify_prehash(shake128, context, &shake128_tampered)
              .expect_err("tampered signature must not verify");
          },
        ),
      );
      group.bench_function(
        "libcrux",
        checked(
          || {
            black_box(libcrux_ml_dsa::$libcrux::verify_pre_hashed_shake128(
              black_box(&libcrux.verification_key),
              black_box(&message),
              black_box(context),
              black_box(&libcrux_signature),
            ))
          },
          |result| result.expect("verify"),
        ),
      );
      group.finish();

      // These rows start from the digest; prepared keys are built untimed.
      let mut group = c.benchmark_group(concat!($name, "/hash-sign/sha512-digest/prepared-key-deterministic"));
      group.bench_function(
        "rscrypto",
        checked(
          || {
            let prehash =
              MlDsaPrehash::new(MlDsaPrehashAlgorithm::Sha512, black_box(&sha512_digest)).expect("digest length");
            black_box(
              prepared_secret
                .sign_prehash_deterministic(prehash, black_box(context))
                .expect("sign"),
            )
          },
          |output| assert_eq!(output.as_bytes(), sha512_signature.as_bytes()),
        ),
      );
      group.finish();
      let mut group = c.benchmark_group(concat!($name, "/hash-verify/sha512-digest/prepared-key"));
      group.bench_function(
        "rscrypto",
        checked(
          || {
            let prehash =
              MlDsaPrehash::new(MlDsaPrehashAlgorithm::Sha512, black_box(&sha512_digest)).expect("digest length");
            black_box(prepared_public.verify_prehash(prehash, black_box(context), black_box(&sha512_signature)))
          },
          |result| {
            result.expect("verify");
            prepared_public
              .verify_prehash(sha512, context, &sha512_tampered)
              .expect_err("tampered signature must not verify");
          },
        ),
      );
      group.finish();
    }
  };
}

prehash_profile!(prehash44, "mldsa44", MlDsa44, MlDsa44Signature, ml_dsa_44);
prehash_profile!(prehash65, "mldsa65", MlDsa65, MlDsa65Signature, ml_dsa_65);
prehash_profile!(prehash87, "mldsa87", MlDsa87, MlDsa87Signature, ml_dsa_87);

#[cfg(any(
  all(
    any(unix, windows),
    not(target_arch = "wasm32"),
    not(any(target_arch = "s390x", target_arch = "powerpc64"))
  ),
  all(
    target_arch = "powerpc64",
    target_endian = "little",
    target_os = "linux",
    target_env = "gnu"
  )
))]
mod aws {
  use super::{Criterion, MlDsa44, MlDsa65, MlDsa87, bench_config, black_box, checked, tampered};
  // AWS-LC exposes randomized pure signing with empty context. Keep these rows
  // separate from the fixed-randomness/deterministic comparisons above.
  macro_rules! aws_profile {
    ($function:ident, $name:literal, $profile:ident, $public:ident, $signature:ident, $sign:ident, $verify:ident) => {
      fn $function(c: &mut Criterion) {
        use aws_lc_rs::{
          encoding::AsRawBytes,
          signature::{KeyPair, PqdsaKeyPair, UnparsedPublicKey},
        };
        if !bench_config::selected(concat!($name, "/cross/")) {
          return;
        }
        let seed = [0x42; 32];
        let message = [0xa5; 32];
        let (public, secret) = $profile::keypair_from_seed(&seed).expect("keygen");
        let mut secret_storage = Default::default();
        let prepared = secret.prepare(&mut secret_storage).expect("prepare secret");
        let mut public_storage = Default::default();
        let prepared_public = public.prepare(&mut public_storage).expect("prepare public");
        let external = PqdsaKeyPair::from_seed(&aws_lc_rs::signature::$sign, &seed).expect("AWS-LC keygen");
        let external_public = UnparsedPublicKey::new(&aws_lc_rs::signature::$verify, external.public_key().as_ref())
          .parse()
          .expect("parse public");
        let signature = prepared.sign_deterministic(&message, &[]).expect("signature");
        let tampered = tampered(signature.to_bytes(), rscrypto::$signature::try_from_slice);

        // Both rows include seed expansion, expanded wire outputs, and key/output
        // destruction. AWS-LC owns heap buffers; rscrypto owns inline arrays.
        let mut group = c.benchmark_group(concat!($name, "/cross/keygen/seed-encoded"));
        group.bench_function(
          "rscrypto",
          checked(
            || {
              let (pk, sk) = $profile::keypair_from_seed(black_box(&seed)).expect("keygen");
              (pk.to_bytes(), sk.expose_secret())
            },
            |(pk, sk)| {
              assert_eq!(external.public_key().as_ref(), pk.as_slice());
              assert_eq!(
                external
                  .private_key()
                  .as_raw_bytes()
                  .expect("expanded key")
                  .as_ref(),
                sk.as_ref()
              );
            },
          ),
        );
        group.bench_function(
          "aws-lc",
          checked(
            || {
              let key = PqdsaKeyPair::from_seed(&aws_lc_rs::signature::$sign, black_box(&seed)).expect("keygen");
              let mut pk = [0u8; rscrypto::$public::LENGTH];
              pk.copy_from_slice(key.public_key().as_ref());
              (pk, key.private_key().as_raw_bytes().expect("expanded key"))
            },
            |(pk, sk)| {
              assert_eq!(pk, public.to_bytes());
              assert_eq!(sk.as_ref(), secret.expose_secret().as_ref());
            },
          ),
        );
        group.finish();

        // Reusable key owners are prepared outside timing. Both request fresh
        // 32-byte randomness from AWS-LC's RNG and return a complete signature.
        // Random rejection counts vary; these rows measure latency distributions.
        // Each randomized signature must verify under the other library.
        let mut group = c.benchmark_group(concat!($name, "/cross/sign/reused-key-randomized-aws-rng"));
        group.bench_function(
          "rscrypto",
          checked(
            || {
              black_box(
                prepared
                  .sign_with(black_box(&message), &[], |out| {
                    aws_lc_rs::rand::fill(out).map_err(|_| rscrypto::MlDsaError::RandomGenerationFailed)
                  })
                  .expect("sign"),
              )
            },
            |output| {
              external_public
                .verify_sig(&message, output.as_bytes())
                .expect("AWS-LC verifies rscrypto");
            },
          ),
        );
        group.bench_function(
          "aws-lc",
          checked(
            || {
              let mut out = [0u8; rscrypto::$signature::LENGTH];
              external.sign(black_box(&message), &mut out).expect("sign");
              black_box(out)
            },
            |output| {
              let parsed = rscrypto::$signature::try_from_slice(&output).expect("parse signature");
              prepared_public
                .verify(&message, &parsed)
                .expect("rscrypto verifies AWS-LC");
            },
          ),
        );
        group.finish();

        // Both retain their public-key owner outside timing and verify the same
        // signature. AWS-LC's API takes encoded signatures; rscrypto takes its
        // validated signature type. Internal work remains charged to each API.
        let mut group = c.benchmark_group(concat!($name, "/cross/verify/reused-key"));
        group.bench_function(
          "rscrypto",
          checked(
            || black_box(prepared_public.verify(black_box(&message), black_box(&signature))),
            |result| {
              result.expect("verify");
              prepared_public
                .verify(&message, &tampered)
                .expect_err("tampered signature must not verify");
              external_public
                .verify_sig(&message, signature.as_bytes())
                .expect("AWS-LC accepts the verified signature");
            },
          ),
        );
        group.bench_function(
          "aws-lc",
          checked(
            || black_box(external_public.verify_sig(black_box(&message), black_box(signature.as_bytes()))),
            |result| result.expect("verify"),
          ),
        );
        group.finish();
      }
    };
  }

  aws_profile!(
    mldsa44,
    "mldsa44",
    MlDsa44,
    MlDsa44PublicKey,
    MlDsa44Signature,
    ML_DSA_44_SIGNING,
    ML_DSA_44
  );
  aws_profile!(
    mldsa65,
    "mldsa65",
    MlDsa65,
    MlDsa65PublicKey,
    MlDsa65Signature,
    ML_DSA_65_SIGNING,
    ML_DSA_65
  );
  aws_profile!(
    mldsa87,
    "mldsa87",
    MlDsa87,
    MlDsa87PublicKey,
    MlDsa87Signature,
    ML_DSA_87_SIGNING,
    ML_DSA_87
  );

  pub(super) fn run(c: &mut Criterion) {
    mldsa44(c);
    mldsa65(c);
    mldsa87(c);
  }
}

fn main() {
  // ML-DSA-87 fixtures and the RustCrypto oracle's key expansion overflow the
  // 1 MiB Windows main-thread stack; run on a thread with an explicit stack.
  std::thread::Builder::new()
    .stack_size(8 << 20)
    .spawn(benchmarks)
    .expect("spawn benchmark thread")
    .join()
    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

fn benchmarks() {
  bench_config::run(&[
    mldsa44,
    mldsa65,
    mldsa87,
    corpus44,
    corpus65,
    corpus87,
    prehash44,
    prehash65,
    prehash87,
    #[cfg(any(
      all(
        any(unix, windows),
        not(target_arch = "wasm32"),
        not(any(target_arch = "s390x", target_arch = "powerpc64"))
      ),
      all(
        target_arch = "powerpc64",
        target_endian = "little",
        target_os = "linux",
        target_env = "gnu"
      )
    ))]
    aws::run,
  ]);
}
