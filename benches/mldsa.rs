//! ML-DSA production workloads. Fixtures use public, deterministic benchmark keys.
//! HashML-DSA rows cover SHAKE128, the prehash libcrux shares, and SHA-512.
//!
//! Before timing, every row checks its output against another implementation:
//! RustCrypto, libcrux, or AWS-LC.

#[path = "common/criterion.rs"]
mod bench_config;

#[path = "common/aws_lc.rs"]
mod aws_lc_support;
use aws_lc_support::aws_lc_bench as with_aws_lc;

use core::hint::black_box;

use bench_config::{checked, checked_routine};
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
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

/// Registers a row that verifies encoded signatures. The check runs the same
/// path: it must accept `signature` and reject `tampered` and `malformed`.
fn verify_row(
  group: &mut BenchmarkGroup<'_, WallTime>,
  implementation: &str,
  [signature, tampered, malformed]: [&[u8]; 3],
  verify: impl Fn(&[u8]) -> bool + Copy,
) {
  group.bench_function(
    implementation,
    checked(
      || verify(black_box(signature)),
      |accepted| {
        assert!(accepted, "{implementation} must accept the signature");
        assert!(!verify(tampered), "{implementation} must reject a tampered signature");
        assert!(!verify(malformed), "{implementation} must reject a malformed signature");
      },
    ),
  );
}

macro_rules! profile {
  (
    $function:ident,
    $corpus:ident,
    $name:literal,
    $profile:ident,
    $public:ident,
    $secret:ident,
    $signature:ident,
    $oracle:ident,
    $libcrux:ident,
    $aws_signing:ident,
    $aws_verification:ident
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
      // Seed to the encoded public key and the expanded FIPS 204 secret key.
      // Timed: key generation, export of both encodings, and destruction. No
      // OS entropy. Each library generates the key once:
      // - rscrypto returns inline arrays and clears the secret on drop.
      // - RustCrypto's `SigningKey` also builds its precomputed signing and
      //   verifying keys and clears them on drop. Its doc-hidden
      //   `expanded_key` is its only export that neither clones the key nor
      //   recomputes the public key. The row clears the secret it exports.
      // - libcrux returns plain arrays.
      // - AWS-LC builds a heap EVP_PKEY and exports the secret into a heap
      //   buffer; the row copies the public key into an array.
      // Checks compare both keys with another library.
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
            let sk = oracle::SigningKey::<oracle::$oracle>::from_seed(black_box(&seed.into()));
            let pk = AsRef::<oracle::VerifyingKey<oracle::$oracle>>::as_ref(&sk).encode();
            #[expect(deprecated, reason = "the measured output is the standard expanded wire encoding")]
            let encoded = sk.expanded_key().to_expanded();
            (pk, rscrypto::SecretBytes::<{ rscrypto::$secret::LENGTH }>::new(encoded.into()))
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
      with_aws_lc! {
        group.bench_function(
          "aws-lc",
          checked(
            || {
              use aws_lc_rs::encoding::AsRawBytes as _;

              let key = aws_lc_rs::signature::PqdsaKeyPair::from_seed(
                &aws_lc_rs::signature::$aws_signing,
                black_box(&seed),
              )
              .expect("key generation");
              let mut pk = [0u8; rscrypto::$public::LENGTH];
              pk.copy_from_slice(aws_lc_rs::signature::KeyPair::public_key(&key).as_ref());
              (pk, key.private_key().as_raw_bytes().expect("expanded key"))
            },
            |(pk, sk)| {
              assert_eq!(pk.as_slice(), external_public.encode().as_slice());
              assert_eq!(sk.as_ref(), expanded.as_slice());
            },
          ),
        );
      }
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
      // Compact keys and explicit prepared keys have distinct memory contracts,
      // so each has its own groups. Sign and verify groups use the 32-byte
      // message and the 18-byte context. Keys are built outside timing.
      //
      // Deterministic signing with a reused compact key. Each call decodes the
      // secret key and expands the matrix: rscrypto per rejection attempt,
      // libcrux once per call. Timed: signing and the signature array.
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
      // Deterministic signing with a reused prepared key. rscrypto's prepared
      // key and RustCrypto's `ExpandedSigningKey` both cache the matrix and the
      // transformed secret vectors. Timed: signing and signature encoding.
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
      // Verification with a reused compact key: each call hashes the key into
      // tr, expands the matrix, and transforms t1. Both libraries take a
      // signature that holds its encoding; parsing it is outside timing.
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
            libcrux_ml_dsa::$libcrux::verify(
              &libcrux.verification_key,
              &message,
              context,
              &libcrux_ml_dsa::MLDSASignature::new(tampered.to_bytes()),
            )
            .expect_err("tampered signature must not verify");
          },
        ),
      );
      group.finish();
      // Verification with a reused prepared key. rscrypto's prepared key and
      // RustCrypto's `VerifyingKey` both cache the matrix, NTT(t1·2^d), and tr.
      // Signatures are parsed outside timing. RustCrypto's parsed signature
      // holds the decoded response and hints, so its timing excludes that
      // decoding; rscrypto's verifier decodes its signature bytes inside timing.
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
          |accepted| {
            assert!(accepted);
            assert_eq!(expected.encode().as_slice(), libcrux_signature.as_ref().as_slice());
            let forged = oracle::Signature::<oracle::$oracle>::try_from(tampered.as_bytes().as_slice())
              .expect("a changed commitment hash keeps the signature well formed");
            assert!(
              !external_public.verify_with_context(&message, context, &forged),
              "tampered signature must not verify"
            );
          },
        ),
      );
      group.finish();

      // Task-view groups: every library that has an API for the task, each
      // with its best form. They use an empty context, the only context that
      // aws-lc-rs takes. The `prepared-key` and `encoded-key` groups above
      // compare equal cache depths instead.
      let empty: &[u8] = &[];
      let encoded_public = public.to_bytes();
      assert_eq!(encoded_public.as_slice(), external_public.encode().as_slice());
      let empty_signature = external
        .sign_deterministic(&message, empty)
        .expect("oracle signature")
        .encode();
      let libcrux_empty_signature =
        libcrux_ml_dsa::$libcrux::sign(&libcrux.signing_key, &message, empty, [0; 32]).expect("libcrux signature");
      assert_eq!(empty_signature.as_slice(), libcrux_empty_signature.as_ref().as_slice());
      let mut empty_tampered = empty_signature.clone();
      empty_tampered[0] ^= 1;
      // The last hint count exceeds omega, so sigDecode rejects the encoding.
      let mut empty_malformed = empty_signature.clone();
      *empty_malformed.last_mut().expect("nonempty signature") = 0xff;
      let signatures = [
        empty_signature.as_slice(),
        empty_tampered.as_slice(),
        empty_malformed.as_slice(),
      ];
      with_aws_lc! {
        let aws_secret = aws_lc_rs::signature::PqdsaKeyPair::from_seed(&aws_lc_rs::signature::$aws_signing, &seed)
          .expect("AWS-LC key generation");
        let aws_encoded_public = aws_lc_rs::signature::KeyPair::public_key(&aws_secret).as_ref();
        assert_eq!(aws_encoded_public, encoded_public.as_slice());
        let aws_public =
          aws_lc_rs::signature::UnparsedPublicKey::new(&aws_lc_rs::signature::$aws_verification, aws_encoded_public)
            .parse()
            .expect("AWS-LC public key");
      }

      // Verification of one received signature under a reused public key.
      // Timed: signature parsing, verification, and destruction. Each
      // library keeps its best reusable key, built outside timing:
      // - rscrypto's prepared key caches tr, the matrix, and NTT(t1).
      // - RustCrypto's `VerifyingKey` caches tr, the matrix, and NTT(t1·2^d).
      // - libcrux's key holds the encoding, which is all it has; each call
      //   hashes tr, expands the matrix, and transforms t1.
      // - AWS-LC's `ParsedPublicKey` holds the decoded key and recomputes the
      //   same values on each call.
      // RustCrypto decodes the response and hints when it parses; the others
      // decode them during verification. Checks: each row accepts the
      // signature, which RustCrypto and libcrux both produce, and rejects a
      // tampered and a malformed signature.
      let mut group = c.benchmark_group(concat!($name, "/verify/reuse-key"));
      verify_row(&mut group, "rscrypto", signatures, |signature| {
        rscrypto::$signature::try_from_slice(signature)
          .is_ok_and(|signature| prepared_public.verify(&message, &signature).is_ok())
      });
      verify_row(&mut group, "rustcrypto", signatures, |signature| {
        oracle::Signature::<oracle::$oracle>::try_from(signature)
          .is_ok_and(|signature| external_public.verify_with_context(&message, empty, &signature))
      });
      verify_row(&mut group, "libcrux", signatures, |signature| {
        <[u8; rscrypto::$signature::LENGTH]>::try_from(signature).is_ok_and(|signature| {
          libcrux_ml_dsa::$libcrux::verify(
            &libcrux.verification_key,
            &message,
            empty,
            &libcrux_ml_dsa::MLDSASignature::new(signature),
          )
          .is_ok()
        })
      });
      with_aws_lc! {
        verify_row(&mut group, "aws-lc", signatures, |signature| {
          aws_public.verify_sig(&message, signature).is_ok()
        });
      }
      group.finish();

      // Verification from encoded bytes, as for a certificate signature.
      // Timed: key import, signature parsing, verification, and destruction.
      // - rscrypto copies the key, checks the signature structure, and
      //   verifies with the compact key, which hashes tr, expands the matrix,
      //   and transforms t1. Its verifier checks the structure again.
      //   Preparing the key would do the same work and also store the matrix.
      // - libcrux copies both encodings into its byte wrappers and decodes
      //   them during verification.
      // - AWS-LC tries SubjectPublicKeyInfo DER, falls back to the raw key,
      //   allocates and frees an EVP_PKEY, and verifies.
      // - RustCrypto decodes the key and precomputes the matrix, NTT(t1·2^d),
      //   and tr, then decodes the response and hints.
      // Checks as in `verify/reuse-key`.
      let mut group = c.benchmark_group(concat!($name, "/verify/import-key"));
      verify_row(&mut group, "rscrypto", signatures, |signature| {
        let (Ok(public), Ok(signature)) = (
          rscrypto::$public::try_from_slice(black_box(encoded_public.as_slice())),
          rscrypto::$signature::try_from_slice(signature),
        ) else {
          return false;
        };
        public.verify(&message, &signature).is_ok()
      });
      verify_row(&mut group, "rustcrypto", signatures, |signature| {
        let (Ok(public), Ok(signature)) = (
          oracle::EncodedVerifyingKey::<oracle::$oracle>::try_from(black_box(encoded_public.as_slice())),
          oracle::Signature::<oracle::$oracle>::try_from(signature),
        ) else {
          return false;
        };
        oracle::VerifyingKey::<oracle::$oracle>::decode(&public).verify_with_context(&message, empty, &signature)
      });
      verify_row(&mut group, "libcrux", signatures, |signature| {
        let (Ok(public), Ok(signature)) = (
          <[u8; rscrypto::$public::LENGTH]>::try_from(black_box(encoded_public.as_slice())),
          <[u8; rscrypto::$signature::LENGTH]>::try_from(signature),
        ) else {
          return false;
        };
        libcrux_ml_dsa::$libcrux::verify(
          &libcrux_ml_dsa::MLDSAVerificationKey::new(public),
          &message,
          empty,
          &libcrux_ml_dsa::MLDSASignature::new(signature),
        )
        .is_ok()
      });
      with_aws_lc! {
        verify_row(&mut group, "aws-lc", signatures, |signature| {
          aws_lc_rs::signature::UnparsedPublicKey::new(
            &aws_lc_rs::signature::$aws_verification,
            black_box(encoded_public.as_slice()),
          )
          .verify(&message, signature)
          .is_ok()
        });
      }
      group.finish();

      // Deterministic signing under a reused secret key. Timed: signing and
      // the encoded signature. Each library keeps its best reusable key,
      // built outside timing:
      // - rscrypto's prepared key and RustCrypto's `ExpandedSigningKey` cache
      //   the matrix and the transformed secret vectors.
      // - libcrux's key holds the encoding, which is all it has; each call
      //   decodes it and expands the matrix.
      // AWS-LC has no deterministic signing API. Checks compare each
      // signature with another library's.
      let mut group = c.benchmark_group(concat!($name, "/sign/reuse-key-deterministic"));
      group.bench_function(
        "rscrypto",
        checked(
          || {
            prepared_secret
              .sign_deterministic(black_box(&message), black_box(empty))
              .expect("sign")
          },
          |output| assert_eq!(output.as_bytes().as_slice(), empty_signature.as_slice()),
        ),
      );
      group.bench_function(
        "rustcrypto",
        checked(
          || {
            external
              .sign_deterministic(black_box(&message), black_box(empty))
              .expect("sign")
              .encode()
          },
          |output| assert_eq!(output.as_slice(), libcrux_empty_signature.as_ref().as_slice()),
        ),
      );
      group.bench_function(
        "libcrux",
        checked(
          || {
            libcrux_ml_dsa::$libcrux::sign(
              black_box(&libcrux.signing_key),
              black_box(&message),
              black_box(empty),
              [0; 32],
            )
            .expect("sign")
          },
          |output| assert_eq!(output.as_ref().as_slice(), empty_signature.as_slice()),
        ),
      );
      group.finish();

      with_aws_lc! {
        // Randomized signing under a reused secret key, with 32 fresh bytes
        // from AWS-LC's RNG in every timed call. Keys are as in
        // `sign/reuse-key-deterministic`, and:
        // - AWS-LC's `PqdsaKeyPair` holds the encoded key; each call unpacks
        //   it and expands the matrix.
        // - `rscrypto-compact` decodes its compact key on each call, like
        //   AWS-LC, and expands matrix rows on each rejection attempt.
        // RustCrypto's randomized API needs its `rand_core` feature, which is
        // off, so its row passes the randomness and the FIPS 204 prefix of an
        // empty context to Sign_internal. libcrux takes the randomness as its
        // signing input. Rejection counts vary with the randomness, so rows
        // measure latency distributions. Each signature must verify under
        // another library.
        let fill =
          |out: &mut [u8]| aws_lc_rs::rand::fill(out).map_err(|_| rscrypto::MlDsaError::RandomGenerationFailed);
        let mut group = c.benchmark_group(concat!($name, "/sign/reuse-key-randomized"));
        group.bench_function(
          "rscrypto",
          checked(
            || black_box(prepared_secret.sign_with(black_box(&message), black_box(empty), fill).expect("sign")),
            |output| {
              aws_public
                .verify_sig(&message, output.as_bytes())
                .expect("AWS-LC verifies rscrypto");
            },
          ),
        );
        group.bench_function(
          "rscrypto-compact",
          checked(
            || black_box(secret.sign_with(black_box(&message), black_box(empty), fill).expect("sign")),
            |output| {
              aws_public
                .verify_sig(&message, output.as_bytes())
                .expect("AWS-LC verifies rscrypto");
            },
          ),
        );
        group.bench_function(
          "rustcrypto",
          checked(
            || {
              let mut random = [0; 32];
              aws_lc_rs::rand::fill(&mut random).expect("randomness");
              let prefix = [0, 0];
              black_box(
                external
                  .sign_internal(&[prefix.as_slice(), black_box(message.as_slice())], &random.into())
                  .encode(),
              )
            },
            |output| {
              aws_public
                .verify_sig(&message, output.as_slice())
                .expect("AWS-LC verifies RustCrypto");
            },
          ),
        );
        group.bench_function(
          "libcrux",
          checked(
            || {
              let mut random = [0; 32];
              aws_lc_rs::rand::fill(&mut random).expect("randomness");
              libcrux_ml_dsa::$libcrux::sign(
                black_box(&libcrux.signing_key),
                black_box(&message),
                black_box(empty),
                random,
              )
              .expect("sign")
            },
            |output| {
              aws_public
                .verify_sig(&message, output.as_ref())
                .expect("AWS-LC verifies libcrux");
            },
          ),
        );
        group.bench_function(
          "aws-lc",
          checked(
            || {
              let mut out = [0; rscrypto::$signature::LENGTH];
              aws_secret.sign(black_box(&message), &mut out).expect("sign");
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
      }
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
  MlDsa44PublicKey,
  MlDsa44SecretKey,
  MlDsa44Signature,
  MlDsa44,
  ml_dsa_44,
  ML_DSA_44_SIGNING,
  ML_DSA_44
);
profile!(
  mldsa65,
  corpus65,
  "mldsa65",
  MlDsa65,
  MlDsa65PublicKey,
  MlDsa65SecretKey,
  MlDsa65Signature,
  MlDsa65,
  ml_dsa_65,
  ML_DSA_65_SIGNING,
  ML_DSA_65
);
profile!(
  mldsa87,
  corpus87,
  "mldsa87",
  MlDsa87,
  MlDsa87PublicKey,
  MlDsa87SecretKey,
  MlDsa87Signature,
  MlDsa87,
  ml_dsa_87,
  ML_DSA_87_SIGNING,
  ML_DSA_87
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
      // RustCrypto's Sign_internal over M', with tiny-keccak's SHAKE128, is
      // independent of libcrux; it is the oracle of the libcrux rows.
      let shake128_oracle = {
        use tiny_keccak::{Hasher as _, Xof as _};

        let mut digest = [0; 32];
        let mut shake = tiny_keccak::Shake::v128();
        shake.update(&message);
        shake.squeeze(&mut digest);
        let formatted = [&prefix[..], context, &SHAKE128_OID, &digest].concat();
        external
          .sign_internal(&[formatted.as_slice()], &[0; 32].into())
          .encode()
      };

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
          |output| assert_eq!(output.as_ref().as_slice(), shake128_oracle.as_slice()),
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
          |result| {
            result.expect("verify");
            assert_eq!(libcrux_signature.as_ref().as_slice(), shake128_oracle.as_slice());
            libcrux_ml_dsa::$libcrux::verify_pre_hashed_shake128(
              &libcrux.verification_key,
              &message,
              context,
              &libcrux_ml_dsa::MLDSASignature::new(shake128_tampered.to_bytes()),
            )
            .expect_err("tampered signature must not verify");
          },
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
    mldsa44, mldsa65, mldsa87, corpus44, corpus65, corpus87, prehash44, prehash65, prehash87,
  ]);
}
