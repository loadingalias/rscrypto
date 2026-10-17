//! SLH-DSA production workloads for all 12 FIPS 205 parameter sets, with
//! RustCrypto `slh-dsa` and `fips205` comparison rows.
//!
//! Fixtures are public, deterministic benchmark keys. Messages are 32 bytes
//! with an empty context, as the standard protocols use.
//!
//! HashSLH-DSA rows sign and verify the same message under each set's RFC 9909
//! profile. Each implementation hashes the message with that profile's
//! function (SHA-256, SHA-512, SHAKE128, or SHAKE256) inside timing. Only
//! `fips205` implements HashSLH-DSA, so it is the sole comparison there.
//!
//! Key generation, pure and HashSLH-DSA signing, and secret-key import compute
//! whole hypertree layers, so those groups use flat sampling: every sample
//! runs the same iteration count, instead of Criterion's linear ramp.
//!
//! Each algorithm builds its keys and reference signature when its first case
//! runs, and every row checks its output against another implementation before
//! timing. Listing cases or measuring another algorithm skips that work.

#[path = "common/criterion.rs"]
mod bench_config;

use core::hint::black_box;

use bench_config::{checked, checked_with};
use criterion::{Bencher, Criterion, SamplingMode};
use fips205::{
  Ph,
  traits::{KeyGen as _, SerDes as _, Signer as _, Verifier as _},
};

const MESSAGE: [u8; 32] = [0xa5; 32];

/// Builds a row's routine when Criterion first runs the row, so listing builds
/// no fixture.
fn deferred<R: FnMut(&mut Bencher<'_>)>(build: impl FnOnce() -> R) -> impl FnMut(&mut Bencher<'_>) {
  let mut build = Some(build);
  let mut routine = None;
  move |b| {
    let built = routine.get_or_insert_with(|| (build.take().expect("a row builds its routine once"))());
    built(b);
  }
}

macro_rules! parameter_set {
  ($function:ident, $name:literal, $profile:ident, $secret:ident, $rustcrypto:ident, $fips:ident, $n:literal) => {
    fn $function(c: &mut Criterion) {
      use rscrypto::{$profile, $secret};
      use rustcrypto_slh_dsa as oracle;

      if !bench_config::selected(concat!($name, "/")) {
        return;
      }
      let seeds: [u8; 3 * $n] =
        core::array::from_fn(|index| u8::try_from(index).expect("seed index").wrapping_mul(0x9d) ^ 0x42);
      let keygen = || {
        $profile::generate_keypair(|out| {
          out.copy_from_slice(black_box(&seeds));
          Ok(())
        })
        .expect("key generation")
      };
      // fips205 checks both key-generation rows.
      let reference_keys = || {
        let (fields, _) = seeds.as_chunks::<$n>();
        let (public, secret) = fips205::$fips::KG::keygen_with_seeds(&fields[0], &fields[1], &fields[2]);
        (public.into_bytes(), secret.into_bytes())
      };
      // Generic fields keep each library's key and signature types inferred.
      struct Fixture<Public, Secret, Encoded, External, ExternalPublic, ExternalSignature, FipsSecret, FipsPublic> {
        public: Public,
        secret: Secret,
        encoded: Encoded,
        external: External,
        external_public: ExternalPublic,
        external_signature: ExternalSignature,
        fips_secret: FipsSecret,
        fips_public: FipsPublic,
        signature: [u8; $profile::SIGNATURE_LENGTH],
      }
      let cell = core::cell::OnceCell::new();
      let fixture = || {
        cell.get_or_init(|| {
          let (public, secret) = keygen();
          let encoded = secret.expose_secret();
          let external = oracle::SigningKey::<oracle::$rustcrypto>::slh_keygen_internal(
            &seeds[..$n],
            &seeds[$n..2 * $n],
            &seeds[2 * $n..],
          );
          let external_public: oracle::VerifyingKey<oracle::$rustcrypto> = external.as_ref().clone();
          let fips_secret = fips205::$fips::PrivateKey::try_from_bytes(encoded.as_bytes()).expect("fips205 secret key");
          let fips_public = fips205::$fips::PublicKey::try_from_bytes(&public.to_bytes()).expect("fips205 public key");
          // RustCrypto's deterministic signature is the reference of the
          // signing rows and the input of the verification rows.
          let external_signature = external
            .try_sign_with_context(&MESSAGE, b"", None)
            .expect("RustCrypto signature");
          let mut signature = [0; $profile::SIGNATURE_LENGTH];
          signature.copy_from_slice(&external_signature.to_bytes());
          Fixture {
            public,
            secret,
            encoded,
            external,
            external_public,
            external_signature,
            fips_secret,
            fips_public,
            signature,
          }
        })
      };

      // The top XMSS tree, typed outputs, and destruction are timed; no OS
      // entropy. fips205 generates keys only from an RNG, so it has no row.
      let mut group = c.benchmark_group(concat!($name, "/keygen/fixed-seeds"));
      group.sampling_mode(SamplingMode::Flat);
      group.bench_function(
        "rscrypto",
        checked(
          || black_box(keygen()),
          |(public, secret)| {
            let (expected_public, expected_secret) = reference_keys();
            assert_eq!(public.as_bytes().as_slice(), expected_public.as_slice());
            assert_eq!(secret.expose_secret().as_bytes().as_slice(), expected_secret.as_slice());
          },
        ),
      );
      group.bench_function(
        "rustcrypto",
        checked(
          || {
            black_box(oracle::SigningKey::<oracle::$rustcrypto>::slh_keygen_internal(
              black_box(&seeds[..$n]),
              &seeds[$n..2 * $n],
              &seeds[2 * $n..],
            ))
          },
          |key| assert_eq!(key.to_bytes().as_slice(), reference_keys().1.as_slice()),
        ),
      );
      group.finish();

      // rscrypto writes into a caller buffer; the others return the signature
      // by value. fips205 checks RustCrypto, the reference.
      let mut group = c.benchmark_group(concat!($name, "/sign/deterministic-32"));
      group.sampling_mode(SamplingMode::Flat);
      group.bench_function(
        "rscrypto",
        deferred(|| {
          let Fixture {
            secret,
            signature: reference,
            ..
          } = fixture();
          checked_with(
            [0; $profile::SIGNATURE_LENGTH],
            move |signature| {
              secret
                .sign_deterministic(black_box(&MESSAGE), b"", signature)
                .expect("signature");
              black_box(&*signature);
            },
            move |(), signature| assert_eq!(signature, reference),
          )
        }),
      );
      group.bench_function(
        "rustcrypto",
        deferred(|| {
          let Fixture {
            external, fips_secret, ..
          } = fixture();
          checked(
            move || {
              black_box(
                external
                  .try_sign_with_context(black_box(&MESSAGE), b"", None)
                  .expect("signature"),
              )
            },
            move |signature| {
              let expected = fips_secret.try_sign(&MESSAGE, b"", false).expect("fips205 signature");
              assert_eq!(signature.to_bytes().as_slice(), expected.as_slice());
            },
          )
        }),
      );
      group.bench_function(
        "fips205",
        deferred(|| {
          let Fixture {
            fips_secret,
            signature: reference,
            ..
          } = fixture();
          checked(
            move || {
              black_box(
                fips_secret
                  .try_sign(black_box(&MESSAGE), b"", false)
                  .expect("signature"),
              )
            },
            move |signature| assert_eq!(&signature, reference),
          )
        }),
      );
      group.finish();

      // RustCrypto's signature value is decoded outside timing.
      let mut group = c.benchmark_group(concat!($name, "/verify/32"));
      group.bench_function(
        "rscrypto",
        deferred(|| {
          let Fixture { public, signature, .. } = fixture();
          checked(
            move || public.verify(black_box(&MESSAGE), black_box(signature)),
            move |result| {
              result.expect("verification");
              let mut tampered = *signature;
              tampered[0] ^= 1;
              public
                .verify(&MESSAGE, &tampered)
                .expect_err("tampered signature must not verify");
            },
          )
        }),
      );
      group.bench_function(
        "rustcrypto",
        deferred(|| {
          let Fixture {
            external_public,
            external_signature,
            ..
          } = fixture();
          checked(
            move || external_public.try_verify_with_context(black_box(&MESSAGE), b"", black_box(external_signature)),
            |result| result.expect("verification"),
          )
        }),
      );
      group.bench_function(
        "fips205",
        deferred(|| {
          let Fixture {
            fips_public, signature, ..
          } = fixture();
          checked(
            move || fips_public.verify(black_box(&MESSAGE), black_box(signature), b""),
            |accepted| assert!(accepted),
          )
        }),
      );
      group.finish();

      // Import regenerates PK.root (FIPS 205 Section 3.1), so it costs one key
      // generation. The comparison libraries import without that check, so
      // they have no equivalent row.
      let mut group = c.benchmark_group(concat!($name, "/import/raw-secret"));
      group.sampling_mode(SamplingMode::Flat);
      group.bench_function(
        "rscrypto",
        deferred(|| {
          let Fixture { encoded, external, .. } = fixture();
          checked(
            move || black_box($secret::try_from_slice(black_box(encoded.as_bytes())).expect("import")),
            move |secret| {
              assert_eq!(
                secret.expose_secret().as_bytes().as_slice(),
                external.to_bytes().as_slice()
              );
              // A PK.root that does not belong to SK.seed and PK.seed fails.
              let mut tampered = *encoded.as_bytes();
              *tampered.last_mut().expect("nonempty key") ^= 1;
              $secret::try_from_slice(&tampered).expect_err("a foreign PK.root must be rejected");
            },
          )
        }),
      );
      group.finish();
    }
  };
}

parameter_set!(
  sha2_128s,
  "slhdsa-sha2-128s",
  SlhDsaSha2_128s,
  SlhDsaSha2_128sSecretKey,
  Sha2_128s,
  slh_dsa_sha2_128s,
  16
);
parameter_set!(
  sha2_128f,
  "slhdsa-sha2-128f",
  SlhDsaSha2_128f,
  SlhDsaSha2_128fSecretKey,
  Sha2_128f,
  slh_dsa_sha2_128f,
  16
);
parameter_set!(
  sha2_192s,
  "slhdsa-sha2-192s",
  SlhDsaSha2_192s,
  SlhDsaSha2_192sSecretKey,
  Sha2_192s,
  slh_dsa_sha2_192s,
  24
);
parameter_set!(
  sha2_192f,
  "slhdsa-sha2-192f",
  SlhDsaSha2_192f,
  SlhDsaSha2_192fSecretKey,
  Sha2_192f,
  slh_dsa_sha2_192f,
  24
);
parameter_set!(
  sha2_256s,
  "slhdsa-sha2-256s",
  SlhDsaSha2_256s,
  SlhDsaSha2_256sSecretKey,
  Sha2_256s,
  slh_dsa_sha2_256s,
  32
);
parameter_set!(
  sha2_256f,
  "slhdsa-sha2-256f",
  SlhDsaSha2_256f,
  SlhDsaSha2_256fSecretKey,
  Sha2_256f,
  slh_dsa_sha2_256f,
  32
);
parameter_set!(
  shake_128s,
  "slhdsa-shake-128s",
  SlhDsaShake128s,
  SlhDsaShake128sSecretKey,
  Shake128s,
  slh_dsa_shake_128s,
  16
);
parameter_set!(
  shake_128f,
  "slhdsa-shake-128f",
  SlhDsaShake128f,
  SlhDsaShake128fSecretKey,
  Shake128f,
  slh_dsa_shake_128f,
  16
);
parameter_set!(
  shake_192s,
  "slhdsa-shake-192s",
  SlhDsaShake192s,
  SlhDsaShake192sSecretKey,
  Shake192s,
  slh_dsa_shake_192s,
  24
);
parameter_set!(
  shake_192f,
  "slhdsa-shake-192f",
  SlhDsaShake192f,
  SlhDsaShake192fSecretKey,
  Shake192f,
  slh_dsa_shake_192f,
  24
);
parameter_set!(
  shake_256s,
  "slhdsa-shake-256s",
  SlhDsaShake256s,
  SlhDsaShake256sSecretKey,
  Shake256s,
  slh_dsa_shake_256s,
  32
);
parameter_set!(
  shake_256f,
  "slhdsa-shake-256f",
  SlhDsaShake256f,
  SlhDsaShake256fSecretKey,
  Shake256f,
  slh_dsa_shake_256f,
  32
);

macro_rules! prehash_set {
  ($function:ident, $name:literal, $profile:ident, $fips:ident, $ph:ident, $prehash:literal, $n:literal) => {
    fn $function(c: &mut Criterion) {
      use rscrypto::$profile;

      if !bench_config::selected(concat!($name, "/hash-")) {
        return;
      }
      let seeds: [u8; 3 * $n] =
        core::array::from_fn(|index| u8::try_from(index).expect("seed index").wrapping_mul(0x9d) ^ 0x42);
      struct Fixture<Public, Secret, FipsSecret, FipsPublic> {
        public: Public,
        secret: Secret,
        fips_secret: FipsSecret,
        fips_public: FipsPublic,
        signature: [u8; $profile::SIGNATURE_LENGTH],
      }
      let cell = core::cell::OnceCell::new();
      let fixture = || {
        cell.get_or_init(|| {
          let (public, secret) = $profile::generate_keypair(|out| {
            out.copy_from_slice(&seeds);
            Ok(())
          })
          .expect("key generation");
          let fips_secret =
            fips205::$fips::PrivateKey::try_from_bytes(secret.expose_secret().as_bytes()).expect("fips205 secret key");
          let fips_public = fips205::$fips::PublicKey::try_from_bytes(&public.to_bytes()).expect("fips205 public key");
          // fips205 is the only other HashSLH-DSA. Its deterministic signature
          // is the reference of the rscrypto signing row and the input of both
          // verification rows; rscrypto checks the fips205 signing row.
          let signature = fips_secret
            .try_hash_sign(&MESSAGE, b"", &Ph::$ph, false)
            .expect("fips205 signature");
          Fixture {
            public,
            secret,
            fips_secret,
            fips_public,
            signature,
          }
        })
      };

      // As in the pure rows, rscrypto writes into a caller buffer and fips205
      // returns the signature by value.
      let mut group = c.benchmark_group(concat!($name, "/hash-sign/", $prehash, "/deterministic-32"));
      group.sampling_mode(SamplingMode::Flat);
      group.bench_function(
        "rscrypto",
        deferred(|| {
          let Fixture {
            secret,
            signature: reference,
            ..
          } = fixture();
          checked_with(
            [0; $profile::SIGNATURE_LENGTH],
            move |signature| {
              secret
                .sign_deterministic(black_box(&MESSAGE), b"", signature)
                .expect("signature");
              black_box(&*signature);
            },
            move |(), signature| assert_eq!(signature, reference),
          )
        }),
      );
      group.bench_function(
        "fips205",
        deferred(|| {
          let Fixture {
            secret, fips_secret, ..
          } = fixture();
          checked(
            move || {
              black_box(
                fips_secret
                  .try_hash_sign(black_box(&MESSAGE), b"", &Ph::$ph, false)
                  .expect("signature"),
              )
            },
            move |signature| {
              let mut expected = [0; $profile::SIGNATURE_LENGTH];
              secret
                .sign_deterministic(&MESSAGE, b"", &mut expected)
                .expect("signature");
              assert_eq!(signature, expected);
            },
          )
        }),
      );
      group.finish();

      let mut group = c.benchmark_group(concat!($name, "/hash-verify/", $prehash, "/32"));
      group.bench_function(
        "rscrypto",
        deferred(|| {
          let Fixture { public, signature, .. } = fixture();
          checked(
            move || public.verify(black_box(&MESSAGE), black_box(signature)),
            move |result| {
              result.expect("verification");
              let mut tampered = *signature;
              tampered[0] ^= 1;
              public
                .verify(&MESSAGE, &tampered)
                .expect_err("tampered signature must not verify");
            },
          )
        }),
      );
      group.bench_function(
        "fips205",
        deferred(|| {
          let Fixture {
            fips_public, signature, ..
          } = fixture();
          checked(
            move || fips_public.hash_verify(black_box(&MESSAGE), black_box(signature), b"", &Ph::$ph),
            |accepted| assert!(accepted),
          )
        }),
      );
      group.finish();
    }
  };
}

prehash_set!(
  hash_sha2_128s,
  "slhdsa-sha2-128s",
  HashSlhDsaSha2_128sWithSha256,
  slh_dsa_sha2_128s,
  SHA256,
  "sha256",
  16
);
prehash_set!(
  hash_sha2_128f,
  "slhdsa-sha2-128f",
  HashSlhDsaSha2_128fWithSha256,
  slh_dsa_sha2_128f,
  SHA256,
  "sha256",
  16
);
prehash_set!(
  hash_sha2_192s,
  "slhdsa-sha2-192s",
  HashSlhDsaSha2_192sWithSha512,
  slh_dsa_sha2_192s,
  SHA512,
  "sha512",
  24
);
prehash_set!(
  hash_sha2_192f,
  "slhdsa-sha2-192f",
  HashSlhDsaSha2_192fWithSha512,
  slh_dsa_sha2_192f,
  SHA512,
  "sha512",
  24
);
prehash_set!(
  hash_sha2_256s,
  "slhdsa-sha2-256s",
  HashSlhDsaSha2_256sWithSha512,
  slh_dsa_sha2_256s,
  SHA512,
  "sha512",
  32
);
prehash_set!(
  hash_sha2_256f,
  "slhdsa-sha2-256f",
  HashSlhDsaSha2_256fWithSha512,
  slh_dsa_sha2_256f,
  SHA512,
  "sha512",
  32
);
prehash_set!(
  hash_shake_128s,
  "slhdsa-shake-128s",
  HashSlhDsaShake128sWithShake128,
  slh_dsa_shake_128s,
  SHAKE128,
  "shake128",
  16
);
prehash_set!(
  hash_shake_128f,
  "slhdsa-shake-128f",
  HashSlhDsaShake128fWithShake128,
  slh_dsa_shake_128f,
  SHAKE128,
  "shake128",
  16
);
prehash_set!(
  hash_shake_192s,
  "slhdsa-shake-192s",
  HashSlhDsaShake192sWithShake256,
  slh_dsa_shake_192s,
  SHAKE256,
  "shake256",
  24
);
prehash_set!(
  hash_shake_192f,
  "slhdsa-shake-192f",
  HashSlhDsaShake192fWithShake256,
  slh_dsa_shake_192f,
  SHAKE256,
  "shake256",
  24
);
prehash_set!(
  hash_shake_256s,
  "slhdsa-shake-256s",
  HashSlhDsaShake256sWithShake256,
  slh_dsa_shake_256s,
  SHAKE256,
  "shake256",
  32
);
prehash_set!(
  hash_shake_256f,
  "slhdsa-shake-256f",
  HashSlhDsaShake256fWithShake256,
  slh_dsa_shake_256f,
  SHAKE256,
  "shake256",
  32
);

fn benchmarks() {
  bench_config::run(&[
    sha2_128s,
    shake_128s,
    sha2_128f,
    shake_128f,
    sha2_192s,
    shake_192s,
    sha2_192f,
    shake_192f,
    sha2_256s,
    shake_256s,
    sha2_256f,
    shake_256f,
    hash_sha2_128s,
    hash_shake_128s,
    hash_sha2_128f,
    hash_shake_128f,
    hash_sha2_192s,
    hash_shake_192s,
    hash_sha2_192f,
    hash_shake_192f,
    hash_sha2_256s,
    hash_shake_256s,
    hash_sha2_256f,
    hash_shake_256f,
  ]);
}

fn main() {
  // A 49,856-byte signature buffer and the comparison libraries' signing
  // frames do not fit the 1 MiB Windows main-thread stack comfortably; run
  // on a thread with an explicit stack.
  std::thread::Builder::new()
    .stack_size(8 << 20)
    .spawn(benchmarks)
    .expect("spawn benchmark thread")
    .join()
    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}
