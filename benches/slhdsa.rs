//! SLH-DSA production workloads for all 12 FIPS 205 parameter sets, with
//! RustCrypto `slh-dsa` and `fips205` comparison rows.
//!
//! Fixtures are public, deterministic benchmark keys. Messages are 32 bytes
//! with an empty context, as the standard protocols use. HashSLH-DSA differs
//! only in hashing the message outside the trees, so it has no separate rows.
//!
//! Key generation, signing, and secret-key import compute whole hypertree
//! layers, so those groups use flat sampling: every sample runs the same
//! iteration count, instead of Criterion's linear ramp.

#[path = "common/criterion.rs"]
mod bench_config;

use core::hint::black_box;
use criterion::{Criterion, SamplingMode};
use fips205::traits::{SerDes as _, Signer as _, Verifier as _};

const MESSAGE: [u8; 32] = [0xa5; 32];

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
      let (public, secret) = keygen();
      let encoded = secret.expose_secret();

      // Every implementation must produce the same keys and deterministic
      // signature, and accept that signature.
      let external = oracle::SigningKey::<oracle::$rustcrypto>::slh_keygen_internal(
        &seeds[..$n],
        &seeds[$n..2 * $n],
        &seeds[2 * $n..],
      );
      assert_eq!(external.to_bytes().as_slice(), encoded.as_bytes());
      let external_public: oracle::VerifyingKey<oracle::$rustcrypto> = external.as_ref().clone();
      let fips_secret = fips205::$fips::PrivateKey::try_from_bytes(encoded.as_bytes()).expect("fips205 secret key");
      let fips_public = fips205::$fips::PublicKey::try_from_bytes(&public.to_bytes()).expect("fips205 public key");
      let mut signature = [0; $profile::SIGNATURE_LENGTH];
      secret
        .sign_deterministic(&MESSAGE, b"", &mut signature)
        .expect("signature");
      let external_signature = external
        .try_sign_with_context(&MESSAGE, b"", None)
        .expect("RustCrypto signature");
      assert_eq!(external_signature.to_bytes().as_slice(), signature.as_slice());
      assert_eq!(
        fips_secret.try_sign(&MESSAGE, b"", false).expect("fips205 signature"),
        signature
      );
      public.verify(&MESSAGE, &signature).expect("verification");
      external_public
        .try_verify_with_context(&MESSAGE, b"", &external_signature)
        .expect("RustCrypto verification");
      assert!(fips_public.verify(&MESSAGE, &signature, b""));

      // The top XMSS tree, typed outputs, and destruction are timed; no OS
      // entropy. fips205 generates keys only from an RNG, so it has no row.
      let mut group = c.benchmark_group(concat!($name, "/keygen/fixed-seeds"));
      group.sampling_mode(SamplingMode::Flat);
      group.bench_function("rscrypto", |b| b.iter(|| black_box(keygen())));
      group.bench_function("rustcrypto", |b| {
        b.iter(|| {
          black_box(oracle::SigningKey::<oracle::$rustcrypto>::slh_keygen_internal(
            black_box(&seeds[..$n]),
            &seeds[$n..2 * $n],
            &seeds[2 * $n..],
          ))
        })
      });
      group.finish();

      // rscrypto writes into a caller buffer; the others return the signature
      // by value.
      let mut group = c.benchmark_group(concat!($name, "/sign/deterministic-32"));
      group.sampling_mode(SamplingMode::Flat);
      group.bench_function("rscrypto", |b| {
        b.iter(|| {
          secret
            .sign_deterministic(black_box(&MESSAGE), b"", &mut signature)
            .expect("signature");
          black_box(&signature);
        })
      });
      group.bench_function("rustcrypto", |b| {
        b.iter(|| {
          black_box(
            external
              .try_sign_with_context(black_box(&MESSAGE), b"", None)
              .expect("signature"),
          )
        })
      });
      group.bench_function("fips205", |b| {
        b.iter(|| {
          black_box(
            fips_secret
              .try_sign(black_box(&MESSAGE), b"", false)
              .expect("signature"),
          )
        })
      });
      group.finish();

      // RustCrypto's signature value is decoded outside timing.
      let mut group = c.benchmark_group(concat!($name, "/verify/32"));
      group.bench_function("rscrypto", |b| {
        b.iter(|| {
          public
            .verify(black_box(&MESSAGE), black_box(&signature))
            .expect("verification")
        })
      });
      group.bench_function("rustcrypto", |b| {
        b.iter(|| {
          external_public
            .try_verify_with_context(black_box(&MESSAGE), b"", black_box(&external_signature))
            .expect("verification")
        })
      });
      group.bench_function("fips205", |b| {
        b.iter(|| assert!(fips_public.verify(black_box(&MESSAGE), black_box(&signature), b"")))
      });
      group.finish();

      // Import regenerates PK.root (FIPS 205 Section 3.1), so it costs one key
      // generation. The comparison libraries import without that check, so
      // they have no equivalent row.
      let mut group = c.benchmark_group(concat!($name, "/import/raw-secret"));
      group.sampling_mode(SamplingMode::Flat);
      group.bench_function("rscrypto", |b| {
        b.iter(|| black_box($secret::try_from_slice(black_box(encoded.as_bytes())).expect("import")))
      });
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

fn benchmarks() {
  bench_config::run(&[
    sha2_128s, shake_128s, sha2_128f, shake_128f, sha2_192s, shake_192s, sha2_192f, shake_192f, sha2_256s, shake_256s,
    sha2_256f, shake_256f,
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
