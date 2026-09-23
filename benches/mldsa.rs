//! ML-DSA production workloads. Fixtures use public, deterministic benchmark keys.

#[path = "common/criterion.rs"]
mod bench_config;

use core::hint::black_box;
use criterion::{BenchmarkId, Criterion};
use rscrypto::{MlDsa44, MlDsa65, MlDsa87};
use rustcrypto_ml_dsa as oracle;

macro_rules! profile {
  ($function:ident, $corpus:ident, $name:literal, $profile:ident, $secret:ident, $oracle:ident) => {
    fn $function(c: &mut Criterion) {
      if !bench_config::selected(concat!($name, "/")) {
        return;
      }
      let seed = [0x42; 32];
      let (public, secret) = $profile::keypair_from_seed(&seed).expect("benchmark key generation");
      let external = oracle::ExpandedSigningKey::<oracle::$oracle>::from_seed(&seed.into());
      let external_public = external.verifying_key();
      let prepared_secret = secret.prepare().expect("prepare secret");
      let prepared_public = public.prepare().expect("prepare public");
      eprintln!(
        "{} retained bytes: compact secret={}, prepared secret including borrowed key={} vs oracle={}, prepared public including borrowed key={} vs oracle={}",
        $name,
        core::mem::size_of_val(&secret),
        core::mem::size_of_val(&secret).strict_add(core::mem::size_of_val(&prepared_secret)),
        core::mem::size_of_val(&external),
        core::mem::size_of_val(&public).strict_add(core::mem::size_of_val(&prepared_public)),
        core::mem::size_of_val(&external_public),
      );
      let message = [0xa5; 32];
      let context = b"rscrypto benchmark";
      let signature = secret
        .sign_deterministic(&message, context)
        .expect("benchmark signature");
      let expected = external
        .sign_deterministic(&message, context)
        .expect("oracle signature");
      assert_eq!(signature.as_bytes().as_slice(), expected.encode().as_slice());
      assert_eq!(public.as_bytes().as_slice(), external_public.encode().as_slice());
      let prepared_signature = prepared_secret
        .sign_deterministic(&message, context)
        .expect("prepared benchmark signature");
      assert_eq!(prepared_signature.as_bytes(), signature.as_bytes());
      public
        .verify_with_context(&message, context, &signature)
        .expect("benchmark verification");
      prepared_public
        .verify_with_context(&message, context, &signature)
        .expect("prepared benchmark verification");
      assert!(external_public.verify_with_context(&message, context, &expected));
      // Seed expansion, output encoding and destruction are timed. No OS entropy.
      let mut group = c.benchmark_group(concat!($name, "/keygen/seed-encoded"));
      group.bench_function("rscrypto", |b| {
        b.iter(|| {
          let (pk, sk) = $profile::keypair_from_seed(black_box(&seed)).expect("key generation");
          black_box((pk.to_bytes(), sk.expose_secret()));
        })
      });
      group.bench_function("rustcrypto", |b| {
        b.iter(|| {
          let sk = oracle::ExpandedSigningKey::<oracle::$oracle>::from_seed(black_box(&seed.into()));
          #[expect(deprecated, reason = "the measured output is the standard expanded wire encoding")]
          let encoded = sk.to_expanded();
          let encoded = rscrypto::SecretBytes::<{ rscrypto::$secret::LENGTH }>::new(encoded.into());
          black_box((sk.verifying_key().encode(), encoded));
        })
      });
      group.finish();
      // Preparation constructs and drops the complete retained owner. Keep the
      // encoded key outside timing, as in the signing and verification rows.
      let mut group = c.benchmark_group(concat!($name, "/prepare/encoded-secret"));
      group.bench_function("rscrypto", |b| {
        b.iter(|| black_box(secret.prepare().expect("prepare secret")))
      });
      group.finish();
      let mut group = c.benchmark_group(concat!($name, "/prepare/encoded-public"));
      group.bench_function("rscrypto", |b| {
        b.iter(|| black_box(public.prepare().expect("prepare public")))
      });
      group.finish();
      // Compact keys and explicit prepared keys have distinct memory contracts.
      // Prepared rows retain transformed polynomials and a complete matrix;
      // neither library's preparation is inside those timings.
      let mut group = c.benchmark_group(concat!($name, "/sign/encoded-key-deterministic"));
      group.bench_function("rscrypto", |b| {
        b.iter(|| {
          black_box(
            secret
              .sign_deterministic(black_box(&message), black_box(context))
              .expect("sign"),
          );
        })
      });
      group.finish();
      let mut group = c.benchmark_group(concat!($name, "/sign/prepared-key-deterministic"));
      group.bench_function("rscrypto", |b| {
        b.iter(|| {
          black_box(
            prepared_secret
              .sign_deterministic(black_box(&message), black_box(context))
              .expect("prepared sign"),
          );
        })
      });
      group.bench_function("rustcrypto", |b| {
        b.iter(|| {
          black_box(
            external
              .sign_deterministic(black_box(&message), black_box(context))
              .expect("sign")
              .encode(),
          );
        })
      });
      group.finish();
      // Signature parsing is outside timing; message hashing and matrix expansion
      // are charged to the production verifier when its API performs them.
      let mut group = c.benchmark_group(concat!($name, "/verify/encoded-key"));
      group.bench_function("rscrypto", |b| {
        b.iter(|| {
          black_box(public.verify_with_context(black_box(&message), black_box(context), black_box(&signature)))
            .expect("verify");
        })
      });
      group.finish();
      let mut group = c.benchmark_group(concat!($name, "/verify/prepared-key"));
      group.bench_function("rscrypto", |b| {
        b.iter(|| {
          black_box(prepared_public.verify_with_context(
            black_box(&message),
            black_box(context),
            black_box(&signature),
          ))
          .expect("prepared verify");
        })
      });
      group.bench_function("rustcrypto", |b| {
        b.iter(|| {
          assert!(black_box(external_public.verify_with_context(
            black_box(&message),
            black_box(context),
            black_box(&expected)
          )));
        })
      });
      group.finish();
    }

    fn $corpus(c: &mut Criterion) {
      if !bench_config::selected(concat!($name, "/corpus/")) {
        return;
      }
      // Each row measures one fixture so rejection tails cannot disappear in
      // an aggregate mean. Both libraries receive identical seeds and inputs.
      for (index, (message_len, context_len)) in [
        (0, 0), (32, 1), (135, 254), (136, 255),
        (137, 17), (1024, 0), (4096, 17), (65536, 255),
      ].into_iter().enumerate() {
        let seed_byte = u8::try_from(index.strict_add(1)).expect("eight corpus seeds");
        let seed = [seed_byte; 32];
        let (public, secret) = $profile::keypair_from_seed(&seed).expect("corpus key generation");
        let prepared = secret.prepare().expect("corpus preparation");
        let external = oracle::ExpandedSigningKey::<oracle::$oracle>::from_seed(&seed.into());
        let message = vec![seed_byte ^ 0xa5; message_len];
        let context = vec![0x55; context_len];
        let random = [seed_byte ^ 0x3c; 32];
        let prefix = [0, u8::try_from(context_len).expect("bounded context")];
        let deterministic = prepared.sign_deterministic(&message, &context).expect("corpus signature");
        let expected = external.sign_deterministic(&message, &context).expect("oracle signature");
        assert_eq!(deterministic.as_bytes().as_slice(), expected.encode().as_slice());
        public.verify_with_context(&message, &context, &deterministic).expect("corpus verification");
        let hedged = prepared.sign_with(&message, &context, |out| {
          out.copy_from_slice(&random);
          Ok(())
        }).expect("hedged corpus signature");
        let expected_hedged = external.sign_internal(&[&prefix, &context, &message], &random.into());
        assert_eq!(hedged.as_bytes().as_slice(), expected_hedged.encode().as_slice());
        public.verify_with_context(&message, &context, &hedged).expect("hedged corpus verification");
        let case = format!("seed{seed_byte}-m{message_len}-c{context_len}");
        let mut group = c.benchmark_group(concat!($name, "/corpus/prepared-deterministic"));
        group.bench_function(BenchmarkId::new("rscrypto", &case), |b| {
          b.iter(|| black_box(prepared.sign_deterministic(black_box(&message), black_box(&context)).expect("sign")))
        });
        group.bench_function(BenchmarkId::new("rustcrypto", &case), |b| {
          b.iter(|| black_box(external.sign_deterministic(black_box(&message), black_box(&context)).expect("sign").encode()))
        });
        group.finish();
        // Entropy is supplied identically from fixed fixture bytes; OS RNG
        // latency is outside both rows. Signature encoding is inside both.
        let mut group = c.benchmark_group(concat!($name, "/corpus/prepared-hedged"));
        group.bench_function(BenchmarkId::new("rscrypto", &case), |b| {
          b.iter(|| black_box(prepared.sign_with(black_box(&message), black_box(&context), |out| {
            out.copy_from_slice(black_box(&random));
            Ok(())
          }).expect("sign")))
        });
        group.bench_function(BenchmarkId::new("rustcrypto", &case), |b| {
          b.iter(|| black_box(external.sign_internal(
            &[black_box(&prefix), black_box(&context), black_box(&message)],
            black_box(&random.into()),
          ).encode()))
        });
        group.finish();
      }
    }
  };
}

profile!(mldsa44, corpus44, "mldsa44", MlDsa44, MlDsa44SecretKey, MlDsa44);
profile!(mldsa65, corpus65, "mldsa65", MlDsa65, MlDsa65SecretKey, MlDsa65);
profile!(mldsa87, corpus87, "mldsa87", MlDsa87, MlDsa87SecretKey, MlDsa87);

#[cfg(all(
  any(unix, windows),
  not(target_arch = "wasm32"),
  not(any(target_arch = "s390x", target_arch = "powerpc64"))
))]
mod aws {
  use super::{Criterion, MlDsa44, MlDsa65, MlDsa87, bench_config, black_box};
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
        let prepared = secret.prepare().expect("prepare secret");
        let prepared_public = public.prepare().expect("prepare public");
        let external = PqdsaKeyPair::from_seed(&aws_lc_rs::signature::$sign, &seed).expect("AWS-LC keygen");
        assert_eq!(external.public_key().as_ref(), public.as_bytes().as_slice());
        assert_eq!(
          external
            .private_key()
            .as_raw_bytes()
            .expect("expanded key")
            .as_ref(),
          secret.expose_secret().as_ref()
        );
        let external_public = UnparsedPublicKey::new(&aws_lc_rs::signature::$verify, external.public_key().as_ref())
          .parse()
          .expect("parse public");
        let signature = prepared.sign_deterministic(&message, &[]).expect("signature");
        external_public
          .verify_sig(&message, signature.as_bytes())
          .expect("AWS-LC verifies rscrypto");
        let mut external_signature = [0u8; rscrypto::$signature::LENGTH];
        assert_eq!(
          external
            .sign(&message, &mut external_signature)
            .expect("AWS-LC signature"),
          external_signature.len()
        );
        let parsed = rscrypto::$signature::try_from_slice(&external_signature).expect("parse signature");
        prepared_public
          .verify(&message, &parsed)
          .expect("rscrypto verifies AWS-LC");
        let randomized = prepared
          .sign_with(&message, &[], |out| {
            aws_lc_rs::rand::fill(out).map_err(|_| rscrypto::MlDsaError::RandomGenerationFailed)
          })
          .expect("randomized signature");
        external_public
          .verify_sig(&message, randomized.as_bytes())
          .expect("randomized interop");

        // Both rows include seed expansion, expanded wire outputs, and key/output
        // destruction. AWS-LC owns heap buffers; rscrypto owns inline arrays.
        let mut group = c.benchmark_group(concat!($name, "/cross/keygen/seed-encoded"));
        group.bench_function("rscrypto", |b| {
          b.iter(|| {
            let (pk, sk) = $profile::keypair_from_seed(black_box(&seed)).expect("keygen");
            black_box((pk.to_bytes(), sk.expose_secret()));
          })
        });
        group.bench_function("aws-lc", |b| {
          b.iter(|| {
            let key = PqdsaKeyPair::from_seed(&aws_lc_rs::signature::$sign, black_box(&seed)).expect("keygen");
            let mut pk = [0u8; rscrypto::$public::LENGTH];
            pk.copy_from_slice(key.public_key().as_ref());
            black_box((pk, key.private_key().as_raw_bytes().expect("expanded key")));
          })
        });
        group.finish();

        // Reusable key owners are prepared outside timing. Both request fresh
        // 32-byte randomness from AWS-LC's RNG and return a complete signature.
        // Random rejection counts vary; these rows measure latency distributions.
        let mut group = c.benchmark_group(concat!($name, "/cross/sign/reused-key-randomized-aws-rng"));
        group.bench_function("rscrypto", |b| {
          b.iter(|| {
            black_box(
              prepared
                .sign_with(black_box(&message), &[], |out| {
                  aws_lc_rs::rand::fill(out).map_err(|_| rscrypto::MlDsaError::RandomGenerationFailed)
                })
                .expect("sign"),
            )
          })
        });
        group.bench_function("aws-lc", |b| {
          b.iter(|| {
            let mut out = [0u8; rscrypto::$signature::LENGTH];
            external.sign(black_box(&message), &mut out).expect("sign");
            black_box(out)
          })
        });
        group.finish();

        // Both retain their public-key owner outside timing and verify the same
        // signature. AWS-LC's API takes encoded signatures; rscrypto takes its
        // validated signature type. Internal work remains charged to each API.
        let mut group = c.benchmark_group(concat!($name, "/cross/verify/reused-key"));
        group.bench_function("rscrypto", |b| {
          b.iter(|| black_box(prepared_public.verify(black_box(&message), black_box(&signature))).expect("verify"))
        });
        group.bench_function("aws-lc", |b| {
          b.iter(|| {
            black_box(external_public.verify_sig(black_box(&message), black_box(signature.as_bytes()))).expect("verify")
          })
        });
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
  let targets: &[fn(&mut Criterion)] = &[
    mldsa44,
    mldsa65,
    mldsa87,
    corpus44,
    corpus65,
    corpus87,
    #[cfg(all(
      any(unix, windows),
      not(target_arch = "wasm32"),
      not(any(target_arch = "s390x", target_arch = "powerpc64"))
    ))]
    aws::run,
  ];
  bench_config::run(targets);
}
