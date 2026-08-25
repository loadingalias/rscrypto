//! AEAD comparison benchmarks: rscrypto vs RustCrypto ecosystem.
//!
//! Measures `encrypt_in_place` (detached tag) for all shipped AEAD primitives
//! across the standard size matrix. Decrypt benchmarks included for the
//! primary primitives to catch asymmetry.

mod common;

use core::hint::black_box;

use aes_gcm::aead::{AeadInOut as _, KeyInit as _};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rscrypto::aead::expert::AeadWithNonce;

#[cfg(all(
  any(unix, windows),
  not(target_arch = "wasm32"),
  not(any(target_arch = "s390x", target_arch = "powerpc64"))
))]
macro_rules! aws_lc_bench {
  ($($tokens:tt)*) => {
    $($tokens)*
  };
}

#[cfg(not(all(
  any(unix, windows),
  not(target_arch = "wasm32"),
  not(any(target_arch = "s390x", target_arch = "powerpc64"))
)))]
macro_rules! aws_lc_bench {
  ($($tokens:tt)*) => {};
}

// Key / nonce material (deterministic, not secret for benchmarking)

const KEY_32: [u8; 32] = [0x42u8; 32];
const NONCE_12: [u8; 12] = [0x07u8; 12];
const KEY_16: [u8; 16] = [0x42u8; 16];
const NONCE_16: [u8; 16] = [0x07u8; 16];
const NONCE_24: [u8; 24] = [0x07u8; 24];
const NONCE_32: [u8; 32] = [0x07u8; 32];
const AAD: &[u8] = b"rscrypto-bench";

// XChaCha20-Poly1305

// dryoc 0.7.2 does not expose `crypto_aead_xchacha20poly1305_ietf` as a single-shot AEAD;
// only the secretstream framing is available, which is not apples-to-apples with rscrypto's
// IETF one-shot XChaCha20-Poly1305. dryoc rows for XChaCha20 are therefore omitted here;
// dryoc participates in BLAKE2b, Ed25519, X25519, and Argon2id benches instead.
fn xchacha20_poly1305_encrypt(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce192::from_bytes(NONCE_24);
  let cipher_rs = rscrypto::XChaCha20Poly1305::new(&rscrypto::XChaCha20Poly1305Key::from_bytes(KEY_32));
  let cipher_rc = chacha20poly1305::XChaCha20Poly1305::new(&KEY_32.into());
  let nonce_rc = chacha20poly1305::XNonce::from(NONCE_24);
  let mut g = c.benchmark_group("xchacha20-poly1305/encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let mut buf = data.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(cipher_rs.encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut buf)))
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(
          cipher_rc
            .encrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
            )
            .expect("valid AEAD benchmark operation must succeed"),
        )
      })
    });
  }

  g.finish();
}

fn xchacha20_poly1305_decrypt(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce192::from_bytes(NONCE_24);
  let cipher_rs = rscrypto::XChaCha20Poly1305::new(&rscrypto::XChaCha20Poly1305Key::from_bytes(KEY_32));
  let cipher_rc = chacha20poly1305::XChaCha20Poly1305::new(&KEY_32.into());
  let nonce_rc = chacha20poly1305::XNonce::from(NONCE_24);
  let mut g = c.benchmark_group("xchacha20-poly1305/decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    // Pre-encrypt with rscrypto to get valid ciphertext + tag.
    let mut ciphertext = data.clone();
    let tag_rs = cipher_rs
      .encrypt_in_place(&nonce_rs, AAD, &mut ciphertext)
      .expect("valid AEAD benchmark operation must succeed");

    // Pre-encrypt with RustCrypto to get its tag format.
    let mut ct_rc = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ct_rc.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");

    let mut buf = ciphertext.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext, |b, ct| {
      b.iter(|| {
        buf.copy_from_slice(ct);
        cipher_rs
          .decrypt_in_place(
            black_box(&nonce_rs),
            black_box(AAD),
            black_box(&mut buf),
            black_box(&tag_rs),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf);
      })
    });

    let mut buf_rc = ct_rc.clone();

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), &ct_rc, |b, ct| {
      b.iter(|| {
        buf_rc.copy_from_slice(ct);
        cipher_rc
          .decrypt_inout_detached(
            black_box(&nonce_rc),
            black_box(AAD),
            black_box(buf_rc.as_mut_slice().into()),
            black_box(&tag_rc),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_rc);
      })
    });
  }

  g.finish();
}

// ChaCha20-Poly1305

fn chacha20_poly1305_encrypt(c: &mut Criterion) {
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }
  use ring::aead as ring_aead;

  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::ChaCha20Poly1305::new(&rscrypto::ChaCha20Poly1305Key::from_bytes(KEY_32));
  let cipher_rc = chacha20poly1305::ChaCha20Poly1305::new(&KEY_32.into());
  let nonce_rc = chacha20poly1305::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key =
      aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::CHACHA20_POLY1305, &KEY_32).expect("valid AEAD benchmark operation must succeed"));
  }
  let ring_key = ring_aead::LessSafeKey::new(
    ring_aead::UnboundKey::new(&ring_aead::CHACHA20_POLY1305, &KEY_32)
      .expect("valid AEAD benchmark operation must succeed"),
  );
  let mut g = c.benchmark_group("chacha20-poly1305/encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let mut buf = data.clone();
    #[cfg(feature = "diag")]
    let mut buf_owned = data.clone();
    #[cfg(all(feature = "diag", target_arch = "x86_64", target_os = "linux"))]
    let mut buf_x86_asm = data.clone();
    let mut buf_combined: Vec<u8> = Vec::with_capacity(data.len().strict_add(16));

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(cipher_rs.encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut buf)))
      })
    });

    #[cfg(feature = "diag")]
    g.bench_with_input(BenchmarkId::new("rscrypto-owned", len), data, |b, d| {
      b.iter(|| {
        buf_owned.copy_from_slice(d);
        black_box(rscrypto::aead::diag_chacha20poly1305_encrypt_in_place_owned(
          black_box(&cipher_rs),
          black_box(&nonce_rs),
          black_box(AAD),
          black_box(&mut buf_owned),
        ))
      })
    });

    #[cfg(all(feature = "diag", target_arch = "x86_64", target_os = "linux"))]
    if *len != 0 {
      g.bench_with_input(BenchmarkId::new("rscrypto-x86-asm", len), data, |b, d| {
        b.iter(|| {
          buf_x86_asm.copy_from_slice(d);
          black_box(
            rscrypto::aead::diag_chacha20poly1305_encrypt_in_place_x86_64_asm(
              black_box(&cipher_rs),
              black_box(&nonce_rs),
              black_box(AAD),
              black_box(&mut buf_x86_asm),
            )
            .expect("x86 asm path must apply to benchmarked non-empty sizes"),
          )
        })
      });
    }

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(
          cipher_rc
            .encrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
            )
            .expect("valid AEAD benchmark operation must succeed"),
        )
      })
    });

    aws_lc_bench! {
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| {
          buf_combined.clear();
          buf_combined.extend_from_slice(black_box(d));
          aws_key
            .seal_in_place_append_tag(
              aws_aead::Nonce::assume_unique_for_key(NONCE_12),
              aws_aead::Aad::from(AAD),
              black_box(&mut buf_combined),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&buf_combined);
        })
      });
    }

    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| {
        buf_combined.clear();
        buf_combined.extend_from_slice(black_box(d));
        ring_key
          .seal_in_place_append_tag(
            ring_aead::Nonce::assume_unique_for_key(NONCE_12),
            ring_aead::Aad::from(AAD),
            black_box(&mut buf_combined),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_combined);
      })
    });
  }

  g.finish();
}

fn chacha20_poly1305_decrypt(c: &mut Criterion) {
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }
  use ring::aead as ring_aead;

  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::ChaCha20Poly1305::new(&rscrypto::ChaCha20Poly1305Key::from_bytes(KEY_32));
  let cipher_rc = chacha20poly1305::ChaCha20Poly1305::new(&KEY_32.into());
  let nonce_rc = chacha20poly1305::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key =
      aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::CHACHA20_POLY1305, &KEY_32).expect("valid AEAD benchmark operation must succeed"));
  }
  let ring_key = ring_aead::LessSafeKey::new(
    ring_aead::UnboundKey::new(&ring_aead::CHACHA20_POLY1305, &KEY_32)
      .expect("valid AEAD benchmark operation must succeed"),
  );
  let mut g = c.benchmark_group("chacha20-poly1305/decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    let mut ciphertext = data.clone();
    let tag_rs = cipher_rs
      .encrypt_in_place(&nonce_rs, AAD, &mut ciphertext)
      .expect("valid AEAD benchmark operation must succeed");

    let mut ct_rc = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ct_rc.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");

    aws_lc_bench! {
      let mut ct_aws: Vec<u8> = data.clone();
      aws_key
        .seal_in_place_append_tag(
          aws_aead::Nonce::assume_unique_for_key(NONCE_12),
          aws_aead::Aad::from(AAD),
          &mut ct_aws,
        )
        .expect("valid AEAD benchmark operation must succeed");
    }

    let mut ct_ring: Vec<u8> = data.clone();
    ring_key
      .seal_in_place_append_tag(
        ring_aead::Nonce::assume_unique_for_key(NONCE_12),
        ring_aead::Aad::from(AAD),
        &mut ct_ring,
      )
      .expect("valid AEAD benchmark operation must succeed");

    let mut buf = ciphertext.clone();
    #[cfg(feature = "diag")]
    let mut buf_owned = ciphertext.clone();
    #[cfg(all(feature = "diag", target_arch = "x86_64", target_os = "linux"))]
    let mut buf_x86_asm = ciphertext.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext, |b, ct| {
      b.iter(|| {
        buf.copy_from_slice(ct);
        cipher_rs
          .decrypt_in_place(
            black_box(&nonce_rs),
            black_box(AAD),
            black_box(&mut buf),
            black_box(&tag_rs),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf);
      })
    });

    #[cfg(feature = "diag")]
    g.bench_with_input(BenchmarkId::new("rscrypto-owned", len), &ciphertext, |b, ct| {
      b.iter(|| {
        buf_owned.copy_from_slice(ct);
        rscrypto::aead::diag_chacha20poly1305_decrypt_in_place_owned(
          black_box(&cipher_rs),
          black_box(&nonce_rs),
          black_box(AAD),
          black_box(&mut buf_owned),
          black_box(&tag_rs),
        )
        .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_owned);
      })
    });

    #[cfg(all(feature = "diag", target_arch = "x86_64", target_os = "linux"))]
    if *len != 0 {
      g.bench_with_input(BenchmarkId::new("rscrypto-x86-asm", len), &ciphertext, |b, ct| {
        b.iter(|| {
          buf_x86_asm.copy_from_slice(ct);
          rscrypto::aead::diag_chacha20poly1305_decrypt_in_place_x86_64_asm(
            black_box(&cipher_rs),
            black_box(&nonce_rs),
            black_box(AAD),
            black_box(&mut buf_x86_asm),
            black_box(&tag_rs),
          )
          .expect("x86 asm path must apply to benchmarked non-empty sizes")
          .expect("valid AEAD benchmark operation must succeed");
          black_box(&buf_x86_asm);
        })
      });
    }

    let mut buf_rc = ct_rc.clone();

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), &ct_rc, |b, ct| {
      b.iter(|| {
        buf_rc.copy_from_slice(ct);
        cipher_rc
          .decrypt_inout_detached(
            black_box(&nonce_rc),
            black_box(AAD),
            black_box(buf_rc.as_mut_slice().into()),
            black_box(&tag_rc),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_rc);
      })
    });

    aws_lc_bench! {
      let mut buf_aws = ct_aws.clone();

      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), &ct_aws, |b, ct| {
        b.iter(|| {
          buf_aws.copy_from_slice(ct);
          aws_key
            .open_in_place(
              aws_aead::Nonce::assume_unique_for_key(NONCE_12),
              aws_aead::Aad::from(AAD),
              black_box(&mut buf_aws),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&buf_aws);
        })
      });
    }

    let mut buf_ring = ct_ring.clone();

    g.bench_with_input(BenchmarkId::new("ring", len), &ct_ring, |b, ct| {
      b.iter(|| {
        buf_ring.copy_from_slice(ct);
        ring_key
          .open_in_place(
            ring_aead::Nonce::assume_unique_for_key(NONCE_12),
            ring_aead::Aad::from(AAD),
            black_box(&mut buf_ring),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_ring);
      })
    });
  }

  g.finish();
}

// AES-256-GCM-SIV

fn aes256_gcm_siv_encrypt(c: &mut Criterion) {
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }

  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::Aes256GcmSiv::new(&rscrypto::Aes256GcmSivKey::from_bytes(KEY_32));
  let cipher_rc = aes_gcm_siv::Aes256GcmSiv::new(&KEY_32.into());
  let nonce_rc = aes_gcm_siv::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key =
      aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::AES_256_GCM_SIV, &KEY_32).expect("valid AEAD benchmark operation must succeed"));
  }
  let mut g = c.benchmark_group("aes-256-gcm-siv/encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let mut buf = data.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(cipher_rs.encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut buf)))
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(
          cipher_rc
            .encrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice()).into(),
            )
            .expect("valid AEAD benchmark operation must succeed"),
        )
      })
    });

    aws_lc_bench! {
      let mut buf_aws = data.clone();
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| {
          buf_aws.copy_from_slice(black_box(d));
          let tag = aws_key
            .seal_in_place_separate_tag(
              aws_aead::Nonce::assume_unique_for_key(NONCE_12),
              aws_aead::Aad::from(AAD),
              black_box(&mut buf_aws),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(tag.as_ref());
          black_box(&buf_aws);
        })
      });
    }
  }

  g.finish();
}

fn aes256_gcm_siv_decrypt(c: &mut Criterion) {
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }

  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::Aes256GcmSiv::new(&rscrypto::Aes256GcmSivKey::from_bytes(KEY_32));
  let cipher_rc = aes_gcm_siv::Aes256GcmSiv::new(&KEY_32.into());
  let nonce_rc = aes_gcm_siv::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key =
      aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::AES_256_GCM_SIV, &KEY_32).expect("valid AEAD benchmark operation must succeed"));
  }
  let mut g = c.benchmark_group("aes-256-gcm-siv/decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    let mut ciphertext = data.clone();
    let tag_rs = cipher_rs
      .encrypt_in_place(&nonce_rs, AAD, &mut ciphertext)
      .expect("valid AEAD benchmark operation must succeed");

    let mut ct_rc = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ct_rc.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");

    aws_lc_bench! {
      let mut ct_aws = data.clone();
      let tag_aws = aws_key
        .seal_in_place_separate_tag(
          aws_aead::Nonce::assume_unique_for_key(NONCE_12),
          aws_aead::Aad::from(AAD),
          &mut ct_aws,
        )
        .expect("valid AEAD benchmark operation must succeed");
      // AWS-LC exposes detached tags for seal, but its in-place open API takes ct||tag.
      ct_aws.extend_from_slice(tag_aws.as_ref());
    }

    let mut buf = ciphertext.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext, |b, ct| {
      b.iter(|| {
        buf.copy_from_slice(ct);
        cipher_rs
          .decrypt_in_place(
            black_box(&nonce_rs),
            black_box(AAD),
            black_box(&mut buf),
            black_box(&tag_rs),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf);
      })
    });

    let mut buf_rc = ct_rc.clone();

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), &ct_rc, |b, ct| {
      b.iter(|| {
        buf_rc.copy_from_slice(ct);
        cipher_rc
          .decrypt_inout_detached(
            black_box(&nonce_rc),
            black_box(AAD),
            black_box(buf_rc.as_mut_slice()).into(),
            black_box(&tag_rc),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_rc);
      })
    });

    aws_lc_bench! {
      let mut buf_aws = ct_aws.clone();

      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), &ct_aws, |b, ct| {
        b.iter(|| {
          buf_aws.copy_from_slice(ct);
          aws_key
            .open_in_place(
              aws_aead::Nonce::assume_unique_for_key(NONCE_12),
              aws_aead::Aad::from(AAD),
              black_box(&mut buf_aws),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&buf_aws);
        })
      });
    }
  }

  g.finish();
}

// AES-128-GCM-SIV

fn aes128_gcm_siv_encrypt(c: &mut Criterion) {
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }

  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::Aes128GcmSiv::new(&rscrypto::Aes128GcmSivKey::from_bytes(KEY_16));
  let cipher_rc = aes_gcm_siv::Aes128GcmSiv::new(&KEY_16.into());
  let nonce_rc = aes_gcm_siv::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key =
      aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::AES_128_GCM_SIV, &KEY_16).expect("valid AEAD benchmark operation must succeed"));
  }
  let mut g = c.benchmark_group("aes-128-gcm-siv/encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let mut buf = data.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(cipher_rs.encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut buf)))
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(
          cipher_rc
            .encrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice()).into(),
            )
            .expect("valid AEAD benchmark operation must succeed"),
        )
      })
    });

    aws_lc_bench! {
      let mut buf_aws = data.clone();
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| {
          buf_aws.copy_from_slice(black_box(d));
          let tag = aws_key
            .seal_in_place_separate_tag(
              aws_aead::Nonce::assume_unique_for_key(NONCE_12),
              aws_aead::Aad::from(AAD),
              black_box(&mut buf_aws),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(tag.as_ref());
          black_box(&buf_aws);
        })
      });
    }
  }

  g.finish();
}

fn aes128_gcm_siv_decrypt(c: &mut Criterion) {
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }

  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::Aes128GcmSiv::new(&rscrypto::Aes128GcmSivKey::from_bytes(KEY_16));
  let cipher_rc = aes_gcm_siv::Aes128GcmSiv::new(&KEY_16.into());
  let nonce_rc = aes_gcm_siv::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key =
      aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::AES_128_GCM_SIV, &KEY_16).expect("valid AEAD benchmark operation must succeed"));
  }
  let mut g = c.benchmark_group("aes-128-gcm-siv/decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    let mut ciphertext = data.clone();
    let tag_rs = cipher_rs
      .encrypt_in_place(&nonce_rs, AAD, &mut ciphertext)
      .expect("valid AEAD benchmark operation must succeed");

    let mut ct_rc = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ct_rc.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");

    aws_lc_bench! {
      let mut ct_aws = data.clone();
      let tag_aws = aws_key
        .seal_in_place_separate_tag(
          aws_aead::Nonce::assume_unique_for_key(NONCE_12),
          aws_aead::Aad::from(AAD),
          &mut ct_aws,
        )
        .expect("valid AEAD benchmark operation must succeed");
      // AWS-LC exposes detached tags for seal, but its in-place open API takes ct||tag.
      ct_aws.extend_from_slice(tag_aws.as_ref());
    }

    let mut buf = ciphertext.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext, |b, ct| {
      b.iter(|| {
        buf.copy_from_slice(ct);
        cipher_rs
          .decrypt_in_place(
            black_box(&nonce_rs),
            black_box(AAD),
            black_box(&mut buf),
            black_box(&tag_rs),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf);
      })
    });

    let mut buf_rc = ct_rc.clone();

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), &ct_rc, |b, ct| {
      b.iter(|| {
        buf_rc.copy_from_slice(ct);
        cipher_rc
          .decrypt_inout_detached(
            black_box(&nonce_rc),
            black_box(AAD),
            black_box(buf_rc.as_mut_slice()).into(),
            black_box(&tag_rc),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_rc);
      })
    });

    aws_lc_bench! {
      let mut buf_aws = ct_aws.clone();

      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), &ct_aws, |b, ct| {
        b.iter(|| {
          buf_aws.copy_from_slice(ct);
          aws_key
            .open_in_place(
              aws_aead::Nonce::assume_unique_for_key(NONCE_12),
              aws_aead::Aad::from(AAD),
              black_box(&mut buf_aws),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&buf_aws);
        })
      });
    }
  }

  g.finish();
}

// AES-256-GCM

fn aes256_gcm_encrypt(c: &mut Criterion) {
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }
  use ring::aead as ring_aead;

  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::Aes256Gcm::new(&rscrypto::Aes256GcmKey::from_bytes(KEY_32));
  let cipher_rc = aes_gcm::Aes256Gcm::new(&KEY_32.into());
  let nonce_rc = aes_gcm::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key = aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::AES_256_GCM, &KEY_32).expect("valid AEAD benchmark operation must succeed"));
  }
  let ring_key = ring_aead::LessSafeKey::new(
    ring_aead::UnboundKey::new(&ring_aead::AES_256_GCM, &KEY_32).expect("valid AEAD benchmark operation must succeed"),
  );
  let mut g = c.benchmark_group("aes-256-gcm/encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let mut buf = data.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(cipher_rs.encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut buf)))
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(
          cipher_rc
            .encrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
            )
            .expect("valid AEAD benchmark operation must succeed"),
        )
      })
    });

    aws_lc_bench! {
      let mut buf_aws = data.clone();
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| {
          buf_aws.copy_from_slice(black_box(d));
          let tag = aws_key
            .seal_in_place_separate_tag(
              aws_aead::Nonce::assume_unique_for_key(NONCE_12),
              aws_aead::Aad::from(AAD),
              black_box(&mut buf_aws),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(tag.as_ref());
          black_box(&buf_aws);
        })
      });
    }

    let mut buf_ring = data.clone();
    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| {
        buf_ring.copy_from_slice(black_box(d));
        let tag = ring_key
          .seal_in_place_separate_tag(
            ring_aead::Nonce::assume_unique_for_key(NONCE_12),
            ring_aead::Aad::from(AAD),
            black_box(&mut buf_ring),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(tag.as_ref());
        black_box(&buf_ring);
      })
    });
  }

  g.finish();
}

fn aes256_gcm_decrypt(c: &mut Criterion) {
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }
  use ring::aead as ring_aead;

  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::Aes256Gcm::new(&rscrypto::Aes256GcmKey::from_bytes(KEY_32));
  let cipher_rc = aes_gcm::Aes256Gcm::new(&KEY_32.into());
  let nonce_rc = aes_gcm::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key = aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::AES_256_GCM, &KEY_32).expect("valid AEAD benchmark operation must succeed"));
  }
  let ring_key = ring_aead::LessSafeKey::new(
    ring_aead::UnboundKey::new(&ring_aead::AES_256_GCM, &KEY_32).expect("valid AEAD benchmark operation must succeed"),
  );
  let mut g = c.benchmark_group("aes-256-gcm/decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    let mut ciphertext = data.clone();
    let tag_rs = cipher_rs
      .encrypt_in_place(&nonce_rs, AAD, &mut ciphertext)
      .expect("valid AEAD benchmark operation must succeed");

    let mut ct_rc = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ct_rc.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");

    aws_lc_bench! {
      let mut ct_aws = data.clone();
      let tag_aws = aws_key
        .seal_in_place_separate_tag(
          aws_aead::Nonce::assume_unique_for_key(NONCE_12),
          aws_aead::Aad::from(AAD),
          &mut ct_aws,
        )
        .expect("valid AEAD benchmark operation must succeed");
      // AWS-LC exposes detached tags for seal, but its in-place open API takes ct||tag.
      ct_aws.extend_from_slice(tag_aws.as_ref());
    }

    let mut ct_ring = data.clone();
    let tag_ring = ring_key
      .seal_in_place_separate_tag(
        ring_aead::Nonce::assume_unique_for_key(NONCE_12),
        ring_aead::Aad::from(AAD),
        &mut ct_ring,
      )
      .expect("valid AEAD benchmark operation must succeed");

    let mut buf = ciphertext.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext, |b, ct| {
      b.iter(|| {
        buf.copy_from_slice(ct);
        cipher_rs
          .decrypt_in_place(
            black_box(&nonce_rs),
            black_box(AAD),
            black_box(&mut buf),
            black_box(&tag_rs),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf);
      })
    });

    let mut buf_rc = ct_rc.clone();

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), &ct_rc, |b, ct| {
      b.iter(|| {
        buf_rc.copy_from_slice(ct);
        cipher_rc
          .decrypt_inout_detached(
            black_box(&nonce_rc),
            black_box(AAD),
            black_box(buf_rc.as_mut_slice().into()),
            black_box(&tag_rc),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_rc);
      })
    });

    aws_lc_bench! {
      let mut buf_aws = ct_aws.clone();

      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), &ct_aws, |b, ct| {
        b.iter(|| {
          buf_aws.copy_from_slice(ct);
          aws_key
            .open_in_place(
              aws_aead::Nonce::assume_unique_for_key(NONCE_12),
              aws_aead::Aad::from(AAD),
              black_box(&mut buf_aws),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&buf_aws);
        })
      });
    }

    let mut buf_ring = ct_ring.clone();

    g.bench_with_input(BenchmarkId::new("ring", len), &ct_ring, |b, ct| {
      b.iter(|| {
        buf_ring.copy_from_slice(ct);
        ring_key
          .open_in_place_separate_tag(
            ring_aead::Nonce::assume_unique_for_key(NONCE_12),
            ring_aead::Aad::from(AAD),
            black_box(tag_ring),
            black_box(&mut buf_ring),
            0..,
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_ring);
      })
    });
  }

  g.finish();
}

// AES-128-GCM

fn aes128_gcm_encrypt(c: &mut Criterion) {
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }
  use ring::aead as ring_aead;

  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::Aes128Gcm::new(&rscrypto::Aes128GcmKey::from_bytes(KEY_16));
  let cipher_rc = aes_gcm::Aes128Gcm::new(&KEY_16.into());
  let nonce_rc = aes_gcm::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key = aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::AES_128_GCM, &KEY_16).expect("valid AEAD benchmark operation must succeed"));
  }
  let ring_key = ring_aead::LessSafeKey::new(
    ring_aead::UnboundKey::new(&ring_aead::AES_128_GCM, &KEY_16).expect("valid AEAD benchmark operation must succeed"),
  );
  let mut g = c.benchmark_group("aes-128-gcm/encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let mut buf = data.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(cipher_rs.encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut buf)))
      })
    });

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(
          cipher_rc
            .encrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
            )
            .expect("valid AEAD benchmark operation must succeed"),
        )
      })
    });

    aws_lc_bench! {
      let mut buf_aws = data.clone();
      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), data, |b, d| {
        b.iter(|| {
          buf_aws.copy_from_slice(black_box(d));
          let tag = aws_key
            .seal_in_place_separate_tag(
              aws_aead::Nonce::assume_unique_for_key(NONCE_12),
              aws_aead::Aad::from(AAD),
              black_box(&mut buf_aws),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(tag.as_ref());
          black_box(&buf_aws);
        })
      });
    }

    let mut buf_ring = data.clone();
    g.bench_with_input(BenchmarkId::new("ring", len), data, |b, d| {
      b.iter(|| {
        buf_ring.copy_from_slice(black_box(d));
        let tag = ring_key
          .seal_in_place_separate_tag(
            ring_aead::Nonce::assume_unique_for_key(NONCE_12),
            ring_aead::Aad::from(AAD),
            black_box(&mut buf_ring),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(tag.as_ref());
        black_box(&buf_ring);
      })
    });
  }

  g.finish();
}

fn aes128_gcm_decrypt(c: &mut Criterion) {
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }
  use ring::aead as ring_aead;

  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::Aes128Gcm::new(&rscrypto::Aes128GcmKey::from_bytes(KEY_16));
  let cipher_rc = aes_gcm::Aes128Gcm::new(&KEY_16.into());
  let nonce_rc = aes_gcm::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key = aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::AES_128_GCM, &KEY_16).expect("valid AEAD benchmark operation must succeed"));
  }
  let ring_key = ring_aead::LessSafeKey::new(
    ring_aead::UnboundKey::new(&ring_aead::AES_128_GCM, &KEY_16).expect("valid AEAD benchmark operation must succeed"),
  );
  let mut g = c.benchmark_group("aes-128-gcm/decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    let mut ciphertext = data.clone();
    let tag_rs = cipher_rs
      .encrypt_in_place(&nonce_rs, AAD, &mut ciphertext)
      .expect("valid AEAD benchmark operation must succeed");

    let mut ct_rc = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ct_rc.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");

    aws_lc_bench! {
      let mut ct_aws = data.clone();
      let tag_aws = aws_key
        .seal_in_place_separate_tag(
          aws_aead::Nonce::assume_unique_for_key(NONCE_12),
          aws_aead::Aad::from(AAD),
          &mut ct_aws,
        )
        .expect("valid AEAD benchmark operation must succeed");
      // AWS-LC exposes detached tags for seal, but its in-place open API takes ct||tag.
      ct_aws.extend_from_slice(tag_aws.as_ref());
    }

    let mut ct_ring = data.clone();
    let tag_ring = ring_key
      .seal_in_place_separate_tag(
        ring_aead::Nonce::assume_unique_for_key(NONCE_12),
        ring_aead::Aad::from(AAD),
        &mut ct_ring,
      )
      .expect("valid AEAD benchmark operation must succeed");

    let mut buf = ciphertext.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext, |b, ct| {
      b.iter(|| {
        buf.copy_from_slice(ct);
        cipher_rs
          .decrypt_in_place(
            black_box(&nonce_rs),
            black_box(AAD),
            black_box(&mut buf),
            black_box(&tag_rs),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf);
      })
    });

    let mut buf_rc = ct_rc.clone();

    g.bench_with_input(BenchmarkId::new("rustcrypto", len), &ct_rc, |b, ct| {
      b.iter(|| {
        buf_rc.copy_from_slice(ct);
        cipher_rc
          .decrypt_inout_detached(
            black_box(&nonce_rc),
            black_box(AAD),
            black_box(buf_rc.as_mut_slice().into()),
            black_box(&tag_rc),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_rc);
      })
    });

    aws_lc_bench! {
      let mut buf_aws = ct_aws.clone();

      g.bench_with_input(BenchmarkId::new("aws-lc-rs", len), &ct_aws, |b, ct| {
        b.iter(|| {
          buf_aws.copy_from_slice(ct);
          aws_key
            .open_in_place(
              aws_aead::Nonce::assume_unique_for_key(NONCE_12),
              aws_aead::Aad::from(AAD),
              black_box(&mut buf_aws),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&buf_aws);
        })
      });
    }

    let mut buf_ring = ct_ring.clone();

    g.bench_with_input(BenchmarkId::new("ring", len), &ct_ring, |b, ct| {
      b.iter(|| {
        buf_ring.copy_from_slice(ct);
        ring_key
          .open_in_place_separate_tag(
            ring_aead::Nonce::assume_unique_for_key(NONCE_12),
            ring_aead::Aad::from(AAD),
            black_box(tag_ring),
            black_box(&mut buf_ring),
            0..,
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_ring);
      })
    });
  }

  g.finish();
}

// AEGIS-256

fn aegis256_encrypt(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce256::from_bytes(NONCE_32);
  let cipher_rs = rscrypto::Aegis256::new(&rscrypto::Aegis256Key::from_bytes(KEY_32));
  let mut g = c.benchmark_group("aegis-256/encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let mut buf = data.clone();
    let cipher_ac = aegis::aegis256::Aegis256::<16>::new(&KEY_32, &NONCE_32);

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(cipher_rs.encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut buf)))
      })
    });

    g.bench_with_input(BenchmarkId::new("aegis-crate", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(cipher_ac.encrypt_in_place(black_box(&mut buf), black_box(AAD)))
      })
    });
  }

  g.finish();
}

fn aegis256_decrypt(c: &mut Criterion) {
  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce256::from_bytes(NONCE_32);
  let cipher_rs = rscrypto::Aegis256::new(&rscrypto::Aegis256Key::from_bytes(KEY_32));
  let mut g = c.benchmark_group("aegis-256/decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    // Pre-encrypt with rscrypto to get valid ciphertext + tag.
    let mut ciphertext = data.clone();
    let tag_rs = cipher_rs
      .encrypt_in_place(&nonce_rs, AAD, &mut ciphertext)
      .expect("valid AEAD benchmark operation must succeed");

    // Pre-encrypt with aegis crate to get its tag format.
    let mut ct_ac = data.clone();
    let cipher_ac = aegis::aegis256::Aegis256::<16>::new(&KEY_32, &NONCE_32);
    let tag_ac = cipher_ac.encrypt_in_place(&mut ct_ac, AAD);

    let mut buf = ciphertext.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext, |b, ct| {
      b.iter(|| {
        buf.copy_from_slice(ct);
        cipher_rs
          .decrypt_in_place(
            black_box(&nonce_rs),
            black_box(AAD),
            black_box(&mut buf),
            black_box(&tag_rs),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf);
      })
    });

    let mut buf_ac = ct_ac.clone();

    g.bench_with_input(BenchmarkId::new("aegis-crate", len), &ct_ac, |b, ct| {
      b.iter(|| {
        buf_ac.copy_from_slice(ct);
        cipher_ac
          .decrypt_in_place(black_box(&mut buf_ac), black_box(&tag_ac), black_box(AAD))
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_ac);
      })
    });
  }

  g.finish();
}

// Ascon-AEAD128

fn ascon_aead128_encrypt(c: &mut Criterion) {
  use ascon_aead::aead::{AeadInOut as _, KeyInit as _, array::Array};

  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce128::from_bytes(NONCE_16);
  let cipher_rs = rscrypto::AsconAead128::new(&rscrypto::AsconAead128Key::from_bytes(KEY_16));
  let cipher_ac = ascon_aead::AsconAead128::new(&Array(KEY_16));
  let nonce_ac = Array(NONCE_16);
  let mut g = c.benchmark_group("ascon-aead128/encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let mut buf = data.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(cipher_rs.encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut buf)))
      })
    });

    g.bench_with_input(BenchmarkId::new("ascon-aead", len), data, |b, d| {
      b.iter(|| {
        buf.copy_from_slice(d);
        black_box(
          cipher_ac
            .encrypt_inout_detached(
              black_box(&nonce_ac),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
            )
            .expect("valid AEAD benchmark operation must succeed"),
        )
      })
    });
  }

  g.finish();
}

fn ascon_aead128_decrypt(c: &mut Criterion) {
  use ascon_aead::aead::{AeadInOut as _, KeyInit as _, array::Array};

  let inputs = common::comp_sizes();
  let nonce_rs = rscrypto::aead::Nonce128::from_bytes(NONCE_16);
  let cipher_rs = rscrypto::AsconAead128::new(&rscrypto::AsconAead128Key::from_bytes(KEY_16));
  let cipher_ac = ascon_aead::AsconAead128::new(&Array(KEY_16));
  let nonce_ac = Array(NONCE_16);
  let mut g = c.benchmark_group("ascon-aead128/decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    let mut ciphertext = data.clone();
    let tag_rs = cipher_rs
      .encrypt_in_place(&nonce_rs, AAD, &mut ciphertext)
      .expect("valid AEAD benchmark operation must succeed");

    let mut ct_ac = data.clone();
    let tag_ac = cipher_ac
      .encrypt_inout_detached(&nonce_ac, AAD, ct_ac.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");

    let mut buf = ciphertext.clone();

    g.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext, |b, ct| {
      b.iter(|| {
        buf.copy_from_slice(ct);
        cipher_rs
          .decrypt_in_place(
            black_box(&nonce_rs),
            black_box(AAD),
            black_box(&mut buf),
            black_box(&tag_rs),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf);
      })
    });

    let mut buf_ac = ct_ac.clone();

    g.bench_with_input(BenchmarkId::new("ascon-aead", len), &ct_ac, |b, ct| {
      b.iter(|| {
        buf_ac.copy_from_slice(ct);
        cipher_ac
          .decrypt_inout_detached(
            black_box(&nonce_ac),
            black_box(AAD),
            black_box(buf_ac.as_mut_slice().into()),
            black_box(&tag_ac),
          )
          .expect("valid AEAD benchmark operation must succeed");
        black_box(&buf_ac);
      })
    });
  }

  g.finish();
}

// Fixed-size header protection

fn header_protection(c: &mut Criterion) {
  use aes::cipher::{Array, BlockCipherEncrypt as _, KeyInit as _};
  use chacha20::cipher::{KeyIvInit as _, StreamCipherCore as _};
  use rscrypto::aead::expert::header_protection::{
    Aes128HeaderProtection, Aes128HeaderProtectionKey, Aes256HeaderProtection, Aes256HeaderProtectionKey,
    ChaCha20HeaderProtection, ChaCha20HeaderProtectionKey,
  };

  let mut construction = c.benchmark_group("header-protection/construct");
  construction.bench_function("rscrypto/aes128", |b| {
    b.iter(|| {
      let key = Aes128HeaderProtectionKey::from_bytes(black_box(KEY_16));
      black_box(Aes128HeaderProtection::new(&key))
    })
  });
  construction.bench_function("rustcrypto/aes128", |b| {
    b.iter(|| aes::Aes128::new(black_box(&Array::from(KEY_16))))
  });
  construction.bench_function("rscrypto/aes256", |b| {
    b.iter(|| {
      let key = Aes256HeaderProtectionKey::from_bytes(black_box(KEY_32));
      black_box(Aes256HeaderProtection::new(&key))
    })
  });
  construction.bench_function("rustcrypto/aes256", |b| {
    b.iter(|| aes::Aes256::new(black_box(&Array::from(KEY_32))))
  });
  construction.bench_function("rscrypto/chacha20", |b| {
    b.iter(|| {
      let key = ChaCha20HeaderProtectionKey::from_bytes(black_box(KEY_32));
      black_box(ChaCha20HeaderProtection::new(&key))
    })
  });
  construction.bench_function("rustcrypto/chacha20", |b| {
    b.iter(|| {
      chacha20::ChaChaCore::<chacha20::R20, chacha20::variants::Ietf>::new(
        black_box(&KEY_32).into(),
        black_box(&NONCE_12).into(),
      )
    })
  });
  construction.finish();

  let sample = [0x07; 16];
  let aes128_rs = Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes(KEY_16));
  let aes256_rs = Aes256HeaderProtection::new(&Aes256HeaderProtectionKey::from_bytes(KEY_32));
  let chacha_rs = ChaCha20HeaderProtection::new(&ChaCha20HeaderProtectionKey::from_bytes(KEY_32));
  let aes128_rc = aes::Aes128::new(&Array::from(KEY_16));
  let aes256_rc = aes::Aes256::new(&Array::from(KEY_32));

  let mut mask = c.benchmark_group("header-protection/mask");
  mask.bench_function("rscrypto/aes128", |b| {
    b.iter(|| black_box(aes128_rs.mask(black_box(&sample))))
  });
  mask.bench_function("rustcrypto/aes128", |b| {
    b.iter(|| {
      let mut block = Array::from(*black_box(&sample));
      aes128_rc.encrypt_block(&mut block);
      black_box(<[u8; 5]>::try_from(&block[..5]).expect("five-byte prefix has fixed length"))
    })
  });
  mask.bench_function("rscrypto/aes256", |b| {
    b.iter(|| black_box(aes256_rs.mask(black_box(&sample))))
  });
  mask.bench_function("rustcrypto/aes256", |b| {
    b.iter(|| {
      let mut block = Array::from(*black_box(&sample));
      aes256_rc.encrypt_block(&mut block);
      black_box(<[u8; 5]>::try_from(&block[..5]).expect("five-byte prefix has fixed length"))
    })
  });
  mask.bench_function("rscrypto/chacha20", |b| {
    b.iter(|| black_box(chacha_rs.mask(black_box(&sample))))
  });
  mask.bench_function("rustcrypto/chacha20", |b| {
    b.iter(|| {
      let counter = u32::from_le_bytes(sample[..4].try_into().expect("counter prefix has fixed length"));
      let nonce: [u8; 12] = sample[4..].try_into().expect("nonce suffix has fixed length");
      let mut core = chacha20::ChaChaCore::<chacha20::R20, chacha20::variants::Ietf>::new(
        black_box(&KEY_32).into(),
        black_box(&nonce).into(),
      );
      core.set_block_pos(counter);
      let mut block = chacha20::cipher::array::Array::<u8, chacha20::cipher::consts::U64>::default();
      core.write_keystream_block(&mut block);
      black_box(<[u8; 5]>::try_from(&block[..5]).expect("five-byte prefix has fixed length"))
    })
  });
  mask.finish();
}

// AES-SIV-CMAC-256 (RFC 5297 nonce-based profile)

fn aes_siv_cmac256(c: &mut Criterion) {
  use aes_siv::{KeyInit as _, siv::Aes128Siv};
  use rscrypto::{AesSivCmac256, AesSivCmac256Key, AesSivCmac256Nonce};

  let nonce_rs = AesSivCmac256Nonce::try_from(NONCE_16.as_slice()).expect("benchmark nonce is non-empty");
  let key_rs = AesSivCmac256Key::from_bytes(KEY_32);
  let key_rc: aes_siv::Key<Aes128Siv> = KEY_32.into();

  let mut construction = c.benchmark_group("aes-siv-cmac-256/construct");
  construction.bench_function("rscrypto", |b| {
    b.iter(|| black_box(AesSivCmac256::new(black_box(&key_rs))))
  });
  construction.bench_function("rustcrypto", |b| {
    b.iter(|| black_box(Aes128Siv::new(black_box(&key_rc))))
  });
  construction.finish();

  // NTS packet shapes: empty authenticator payload, short field, extension-field,
  // cookie-shaped, and near-MTU protected bodies.
  let inputs = [0usize, 16, 64, 256, 1232]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let cipher_rs = AesSivCmac256::new(&key_rs);
  let mut cipher_rc = Aes128Siv::new(&key_rc);
  let headers: [&[u8]; 2] = [AAD, &NONCE_16];

  // Construction prepares different amounts of reusable work in the two libraries. Measure the
  // complete one-context/one-seal lifecycle separately so setup deferral cannot skew the result.
  let mut construct_and_seal = c.benchmark_group("aes-siv-cmac-256/construct-and-seal");
  for (len, data) in &inputs {
    common::set_throughput(&mut construct_and_seal, *len);
    let mut buffer_rs = data.clone();
    construct_and_seal.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, input| {
      b.iter(|| {
        buffer_rs.copy_from_slice(input);
        let cipher = AesSivCmac256::new(black_box(&key_rs));
        black_box(cipher.seal_in_place(black_box(nonce_rs), black_box(AAD), black_box(&mut buffer_rs)))
      })
    });

    let mut buffer_rc = data.clone();
    construct_and_seal.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, input| {
      b.iter(|| {
        buffer_rc.copy_from_slice(input);
        let mut cipher = Aes128Siv::new(black_box(&key_rc));
        black_box(
          cipher
            .encrypt_inout_detached(black_box(headers), black_box(buffer_rc.as_mut_slice().into()))
            .expect("valid benchmark input must seal"),
        )
      })
    });
  }
  construct_and_seal.finish();

  let mut seal = c.benchmark_group("aes-siv-cmac-256/seal");

  for (len, data) in &inputs {
    common::set_throughput(&mut seal, *len);
    let mut buffer_rs = data.clone();
    seal.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, input| {
      b.iter(|| {
        buffer_rs.copy_from_slice(input);
        black_box(cipher_rs.seal_in_place(black_box(nonce_rs), black_box(AAD), black_box(&mut buffer_rs)))
      })
    });

    let mut buffer_rc = data.clone();
    seal.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, input| {
      b.iter(|| {
        buffer_rc.copy_from_slice(input);
        black_box(
          cipher_rc
            .encrypt_inout_detached(black_box(headers), black_box(buffer_rc.as_mut_slice().into()))
            .expect("valid benchmark input must seal"),
        )
      })
    });
  }
  seal.finish();

  let mut open = c.benchmark_group("aes-siv-cmac-256/open");
  for (len, data) in &inputs {
    common::set_throughput(&mut open, *len);

    let mut ciphertext_rs = data.clone();
    let tag_rs = cipher_rs.seal_in_place(nonce_rs, AAD, &mut ciphertext_rs);
    let mut buffer_rs = ciphertext_rs.clone();
    open.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext_rs, |b, ciphertext| {
      b.iter(|| {
        buffer_rs.copy_from_slice(ciphertext);
        cipher_rs
          .open_in_place(
            black_box(nonce_rs),
            black_box(AAD),
            black_box(&mut buffer_rs),
            black_box(&tag_rs),
          )
          .expect("valid benchmark ciphertext must open");
        black_box(&buffer_rs);
      })
    });

    let mut ciphertext_rc = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(headers, ciphertext_rc.as_mut_slice().into())
      .expect("valid benchmark input must seal");
    let mut buffer_rc = ciphertext_rc.clone();
    open.bench_with_input(BenchmarkId::new("rustcrypto", len), &ciphertext_rc, |b, ciphertext| {
      b.iter(|| {
        buffer_rc.copy_from_slice(ciphertext);
        cipher_rc
          .decrypt_inout_detached(
            black_box(headers),
            black_box(buffer_rc.as_mut_slice().into()),
            black_box(&tag_rc),
          )
          .expect("valid benchmark ciphertext must open");
        black_box(&buffer_rc);
      })
    });
  }
  open.finish();
}

// Criterion harness

criterion_group!(
  benches,
  xchacha20_poly1305_encrypt,
  xchacha20_poly1305_decrypt,
  chacha20_poly1305_encrypt,
  chacha20_poly1305_decrypt,
  aes256_gcm_siv_encrypt,
  aes256_gcm_siv_decrypt,
  aes128_gcm_siv_encrypt,
  aes128_gcm_siv_decrypt,
  aes256_gcm_encrypt,
  aes256_gcm_decrypt,
  aes128_gcm_encrypt,
  aes128_gcm_decrypt,
  aegis256_encrypt,
  aegis256_decrypt,
  ascon_aead128_encrypt,
  ascon_aead128_decrypt,
  header_protection,
  aes_siv_cmac256,
);
criterion_main!(benches);
