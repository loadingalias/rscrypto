//! AEAD comparison benchmarks: rscrypto vs RustCrypto ecosystem.
//!
//! Copy-plus-operation workloads over preallocated buffers and reusable keys.
//! Input restoration is timed; initial allocation and fixture generation are not.
//! `appended-tag` rows handle ciphertext || tag; other rows use detached tags.
//! Before timing, every seal row must reproduce an independent library's
//! ciphertext and tag, and every open row must recover the plaintext from them.
//! See docs/benchmarking.md for the suite timing policy.

#[path = "common/criterion.rs"]
mod bench_config;

mod common;

use core::hint::black_box;

use aes_gcm::aead::{AeadInOut as _, KeyInit as _};
use bench_config::{checked, checked_with};
use criterion::{BenchmarkId, Criterion};
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

/// The `ciphertext || tag` form in which every sealed output is compared.
fn sealed(ciphertext: &[u8], tag: impl AsRef<[u8]>) -> Vec<u8> {
  [ciphertext, tag.as_ref()].concat()
}

// XChaCha20-Poly1305

// dryoc 0.7.2 does not expose `crypto_aead_xchacha20poly1305_ietf` as a single-shot AEAD;
// only the secretstream framing is available, which is not apples-to-apples with rscrypto's
// IETF one-shot XChaCha20-Poly1305. dryoc rows for XChaCha20 are therefore omitted here;
// dryoc participates in BLAKE2b, Ed25519, X25519, and Argon2id benches instead.
fn xchacha20_poly1305_encrypt(c: &mut Criterion) {
  if !bench_config::selected("xchacha20-poly1305/copy-and-encrypt") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let nonce_rs = rscrypto::aead::Nonce192::from_bytes(NONCE_24);
  let cipher_rs = rscrypto::XChaCha20Poly1305::new(&rscrypto::XChaCha20Poly1305Key::from_bytes(KEY_32));
  let cipher_rc = chacha20poly1305::XChaCha20Poly1305::new(&KEY_32.into());
  let nonce_rc = chacha20poly1305::XNonce::from(NONCE_24);
  // Timed: restore input into a preallocated buffer, encrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("xchacha20-poly1305/copy-and-encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    // Every row must produce RustCrypto's ciphertext and tag.
    let mut ciphertext = data.clone();
    let tag = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let expected = sealed(&ciphertext, tag);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rs
            .encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut *buf))
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rc
            .encrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
            )
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );
  }

  g.finish();
}

fn xchacha20_poly1305_decrypt(c: &mut Criterion) {
  if !bench_config::selected("xchacha20-poly1305/copy-and-decrypt") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let nonce_rs = rscrypto::aead::Nonce192::from_bytes(NONCE_24);
  let cipher_rs = rscrypto::XChaCha20Poly1305::new(&rscrypto::XChaCha20Poly1305Key::from_bytes(KEY_32));
  let cipher_rc = chacha20poly1305::XChaCha20Poly1305::new(&KEY_32.into());
  let nonce_rc = chacha20poly1305::XNonce::from(NONCE_24);
  // Timed: restore input into a preallocated buffer, decrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("xchacha20-poly1305/copy-and-decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    // Every row opens RustCrypto's ciphertext and tag and must recover the input.
    let mut ciphertext = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let tag_rs = rscrypto::XChaCha20Poly1305::tag_from_slice(&tag_rc).expect("AEAD tags have 16 bytes");

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rs
            .decrypt_in_place(
              black_box(&nonce_rs),
              black_box(AAD),
              black_box(&mut *buf),
              black_box(&tag_rs),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rc
            .decrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
              black_box(&tag_rc),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );
  }

  g.finish();
}

// ChaCha20-Poly1305

fn chacha20_poly1305_encrypt(c: &mut Criterion) {
  if !bench_config::selected("chacha20-poly1305/copy-and-encrypt") {
    return;
  }
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }
  use ring::aead as ring_aead;

  let inputs = common::comp_sizes(&[]);
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
  // Timed: restore input into a preallocated buffer, encrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("chacha20-poly1305/copy-and-encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    // Every row must produce RustCrypto's ciphertext and tag.
    let mut ciphertext = data.clone();
    let tag = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let expected = sealed(&ciphertext, tag);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rs
            .encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut *buf))
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    #[cfg(all(rscrypto_internal, feature = "diag"))]
    g.bench_function(
      BenchmarkId::new("rscrypto-owned", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          rscrypto::aead::diag_chacha20poly1305_encrypt_in_place_owned(
            black_box(&cipher_rs),
            black_box(&nonce_rs),
            black_box(AAD),
            black_box(&mut *buf),
          )
          .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    #[cfg(all(rscrypto_internal, feature = "diag", target_arch = "x86_64", target_os = "linux"))]
    if *len != 0 {
      g.bench_function(
        BenchmarkId::new("rscrypto-x86-asm", len),
        checked_with(
          data.clone(),
          |buf| {
            buf.copy_from_slice(data);
            rscrypto::aead::diag_chacha20poly1305_encrypt_in_place_x86_64_asm(
              black_box(&cipher_rs),
              black_box(&nonce_rs),
              black_box(AAD),
              black_box(&mut *buf),
            )
            .expect("x86 asm path must apply to benchmarked non-empty sizes")
            .expect("valid AEAD benchmark operation must succeed")
          },
          |tag, buf| assert_eq!(sealed(buf, tag), expected),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rc
            .encrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
            )
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs-appended-tag", len),
        checked_with(
          Vec::<u8>::with_capacity(data.len().strict_add(16)),
          |buf| {
            buf.clear();
            buf.extend_from_slice(black_box(data));
            aws_key
              .seal_in_place_append_tag(
                aws_aead::Nonce::assume_unique_for_key(NONCE_12),
                aws_aead::Aad::from(AAD),
                black_box(&mut *buf),
              )
              .expect("valid AEAD benchmark operation must succeed");
            black_box(&*buf);
          },
          |(), buf| assert_eq!(*buf, expected),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("ring-appended-tag", len),
      checked_with(
        Vec::<u8>::with_capacity(data.len().strict_add(16)),
        |buf| {
          buf.clear();
          buf.extend_from_slice(black_box(data));
          ring_key
            .seal_in_place_append_tag(
              ring_aead::Nonce::assume_unique_for_key(NONCE_12),
              ring_aead::Aad::from(AAD),
              black_box(&mut *buf),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(*buf, expected),
      ),
    );
  }

  g.finish();
}

fn chacha20_poly1305_decrypt(c: &mut Criterion) {
  if !bench_config::selected("chacha20-poly1305/copy-and-decrypt") {
    return;
  }
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }
  use ring::aead as ring_aead;

  let inputs = common::comp_sizes(&[]);
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
  // Timed: restore input into a preallocated buffer, decrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("chacha20-poly1305/copy-and-decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    // Every row opens RustCrypto's ciphertext and tag and must recover the input.
    let mut ciphertext = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let tag_rs = rscrypto::ChaCha20Poly1305::tag_from_slice(&tag_rc).expect("AEAD tags have 16 bytes");
    let appended = sealed(&ciphertext, tag_rc);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rs
            .decrypt_in_place(
              black_box(&nonce_rs),
              black_box(AAD),
              black_box(&mut *buf),
              black_box(&tag_rs),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    #[cfg(all(rscrypto_internal, feature = "diag"))]
    g.bench_function(
      BenchmarkId::new("rscrypto-owned", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          rscrypto::aead::diag_chacha20poly1305_decrypt_in_place_owned(
            black_box(&cipher_rs),
            black_box(&nonce_rs),
            black_box(AAD),
            black_box(&mut *buf),
            black_box(&tag_rs),
          )
          .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    #[cfg(all(rscrypto_internal, feature = "diag", target_arch = "x86_64", target_os = "linux"))]
    if *len != 0 {
      g.bench_function(
        BenchmarkId::new("rscrypto-x86-asm", len),
        checked_with(
          ciphertext.clone(),
          |buf| {
            buf.copy_from_slice(&ciphertext);
            rscrypto::aead::diag_chacha20poly1305_decrypt_in_place_x86_64_asm(
              black_box(&cipher_rs),
              black_box(&nonce_rs),
              black_box(AAD),
              black_box(&mut *buf),
              black_box(&tag_rs),
            )
            .expect("x86 asm path must apply to benchmarked non-empty sizes")
            .expect("valid AEAD benchmark operation must succeed");
            black_box(&*buf);
          },
          |(), buf| assert_eq!(buf, data),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rc
            .decrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
              black_box(&tag_rc),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs-appended-tag", len),
        checked_with(
          appended.clone(),
          |buf| {
            buf.copy_from_slice(&appended);
            aws_key
              .open_in_place(
                aws_aead::Nonce::assume_unique_for_key(NONCE_12),
                aws_aead::Aad::from(AAD),
                black_box(&mut *buf),
              )
              .expect("valid AEAD benchmark operation must succeed");
            black_box(&*buf);
          },
          |(), buf| assert_eq!(&buf[..*len], data.as_slice()),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("ring-appended-tag", len),
      checked_with(
        appended.clone(),
        |buf| {
          buf.copy_from_slice(&appended);
          ring_key
            .open_in_place(
              ring_aead::Nonce::assume_unique_for_key(NONCE_12),
              ring_aead::Aad::from(AAD),
              black_box(&mut *buf),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(&buf[..*len], data.as_slice()),
      ),
    );
  }

  g.finish();
}

// AES-256-GCM-SIV

fn aes256_gcm_siv_encrypt(c: &mut Criterion) {
  if !bench_config::selected("aes-256-gcm-siv/copy-and-encrypt") {
    return;
  }
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }

  let inputs = common::comp_sizes(&[]);
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::Aes256GcmSiv::new(&rscrypto::Aes256GcmSivKey::from_bytes(KEY_32));
  let cipher_rc = aes_gcm_siv::Aes256GcmSiv::new(&KEY_32.into());
  let nonce_rc = aes_gcm_siv::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key =
      aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::AES_256_GCM_SIV, &KEY_32).expect("valid AEAD benchmark operation must succeed"));
  }
  // Timed: restore input into a preallocated buffer, encrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("aes-256-gcm-siv/copy-and-encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    // Every row must produce RustCrypto's ciphertext and tag.
    let mut ciphertext = data.clone();
    let tag = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let expected = sealed(&ciphertext, tag);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rs
            .encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut *buf))
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rc
            .encrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice()).into(),
            )
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked_with(
          data.clone(),
          |buf| {
            buf.copy_from_slice(black_box(data));
            let tag = aws_key
              .seal_in_place_separate_tag(
                aws_aead::Nonce::assume_unique_for_key(NONCE_12),
                aws_aead::Aad::from(AAD),
                black_box(&mut *buf),
              )
              .expect("valid AEAD benchmark operation must succeed");
            black_box(&*buf);
            tag
          },
          |tag, buf| assert_eq!(sealed(buf, tag), expected),
        ),
      );
    }
  }

  g.finish();
}

fn aes256_gcm_siv_decrypt(c: &mut Criterion) {
  if !bench_config::selected("aes-256-gcm-siv/copy-and-decrypt") {
    return;
  }
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }

  let inputs = common::comp_sizes(&[]);
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::Aes256GcmSiv::new(&rscrypto::Aes256GcmSivKey::from_bytes(KEY_32));
  let cipher_rc = aes_gcm_siv::Aes256GcmSiv::new(&KEY_32.into());
  let nonce_rc = aes_gcm_siv::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key =
      aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::AES_256_GCM_SIV, &KEY_32).expect("valid AEAD benchmark operation must succeed"));
  }
  // Timed: restore input into a preallocated buffer, decrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("aes-256-gcm-siv/copy-and-decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    // Every row opens RustCrypto's ciphertext and tag and must recover the input.
    let mut ciphertext = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let tag_rs = rscrypto::Aes256GcmSiv::tag_from_slice(&tag_rc).expect("AEAD tags have 16 bytes");

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rs
            .decrypt_in_place(
              black_box(&nonce_rs),
              black_box(AAD),
              black_box(&mut *buf),
              black_box(&tag_rs),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rc
            .decrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice()).into(),
              black_box(&tag_rc),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    aws_lc_bench! {
      // AWS-LC exposes detached tags for seal, but its in-place open API takes ct||tag.
      let appended = sealed(&ciphertext, tag_rc);

      g.bench_function(
        BenchmarkId::new("aws-lc-rs-appended-tag", len),
        checked_with(
          appended.clone(),
          |buf| {
            buf.copy_from_slice(&appended);
            aws_key
              .open_in_place(
                aws_aead::Nonce::assume_unique_for_key(NONCE_12),
                aws_aead::Aad::from(AAD),
                black_box(&mut *buf),
              )
              .expect("valid AEAD benchmark operation must succeed");
            black_box(&*buf);
          },
          |(), buf| assert_eq!(&buf[..*len], data.as_slice()),
        ),
      );
    }
  }

  g.finish();
}

// AES-128-GCM-SIV

fn aes128_gcm_siv_encrypt(c: &mut Criterion) {
  if !bench_config::selected("aes-128-gcm-siv/copy-and-encrypt") {
    return;
  }
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }

  let inputs = common::comp_sizes(&[]);
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::Aes128GcmSiv::new(&rscrypto::Aes128GcmSivKey::from_bytes(KEY_16));
  let cipher_rc = aes_gcm_siv::Aes128GcmSiv::new(&KEY_16.into());
  let nonce_rc = aes_gcm_siv::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key =
      aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::AES_128_GCM_SIV, &KEY_16).expect("valid AEAD benchmark operation must succeed"));
  }
  // Timed: restore input into a preallocated buffer, encrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("aes-128-gcm-siv/copy-and-encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    // Every row must produce RustCrypto's ciphertext and tag.
    let mut ciphertext = data.clone();
    let tag = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let expected = sealed(&ciphertext, tag);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rs
            .encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut *buf))
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rc
            .encrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice()).into(),
            )
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked_with(
          data.clone(),
          |buf| {
            buf.copy_from_slice(black_box(data));
            let tag = aws_key
              .seal_in_place_separate_tag(
                aws_aead::Nonce::assume_unique_for_key(NONCE_12),
                aws_aead::Aad::from(AAD),
                black_box(&mut *buf),
              )
              .expect("valid AEAD benchmark operation must succeed");
            black_box(&*buf);
            tag
          },
          |tag, buf| assert_eq!(sealed(buf, tag), expected),
        ),
      );
    }
  }

  g.finish();
}

fn aes128_gcm_siv_decrypt(c: &mut Criterion) {
  if !bench_config::selected("aes-128-gcm-siv/copy-and-decrypt") {
    return;
  }
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }

  let inputs = common::comp_sizes(&[]);
  let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
  let cipher_rs = rscrypto::Aes128GcmSiv::new(&rscrypto::Aes128GcmSivKey::from_bytes(KEY_16));
  let cipher_rc = aes_gcm_siv::Aes128GcmSiv::new(&KEY_16.into());
  let nonce_rc = aes_gcm_siv::Nonce::from(NONCE_12);
  aws_lc_bench! {
    let aws_key =
      aws_aead::LessSafeKey::new(aws_aead::UnboundKey::new(&aws_aead::AES_128_GCM_SIV, &KEY_16).expect("valid AEAD benchmark operation must succeed"));
  }
  // Timed: restore input into a preallocated buffer, decrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("aes-128-gcm-siv/copy-and-decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    // Every row opens RustCrypto's ciphertext and tag and must recover the input.
    let mut ciphertext = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let tag_rs = rscrypto::Aes128GcmSiv::tag_from_slice(&tag_rc).expect("AEAD tags have 16 bytes");

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rs
            .decrypt_in_place(
              black_box(&nonce_rs),
              black_box(AAD),
              black_box(&mut *buf),
              black_box(&tag_rs),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rc
            .decrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice()).into(),
              black_box(&tag_rc),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    aws_lc_bench! {
      // AWS-LC exposes detached tags for seal, but its in-place open API takes ct||tag.
      let appended = sealed(&ciphertext, tag_rc);

      g.bench_function(
        BenchmarkId::new("aws-lc-rs-appended-tag", len),
        checked_with(
          appended.clone(),
          |buf| {
            buf.copy_from_slice(&appended);
            aws_key
              .open_in_place(
                aws_aead::Nonce::assume_unique_for_key(NONCE_12),
                aws_aead::Aad::from(AAD),
                black_box(&mut *buf),
              )
              .expect("valid AEAD benchmark operation must succeed");
            black_box(&*buf);
          },
          |(), buf| assert_eq!(&buf[..*len], data.as_slice()),
        ),
      );
    }
  }

  g.finish();
}

// AES-256-GCM

fn aes256_gcm_encrypt(c: &mut Criterion) {
  if !bench_config::selected("aes-256-gcm/copy-and-encrypt") {
    return;
  }
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }
  use ring::aead as ring_aead;

  let inputs = common::comp_sizes(&[]);
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
  // Timed: restore input into a preallocated buffer, encrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("aes-256-gcm/copy-and-encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    // Every row must produce RustCrypto's ciphertext and tag.
    let mut ciphertext = data.clone();
    let tag = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let expected = sealed(&ciphertext, tag);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rs
            .encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut *buf))
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rc
            .encrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
            )
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked_with(
          data.clone(),
          |buf| {
            buf.copy_from_slice(black_box(data));
            let tag = aws_key
              .seal_in_place_separate_tag(
                aws_aead::Nonce::assume_unique_for_key(NONCE_12),
                aws_aead::Aad::from(AAD),
                black_box(&mut *buf),
              )
              .expect("valid AEAD benchmark operation must succeed");
            black_box(&*buf);
            tag
          },
          |tag, buf| assert_eq!(sealed(buf, tag), expected),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("ring", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(black_box(data));
          let tag = ring_key
            .seal_in_place_separate_tag(
              ring_aead::Nonce::assume_unique_for_key(NONCE_12),
              ring_aead::Aad::from(AAD),
              black_box(&mut *buf),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
          tag
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );
  }

  g.finish();
}

fn aes256_gcm_decrypt(c: &mut Criterion) {
  if !bench_config::selected("aes-256-gcm/copy-and-decrypt") {
    return;
  }
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }
  use ring::aead as ring_aead;

  let inputs = common::comp_sizes(&[]);
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
  // Timed: restore input into a preallocated buffer, decrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("aes-256-gcm/copy-and-decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    // Every row opens RustCrypto's ciphertext and tag and must recover the input.
    let mut ciphertext = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let tag_rs = rscrypto::Aes256Gcm::tag_from_slice(&tag_rc).expect("AEAD tags have 16 bytes");
    let tag_ring = ring_aead::Tag::try_from(&tag_rc[..]).expect("AEAD tags have 16 bytes");

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rs
            .decrypt_in_place(
              black_box(&nonce_rs),
              black_box(AAD),
              black_box(&mut *buf),
              black_box(&tag_rs),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rc
            .decrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
              black_box(&tag_rc),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    aws_lc_bench! {
      // AWS-LC exposes detached tags for seal, but its in-place open API takes ct||tag.
      let appended = sealed(&ciphertext, tag_rc);

      g.bench_function(
        BenchmarkId::new("aws-lc-rs-appended-tag", len),
        checked_with(
          appended.clone(),
          |buf| {
            buf.copy_from_slice(&appended);
            aws_key
              .open_in_place(
                aws_aead::Nonce::assume_unique_for_key(NONCE_12),
                aws_aead::Aad::from(AAD),
                black_box(&mut *buf),
              )
              .expect("valid AEAD benchmark operation must succeed");
            black_box(&*buf);
          },
          |(), buf| assert_eq!(&buf[..*len], data.as_slice()),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("ring", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          ring_key
            .open_in_place_separate_tag(
              ring_aead::Nonce::assume_unique_for_key(NONCE_12),
              ring_aead::Aad::from(AAD),
              black_box(tag_ring),
              black_box(&mut *buf),
              0..,
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );
  }

  g.finish();
}

// AES-128-GCM

fn aes128_gcm_encrypt(c: &mut Criterion) {
  if !bench_config::selected("aes-128-gcm/copy-and-encrypt") {
    return;
  }
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }
  use ring::aead as ring_aead;

  let inputs = common::comp_sizes(&[]);
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
  // Timed: restore input into a preallocated buffer, encrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("aes-128-gcm/copy-and-encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    // Every row must produce RustCrypto's ciphertext and tag.
    let mut ciphertext = data.clone();
    let tag = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let expected = sealed(&ciphertext, tag);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rs
            .encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut *buf))
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rc
            .encrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
            )
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc-rs", len),
        checked_with(
          data.clone(),
          |buf| {
            buf.copy_from_slice(black_box(data));
            let tag = aws_key
              .seal_in_place_separate_tag(
                aws_aead::Nonce::assume_unique_for_key(NONCE_12),
                aws_aead::Aad::from(AAD),
                black_box(&mut *buf),
              )
              .expect("valid AEAD benchmark operation must succeed");
            black_box(&*buf);
            tag
          },
          |tag, buf| assert_eq!(sealed(buf, tag), expected),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("ring", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(black_box(data));
          let tag = ring_key
            .seal_in_place_separate_tag(
              ring_aead::Nonce::assume_unique_for_key(NONCE_12),
              ring_aead::Aad::from(AAD),
              black_box(&mut *buf),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
          tag
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );
  }

  g.finish();
}

fn aes128_gcm_decrypt(c: &mut Criterion) {
  if !bench_config::selected("aes-128-gcm/copy-and-decrypt") {
    return;
  }
  aws_lc_bench! {
    use aws_lc_rs::aead as aws_aead;
  }
  use ring::aead as ring_aead;

  let inputs = common::comp_sizes(&[]);
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
  // Timed: restore input into a preallocated buffer, decrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("aes-128-gcm/copy-and-decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    // Every row opens RustCrypto's ciphertext and tag and must recover the input.
    let mut ciphertext = data.clone();
    let tag_rc = cipher_rc
      .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let tag_rs = rscrypto::Aes128Gcm::tag_from_slice(&tag_rc).expect("AEAD tags have 16 bytes");
    let tag_ring = ring_aead::Tag::try_from(&tag_rc[..]).expect("AEAD tags have 16 bytes");

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rs
            .decrypt_in_place(
              black_box(&nonce_rs),
              black_box(AAD),
              black_box(&mut *buf),
              black_box(&tag_rs),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    g.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rc
            .decrypt_inout_detached(
              black_box(&nonce_rc),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
              black_box(&tag_rc),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    aws_lc_bench! {
      // AWS-LC exposes detached tags for seal, but its in-place open API takes ct||tag.
      let appended = sealed(&ciphertext, tag_rc);

      g.bench_function(
        BenchmarkId::new("aws-lc-rs-appended-tag", len),
        checked_with(
          appended.clone(),
          |buf| {
            buf.copy_from_slice(&appended);
            aws_key
              .open_in_place(
                aws_aead::Nonce::assume_unique_for_key(NONCE_12),
                aws_aead::Aad::from(AAD),
                black_box(&mut *buf),
              )
              .expect("valid AEAD benchmark operation must succeed");
            black_box(&*buf);
          },
          |(), buf| assert_eq!(&buf[..*len], data.as_slice()),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("ring", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          ring_key
            .open_in_place_separate_tag(
              ring_aead::Nonce::assume_unique_for_key(NONCE_12),
              ring_aead::Aad::from(AAD),
              black_box(tag_ring),
              black_box(&mut *buf),
              0..,
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );
  }

  g.finish();
}

// AEGIS-256

fn aegis256_encrypt(c: &mut Criterion) {
  if !bench_config::selected("aegis-256/copy-and-encrypt") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let nonce_rs = rscrypto::aead::Nonce256::from_bytes(NONCE_32);
  let cipher_rs = rscrypto::Aegis256::new(&rscrypto::Aegis256Key::from_bytes(KEY_32));
  // Timed: restore input into a preallocated buffer, encrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("aegis-256/copy-and-encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let cipher_ac = aegis::aegis256::Aegis256::<16>::new(&KEY_32, &NONCE_32);
    // Every row must produce the aegis crate's ciphertext and tag.
    let mut ciphertext = data.clone();
    let tag = cipher_ac.encrypt_in_place(&mut ciphertext, AAD);
    let expected = sealed(&ciphertext, tag);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rs
            .encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut *buf))
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    g.bench_function(
      BenchmarkId::new("aegis-crate", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_ac.encrypt_in_place(black_box(&mut *buf), black_box(AAD))
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );
  }

  g.finish();
}

fn aegis256_decrypt(c: &mut Criterion) {
  if !bench_config::selected("aegis-256/copy-and-decrypt") {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let nonce_rs = rscrypto::aead::Nonce256::from_bytes(NONCE_32);
  let cipher_rs = rscrypto::Aegis256::new(&rscrypto::Aegis256Key::from_bytes(KEY_32));
  // Timed: restore input into a preallocated buffer, decrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("aegis-256/copy-and-decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    // Every row opens the aegis crate's ciphertext and tag and must recover the input.
    let cipher_ac = aegis::aegis256::Aegis256::<16>::new(&KEY_32, &NONCE_32);
    let mut ciphertext = data.clone();
    let tag_ac = cipher_ac.encrypt_in_place(&mut ciphertext, AAD);
    let tag_rs = rscrypto::Aegis256::tag_from_slice(&tag_ac).expect("AEAD tags have 16 bytes");

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rs
            .decrypt_in_place(
              black_box(&nonce_rs),
              black_box(AAD),
              black_box(&mut *buf),
              black_box(&tag_rs),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    g.bench_function(
      BenchmarkId::new("aegis-crate", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_ac
            .decrypt_in_place(black_box(&mut *buf), black_box(&tag_ac), black_box(AAD))
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );
  }

  g.finish();
}

// Ascon-AEAD128

fn ascon_aead128_encrypt(c: &mut Criterion) {
  if !bench_config::selected("ascon-aead128/copy-and-encrypt") {
    return;
  }
  use ascon_aead::aead::{AeadInOut as _, KeyInit as _, array::Array};

  let inputs = common::comp_sizes(&[]);
  let nonce_rs = rscrypto::aead::Nonce128::from_bytes(NONCE_16);
  let cipher_rs = rscrypto::AsconAead128::new(&rscrypto::AsconAead128Key::from_bytes(KEY_16));
  let cipher_ac = ascon_aead::AsconAead128::new(&Array(KEY_16));
  let nonce_ac = Array(NONCE_16);
  // Timed: restore input into a preallocated buffer, encrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("ascon-aead128/copy-and-encrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    // Every row must produce the ascon-aead crate's ciphertext and tag.
    let mut ciphertext = data.clone();
    let tag = cipher_ac
      .encrypt_inout_detached(&nonce_ac, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let expected = sealed(&ciphertext, tag);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_rs
            .encrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut *buf))
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );

    g.bench_function(
      BenchmarkId::new("ascon-aead", len),
      checked_with(
        data.clone(),
        |buf| {
          buf.copy_from_slice(data);
          cipher_ac
            .encrypt_inout_detached(
              black_box(&nonce_ac),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
            )
            .expect("valid AEAD benchmark operation must succeed")
        },
        |tag, buf| assert_eq!(sealed(buf, tag), expected),
      ),
    );
  }

  g.finish();
}

fn ascon_aead128_decrypt(c: &mut Criterion) {
  if !bench_config::selected("ascon-aead128/copy-and-decrypt") {
    return;
  }
  use ascon_aead::aead::{AeadInOut as _, KeyInit as _, array::Array};

  let inputs = common::comp_sizes(&[]);
  let nonce_rs = rscrypto::aead::Nonce128::from_bytes(NONCE_16);
  let cipher_rs = rscrypto::AsconAead128::new(&rscrypto::AsconAead128Key::from_bytes(KEY_16));
  let cipher_ac = ascon_aead::AsconAead128::new(&Array(KEY_16));
  let nonce_ac = Array(NONCE_16);
  // Timed: restore input into a preallocated buffer, decrypt, and handle/drop per-call outputs.
  // Untimed: fixtures, buffer allocation, and reusable cipher construction/destruction.
  let mut g = c.benchmark_group("ascon-aead128/copy-and-decrypt");

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);

    // Every row opens the ascon-aead crate's ciphertext and tag and must recover the input.
    let mut ciphertext = data.clone();
    let tag_ac = cipher_ac
      .encrypt_inout_detached(&nonce_ac, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let tag_rs = rscrypto::AsconAead128::tag_from_slice(&tag_ac).expect("AEAD tags have 16 bytes");

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_rs
            .decrypt_in_place(
              black_box(&nonce_rs),
              black_box(AAD),
              black_box(&mut *buf),
              black_box(&tag_rs),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );

    g.bench_function(
      BenchmarkId::new("ascon-aead", len),
      checked_with(
        ciphertext.clone(),
        |buf| {
          buf.copy_from_slice(&ciphertext);
          cipher_ac
            .decrypt_inout_detached(
              black_box(&nonce_ac),
              black_box(AAD),
              black_box(buf.as_mut_slice().into()),
              black_box(&tag_ac),
            )
            .expect("valid AEAD benchmark operation must succeed");
          black_box(&*buf);
        },
        |(), buf| assert_eq!(buf, data),
      ),
    );
  }

  g.finish();
}

// Fixed-size header protection (RFC 9001 section 5.4)

/// A header-protection vector: key, 16-byte sample, and five-byte mask.
type MaskVector<const N: usize> = ([u8; N], [u8; 16], [u8; 5]);

/// Decodes a fixed-length hexadecimal test vector.
fn hex<const N: usize>(text: &str) -> [u8; N] {
  assert_eq!(text.len(), N.strict_mul(2), "hex vector length");
  let mut out = [0u8; N];
  for (byte, pair) in out.iter_mut().zip(text.as_bytes().as_chunks::<2>().0) {
    let pair = core::str::from_utf8(pair).expect("hex vector is ASCII");
    *byte = u8::from_str_radix(pair, 16).expect("hex vector digit pair");
  }
  out
}

/// One library's header-protection API for one algorithm.
struct MaskApi<New, Mask> {
  /// Builds the reusable context from key bytes.
  new: New,
  /// Computes the mask of one sample with that context.
  mask: Mask,
}

impl<New, Mask> MaskApi<New, Mask> {
  /// The mask of `sample` under a new context for `key`.
  fn mask_of<const N: usize, C>(&self, key: &[u8; N], sample: &[u8; 16]) -> [u8; 5]
  where
    New: Fn(&[u8; N]) -> C,
    Mask: Fn(&C, &[u8; 16]) -> [u8; 5],
  {
    (self.mask)(&(self.new)(key), sample)
  }
}

/// One algorithm's header-protection fixtures and expected masks.
struct MaskCase<const N: usize> {
  algorithm: &'static str,
  key: [u8; N],
  sample: [u8; 16],
  /// The mask of `sample` under `key`.
  expected: [u8; 5],
  /// Published vectors that every row's API must reproduce.
  published: Vec<MaskVector<N>>,
}

impl<const N: usize> MaskCase<N> {
  /// Builds a case whose expected mask is the one that every library in
  /// `independent` computed for `key` and `sample`.
  fn agreed(
    algorithm: &'static str,
    key: [u8; N],
    sample: [u8; 16],
    published: Vec<MaskVector<N>>,
    independent: &[[u8; 5]],
  ) -> Self {
    let expected = independent[0];
    assert!(
      independent.iter().all(|mask| *mask == expected),
      "independent libraries disagree on the {algorithm} mask: {independent:02x?}"
    );
    Self {
      algorithm,
      key,
      sample,
      expected,
      published,
    }
  }

  /// Asserts that `mask`, computed for this case's key and sample, is the
  /// expected mask, and that `api` reproduces every published vector.
  fn check<C, New, Mask>(&self, api: &MaskApi<New, Mask>, mask: [u8; 5])
  where
    New: Fn(&[u8; N]) -> C,
    Mask: Fn(&C, &[u8; 16]) -> [u8; 5],
  {
    assert_eq!(mask, self.expected, "{} mask of the bench sample", self.algorithm);
    for (key, sample, published) in &self.published {
      assert_eq!(
        api.mask_of(key, sample),
        *published,
        "{} published mask",
        self.algorithm
      );
    }
  }
}

/// Registers `library`'s row that builds a context from key bytes, then destroys it.
fn construct_row<const N: usize, C, New, Mask>(
  group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
  library: &str,
  case: &MaskCase<N>,
  api: &MaskApi<New, Mask>,
) where
  New: Fn(&[u8; N]) -> C,
  Mask: Fn(&C, &[u8; 16]) -> [u8; 5],
{
  group.bench_function(
    BenchmarkId::new(library, case.algorithm),
    checked(
      || (api.new)(black_box(&case.key)),
      |context| case.check(api, (api.mask)(&context, &case.sample)),
    ),
  );
}

/// Registers `library`'s row that computes one sample's mask with a reused context.
fn mask_row<const N: usize, C, New, Mask>(
  group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
  library: &str,
  case: &MaskCase<N>,
  api: &MaskApi<New, Mask>,
) where
  New: Fn(&[u8; N]) -> C,
  Mask: Fn(&C, &[u8; 16]) -> [u8; 5],
{
  let context = (api.new)(&case.key);
  group.bench_function(
    BenchmarkId::new(library, case.algorithm),
    checked(
      || (api.mask)(black_box(&context), black_box(&case.sample)),
      |mask| case.check(api, mask),
    ),
  );
}

/// RustCrypto's composed AES mask: encrypt the sample as one block, then keep
/// its first five bytes.
fn rustcrypto_aes_mask<B>(cipher: &B, sample: &[u8; 16]) -> [u8; 5]
where
  B: aes::cipher::BlockCipherEncrypt<BlockSize = aes::cipher::consts::U16>,
{
  let mut block = aes::cipher::Array::from(*sample);
  cipher.encrypt_block(&mut block);
  block[..5].try_into().expect("five-byte prefix has fixed length")
}

/// RustCrypto's composed ChaCha20 mask. RustCrypto has no key-only ChaCha20
/// object, so each call keys a new core with the sample's nonce, sets the
/// sample's block counter, and keeps the first five keystream bytes.
fn rustcrypto_chacha20_mask(key: &[u8; 32], sample: &[u8; 16]) -> [u8; 5] {
  use chacha20::cipher::{KeyIvInit as _, StreamCipherCore as _};
  let counter = u32::from_le_bytes(sample[..4].try_into().expect("counter prefix has fixed length"));
  let nonce: &[u8; 12] = sample[4..].try_into().expect("nonce suffix has fixed length");
  let mut core = chacha20::ChaChaCore::<chacha20::R20, chacha20::variants::Ietf>::new(key.into(), nonce.into());
  core.set_block_pos(counter);
  let mut block = chacha20::cipher::array::Array::<u8, chacha20::cipher::consts::U64>::default();
  core.write_keystream_block(&mut block);
  block[..5].try_into().expect("five-byte prefix has fixed length")
}

/// ring's QUIC header-protection mask.
fn ring_mask(key: &ring::aead::quic::HeaderProtectionKey, sample: &[u8; 16]) -> [u8; 5] {
  key.new_mask(sample).expect("a 16-byte sample is valid")
}

aws_lc_bench! {
  /// aws-lc-rs's QUIC header-protection mask.
  fn aws_lc_mask(key: &aws_lc_rs::aead::quic::HeaderProtectionKey, sample: &[u8; 16]) -> [u8; 5] {
    key.new_mask(sample).expect("a 16-byte sample is valid")
  }
}

fn header_protection(c: &mut Criterion) {
  if !bench_config::selected("header-protection/") {
    return;
  }
  use aes::cipher::{Array, KeyInit as _};
  aws_lc_bench! {
    use aws_lc_rs::aead::quic as aws_quic;
  }
  use ring::aead::quic as ring_quic;
  use rscrypto::aead::expert::header_protection::{
    Aes128HeaderProtection, Aes128HeaderProtectionKey, Aes256HeaderProtection, Aes256HeaderProtectionKey,
    ChaCha20HeaderProtection, ChaCha20HeaderProtectionKey,
  };

  // rscrypto, ring, and aws-lc-rs expose a header-protection key type. RustCrypto has none:
  // its AES context is an encrypt-only block cipher, and its ChaCha20 context is the raw key.
  let rscrypto_aes128 = MaskApi {
    new: |key: &[u8; 16]| Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes(*key)),
    mask: Aes128HeaderProtection::mask,
  };
  let rscrypto_aes256 = MaskApi {
    new: |key: &[u8; 32]| Aes256HeaderProtection::new(&Aes256HeaderProtectionKey::from_bytes(*key)),
    mask: Aes256HeaderProtection::mask,
  };
  let rscrypto_chacha20 = MaskApi {
    new: |key: &[u8; 32]| ChaCha20HeaderProtection::new(&ChaCha20HeaderProtectionKey::from_bytes(*key)),
    mask: ChaCha20HeaderProtection::mask,
  };
  let rustcrypto_aes128 = MaskApi {
    new: |key: &[u8; 16]| aes::Aes128Enc::new(&Array::from(*key)),
    mask: rustcrypto_aes_mask::<aes::Aes128Enc>,
  };
  let rustcrypto_aes256 = MaskApi {
    new: |key: &[u8; 32]| aes::Aes256Enc::new(&Array::from(*key)),
    mask: rustcrypto_aes_mask::<aes::Aes256Enc>,
  };
  let rustcrypto_chacha20 = MaskApi {
    new: |key: &[u8; 32]| *key,
    mask: rustcrypto_chacha20_mask,
  };
  let ring_aes128 = MaskApi {
    new: |key: &[u8; 16]| {
      ring_quic::HeaderProtectionKey::new(&ring_quic::AES_128, key).expect("valid header-protection key")
    },
    mask: ring_mask,
  };
  let ring_aes256 = MaskApi {
    new: |key: &[u8; 32]| {
      ring_quic::HeaderProtectionKey::new(&ring_quic::AES_256, key).expect("valid header-protection key")
    },
    mask: ring_mask,
  };
  let ring_chacha20 = MaskApi {
    new: |key: &[u8; 32]| {
      ring_quic::HeaderProtectionKey::new(&ring_quic::CHACHA20, key).expect("valid header-protection key")
    },
    mask: ring_mask,
  };
  aws_lc_bench! {
    let aws_aes128 = MaskApi {
      new: |key: &[u8; 16]| {
        aws_quic::HeaderProtectionKey::new(&aws_quic::AES_128, key).expect("valid header-protection key")
      },
      mask: aws_lc_mask,
    };
    let aws_aes256 = MaskApi {
      new: |key: &[u8; 32]| {
        aws_quic::HeaderProtectionKey::new(&aws_quic::AES_256, key).expect("valid header-protection key")
      },
      mask: aws_lc_mask,
    };
    let aws_chacha20 = MaskApi {
      new: |key: &[u8; 32]| {
        aws_quic::HeaderProtectionKey::new(&aws_quic::CHACHA20, key).expect("valid header-protection key")
      },
      mask: aws_lc_mask,
    };
  }

  // Before timing, every row's API must reproduce the RFC 9001 Appendix A.2 (client
  // Initial) AES-128 mask and the A.5 (ChaCha20-Poly1305 short header) ChaCha20 mask.
  // Its bench mask must also match the one on which RustCrypto and ring agree. That
  // agreement is the only oracle for AES-256, which has no published vector.
  let sample = [0x07; 16];
  let aes128 = MaskCase::agreed(
    "aes128",
    KEY_16,
    sample,
    vec![(
      hex("9f50449e04a0e810283a1e9933adedd2"),
      hex("d1b1c98dd7689fb8ec11d242b123dc9b"),
      hex("437b9aec36"),
    )],
    &[
      rustcrypto_aes128.mask_of(&KEY_16, &sample),
      ring_aes128.mask_of(&KEY_16, &sample),
    ],
  );
  let aes256 = MaskCase::agreed(
    "aes256",
    KEY_32,
    sample,
    Vec::new(),
    &[
      rustcrypto_aes256.mask_of(&KEY_32, &sample),
      ring_aes256.mask_of(&KEY_32, &sample),
    ],
  );
  let chacha20 = MaskCase::agreed(
    "chacha20",
    KEY_32,
    sample,
    vec![(
      hex("25a282b9e82f06f21f488917a4fc8f1b73573685608597d0efcb076b0ab7a7a4"),
      hex("5e5cd55c41f69080575d7999c25a5bfb"),
      hex("aefefe7d03"),
    )],
    &[
      rustcrypto_chacha20.mask_of(&KEY_32, &sample),
      ring_chacha20.mask_of(&KEY_32, &sample),
    ],
  );

  // Task: build a reusable header-protection context from borrowed key bytes, then destroy it.
  // Timed: construction and the context's drop. Untimed: key and sample fixtures.
  // rscrypto copies the bytes into its typed key, expands an encrypt-only AES schedule or
  // copies the ChaCha20 key, and zeroizes the typed key and the context on drop.
  // ring builds an encrypt-only AES schedule or copies the ChaCha20 key. It does not zeroize.
  // aws-lc-rs builds the AES encrypt schedule plus a decrypt schedule that masks never use.
  // It zeroizes both AES schedules or the ChaCha20 key on drop. ring and aws-lc-rs return one enum for all three
  // algorithms, so their ChaCha20 contexts are as large as their AES contexts.
  // RustCrypto builds an encrypt-only `aes` cipher. This build does not enable the `aes`
  // `zeroize` feature, so its drop leaves the schedule in memory. RustCrypto has no key-only
  // ChaCha20 object, so it has no ChaCha20 row.
  let mut construction = c.benchmark_group("header-protection/construct");
  construct_row(&mut construction, "rscrypto", &aes128, &rscrypto_aes128);
  construct_row(&mut construction, "rustcrypto-composed", &aes128, &rustcrypto_aes128);
  construct_row(&mut construction, "ring", &aes128, &ring_aes128);
  construct_row(&mut construction, "rscrypto", &aes256, &rscrypto_aes256);
  construct_row(&mut construction, "rustcrypto-composed", &aes256, &rustcrypto_aes256);
  construct_row(&mut construction, "ring", &aes256, &ring_aes256);
  construct_row(&mut construction, "rscrypto", &chacha20, &rscrypto_chacha20);
  construct_row(&mut construction, "ring", &chacha20, &ring_chacha20);
  aws_lc_bench! {
    construct_row(&mut construction, "aws-lc-rs", &aes128, &aws_aes128);
    construct_row(&mut construction, "aws-lc-rs", &aes256, &aws_aes256);
    construct_row(&mut construction, "aws-lc-rs", &chacha20, &aws_chacha20);
  }
  construction.finish();

  // Task: compute the five-byte mask of one 16-byte sample with a reused context.
  // Timed: one mask call through an opaque context reference. Untimed: fixtures, and context
  // construction and destruction.
  // rscrypto takes a sample array. ring and aws-lc-rs take a slice, check its length, and
  // return a `Result`. rscrypto clears its temporary ChaCha20 block and, without AES-NI or
  // AES-CE, its temporary AES block. ring's hardware AES path computes the block with its
  // CTR routine over one zero block.
  // RustCrypto composes primitives. Its AES row encrypts a full block and keeps five bytes.
  // Its ChaCha20 row keys a new core with the sample's nonce on every call, and its SIMD
  // backends compute a batch (eight blocks on NEON, four on SSE2 and AVX2) to return one
  // block. This build does not enable the `chacha20` `zeroize` feature, so that core is not
  // cleared.
  let mut mask = c.benchmark_group("header-protection/mask");
  mask_row(&mut mask, "rscrypto", &aes128, &rscrypto_aes128);
  mask_row(&mut mask, "rustcrypto-composed", &aes128, &rustcrypto_aes128);
  mask_row(&mut mask, "ring", &aes128, &ring_aes128);
  mask_row(&mut mask, "rscrypto", &aes256, &rscrypto_aes256);
  mask_row(&mut mask, "rustcrypto-composed", &aes256, &rustcrypto_aes256);
  mask_row(&mut mask, "ring", &aes256, &ring_aes256);
  mask_row(&mut mask, "rscrypto", &chacha20, &rscrypto_chacha20);
  mask_row(&mut mask, "rustcrypto-composed", &chacha20, &rustcrypto_chacha20);
  mask_row(&mut mask, "ring", &chacha20, &ring_chacha20);
  aws_lc_bench! {
    mask_row(&mut mask, "aws-lc-rs", &aes128, &aws_aes128);
    mask_row(&mut mask, "aws-lc-rs", &aes256, &aws_aes256);
    mask_row(&mut mask, "aws-lc-rs", &chacha20, &aws_chacha20);
  }
  mask.finish();
}

// AES-SIV-CMAC-256 (RFC 5297 nonce-based profile)

fn aes_siv_cmac256(c: &mut Criterion) {
  if !bench_config::selected("aes-siv-cmac-256/") {
    return;
  }
  use aes_siv::{KeyInit as _, siv::Aes128Siv};
  use rscrypto::{AesSivCmac256, AesSivCmac256Key, AesSivCmac256Nonce};

  let nonce_rs = AesSivCmac256Nonce::try_from(NONCE_16.as_slice()).expect("benchmark nonce is non-empty");
  let key_rs = AesSivCmac256Key::from_bytes(KEY_32);
  let key_rc: aes_siv::Key<Aes128Siv> = KEY_32.into();

  // NTS packet shapes: empty authenticator payload, short field, extension-field,
  // cookie-shaped, and near-MTU protected bodies.
  let inputs = [0usize, 16, 64, 256, 1232]
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect::<Vec<_>>();
  let cipher_rs = AesSivCmac256::new(&key_rs);
  let mut cipher_rc = Aes128Siv::new(&key_rc);
  let headers: [&[u8]; 2] = [AAD, &NONCE_16];
  // RustCrypto's ciphertext and tag for one input; every row must produce or open them.
  let reference = |data: &[u8]| {
    let mut ciphertext = data.to_vec();
    let tag = Aes128Siv::new(&key_rc)
      .encrypt_inout_detached(headers, ciphertext.as_mut_slice().into())
      .expect("valid benchmark input must seal");
    (ciphertext, tag)
  };

  // Timed: restore plaintext, construct a cipher, seal, and destroy per-call objects.
  // Untimed: key/nonce fixtures and buffer allocation. This is one cipher lifecycle, not packet I/O.
  let mut construct_and_seal = c.benchmark_group("aes-siv-cmac-256/copy-and-construct-and-seal");
  for (len, data) in &inputs {
    common::set_throughput(&mut construct_and_seal, *len);
    let (ciphertext, tag) = reference(data.as_slice());
    let expected = sealed(&ciphertext, tag);

    construct_and_seal.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        data.clone(),
        |buffer| {
          buffer.copy_from_slice(data);
          let cipher = AesSivCmac256::new(black_box(&key_rs));
          cipher.seal_in_place(black_box(nonce_rs), black_box(AAD), black_box(&mut *buffer))
        },
        |tag, buffer| assert_eq!(sealed(buffer, tag), expected),
      ),
    );

    construct_and_seal.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        data.clone(),
        |buffer| {
          buffer.copy_from_slice(data);
          let mut cipher = Aes128Siv::new(black_box(&key_rc));
          cipher
            .encrypt_inout_detached(black_box(headers), black_box(buffer.as_mut_slice().into()))
            .expect("valid benchmark input must seal")
        },
        |tag, buffer| assert_eq!(sealed(buffer, tag), expected),
      ),
    );
  }
  construct_and_seal.finish();

  // Timed: restore plaintext and seal; buffer allocation and reusable ciphers are untimed.
  let mut seal = c.benchmark_group("aes-siv-cmac-256/copy-and-seal");

  for (len, data) in &inputs {
    common::set_throughput(&mut seal, *len);
    let (ciphertext, tag) = reference(data.as_slice());
    let expected = sealed(&ciphertext, tag);

    seal.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        data.clone(),
        |buffer| {
          buffer.copy_from_slice(data);
          cipher_rs.seal_in_place(black_box(nonce_rs), black_box(AAD), black_box(&mut *buffer))
        },
        |tag, buffer| assert_eq!(sealed(buffer, tag), expected),
      ),
    );

    seal.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        data.clone(),
        |buffer| {
          buffer.copy_from_slice(data);
          cipher_rc
            .encrypt_inout_detached(black_box(headers), black_box(buffer.as_mut_slice().into()))
            .expect("valid benchmark input must seal")
        },
        |tag, buffer| assert_eq!(sealed(buffer, tag), expected),
      ),
    );
  }
  seal.finish();

  // Timed: restore ciphertext and open; fixture generation, buffers, and reusable ciphers are untimed.
  let mut open = c.benchmark_group("aes-siv-cmac-256/copy-and-open");
  for (len, data) in &inputs {
    common::set_throughput(&mut open, *len);
    let (ciphertext, tag_rc) = reference(data.as_slice());
    let tag_rs = AesSivCmac256::tag_from_slice(&tag_rc).expect("SIV tags have 16 bytes");

    open.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        ciphertext.clone(),
        |buffer| {
          buffer.copy_from_slice(&ciphertext);
          cipher_rs
            .open_in_place(
              black_box(nonce_rs),
              black_box(AAD),
              black_box(&mut *buffer),
              black_box(&tag_rs),
            )
            .expect("valid benchmark ciphertext must open");
          black_box(&*buffer);
        },
        |(), buffer| assert_eq!(buffer, data),
      ),
    );

    open.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked_with(
        ciphertext.clone(),
        |buffer| {
          buffer.copy_from_slice(&ciphertext);
          cipher_rc
            .decrypt_inout_detached(
              black_box(headers),
              black_box(buffer.as_mut_slice().into()),
              black_box(&tag_rc),
            )
            .expect("valid benchmark ciphertext must open");
          black_box(&*buffer);
        },
        |(), buffer| assert_eq!(buffer, data),
      ),
    );
  }
  open.finish();
}

// One message per key

/// Message lengths for one message per key: a short record and a typical one. Key setup is a fixed cost, so a longer
/// message only dilutes it; its `copy-and-encrypt` row gives the remaining cost.
const CONSTRUCT_SIZES: [usize; 2] = [64, 1024];

fn construct_inputs() -> Vec<(usize, Vec<u8>)> {
  CONSTRUCT_SIZES
    .into_iter()
    .map(|len| (len, common::random_bytes(len)))
    .collect()
}

/// Registers `library`'s one-message-per-key row. `encrypt` builds a cipher from the key bytes, encrypts the buffer
/// in place, and returns the detached tag; the cipher is destroyed inside the call. The ciphertext and tag must
/// equal `expected`.
fn construct_and_encrypt_row<T: AsRef<[u8]>>(
  group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
  library: &str,
  data: &[u8],
  expected: &[u8],
  mut encrypt: impl FnMut(&mut [u8]) -> T,
) {
  group.bench_function(
    BenchmarkId::new(library, data.len()),
    checked_with(
      data.to_vec(),
      |buf| {
        buf.copy_from_slice(data);
        encrypt(black_box(buf.as_mut_slice()))
      },
      |tag, buf| assert_eq!(sealed(buf, tag), expected),
    ),
  );
}

/// ring: build an `UnboundKey` and a `LessSafeKey` from key bytes, then seal in place with a detached tag.
fn ring_construct_and_seal(algorithm: &'static ring::aead::Algorithm, key: &[u8], buf: &mut [u8]) -> ring::aead::Tag {
  use ring::aead as ring_aead;

  ring_aead::LessSafeKey::new(
    ring_aead::UnboundKey::new(algorithm, key).expect("valid AEAD benchmark operation must succeed"),
  )
  .seal_in_place_separate_tag(
    ring_aead::Nonce::assume_unique_for_key(NONCE_12),
    ring_aead::Aad::from(AAD),
    buf,
  )
  .expect("valid AEAD benchmark operation must succeed")
}

aws_lc_bench! {
  /// aws-lc-rs: build an `UnboundKey` and a `LessSafeKey` from key bytes, then seal in place with a detached tag.
  fn aws_construct_and_seal(
    algorithm: &'static aws_lc_rs::aead::Algorithm,
    key: &[u8],
    buf: &mut [u8],
  ) -> aws_lc_rs::aead::Tag {
    use aws_lc_rs::aead as aws_aead;

    aws_aead::LessSafeKey::new(
      aws_aead::UnboundKey::new(algorithm, key).expect("valid AEAD benchmark operation must succeed"),
    )
    .seal_in_place_separate_tag(
      aws_aead::Nonce::assume_unique_for_key(NONCE_12),
      aws_aead::Aad::from(AAD),
      buf,
    )
    .expect("valid AEAD benchmark operation must succeed")
  }
}

// One message per key: restore the input into a reused buffer, build a cipher from the key bytes, encrypt with a
// detached tag, and destroy the cipher, in each timed call. This is the `copy-and-encrypt` workload plus key setup.
// - rscrypto copies the key into its typed key, expands it, and zeroizes both on drop.
// - RustCrypto builds its cipher with `KeyInit::new`; ring and aws-lc-rs build an `UnboundKey` and a `LessSafeKey`.
// - The AES-GCM rows also derive the GHASH key, and each library precomputes what its GHASH backend needs.
// Untimed: fixtures and the buffer allocation. Every row must produce RustCrypto's ciphertext and tag.
macro_rules! construct_and_encrypt_bench {
  (
    $name:ident,
    $group:literal,
    rscrypto: $cipher:ty, $key:ty, $nonce:ty,
    rustcrypto: $rustcrypto:ty, $rustcrypto_nonce:expr,
    key: $key_bytes:expr,
    nonce: $nonce_bytes:expr,
    ring: [$($ring_algorithm:path)?],
    aws_lc: [$($aws_algorithm:path)?] $(,)?
  ) => {
    fn $name(c: &mut Criterion) {
      if !bench_config::selected($group) {
        return;
      }
      let nonce_rs = <$nonce>::from_bytes($nonce_bytes);
      let nonce_rc = $rustcrypto_nonce;
      let mut g = c.benchmark_group($group);

      for (len, data) in construct_inputs() {
        common::set_throughput(&mut g, len);
        let mut ciphertext = data.clone();
        let tag = <$rustcrypto>::new(&$key_bytes.into())
          .encrypt_inout_detached(&nonce_rc, AAD, ciphertext.as_mut_slice().into())
          .expect("valid AEAD benchmark operation must succeed");
        let expected = sealed(&ciphertext, tag);

        construct_and_encrypt_row(&mut g, "rscrypto", &data, &expected, |buf| {
          <$cipher>::new(&<$key>::from_bytes(black_box($key_bytes)))
            .encrypt_in_place(black_box(&nonce_rs), black_box(AAD), buf)
            .expect("valid AEAD benchmark operation must succeed")
        });
        construct_and_encrypt_row(&mut g, "rustcrypto", &data, &expected, |buf| {
          <$rustcrypto>::new(&black_box($key_bytes).into())
            .encrypt_inout_detached(black_box(&nonce_rc), black_box(AAD), buf.into())
            .expect("valid AEAD benchmark operation must succeed")
        });
        $(
          aws_lc_bench! {
            construct_and_encrypt_row(&mut g, "aws-lc-rs", &data, &expected, |buf| {
              aws_construct_and_seal(&$aws_algorithm, &black_box($key_bytes), buf)
            });
          }
        )?
        $(
          construct_and_encrypt_row(&mut g, "ring", &data, &expected, |buf| {
            ring_construct_and_seal(&$ring_algorithm, &black_box($key_bytes), buf)
          });
        )?
      }

      g.finish();
    }
  };
}

construct_and_encrypt_bench!(
  aes128_gcm_construct_and_encrypt,
  "aes-128-gcm/copy-and-construct-and-encrypt",
  rscrypto: rscrypto::Aes128Gcm, rscrypto::Aes128GcmKey, rscrypto::aead::Nonce96,
  rustcrypto: aes_gcm::Aes128Gcm, aes_gcm::Nonce::from(NONCE_12),
  key: KEY_16,
  nonce: NONCE_12,
  ring: [ring::aead::AES_128_GCM],
  aws_lc: [aws_lc_rs::aead::AES_128_GCM],
);
construct_and_encrypt_bench!(
  aes256_gcm_construct_and_encrypt,
  "aes-256-gcm/copy-and-construct-and-encrypt",
  rscrypto: rscrypto::Aes256Gcm, rscrypto::Aes256GcmKey, rscrypto::aead::Nonce96,
  rustcrypto: aes_gcm::Aes256Gcm, aes_gcm::Nonce::from(NONCE_12),
  key: KEY_32,
  nonce: NONCE_12,
  ring: [ring::aead::AES_256_GCM],
  aws_lc: [aws_lc_rs::aead::AES_256_GCM],
);
construct_and_encrypt_bench!(
  aes128_gcm_siv_construct_and_encrypt,
  "aes-128-gcm-siv/copy-and-construct-and-encrypt",
  rscrypto: rscrypto::Aes128GcmSiv, rscrypto::Aes128GcmSivKey, rscrypto::aead::Nonce96,
  rustcrypto: aes_gcm_siv::Aes128GcmSiv, aes_gcm_siv::Nonce::from(NONCE_12),
  key: KEY_16,
  nonce: NONCE_12,
  ring: [],
  aws_lc: [aws_lc_rs::aead::AES_128_GCM_SIV],
);
construct_and_encrypt_bench!(
  aes256_gcm_siv_construct_and_encrypt,
  "aes-256-gcm-siv/copy-and-construct-and-encrypt",
  rscrypto: rscrypto::Aes256GcmSiv, rscrypto::Aes256GcmSivKey, rscrypto::aead::Nonce96,
  rustcrypto: aes_gcm_siv::Aes256GcmSiv, aes_gcm_siv::Nonce::from(NONCE_12),
  key: KEY_32,
  nonce: NONCE_12,
  ring: [],
  aws_lc: [aws_lc_rs::aead::AES_256_GCM_SIV],
);
construct_and_encrypt_bench!(
  chacha20_poly1305_construct_and_encrypt,
  "chacha20-poly1305/copy-and-construct-and-encrypt",
  rscrypto: rscrypto::ChaCha20Poly1305, rscrypto::ChaCha20Poly1305Key, rscrypto::aead::Nonce96,
  rustcrypto: chacha20poly1305::ChaCha20Poly1305, chacha20poly1305::Nonce::from(NONCE_12),
  key: KEY_32,
  nonce: NONCE_12,
  ring: [ring::aead::CHACHA20_POLY1305],
  aws_lc: [aws_lc_rs::aead::CHACHA20_POLY1305],
);
construct_and_encrypt_bench!(
  xchacha20_poly1305_construct_and_encrypt,
  "xchacha20-poly1305/copy-and-construct-and-encrypt",
  rscrypto: rscrypto::XChaCha20Poly1305, rscrypto::XChaCha20Poly1305Key, rscrypto::aead::Nonce192,
  rustcrypto: chacha20poly1305::XChaCha20Poly1305, chacha20poly1305::XNonce::from(NONCE_24),
  key: KEY_32,
  nonce: NONCE_24,
  ring: [],
  aws_lc: [],
);

// AEGIS-256 one message per key: as above. The aegis crate binds the nonce when it builds its cipher, so its row
// builds one in every call as well. Every row must produce the aegis crate's ciphertext and tag.
fn aegis256_construct_and_encrypt(c: &mut Criterion) {
  if !bench_config::selected("aegis-256/copy-and-construct-and-encrypt") {
    return;
  }
  let nonce_rs = rscrypto::aead::Nonce256::from_bytes(NONCE_32);
  let mut g = c.benchmark_group("aegis-256/copy-and-construct-and-encrypt");

  for (len, data) in construct_inputs() {
    common::set_throughput(&mut g, len);
    let mut ciphertext = data.clone();
    let tag = aegis::aegis256::Aegis256::<16>::new(&KEY_32, &NONCE_32).encrypt_in_place(&mut ciphertext, AAD);
    let expected = sealed(&ciphertext, tag);

    construct_and_encrypt_row(&mut g, "rscrypto", &data, &expected, |buf| {
      rscrypto::Aegis256::new(&rscrypto::Aegis256Key::from_bytes(black_box(KEY_32)))
        .encrypt_in_place(black_box(&nonce_rs), black_box(AAD), buf)
        .expect("valid AEAD benchmark operation must succeed")
    });
    construct_and_encrypt_row(&mut g, "aegis-crate", &data, &expected, |buf| {
      aegis::aegis256::Aegis256::<16>::new(&black_box(KEY_32), black_box(&NONCE_32))
        .encrypt_in_place(buf, black_box(AAD))
    });
  }

  g.finish();
}

// Ascon-AEAD128 one message per key: as above. Every row must produce the ascon-aead crate's ciphertext and tag.
fn ascon_aead128_construct_and_encrypt(c: &mut Criterion) {
  if !bench_config::selected("ascon-aead128/copy-and-construct-and-encrypt") {
    return;
  }
  use ascon_aead::aead::{AeadInOut as _, KeyInit as _, array::Array};

  let nonce_rs = rscrypto::aead::Nonce128::from_bytes(NONCE_16);
  let nonce_ac = Array(NONCE_16);
  let mut g = c.benchmark_group("ascon-aead128/copy-and-construct-and-encrypt");

  for (len, data) in construct_inputs() {
    common::set_throughput(&mut g, len);
    let mut ciphertext = data.clone();
    let tag = ascon_aead::AsconAead128::new(&Array(KEY_16))
      .encrypt_inout_detached(&nonce_ac, AAD, ciphertext.as_mut_slice().into())
      .expect("valid AEAD benchmark operation must succeed");
    let expected = sealed(&ciphertext, tag);

    construct_and_encrypt_row(&mut g, "rscrypto", &data, &expected, |buf| {
      rscrypto::AsconAead128::new(&rscrypto::AsconAead128Key::from_bytes(black_box(KEY_16)))
        .encrypt_in_place(black_box(&nonce_rs), black_box(AAD), buf)
        .expect("valid AEAD benchmark operation must succeed")
    });
    construct_and_encrypt_row(&mut g, "ascon-aead", &data, &expected, |buf| {
      ascon_aead::AsconAead128::new(&Array(black_box(KEY_16)))
        .encrypt_inout_detached(black_box(&nonce_ac), black_box(AAD), buf.into())
        .expect("valid AEAD benchmark operation must succeed")
    });
  }

  g.finish();
}

// Criterion harness

fn main() {
  bench_config::run(&[
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
    aes128_gcm_construct_and_encrypt,
    aes256_gcm_construct_and_encrypt,
    aes128_gcm_siv_construct_and_encrypt,
    aes256_gcm_siv_construct_and_encrypt,
    chacha20_poly1305_construct_and_encrypt,
    xchacha20_poly1305_construct_and_encrypt,
    aegis256_construct_and_encrypt,
    ascon_aead128_construct_and_encrypt,
    header_protection,
    aes_siv_cmac256,
  ]);
}
