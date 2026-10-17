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
  let inputs = common::comp_sizes();
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
  let inputs = common::comp_sizes();
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

  let inputs = common::comp_sizes();
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

  let inputs = common::comp_sizes();
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

  let inputs = common::comp_sizes();
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

  let inputs = common::comp_sizes();
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
  let inputs = common::comp_sizes();
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
  let inputs = common::comp_sizes();
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

  let inputs = common::comp_sizes();
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

  let inputs = common::comp_sizes();
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

// Fixed-size header protection

fn header_protection(c: &mut Criterion) {
  if !bench_config::selected("header-protection/") {
    return;
  }
  use aes::cipher::{Array, BlockCipherEncrypt as _, KeyInit as _};
  use chacha20::cipher::{KeyIvInit as _, StreamCipher as _, StreamCipherCore as _, StreamCipherSeek as _};
  use rscrypto::aead::expert::header_protection::{
    Aes128HeaderProtection, Aes128HeaderProtectionKey, Aes256HeaderProtection, Aes256HeaderProtectionKey,
    ChaCha20HeaderProtection, ChaCha20HeaderProtectionKey,
  };

  // The sample's nonce is `NONCE_12`, which keys the RustCrypto ChaCha20 construction row.
  let sample = [0x07; 16];
  let counter = u32::from_le_bytes(sample[..4].try_into().expect("counter prefix has fixed length"));
  let nonce: [u8; 12] = sample[4..].try_into().expect("nonce suffix has fixed length");
  let aes128_rc = aes::Aes128::new(&Array::from(KEY_16));
  let aes256_rc = aes::Aes256::new(&Array::from(KEY_32));
  // Every row's mask, or its constructed context's mask for `sample`, must
  // match RustCrypto's: AES encrypts the sample; ChaCha20 takes its counter
  // and nonce from the sample.
  let aes_mask = |block: Array<u8, aes::cipher::consts::U16>| {
    <[u8; 5]>::try_from(&block[..5]).expect("five-byte prefix has fixed length")
  };
  let expected_aes128 = {
    let mut block = Array::from(sample);
    aes128_rc.encrypt_block(&mut block);
    aes_mask(block)
  };
  let expected_aes256 = {
    let mut block = Array::from(sample);
    aes256_rc.encrypt_block(&mut block);
    aes_mask(block)
  };
  let expected_chacha = {
    let mut mask = [0u8; 5];
    let mut cipher = chacha20::ChaCha20::new(&KEY_32.into(), &nonce.into());
    cipher.seek(u64::from(counter).strict_mul(64));
    cipher.apply_keystream(&mut mask);
    mask
  };

  // Timed: construct and destroy each returned context; key fixtures are untimed.
  let mut construction = c.benchmark_group("header-protection/construct");
  construction.bench_function(
    "rscrypto/aes128",
    checked(
      || {
        let key = Aes128HeaderProtectionKey::from_bytes(black_box(KEY_16));
        Aes128HeaderProtection::new(&key)
      },
      |context| assert_eq!(context.mask(&sample), expected_aes128),
    ),
  );
  construction.bench_function(
    "rustcrypto/aes128",
    checked(
      || aes::Aes128::new(black_box(&Array::from(KEY_16))),
      |cipher| {
        let mut block = Array::from(sample);
        cipher.encrypt_block(&mut block);
        assert_eq!(aes_mask(block), expected_aes128);
      },
    ),
  );
  construction.bench_function(
    "rscrypto/aes256",
    checked(
      || {
        let key = Aes256HeaderProtectionKey::from_bytes(black_box(KEY_32));
        Aes256HeaderProtection::new(&key)
      },
      |context| assert_eq!(context.mask(&sample), expected_aes256),
    ),
  );
  construction.bench_function(
    "rustcrypto/aes256",
    checked(
      || aes::Aes256::new(black_box(&Array::from(KEY_32))),
      |cipher| {
        let mut block = Array::from(sample);
        cipher.encrypt_block(&mut block);
        assert_eq!(aes_mask(block), expected_aes256);
      },
    ),
  );
  construction.bench_function(
    "rscrypto/chacha20",
    checked(
      || {
        let key = ChaCha20HeaderProtectionKey::from_bytes(black_box(KEY_32));
        ChaCha20HeaderProtection::new(&key)
      },
      |context| assert_eq!(context.mask(&sample), expected_chacha),
    ),
  );
  construction.bench_function(
    "rustcrypto/chacha20",
    checked(
      || {
        chacha20::ChaChaCore::<chacha20::R20, chacha20::variants::Ietf>::new(
          black_box(&KEY_32).into(),
          black_box(&NONCE_12).into(),
        )
      },
      |mut core| {
        core.set_block_pos(counter);
        let mut block = chacha20::cipher::array::Array::<u8, chacha20::cipher::consts::U64>::default();
        core.write_keystream_block(&mut block);
        assert_eq!(block[..5], expected_chacha);
      },
    ),
  );
  construction.finish();

  let aes128_rs = Aes128HeaderProtection::new(&Aes128HeaderProtectionKey::from_bytes(KEY_16));
  let aes256_rs = Aes256HeaderProtection::new(&Aes256HeaderProtectionKey::from_bytes(KEY_32));
  let chacha_rs = ChaCha20HeaderProtection::new(&ChaCha20HeaderProtectionKey::from_bytes(KEY_32));

  // Timed: sample processing and mask generation, including local block/core setup.
  // Untimed: reusable key contexts and the sample fixture.
  let mut mask = c.benchmark_group("header-protection/mask");
  mask.bench_function(
    "rscrypto/aes128",
    checked(
      || aes128_rs.mask(black_box(&sample)),
      |output| assert_eq!(output, expected_aes128),
    ),
  );
  mask.bench_function(
    "rustcrypto/aes128",
    checked(
      || {
        let mut block = Array::from(*black_box(&sample));
        aes128_rc.encrypt_block(&mut block);
        <[u8; 5]>::try_from(&block[..5]).expect("five-byte prefix has fixed length")
      },
      |output| assert_eq!(output, expected_aes128),
    ),
  );
  mask.bench_function(
    "rscrypto/aes256",
    checked(
      || aes256_rs.mask(black_box(&sample)),
      |output| assert_eq!(output, expected_aes256),
    ),
  );
  mask.bench_function(
    "rustcrypto/aes256",
    checked(
      || {
        let mut block = Array::from(*black_box(&sample));
        aes256_rc.encrypt_block(&mut block);
        <[u8; 5]>::try_from(&block[..5]).expect("five-byte prefix has fixed length")
      },
      |output| assert_eq!(output, expected_aes256),
    ),
  );
  mask.bench_function(
    "rscrypto/chacha20",
    checked(
      || chacha_rs.mask(black_box(&sample)),
      |output| assert_eq!(output, expected_chacha),
    ),
  );
  mask.bench_function(
    "rustcrypto/chacha20",
    checked(
      || {
        let counter = u32::from_le_bytes(sample[..4].try_into().expect("counter prefix has fixed length"));
        let nonce: [u8; 12] = sample[4..].try_into().expect("nonce suffix has fixed length");
        let mut core = chacha20::ChaChaCore::<chacha20::R20, chacha20::variants::Ietf>::new(
          black_box(&KEY_32).into(),
          black_box(&nonce).into(),
        );
        core.set_block_pos(counter);
        let mut block = chacha20::cipher::array::Array::<u8, chacha20::cipher::consts::U64>::default();
        core.write_keystream_block(&mut block);
        <[u8; 5]>::try_from(&block[..5]).expect("five-byte prefix has fixed length")
      },
      |output| assert_eq!(output, expected_chacha),
    ),
  );
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
    header_protection,
    aes_siv_cmac256,
  ]);
}
