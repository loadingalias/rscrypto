//! AEAD comparison benchmarks: rscrypto vs RustCrypto ecosystem.
//!
//! Measures `encrypt_in_place` (detached tag) for all shipped AEAD primitives
//! across the standard size matrix. Decrypt benchmarks included for the
//! primary primitives to catch asymmetry.

mod common;

use core::hint::black_box;

// Both aes-gcm and aes-gcm-siv re-export the same `aead` crate traits.
use aes_gcm_siv::aead::{AeadInPlace as _, KeyInit as _};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};


// ---------------------------------------------------------------------------
// Key / nonce material (deterministic, not secret for benchmarking)
// ---------------------------------------------------------------------------

const KEY_32: [u8; 32] = [0x42u8; 32];
const NONCE_12: [u8; 12] = [0x07u8; 12];
const NONCE_24: [u8; 24] = [0x07u8; 24];
const AAD: &[u8] = b"rscrypto-bench";

// ---------------------------------------------------------------------------
// XChaCha20-Poly1305
// ---------------------------------------------------------------------------

fn xchacha20_poly1305_encrypt(c: &mut Criterion) {
    let inputs = common::comp_sizes();
    let nonce_rs = rscrypto::aead::Nonce192::from_bytes(NONCE_24);
    let cipher_rs = rscrypto::XChaCha20Poly1305::new(&rscrypto::XChaCha20Poly1305Key::from_bytes(KEY_32));
    let cipher_rc = chacha20poly1305::XChaCha20Poly1305::new(&KEY_32.into());
    let nonce_rc = chacha20poly1305::XNonce::from_slice(&NONCE_24);
    let mut g = c.benchmark_group("xchacha20-poly1305/encrypt");

    for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);
        let mut buf = data.clone();

        g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
            b.iter(|| {
                buf.copy_from_slice(d);
                black_box(cipher_rs.encrypt_in_place(
                    black_box(&nonce_rs),
                    black_box(AAD),
                    black_box(&mut buf),
                ))
            })
        });

        g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
            b.iter(|| {
                buf.copy_from_slice(d);
                black_box(
                    cipher_rc
                        .encrypt_in_place_detached(black_box(nonce_rc), black_box(AAD), black_box(&mut buf))
                        .unwrap(),
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
    let nonce_rc = chacha20poly1305::XNonce::from_slice(&NONCE_24);
    let mut g = c.benchmark_group("xchacha20-poly1305/decrypt");

    for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);

        // Pre-encrypt with rscrypto to get valid ciphertext + tag.
        let mut ciphertext = data.clone();
        let tag_rs = cipher_rs.encrypt_in_place(&nonce_rs, AAD, &mut ciphertext);

        // Pre-encrypt with RustCrypto to get its tag format.
        let mut ct_rc = data.clone();
        let tag_rc = cipher_rc
            .encrypt_in_place_detached(nonce_rc, AAD, &mut ct_rc)
            .unwrap();

        let mut buf = ciphertext.clone();

        g.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext, |b, ct| {
            b.iter(|| {
                buf.copy_from_slice(ct);
                cipher_rs
                    .decrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut buf), black_box(&tag_rs))
                    .unwrap();
                black_box(&buf);
            })
        });

        let mut buf_rc = ct_rc.clone();

        g.bench_with_input(BenchmarkId::new("rustcrypto", len), &ct_rc, |b, ct| {
            b.iter(|| {
                buf_rc.copy_from_slice(ct);
                cipher_rc
                    .decrypt_in_place_detached(black_box(nonce_rc), black_box(AAD), black_box(&mut buf_rc), black_box(&tag_rc))
                    .unwrap();
                black_box(&buf_rc);
            })
        });
    }

    g.finish();
}

// ---------------------------------------------------------------------------
// ChaCha20-Poly1305
// ---------------------------------------------------------------------------

fn chacha20_poly1305_encrypt(c: &mut Criterion) {
    let inputs = common::comp_sizes();
    let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
    let cipher_rs = rscrypto::ChaCha20Poly1305::new(&rscrypto::ChaCha20Poly1305Key::from_bytes(KEY_32));
    let cipher_rc = chacha20poly1305::ChaCha20Poly1305::new(&KEY_32.into());
    let nonce_rc = chacha20poly1305::Nonce::from_slice(&NONCE_12);
    let mut g = c.benchmark_group("chacha20-poly1305/encrypt");

    for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);
        let mut buf = data.clone();

        g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
            b.iter(|| {
                buf.copy_from_slice(d);
                black_box(cipher_rs.encrypt_in_place(
                    black_box(&nonce_rs),
                    black_box(AAD),
                    black_box(&mut buf),
                ))
            })
        });

        g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
            b.iter(|| {
                buf.copy_from_slice(d);
                black_box(
                    cipher_rc
                        .encrypt_in_place_detached(black_box(nonce_rc), black_box(AAD), black_box(&mut buf))
                        .unwrap(),
                )
            })
        });
    }

    g.finish();
}

fn chacha20_poly1305_decrypt(c: &mut Criterion) {
    let inputs = common::comp_sizes();
    let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
    let cipher_rs = rscrypto::ChaCha20Poly1305::new(&rscrypto::ChaCha20Poly1305Key::from_bytes(KEY_32));
    let cipher_rc = chacha20poly1305::ChaCha20Poly1305::new(&KEY_32.into());
    let nonce_rc = chacha20poly1305::Nonce::from_slice(&NONCE_12);
    let mut g = c.benchmark_group("chacha20-poly1305/decrypt");

    for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);

        let mut ciphertext = data.clone();
        let tag_rs = cipher_rs.encrypt_in_place(&nonce_rs, AAD, &mut ciphertext);

        let mut ct_rc = data.clone();
        let tag_rc = cipher_rc
            .encrypt_in_place_detached(nonce_rc, AAD, &mut ct_rc)
            .unwrap();

        let mut buf = ciphertext.clone();

        g.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext, |b, ct| {
            b.iter(|| {
                buf.copy_from_slice(ct);
                cipher_rs
                    .decrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut buf), black_box(&tag_rs))
                    .unwrap();
                black_box(&buf);
            })
        });

        let mut buf_rc = ct_rc.clone();

        g.bench_with_input(BenchmarkId::new("rustcrypto", len), &ct_rc, |b, ct| {
            b.iter(|| {
                buf_rc.copy_from_slice(ct);
                cipher_rc
                    .decrypt_in_place_detached(black_box(nonce_rc), black_box(AAD), black_box(&mut buf_rc), black_box(&tag_rc))
                    .unwrap();
                black_box(&buf_rc);
            })
        });
    }

    g.finish();
}

// ---------------------------------------------------------------------------
// AES-256-GCM-SIV
// ---------------------------------------------------------------------------

fn aes256_gcm_siv_encrypt(c: &mut Criterion) {
    let inputs = common::comp_sizes();
    let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
    let cipher_rs = rscrypto::Aes256GcmSiv::new(&rscrypto::Aes256GcmSivKey::from_bytes(KEY_32));
    let cipher_rc = aes_gcm_siv::Aes256GcmSiv::new(&KEY_32.into());
    let nonce_rc = aes_gcm_siv::Nonce::from_slice(&NONCE_12);
    let mut g = c.benchmark_group("aes-256-gcm-siv/encrypt");

    for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);
        let mut buf = data.clone();

        g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
            b.iter(|| {
                buf.copy_from_slice(d);
                black_box(cipher_rs.encrypt_in_place(
                    black_box(&nonce_rs),
                    black_box(AAD),
                    black_box(&mut buf),
                ))
            })
        });

        g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
            b.iter(|| {
                buf.copy_from_slice(d);
                black_box(
                    cipher_rc
                        .encrypt_in_place_detached(black_box(nonce_rc), black_box(AAD), black_box(&mut buf))
                        .unwrap(),
                )
            })
        });
    }

    g.finish();
}

fn aes256_gcm_siv_decrypt(c: &mut Criterion) {
    let inputs = common::comp_sizes();
    let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
    let cipher_rs = rscrypto::Aes256GcmSiv::new(&rscrypto::Aes256GcmSivKey::from_bytes(KEY_32));
    let cipher_rc = aes_gcm_siv::Aes256GcmSiv::new(&KEY_32.into());
    let nonce_rc = aes_gcm_siv::Nonce::from_slice(&NONCE_12);
    let mut g = c.benchmark_group("aes-256-gcm-siv/decrypt");

    for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);

        let mut ciphertext = data.clone();
        let tag_rs = cipher_rs.encrypt_in_place(&nonce_rs, AAD, &mut ciphertext);

        let mut ct_rc = data.clone();
        let tag_rc = cipher_rc
            .encrypt_in_place_detached(nonce_rc, AAD, &mut ct_rc)
            .unwrap();

        let mut buf = ciphertext.clone();

        g.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext, |b, ct| {
            b.iter(|| {
                buf.copy_from_slice(ct);
                cipher_rs
                    .decrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut buf), black_box(&tag_rs))
                    .unwrap();
                black_box(&buf);
            })
        });

        let mut buf_rc = ct_rc.clone();

        g.bench_with_input(BenchmarkId::new("rustcrypto", len), &ct_rc, |b, ct| {
            b.iter(|| {
                buf_rc.copy_from_slice(ct);
                cipher_rc
                    .decrypt_in_place_detached(black_box(nonce_rc), black_box(AAD), black_box(&mut buf_rc), black_box(&tag_rc))
                    .unwrap();
                black_box(&buf_rc);
            })
        });
    }

    g.finish();
}

// ---------------------------------------------------------------------------
// AES-256-GCM
// ---------------------------------------------------------------------------

fn aes256_gcm_encrypt(c: &mut Criterion) {
    let inputs = common::comp_sizes();
    let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
    let cipher_rs = rscrypto::Aes256Gcm::new(&rscrypto::Aes256GcmKey::from_bytes(KEY_32));
    let cipher_rc = aes_gcm::Aes256Gcm::new(&KEY_32.into());
    let nonce_rc = aes_gcm::Nonce::from_slice(&NONCE_12);
    let mut g = c.benchmark_group("aes-256-gcm/encrypt");

    for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);
        let mut buf = data.clone();

        g.bench_with_input(BenchmarkId::new("rscrypto", len), data, |b, d| {
            b.iter(|| {
                buf.copy_from_slice(d);
                black_box(cipher_rs.encrypt_in_place(
                    black_box(&nonce_rs),
                    black_box(AAD),
                    black_box(&mut buf),
                ))
            })
        });

        g.bench_with_input(BenchmarkId::new("rustcrypto", len), data, |b, d| {
            b.iter(|| {
                buf.copy_from_slice(d);
                black_box(
                    cipher_rc
                        .encrypt_in_place_detached(black_box(nonce_rc), black_box(AAD), black_box(&mut buf))
                        .unwrap(),
                )
            })
        });
    }

    g.finish();
}

fn aes256_gcm_decrypt(c: &mut Criterion) {
    let inputs = common::comp_sizes();
    let nonce_rs = rscrypto::aead::Nonce96::from_bytes(NONCE_12);
    let cipher_rs = rscrypto::Aes256Gcm::new(&rscrypto::Aes256GcmKey::from_bytes(KEY_32));
    let cipher_rc = aes_gcm::Aes256Gcm::new(&KEY_32.into());
    let nonce_rc = aes_gcm::Nonce::from_slice(&NONCE_12);
    let mut g = c.benchmark_group("aes-256-gcm/decrypt");

    for (len, data) in &inputs {
        common::set_throughput(&mut g, *len);

        let mut ciphertext = data.clone();
        let tag_rs = cipher_rs.encrypt_in_place(&nonce_rs, AAD, &mut ciphertext);

        let mut ct_rc = data.clone();
        let tag_rc = cipher_rc
            .encrypt_in_place_detached(nonce_rc, AAD, &mut ct_rc)
            .unwrap();

        let mut buf = ciphertext.clone();

        g.bench_with_input(BenchmarkId::new("rscrypto", len), &ciphertext, |b, ct| {
            b.iter(|| {
                buf.copy_from_slice(ct);
                cipher_rs
                    .decrypt_in_place(black_box(&nonce_rs), black_box(AAD), black_box(&mut buf), black_box(&tag_rs))
                    .unwrap();
                black_box(&buf);
            })
        });

        let mut buf_rc = ct_rc.clone();

        g.bench_with_input(BenchmarkId::new("rustcrypto", len), &ct_rc, |b, ct| {
            b.iter(|| {
                buf_rc.copy_from_slice(ct);
                cipher_rc
                    .decrypt_in_place_detached(black_box(nonce_rc), black_box(AAD), black_box(&mut buf_rc), black_box(&tag_rc))
                    .unwrap();
                black_box(&buf_rc);
            })
        });
    }

    g.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    xchacha20_poly1305_encrypt,
    xchacha20_poly1305_decrypt,
    chacha20_poly1305_encrypt,
    chacha20_poly1305_decrypt,
    aes256_gcm_siv_encrypt,
    aes256_gcm_siv_decrypt,
    aes256_gcm_encrypt,
    aes256_gcm_decrypt,
);
criterion_main!(benches);
