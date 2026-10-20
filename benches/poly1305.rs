//! Standalone Poly1305 construction, authentication, finalization, and cleanup.
//!
//! Before timing, every row must return RustCrypto's tag, and the RustCrypto row must return dryoc's.

#[path = "common/criterion.rs"]
mod bench_config;

#[path = "common/aws_lc.rs"]
mod aws_lc_support;
use aws_lc_support::aws_lc_bench;

use core::hint::black_box;

use bench_config::checked;
use criterion::{BenchmarkId, Criterion, Throughput};
use dryoc::classic::crypto_onetimeauth::crypto_onetimeauth;
use poly1305::universal_hash::KeyInit as _;
use rscrypto::{Poly1305, Poly1305OneTimeKey};

aws_lc_bench! {
  use core::mem::MaybeUninit;

  use aws_lc_sys as aws_lc;

  /// One Poly1305 tag through AWS-LC's public `CRYPTO_poly1305_*` API, with the state on this function's stack.
  ///
  /// Citations name files under aws-lc-sys 0.46.0's `aws-lc/` tree: `include/openssl/poly1305.h` and
  /// `crypto/poly1305/`. Every backend aligns the 512-byte `poly1305_state` to 64 bytes inside the buffer
  /// (`internal.h:16-19`, `poly1305_vec.c:72-75`) and asserts that the aligned state fits
  /// (`poly1305.c:30-32`, `poly1305_vec.c:68-70`, `poly1305_arm.c:175-177`), so the buffer may have any
  /// alignment. The aligned offset depends on the buffer's address, so `state` stays in place from
  /// `init` through `finish`. AWS-LC keeps no pointer to the key or the input, and does not clear the state.
  fn aws_lc_poly1305(key: &[u8; 32], data: &[u8]) -> [u8; 16] {
    let mut state = MaybeUninit::<aws_lc::poly1305_state>::uninit();
    let mut tag = [0u8; 16];
    // SAFETY: Keys a new state because:
    // 1. `state.as_mut_ptr()` points to 512 writable bytes that only this function reaches; `init` writes every
    //    field that `update` and `finish` read (poly1305.h:16-21), so no uninitialized byte is read.
    // 2. `key.as_ptr()` is readable for the 32 bytes that `init` reads; every backend copies what it
    //    needs into the state (poly1305.c:132-176, poly1305_vec.c:81-110, poly1305_arm.c:179-200).
    unsafe { aws_lc::CRYPTO_poly1305_init(state.as_mut_ptr(), key.as_ptr()) };
    // SAFETY: Absorbs the message because:
    // 1. `state` was set up by `init` above and has not moved.
    // 2. `data.as_ptr()` is readable for `data.len()` bytes; an empty input returns before any read
    //    (poly1305.c:182-185).
    unsafe { aws_lc::CRYPTO_poly1305_update(state.as_mut_ptr(), data.as_ptr(), data.len()) };
    // SAFETY: Finishes the tag because `state` was set up by `init` and has not moved, and `tag.as_mut_ptr()` is
    // writable for the 16 bytes that `finish` writes (poly1305.h:28-31).
    unsafe { aws_lc::CRYPTO_poly1305_finish(state.as_mut_ptr(), tag.as_mut_ptr()) };
    tag
  }
}

fn authenticate(c: &mut Criterion) {
  if !bench_config::selected("poly1305/authenticate") {
    return;
  }
  let key = [0x42; 32];
  let rustcrypto = |data: &[u8]| -> [u8; 16] { poly1305::Poly1305::new(&key.into()).compute_unpadded(data).into() };
  let mut group = c.benchmark_group("poly1305/authenticate");
  // Empty-message setup; both sides of the 16-byte block boundary; short and
  // sustained block processing. Fixtures and expected tags stay outside timing.
  for len in [0, 15, 16, 17, 64, 1024, 16384] {
    let data = vec![0xff; len];
    let expected = rustcrypto(&data);
    let mut dryoc_tag = [0; 16];
    crypto_onetimeauth(&mut dryoc_tag, &data, &key);
    if len != 0 {
      group.throughput(Throughput::Bytes(len as u64));
    }
    // Timed: construct the one-time key and state, authenticate, finalize, and observe the tag. There is no
    // per-operation allocation or input copy. rscrypto also clears its key and state; cleanup policies differ by
    // library. AWS-LC is called through its public `CRYPTO_poly1305_*` API.
    group.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || Poly1305::authenticate_once(Poly1305OneTimeKey::from_bytes(*black_box(&key)), black_box(&data)),
        |tag| assert_eq!(tag.to_bytes(), expected),
      ),
    );
    group.bench_function(
      BenchmarkId::new("rustcrypto", len),
      checked(|| rustcrypto(black_box(&data)), |tag| assert_eq!(tag, dryoc_tag)),
    );
    group.bench_function(
      BenchmarkId::new("dryoc", len),
      checked(
        || {
          let mut tag = [0; 16];
          crypto_onetimeauth(&mut tag, black_box(&data), black_box(&key));
          tag
        },
        |tag| assert_eq!(tag, expected),
      ),
    );
    aws_lc_bench! {
      group.bench_function(
        BenchmarkId::new("aws-lc", len),
        checked(|| aws_lc_poly1305(black_box(&key), black_box(&data)), |tag| assert_eq!(tag, expected)),
      );
    }
  }
  group.finish();
}

fn main() {
  bench_config::run(&[authenticate]);
}
