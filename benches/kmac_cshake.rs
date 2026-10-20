//! SP 800-185 and HMAC-SHA3 comparison benchmarks.
//!
//! Every group runs the shared comparison size matrix and times one complete
//! output per iteration:
//!
//! - `kmac128`, `kmac256`: one-shot KMAC with a fixed key and customization
//!   string and a 32-byte output. Key absorption is timed. rscrypto vs
//!   tiny-keccak.
//! - `cshake128`, `cshake256`: one-shot cSHAKE with an empty function name, a
//!   fixed customization string, and a 32-byte output. rscrypto vs
//!   tiny-keccak.
//! - `hmac-sha3-224` through `hmac-sha3-512`: one HMAC-SHA3 tag from a key
//!   set up once outside timing. rscrypto vs AWS-LC's C `HMAC_CTX` through
//!   aws-lc-sys vs RustCrypto `hmac::SimpleHmac` over `sha3`. The boundary
//!   comment on `hmac_sha3_group` states what each library caches. The
//!   aws-lc-rs and ring APIs do not offer HMAC-SHA3.
//!
//! Before timing, every row must match another implementation's output for its
//! input.

#[path = "common/criterion.rs"]
mod bench_config;

#[path = "common/aws_lc.rs"]
mod aws_lc_support;
use aws_lc_support::aws_lc_bench;

mod common;

use core::hint::black_box;

use bench_config::{checked, checked_with};
use criterion::{BenchmarkId, Criterion};
use hmac::{KeyInit, SimpleHmac};
use rscrypto::{HmacSha3_224, HmacSha3_256, HmacSha3_384, HmacSha3_512, Mac};

aws_lc_bench! {
  use core::{
    ffi::c_uint,
    ptr::{self, NonNull},
  };

  use aws_lc_sys as aws_lc;

  /// An AWS-LC `HMAC_CTX` keyed once for the SHA3 digest with a `TAG`-byte
  /// output. Each [`tag`](Self::tag) call reuses the key.
  ///
  /// Invariant: `ctx` came from `HMAC_CTX_new`, is owned only by this value,
  /// and was keyed by a successful `HMAC_Init_ex` with `HMAC_size(ctx) == TAG`.
  /// Citations name files under aws-lc-sys 0.46.0's `aws-lc/` tree:
  /// `include/openssl/hmac.h`, `crypto/fipsmodule/hmac/hmac.c`, and
  /// `crypto/fipsmodule/sha/sha3.c`.
  struct AwsLcHmacSha3<const TAG: usize> {
    ctx: NonNull<aws_lc::HMAC_CTX>,
  }

  impl<const TAG: usize> AwsLcHmacSha3<TAG> {
    fn new(key: &[u8]) -> Self {
      // SAFETY: Allocates a zeroed AWS-LC HMAC_CTX because:
      // 1. The function takes no arguments.
      // 2. The result is checked for null before ownership is wrapped.
      // 3. `Drop` frees the context exactly once; the type is neither `Clone` nor `Copy`.
      let ctx = NonNull::new(unsafe { aws_lc::HMAC_CTX_new() }).expect("AWS-LC must allocate an HMAC_CTX");
      // Owning the context first frees it if a later check panics.
      let mac = Self { ctx };

      let md = Self::sha3_md();
      assert!(!md.is_null(), "AWS-LC has no SHA3 digest with a {TAG}-byte output");
      // SAFETY: Keys the owned context because:
      // 1. `mac.ctx` is live, zero-initialized, and reached only through this call.
      // 2. `key.as_ptr()` is non-null and readable for `key.len()` bytes for the call. AWS-LC
      //    copies or hashes the key into its pads and keeps no pointer to it (hmac.c:377-387).
      //    A non-null key also avoids the NULL-key reuse rule, so this call always sets the key.
      // 3. `md` is a non-null, process-static AWS-LC descriptor that must not be freed.
      // 4. `impl_` must be null; AWS-LC asserts this (hmac.c:324).
      let rc = unsafe {
        aws_lc::HMAC_Init_ex(mac.ctx.as_ptr(), key.as_ptr().cast(), key.len(), md, ptr::null_mut())
      };
      assert_eq!(rc, 1, "AWS-LC HMAC_Init_ex must key the context");
      // SAFETY: Reads the tag size because `mac.ctx` is live and was just keyed by
      // `HMAC_Init_ex`, which hmac.h requires before `HMAC_size`.
      let size = unsafe { aws_lc::HMAC_size(mac.ctx.as_ptr()) };
      assert_eq!(size, TAG, "AWS-LC HMAC tag size must match the requested SHA3 digest");
      mac
    }

    /// Returns AWS-LC's SHA3 digest with a `TAG`-byte output, or null if none exists.
    fn sha3_md() -> *const aws_lc::EVP_MD {
      match TAG {
        // SAFETY: Takes no arguments and returns AWS-LC's process-static SHA3-224 descriptor.
        28 => unsafe { aws_lc::EVP_sha3_224() },
        // SAFETY: Takes no arguments and returns AWS-LC's process-static SHA3-256 descriptor.
        32 => unsafe { aws_lc::EVP_sha3_256() },
        // SAFETY: Takes no arguments and returns AWS-LC's process-static SHA3-384 descriptor.
        48 => unsafe { aws_lc::EVP_sha3_384() },
        // SAFETY: Takes no arguments and returns AWS-LC's process-static SHA3-512 descriptor.
        64 => unsafe { aws_lc::EVP_sha3_512() },
        _ => ptr::null(),
      }
    }

    /// Computes one tag of `data` with the stored key.
    fn tag(&mut self, data: &[u8]) -> [u8; TAG] {
      let ctx = self.ctx.as_ptr();
      // SAFETY: Restarts the keyed context because:
      // 1. `ctx` is live and keyed (type invariant); `&mut self` gives this call sole access.
      // 2. A null key with a null digest keeps the stored key and digest and resets the
      //    context for new data (hmac.h:63-73). After `HMAC_Final` this only clears the
      //    needs-init state, because `HMAC_Final` already restored the inner state (hmac.c:451).
      // 3. `impl_` must be null; AWS-LC asserts this (hmac.c:324).
      let rc = unsafe { aws_lc::HMAC_Init_ex(ctx, ptr::null(), 0, ptr::null(), ptr::null_mut()) };
      assert_eq!(rc, 1, "AWS-LC HMAC_Init_ex must reuse the key");
      // SAFETY: Absorbs the message because:
      // 1. `ctx` is live and was just reinitialized; `&mut self` gives this call sole access.
      // 2. `data.as_ptr()` is non-null and readable for `data.len()` bytes for the call. For an
      //    empty slice, SHA3_Update returns before reading (sha3.c:239-253).
      // 3. `data` cannot overlap the AWS-LC-owned heap context, and AWS-LC keeps no pointer to it.
      let rc = unsafe { aws_lc::HMAC_Update(ctx, data.as_ptr(), data.len()) };
      assert_eq!(rc, 1, "AWS-LC HMAC_Update must absorb the message");
      let mut tag = [0u8; TAG];
      let mut tag_len: c_uint = 0;
      // SAFETY: Writes the tag because:
      // 1. `ctx` is live and has absorbed this message; `&mut self` gives this call sole access.
      // 2. `tag` is writable for `TAG` bytes, and `HMAC_size(ctx) == TAG` (type invariant): the
      //    digest never changes after `new`, so `HMAC_Final` writes exactly `TAG` bytes.
      // 3. `tag_len` is a writable local `c_uint`.
      // 4. Both outputs are distinct locals that cannot overlap the context.
      let rc = unsafe { aws_lc::HMAC_Final(ctx, tag.as_mut_ptr(), &raw mut tag_len) };
      assert_eq!(rc, 1, "AWS-LC HMAC_Final must produce a tag");
      assert_eq!(usize::try_from(tag_len), Ok(TAG), "AWS-LC HMAC_Final must write the full tag");
      tag
    }
  }

  impl<const TAG: usize> Drop for AwsLcHmacSha3<TAG> {
    fn drop(&mut self) {
      // SAFETY: Releases the context because:
      // 1. `self.ctx` came from `HMAC_CTX_new` and this value is its only owner.
      // 2. `HMAC_CTX_free` accepts a context in any state, including after a failed call,
      //    which leaves it zeroed (hmac.c:292-299).
      // 3. Nothing uses the pointer after `drop`.
      unsafe { aws_lc::HMAC_CTX_free(self.ctx.as_ptr()) }
    }
  }
}

const KMAC_KEY: &[u8] = b"rscrypto-kmac-bench-key-32bytes";
const CUSTOMIZATION: &[u8] = b"rscrypto-bench";
const FUNCTION_NAME: &[u8] = b"";
const OUTPUT_LEN: usize = 32;

/// Registers the rscrypto and tiny-keccak rows of one SP 800-185 group.
/// `key_or_name` is the KMAC key or the cSHAKE function name.
///
/// Boundary: each timed call is one complete one-shot operation that absorbs
/// the key or function name, the customization string, and the message, then
/// squeezes 32 bytes. Neither library caches state between calls. Each row
/// must match the other library's output.
fn sp800_185_group(
  c: &mut Criterion,
  group: &str,
  key_or_name: &[u8],
  rscrypto: impl Fn(&[u8], &[u8], &[u8]) -> [u8; OUTPUT_LEN],
  tiny_keccak: impl Fn(&[u8], &[u8], &[u8]) -> [u8; OUTPUT_LEN],
) {
  let inputs = common::comp_sizes(&[]);
  let mut g = c.benchmark_group(group);

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let tiny_keccak_output = tiny_keccak(key_or_name, CUSTOMIZATION, data);
    let rscrypto_output = rscrypto(key_or_name, CUSTOMIZATION, data);

    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked(
        || rscrypto(black_box(key_or_name), black_box(CUSTOMIZATION), black_box(data)),
        |out| assert_eq!(out, tiny_keccak_output),
      ),
    );

    g.bench_function(
      BenchmarkId::new("tiny-keccak", len),
      checked(
        || tiny_keccak(black_box(key_or_name), black_box(CUSTOMIZATION), black_box(data)),
        |out| assert_eq!(out, rscrypto_output),
      ),
    );
  }

  g.finish();
}

fn kmac128(c: &mut Criterion) {
  if !bench_config::selected("kmac128") {
    return;
  }
  sp800_185_group(
    c,
    "kmac128",
    KMAC_KEY,
    rscrypto::Kmac128::mac_array,
    |key, customization, data| {
      use tiny_keccak::Hasher as _;

      let mut out = [0u8; OUTPUT_LEN];
      let mut kmac = tiny_keccak::Kmac::v128(key, customization);
      kmac.update(data);
      kmac.finalize(&mut out);
      out
    },
  );
}

fn kmac256(c: &mut Criterion) {
  if !bench_config::selected("kmac256") {
    return;
  }
  sp800_185_group(
    c,
    "kmac256",
    KMAC_KEY,
    rscrypto::Kmac256::mac_array,
    |key, customization, data| {
      use tiny_keccak::Hasher as _;

      let mut out = [0u8; OUTPUT_LEN];
      let mut kmac = tiny_keccak::Kmac::v256(key, customization);
      kmac.update(data);
      kmac.finalize(&mut out);
      out
    },
  );
}

fn cshake128(c: &mut Criterion) {
  if !bench_config::selected("cshake128") {
    return;
  }
  sp800_185_group(
    c,
    "cshake128",
    FUNCTION_NAME,
    |function_name, customization, data| {
      let mut out = [0u8; OUTPUT_LEN];
      rscrypto::Cshake128::hash_into(function_name, customization, data, &mut out);
      out
    },
    |function_name, customization, data| {
      use tiny_keccak::Hasher as _;

      let mut out = [0u8; OUTPUT_LEN];
      let mut cshake = tiny_keccak::CShake::v128(function_name, customization);
      cshake.update(data);
      cshake.finalize(&mut out);
      out
    },
  );
}

fn cshake256(c: &mut Criterion) {
  if !bench_config::selected("cshake256") {
    return;
  }
  sp800_185_group(
    c,
    "cshake256",
    FUNCTION_NAME,
    |function_name, customization, data| {
      let mut out = [0u8; OUTPUT_LEN];
      rscrypto::Cshake256::hash_into(function_name, customization, data, &mut out);
      out
    },
    |function_name, customization, data| {
      use tiny_keccak::Hasher as _;

      let mut out = [0u8; OUTPUT_LEN];
      let mut cshake = tiny_keccak::CShake::v256(function_name, customization);
      cshake.update(data);
      cshake.finalize(&mut out);
      out
    },
  );
}

/// Registers one HMAC-SHA3 group. `TAG` is the tag size in bytes.
///
/// Boundary: the task is one tag from a key that is already set up. Each
/// library keys once with a `TAG`-byte key, outside timing. Each timed call
/// absorbs the message, finalizes one tag, and leaves the keyed state ready
/// for the next call.
///
/// - `rscrypto` caches the Keccak states after the ipad and opad blocks. It
///   updates, finalizes from a copy of the opad state, and resets by copying
///   the ipad state.
/// - `aws-lc` is AWS-LC's C `HMAC_CTX` through aws-lc-sys, which caches the
///   same two states. It calls `HMAC_Init_ex` with a null key and digest to
///   reuse them, then `HMAC_Update` and `HMAC_Final`. The row exists only where
///   aws-lc-sys is a dev-dependency.
/// - `rustcrypto-simplehmac` is `hmac::SimpleHmac` over `sha3`. It caches the
///   ipad state and the raw opad block. Each call clones both, updates, and
///   finalizes. Finalization starts a new outer hash and absorbs the opad block
///   again: one more Keccak permutation per tag than the other rows. For an
///   n-byte message at rate r, it runs floor(n/r) + 3 permutations instead of
///   floor(n/r) + 2. That is 1.5x the work below r bytes and under 1.01x from
///   16 KiB. `hmac::Hmac` caches the opad state, but it needs a block-level
///   core (`EagerHash`) that `sha3` 0.12 does not expose.
///
/// Checks: `rscrypto` and `aws-lc` must match RustCrypto's tag. RustCrypto must
/// match rscrypto's one-shot `Mac::mac` tag and, where it links, AWS-LC's tag.
fn hmac_sha3_group<Ours: Mac, Theirs: hmac::Mac + KeyInit + Clone, const TAG: usize>(c: &mut Criterion, group: &str) {
  if !bench_config::selected(group) {
    return;
  }
  let inputs = common::comp_sizes(&[]);
  let key = vec![0x42u8; Ours::TAG_SIZE];
  let base_mac = Theirs::new_from_slice(&key).expect("valid authentication benchmark operation must succeed");
  let rscrypto_tag = |mac: &mut Ours, data: &[u8]| {
    mac.update(data);
    let tag = mac.finalize();
    mac.reset();
    tag
  };
  let rustcrypto_tag = |data: &[u8]| {
    let mut mac = base_mac.clone();
    mac.update(data);
    mac.finalize().into_bytes()
  };

  let mut g = c.benchmark_group(group);

  for (len, data) in &inputs {
    common::set_throughput(&mut g, *len);
    let rustcrypto_output = rustcrypto_tag(data);
    let rscrypto_one_shot = Ours::mac(&key, data);
    aws_lc_bench! {
      let aws_lc_output = AwsLcHmacSha3::<TAG>::new(&key).tag(data);
    }

    // The check sees the second tag, after one reset of the reused state.
    g.bench_function(
      BenchmarkId::new("rscrypto", len),
      checked_with(
        Ours::new(&key),
        |mac| rscrypto_tag(mac, black_box(data)),
        |tag, _| assert_eq!(tag.as_ref(), rustcrypto_output.as_slice()),
      ),
    );

    // The check sees the second tag, after one key-reusing `HMAC_Init_ex`.
    aws_lc_bench! {
      g.bench_function(
        BenchmarkId::new("aws-lc", len),
        checked_with(
          AwsLcHmacSha3::<TAG>::new(&key),
          |mac| mac.tag(black_box(data)),
          |tag, _| assert_eq!(tag.as_slice(), rustcrypto_output.as_slice()),
        ),
      );
    }

    g.bench_function(
      BenchmarkId::new("rustcrypto-simplehmac", len),
      checked(
        || rustcrypto_tag(black_box(data)),
        |tag| {
          assert_eq!(tag.as_slice(), rscrypto_one_shot.as_ref());
          aws_lc_bench! {
            assert_eq!(tag.as_slice(), aws_lc_output.as_slice());
          }
        },
      ),
    );
  }

  g.finish();
}

fn hmac_sha3_224(c: &mut Criterion) {
  hmac_sha3_group::<HmacSha3_224, SimpleHmac<sha3::Sha3_224>, { HmacSha3_224::TAG_SIZE }>(c, "hmac-sha3-224");
}

fn hmac_sha3_256(c: &mut Criterion) {
  hmac_sha3_group::<HmacSha3_256, SimpleHmac<sha3::Sha3_256>, { HmacSha3_256::TAG_SIZE }>(c, "hmac-sha3-256");
}

fn hmac_sha3_384(c: &mut Criterion) {
  hmac_sha3_group::<HmacSha3_384, SimpleHmac<sha3::Sha3_384>, { HmacSha3_384::TAG_SIZE }>(c, "hmac-sha3-384");
}

fn hmac_sha3_512(c: &mut Criterion) {
  hmac_sha3_group::<HmacSha3_512, SimpleHmac<sha3::Sha3_512>, { HmacSha3_512::TAG_SIZE }>(c, "hmac-sha3-512");
}

fn main() {
  bench_config::run(&[
    kmac128,
    kmac256,
    cshake128,
    cshake256,
    hmac_sha3_224,
    hmac_sha3_256,
    hmac_sha3_384,
    hmac_sha3_512,
  ]);
}
