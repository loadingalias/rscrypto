//! FIPS 205 Section 11 hash-function instantiations.
//!
//! A suite binds PK.seed for one key operation. The SHA-2 suites compress the
//! PK.seed block once (Section 11.2: PK.seed padded to the hash block) and
//! finish each call from that state. Every call that may hold a secret input
//! clears its block and state copies before returning.

use super::address::Address;
use crate::auth::hmac::{HmacSha256, HmacSha512};
use crate::backend::mgf1::mgf1;
use crate::hashes::crypto::{
  Sha256, Sha512,
  keccak::{KeccakCore, PlatformPermuter},
  sha256, sha512,
};
use crate::traits::{Digest, Mac, ct};

/// The functions of one hash family, for `N`-byte values.
pub(crate) trait Suite<const N: usize>: Sized {
  /// Bind PK.seed.
  fn new(pk_seed: &[u8; N]) -> Self;

  /// `PRF(PK.seed, SK.seed, ADRS)` into `out`.
  fn prf(&self, address: &Address, sk_seed: &[u8; N], out: &mut [u8; N]);

  /// `F(PK.seed, ADRS, value)`, replacing `value`.
  fn f(&self, address: &Address, value: &mut [u8; N]);

  /// `H(PK.seed, ADRS, left || right)` into `out`.
  fn h(&self, address: &Address, left: &[u8; N], right: &[u8; N], out: &mut [u8; N]);

  /// `T_l(PK.seed, ADRS, values)` into `out`.
  fn t(&self, address: &Address, values: &[[u8; N]], out: &mut [u8; N]);

  /// `H_msg(R, PK.seed, PK.root, M)` into `out`, where `M` is the
  /// concatenation of `message` and `out` holds m bytes.
  fn h_msg(r: &[u8; N], pk_seed: &[u8; N], pk_root: &[u8; N], message: &[&[u8]], out: &mut [u8]);

  /// `PRF_msg(SK.prf, opt_rand, M)` into `out`.
  fn prf_msg(sk_prf: &[u8; N], opt_rand: &[u8; N], message: &[&[u8]], out: &mut [u8; N]);
}

/// SHA-256 resumed after the 64-byte PK.seed block.
#[derive(Clone, Copy)]
struct SeededSha256 {
  compress: sha256::kernels::CompressBlocksFn,
  state: [u32; 8],
}

impl SeededSha256 {
  fn new(pk_seed: &[u8]) -> Self {
    let compress = sha256::dispatch::compress_dispatch();
    let mut block = [0; 64];
    block[..pk_seed.len()].copy_from_slice(pk_seed);
    let mut state = sha256::H0;
    compress(&mut state, &block);
    Self { compress, state }
  }

  /// `Trunc_n(SHA-256(PK.seed || toByte(0, 64 - n) || parts))` into `out`.
  fn finish(&self, parts: &[&[u8]], out: &mut [u8]) {
    let mut state = self.state;
    let mut block = [0u8; 64];
    let mut filled = 0usize;
    let mut total = 64u64;
    for &part in parts {
      total = total.strict_add(part.len() as u64);
      let mut rest = part;
      while !rest.is_empty() {
        let take = core::cmp::min(64usize.strict_sub(filled), rest.len());
        let (head, tail) = rest.split_at(take);
        block[filled..filled.strict_add(take)].copy_from_slice(head);
        filled = filled.strict_add(take);
        rest = tail;
        if filled == 64 {
          (self.compress)(&mut state, &block);
          filled = 0;
        }
      }
    }
    block[filled] = 0x80;
    filled = filled.strict_add(1);
    if filled > 56 {
      block[filled..].fill(0);
      (self.compress)(&mut state, &block);
      filled = 0;
    }
    block[filled..56].fill(0);
    block[56..].copy_from_slice(&total.strict_mul(8).to_be_bytes());
    (self.compress)(&mut state, &block);
    for (chunk, word) in out.chunks_mut(4).zip(state) {
      chunk.copy_from_slice(&word.to_be_bytes()[..chunk.len()]);
    }
    ct::zeroize_words_no_fence(&mut state);
    ct::zeroize(&mut block);
  }
}

/// SHA-512 resumed after the 128-byte PK.seed block.
#[derive(Clone, Copy)]
struct SeededSha512 {
  compress: sha512::kernels::CompressBlocksFn,
  state: [u64; 8],
}

impl SeededSha512 {
  fn new(pk_seed: &[u8]) -> Self {
    let compress = sha512::dispatch::compress_dispatch();
    let mut block = [0; 128];
    block[..pk_seed.len()].copy_from_slice(pk_seed);
    let mut state = sha512::H0;
    compress(&mut state, &block);
    Self { compress, state }
  }

  /// `Trunc_n(SHA-512(PK.seed || toByte(0, 128 - n) || parts))` into `out`.
  fn finish(&self, parts: &[&[u8]], out: &mut [u8]) {
    let mut state = self.state;
    let mut block = [0u8; 128];
    let mut filled = 0usize;
    let mut total = 128u128;
    for &part in parts {
      total = total.strict_add(part.len() as u128);
      let mut rest = part;
      while !rest.is_empty() {
        let take = core::cmp::min(128usize.strict_sub(filled), rest.len());
        let (head, tail) = rest.split_at(take);
        block[filled..filled.strict_add(take)].copy_from_slice(head);
        filled = filled.strict_add(take);
        rest = tail;
        if filled == 128 {
          (self.compress)(&mut state, &block);
          filled = 0;
        }
      }
    }
    block[filled] = 0x80;
    filled = filled.strict_add(1);
    if filled > 112 {
      block[filled..].fill(0);
      (self.compress)(&mut state, &block);
      filled = 0;
    }
    block[filled..112].fill(0);
    block[112..].copy_from_slice(&total.strict_mul(8).to_be_bytes());
    (self.compress)(&mut state, &block);
    for (chunk, word) in out.chunks_mut(8).zip(state) {
      chunk.copy_from_slice(&word.to_be_bytes()[..chunk.len()]);
    }
    ct::zeroize_words_no_fence(&mut state);
    ct::zeroize(&mut block);
  }
}

/// MGF1 seed `R || PK.seed || digest`, at most 2 * 32 + 64 bytes.
fn mgf1_seed<const N: usize>(r: &[u8; N], pk_seed: &[u8; N], digest: &[u8], buffer: &mut [u8; 128]) -> usize {
  let len = N.strict_mul(2).strict_add(digest.len());
  buffer[..N].copy_from_slice(r);
  buffer[N..N.strict_mul(2)].copy_from_slice(pk_seed);
  buffer[N.strict_mul(2)..len].copy_from_slice(digest);
  len
}

/// Section 11.2.1: SHA-256 throughout, for n = 16.
pub(crate) struct Sha2Category1<const N: usize> {
  sha256: SeededSha256,
}

impl<const N: usize> Suite<N> for Sha2Category1<N> {
  fn new(pk_seed: &[u8; N]) -> Self {
    Self {
      sha256: SeededSha256::new(pk_seed),
    }
  }

  fn prf(&self, address: &Address, sk_seed: &[u8; N], out: &mut [u8; N]) {
    self.sha256.finish(&[&address.compressed(), sk_seed], out);
  }

  fn f(&self, address: &Address, value: &mut [u8; N]) {
    let mut input = *value;
    self.sha256.finish(&[&address.compressed(), &input], value);
    ct::zeroize(&mut input);
  }

  fn h(&self, address: &Address, left: &[u8; N], right: &[u8; N], out: &mut [u8; N]) {
    self.sha256.finish(&[&address.compressed(), left, right], out);
  }

  fn t(&self, address: &Address, values: &[[u8; N]], out: &mut [u8; N]) {
    self.sha256.finish(&[&address.compressed(), values.as_flattened()], out);
  }

  fn h_msg(r: &[u8; N], pk_seed: &[u8; N], pk_root: &[u8; N], message: &[&[u8]], out: &mut [u8]) {
    let mut inner = Sha256::new();
    inner.update(r);
    inner.update(pk_seed);
    inner.update(pk_root);
    for part in message {
      inner.update(part);
    }
    let mut digest = inner.finalize();
    let mut seed = [0; 128];
    let len = mgf1_seed(r, pk_seed, &digest, &mut seed);
    mgf1::<Sha256>(&seed[..len], out);
    ct::zeroize(&mut digest);
    ct::zeroize(&mut seed);
  }

  fn prf_msg(sk_prf: &[u8; N], opt_rand: &[u8; N], message: &[&[u8]], out: &mut [u8; N]) {
    let mut mac = HmacSha256::new(sk_prf);
    mac.update(opt_rand);
    for part in message {
      mac.update(part);
    }
    out.copy_from_slice(&mac.finalize().as_bytes()[..N]);
  }
}

/// Section 11.2.2: SHA-256 for PRF and F, SHA-512 for the rest, for n = 24
/// and n = 32.
pub(crate) struct Sha2Category3And5<const N: usize> {
  sha256: SeededSha256,
  sha512: SeededSha512,
}

impl<const N: usize> Suite<N> for Sha2Category3And5<N> {
  fn new(pk_seed: &[u8; N]) -> Self {
    Self {
      sha256: SeededSha256::new(pk_seed),
      sha512: SeededSha512::new(pk_seed),
    }
  }

  fn prf(&self, address: &Address, sk_seed: &[u8; N], out: &mut [u8; N]) {
    self.sha256.finish(&[&address.compressed(), sk_seed], out);
  }

  fn f(&self, address: &Address, value: &mut [u8; N]) {
    let mut input = *value;
    self.sha256.finish(&[&address.compressed(), &input], value);
    ct::zeroize(&mut input);
  }

  fn h(&self, address: &Address, left: &[u8; N], right: &[u8; N], out: &mut [u8; N]) {
    self.sha512.finish(&[&address.compressed(), left, right], out);
  }

  fn t(&self, address: &Address, values: &[[u8; N]], out: &mut [u8; N]) {
    self.sha512.finish(&[&address.compressed(), values.as_flattened()], out);
  }

  fn h_msg(r: &[u8; N], pk_seed: &[u8; N], pk_root: &[u8; N], message: &[&[u8]], out: &mut [u8]) {
    let mut inner = Sha512::new();
    inner.update(r);
    inner.update(pk_seed);
    inner.update(pk_root);
    for part in message {
      inner.update(part);
    }
    let mut digest = inner.finalize();
    let mut seed = [0; 128];
    let len = mgf1_seed(r, pk_seed, &digest, &mut seed);
    mgf1::<Sha512>(&seed[..len], out);
    ct::zeroize(&mut digest);
    ct::zeroize(&mut seed);
  }

  fn prf_msg(sk_prf: &[u8; N], opt_rand: &[u8; N], message: &[&[u8]], out: &mut [u8; N]) {
    let mut mac = HmacSha512::new(sk_prf);
    mac.update(opt_rand);
    for part in message {
      mac.update(part);
    }
    out.copy_from_slice(&mac.finalize().as_bytes()[..N]);
  }
}

/// SHAKE256 rate in bytes.
const SHAKE256_RATE: usize = 136;
/// SHAKE domain-separation and first padding bits.
const SHAKE_SUFFIX: u8 = 0x1f;

/// Section 11.1: SHAKE256 throughout.
pub(crate) struct Shake<const N: usize> {
  pk_seed: [u8; N],
  permuter: PlatformPermuter,
}

impl<const N: usize> Shake<N> {
  fn sponge(permuter: PlatformPermuter) -> KeccakCore<SHAKE256_RATE> {
    KeccakCore::with_permuter(permuter)
  }

  /// `SHAKE256(PK.seed || ADRS || parts, 8n)`; the sponge clears its state.
  fn tweak(&self, address: &Address, parts: &[&[u8]], out: &mut [u8; N]) {
    let mut sponge = Self::sponge(self.permuter);
    sponge.update(&self.pk_seed);
    sponge.update(&address.full());
    for part in parts {
      sponge.update(part);
    }
    sponge.finalize_into_fixed(SHAKE_SUFFIX, out);
  }
}

impl<const N: usize> Suite<N> for Shake<N> {
  fn new(pk_seed: &[u8; N]) -> Self {
    Self {
      pk_seed: *pk_seed,
      permuter: PlatformPermuter::default(),
    }
  }

  fn prf(&self, address: &Address, sk_seed: &[u8; N], out: &mut [u8; N]) {
    self.tweak(address, &[sk_seed], out);
  }

  fn f(&self, address: &Address, value: &mut [u8; N]) {
    let mut input = *value;
    self.tweak(address, &[&input], value);
    ct::zeroize(&mut input);
  }

  fn h(&self, address: &Address, left: &[u8; N], right: &[u8; N], out: &mut [u8; N]) {
    self.tweak(address, &[left, right], out);
  }

  fn t(&self, address: &Address, values: &[[u8; N]], out: &mut [u8; N]) {
    self.tweak(address, &[values.as_flattened()], out);
  }

  fn h_msg(r: &[u8; N], pk_seed: &[u8; N], pk_root: &[u8; N], message: &[&[u8]], out: &mut [u8]) {
    let mut sponge = Self::sponge(PlatformPermuter::default());
    sponge.update(r);
    sponge.update(pk_seed);
    sponge.update(pk_root);
    for part in message {
      sponge.update(part);
    }
    // m is below the rate, so the output is a prefix of one squeezed block.
    let mut block = [0; SHAKE256_RATE];
    sponge.finalize_into_fixed(SHAKE_SUFFIX, &mut block);
    out.copy_from_slice(&block[..out.len()]);
    ct::zeroize(&mut block);
  }

  fn prf_msg(sk_prf: &[u8; N], opt_rand: &[u8; N], message: &[&[u8]], out: &mut [u8; N]) {
    let mut sponge = Self::sponge(PlatformPermuter::default());
    sponge.update(sk_prf);
    sponge.update(opt_rand);
    for part in message {
      sponge.update(part);
    }
    sponge.finalize_into_fixed(SHAKE_SUFFIX, out);
  }
}
