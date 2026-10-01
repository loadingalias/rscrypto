//! Secret SHA-3 and SHAKE hashing in a scrubbed out-of-line worker.
//!
//! ML-DSA and the ML-KEM G, J, and PRF functions hash secrets through this
//! module. Each entry point builds the permuter in its own frame, so a
//! process's first capability detection runs and returns before any secret is
//! loaded. An `#[inline(never)]` worker then absorbs and squeezes, and
//! `scrub_dead_stack`, called from the same frame, overwrites the worker's dead
//! frame. That covers compiler-created Keccak lane spills that no named state
//! owner can clear. `just stack-frames` checks on each reviewed target that
//! every worker reaches no detection and fits within the scrub; changes to
//! these functions require that review. Public SHA-3 and SHAKE hashing does
//! not use this module.

use super::{KeccakCoreImpl, PlatformPermuter};
#[cfg(feature = "ml-kem")]
use super::{KeccakXofImpl, xof_seeded_32_1_pair_secret, xof_seeded_32_1_quad_secret, xof_seeded_32_1_secret};

/// Dead stack cleared after each worker. Every worker's linked frame,
/// including its Keccak callees and any leaf red zone, must fit within it.
const STACK_SCRUB_WORDS: usize = 256;
/// The four-state worker holds four sponges plus vector spills, which reach
/// 2.0-2.7 KiB on the reviewed targets, so it gets its own larger bound.
#[cfg(feature = "ml-kem")]
const QUAD_STACK_SCRUB_WORDS: usize = 512;

const SHAKE256_RATE: usize = 136;
const SHAKE_SUFFIX: u8 = 0x1f;

/// SHAKE256 of the concatenated `parts`, squeezed into `out`.
pub(crate) fn shake256(parts: &[&[u8]], out: &mut [u8]) {
  let permuter = PlatformPermuter::default();
  absorb_and_squeeze::<SHAKE256_RATE>(permuter, SHAKE_SUFFIX, parts, out);
  scrub_dead_stack::<STACK_SCRUB_WORDS>();
}

/// SHA3-512 of `input`.
#[cfg(feature = "ml-kem")]
pub(crate) fn sha3_512(input: &[u8], out: &mut [u8; 64]) {
  let permuter = PlatformPermuter::default();
  // A SHA-3 digest no longer than the rate is the first squeeze of the sponge
  // padded with the SHA-3 suffix.
  absorb_and_squeeze::<72>(permuter, 0x06, &[input], out);
  scrub_dead_stack::<STACK_SCRUB_WORDS>();
}

/// SHAKE256(`seed` || `nonce`), the ML-KEM PRF.
#[cfg(feature = "ml-kem")]
pub(crate) fn shake256_seeded(seed: &[u8; 32], nonce: u8, out: &mut [u8]) {
  let permuter = PlatformPermuter::default();
  squeeze_seeded(permuter, seed, nonce, out);
  scrub_dead_stack::<STACK_SCRUB_WORDS>();
}

/// [`shake256_seeded`] for two nonces, permuted together where supported.
/// The lane-parallel squeeze needs equal output lengths.
#[cfg(feature = "ml-kem")]
pub(crate) fn shake256_seeded_pair<const N: usize>(seed: &[u8; 32], nonces: [u8; 2], outputs: [&mut [u8; N]; 2]) {
  let permuter = PlatformPermuter::default();
  squeeze_seeded_pair(permuter, seed, nonces, outputs.map(|out| out.as_mut_slice()));
  scrub_dead_stack::<STACK_SCRUB_WORDS>();
}

/// [`shake256_seeded`] for four nonces, permuted together where supported.
/// The lane-parallel squeeze needs equal output lengths.
#[cfg(feature = "ml-kem")]
pub(crate) fn shake256_seeded_quad<const N: usize>(seed: &[u8; 32], nonces: [u8; 4], outputs: [&mut [u8; N]; 4]) {
  let permuter = PlatformPermuter::default();
  squeeze_seeded_quad(permuter, seed, nonces, outputs.map(|out| out.as_mut_slice()));
  scrub_dead_stack::<QUAD_STACK_SCRUB_WORDS>();
}

#[inline(never)]
fn absorb_and_squeeze<const RATE: usize>(permuter: PlatformPermuter, suffix: u8, parts: &[&[u8]], out: &mut [u8]) {
  let mut state = KeccakCoreImpl::<RATE, PlatformPermuter, true>::with_permuter(permuter);
  for part in parts {
    state.update(part);
  }
  state.finalize_xof_into(suffix, out);
}

#[cfg(feature = "ml-kem")]
#[inline(never)]
fn squeeze_seeded(permuter: PlatformPermuter, seed: &[u8; 32], nonce: u8, out: &mut [u8]) {
  let mut reader = xof_seeded_32_1_secret::<SHAKE256_RATE>(permuter, SHAKE_SUFFIX, seed, nonce);
  reader.squeeze_into(out);
}

#[cfg(feature = "ml-kem")]
#[inline(never)]
fn squeeze_seeded_pair(permuter: PlatformPermuter, seed: &[u8; 32], [a, b]: [u8; 2], [out_a, out_b]: [&mut [u8]; 2]) {
  let (mut reader_a, mut reader_b) = xof_seeded_32_1_pair_secret::<SHAKE256_RATE>(permuter, SHAKE_SUFFIX, seed, a, b);
  KeccakXofImpl::squeeze_pair_into(&mut reader_a, &mut reader_b, out_a, out_b);
}

#[cfg(feature = "ml-kem")]
#[inline(never)]
fn squeeze_seeded_quad(permuter: PlatformPermuter, seed: &[u8; 32], [a, b, c, d]: [u8; 4], outputs: [&mut [u8]; 4]) {
  let (mut reader_a, mut reader_b, mut reader_c, mut reader_d) =
    xof_seeded_32_1_quad_secret::<SHAKE256_RATE>(permuter, SHAKE_SUFFIX, seed, a, b, c, d);
  KeccakXofImpl::squeeze_quad_into([&mut reader_a, &mut reader_b, &mut reader_c, &mut reader_d], outputs);
}

#[inline(never)]
fn scrub_dead_stack<const WORDS: usize>() {
  const { assert!(WORDS.is_multiple_of(4)) };
  // Leave the buffer uninitialized so only the volatile stores write it.
  // Volatile stores cannot be elided, and no later access depends on their
  // order, so no fence is needed.
  let mut scratch = core::mem::MaybeUninit::<[u64; WORDS]>::uninit();
  let words = scratch.as_mut_ptr().cast::<u64>();
  // Four stores per iteration let cores with two store ports retire the
  // scrub faster than a one-store loop.
  for index in (0..WORDS).step_by(4) {
    // SAFETY: the word count is a multiple of four, so `index + 3` is below
    // the array length and every pointer stays inside this local, properly
    // aligned `[u64; N]` allocation. Writing a `u64` needs no prior
    // initialization, and the buffer is never read.
    unsafe {
      words.add(index).write_volatile(0);
      words.add(index.strict_add(1)).write_volatile(0);
      words.add(index.strict_add(2)).write_volatile(0);
      words.add(index.strict_add(3)).write_volatile(0);
    }
  }
}

#[cfg(test)]
mod tests {
  use tiny_keccak::Hasher as _;

  const GUARD: u8 = 0xa5;

  fn message<const N: usize>() -> [u8; N] {
    core::array::from_fn(|index| u8::try_from(index.wrapping_mul(131) % 251).expect("test byte fits u8"))
  }

  fn independent_shake256(parts: &[&[u8]], out: &mut [u8]) {
    let mut oracle = tiny_keccak::Shake::v256();
    for part in parts {
      oracle.update(part);
    }
    oracle.finalize(out);
  }

  /// Runs `hash` into the middle of a guarded buffer and checks the guards.
  fn guarded<const N: usize>(len: usize, hash: impl FnOnce(&mut [u8])) -> [u8; N] {
    let mut buffer = [GUARD; N];
    hash(&mut buffer[1..=len]);
    assert_eq!(buffer[0], GUARD);
    assert!(buffer[len.strict_add(1)..].iter().all(|&byte| byte == GUARD));
    buffer
  }

  #[test]
  fn shake256_matches_independent_shake256_across_rate_boundaries() {
    let input = message::<300>();
    for split in [0, 1, 135, 136, 137, 271, 272, 300] {
      let parts = [&input[..split], &input[split..]];
      for len in [0, 1, 32, 64, 135, 136, 137, 272, 273] {
        let mut expected = [0u8; 273];
        independent_shake256(&parts, &mut expected[..len]);
        let actual = guarded::<275>(len, |out| super::shake256(&parts, out));
        assert_eq!(&actual[1..=len], &expected[..len], "split {split}, output {len}");
      }
    }
  }

  #[cfg(feature = "ml-kem")]
  #[test]
  fn sha3_512_matches_independent_sha3_512_across_rate_boundaries() {
    use sha3::Digest as _;

    let input = message::<300>();
    for len in [0, 1, 71, 72, 73, 143, 144, 145, 300] {
      let mut actual = [0u8; 64];
      super::sha3_512(&input[..len], &mut actual);
      assert_eq!(
        actual.as_slice(),
        sha3::Sha3_512::digest(&input[..len]).as_slice(),
        "input {len}"
      );
    }
  }

  #[cfg(feature = "ml-kem")]
  fn seeded_matches<const N: usize>(seed: &[u8; 32]) {
    let expected = |nonce: u8| {
      let mut out = [0u8; N];
      independent_shake256(&[seed, &[nonce]], &mut out);
      out
    };
    let single = guarded::<386>(N, |out| super::shake256_seeded(seed, 0x7f, out));
    assert_eq!(&single[1..=N], &expected(0x7f), "single, output {N}");

    let (mut a, mut b) = ([0u8; N], [0u8; N]);
    super::shake256_seeded_pair(seed, [0, 0xff], [&mut a, &mut b]);
    assert_eq!((a, b), (expected(0), expected(0xff)), "pair, output {N}");

    let mut quad = [[0u8; N]; 4];
    let [q0, q1, q2, q3] = &mut quad;
    super::shake256_seeded_quad(seed, [1, 2, 3, 4], [q0, q1, q2, q3]);
    for (nonce, out) in (1u8..).zip(quad) {
      assert_eq!(out, expected(nonce), "quad nonce {nonce}, output {N}");
    }
  }

  #[cfg(feature = "ml-kem")]
  #[test]
  fn seeded_shapes_match_independent_shake256() {
    let seed = message::<32>();
    // 128 and 192 bytes are the ML-KEM noise widths and 384 the decryption
    // mask; 136, 137, and 300 meet or cross squeeze-block boundaries.
    seeded_matches::<1>(&seed);
    seeded_matches::<128>(&seed);
    seeded_matches::<136>(&seed);
    seeded_matches::<137>(&seed);
    seeded_matches::<192>(&seed);
    seeded_matches::<300>(&seed);
    seeded_matches::<384>(&seed);
  }
}
