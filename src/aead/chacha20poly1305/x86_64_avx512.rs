//! Owned ChaCha20-Poly1305 on x86-64 with AVX-512.
//!
//! Sealing XORs the keystream from block 1 with masked loads and stores and
//! takes the Poly1305 key from block 0 in the same final pass. Short messages
//! need one row pass of at most eight blocks; longer ones add sixteen-block
//! column batches. Short opens keep the keystream in registers, check the tag
//! over the ciphertext, and only then write plaintext. Lengths are public and
//! select only masks, register groups, and the Poly1305 kernel. Workers whose
//! generated code spills keystream run out of line, and a fixed dead-stack
//! scrub follows each from the same frame; `just stack-frames` checks that the
//! linked worker fits within it.

use core::arch::x86_64::{_mm512_mask_storeu_epi8, _mm512_setzero_si512, _mm512_xor_si512};

use super::super::poly1305;
use crate::{
  aead::targets::AeadPrimitive,
  backend::{
    chacha20::{KEY_SIZE, NONCE_SIZE, X86Source, x86_byte_mask, x86_rows, x86_xor_from},
    poly1305::{aead_tag_scalar, aead_tag_scalar_one_chain},
    stack::scrub_dead_stack_avx512,
  },
  traits::ct,
};

/// Messages that open in one row pass: block 0 and at most seven message blocks.
pub(super) const ONE_PASS_MAX: usize = 7 * 64;

/// Ciphertext lengths from which the vector Poly1305 kernels outrun one scalar chain.
const VECTOR_POLY_MIN: usize = 1024;

/// Dead stack cleared after [`open_one_pass`]. Its linked frame holds spilled
/// keystream on the reviewed targets and must fit within this bound.
const ONE_PASS_STACK_SCRUB_WORDS: usize = 40;

/// RFC 8439 Poly1305 over `aad` and `ciphertext` with the one-time `poly_key`.
#[inline]
fn tag(poly_key: &[u8; 32], aad: &[u8], ciphertext: &[u8]) -> [u8; 16] {
  if ciphertext.len() >= VECTOR_POLY_MIN {
    poly1305::authenticate_aead(AeadPrimitive::ChaCha20Poly1305, aad, ciphertext, poly_key)
  } else {
    let lengths = super::super::AeadByteLengths::from_usize(aad.len(), ciphertext.len()).to_le_bytes_block();
    aead_tag_scalar(poly_key, aad, ciphertext, &lengths)
  }
}

/// Seal the `len`-byte message of `source` into `dst` and return the tag.
///
/// # Safety
///
/// The caller must establish AVX512F, AVX512VL, AVX512BW, and AVX512DQ. The
/// source's head is valid for its reads and `dst` for `len` writes; `dst`
/// equals the head or is disjoint from the source. The message holds `len`
/// bytes within ChaCha20's 2³²-block limit.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub(super) unsafe fn seal(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  source: X86Source<'_>,
  dst: *mut u8,
  len: usize,
) -> [u8; 16] {
  let mut poly_key = [0u8; 32];
  // SAFETY: the caller establishes the features and ranges.
  unsafe { x86_xor_from(key, nonce, 1, source, dst, len, Some(&mut poly_key)) };
  // SAFETY: the pass above initialized all `len` bytes of `dst`, and no other
  // borrow of them is live.
  let ciphertext = unsafe { core::slice::from_raw_parts(dst, len) };
  let tag = tag(&poly_key, aad, ciphertext);
  ct::zeroize(&mut poly_key);
  tag
}

/// Open `data[..len]` in place if the tag equals `expected`.
///
/// Returns whether the tag matched. On a match `data` holds the plaintext;
/// otherwise every byte of `data` is zero.
///
/// # Safety
///
/// The caller must establish AVX512F, AVX512VL, AVX512BW, and AVX512DQ, and
/// `data` valid for `len` reads and writes within ChaCha20's 2³²-block limit.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
pub(super) unsafe fn open(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  data: *mut u8,
  len: usize,
  expected: &[u8; 16],
) -> bool {
  let source = X86Source {
    head: data,
    head_len: len,
    tail: &[],
  };
  if len <= ONE_PASS_MAX {
    // SAFETY: the caller establishes the features and the range.
    let matched = unsafe {
      if len <= 3 * 64 {
        open_one_pass::<1>(key, nonce, aad, source, data, len, expected)
      } else {
        open_one_pass::<2>(key, nonce, aad, source, data, len, expected)
      }
    };
    // SAFETY: the caller establishes AVX512F.
    unsafe { scrub_dead_stack_avx512::<ONE_PASS_STACK_SCRUB_WORDS>() };
    return matched;
  }
  let mut poly_key = [0u8; 32];
  // SAFETY: the caller establishes the features. An empty message computes
  // only block 0 and writes only `poly_key`.
  unsafe {
    let empty = X86Source {
      head: data,
      head_len: 0,
      tail: &[],
    };
    x86_xor_from(key, nonce, 1, empty, data, 0, Some(&mut poly_key));
  }
  // SAFETY: `data` is valid for `len` reads and no mutable borrow is live.
  let computed = tag(&poly_key, aad, unsafe { core::slice::from_raw_parts(data, len) });
  ct::zeroize(&mut poly_key);
  if !ct::fixed_eq(&computed, expected).declassify() {
    // SAFETY: `data` is valid for `len` writes and no other borrow is live.
    ct::zeroize(unsafe { core::slice::from_raw_parts_mut(data, len) });
    return false;
  }
  // SAFETY: the caller establishes the features and the range; the source and
  // destination are the same buffer.
  unsafe { x86_xor_from(key, nonce, 1, source, data, len, None) };
  true
}

/// [`open`] for messages that fit `G` row groups with block 0.
///
/// The keystream stays live across the Poly1305 call, so the compiler spills
/// it; this runs out of line and its caller scrubs the released frame.
///
/// # Safety
///
/// The contract of [`open`], with `len.div_ceil(64) + 1 <= 4 * G`.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512dq")]
#[inline(never)]
unsafe fn open_one_pass<const G: usize>(
  key: &[u8; KEY_SIZE],
  nonce: &[u8; NONCE_SIZE],
  aad: &[u8],
  source: X86Source<'_>,
  data: *mut u8,
  len: usize,
  expected: &[u8; 16],
) -> bool {
  let counters = core::array::from_fn(|lane| u32::try_from(lane).unwrap_or(u32::MAX));
  // SAFETY: the caller establishes AVX512F; `G <= 2`.
  let keystream = unsafe { x86_rows::<G>(key, nonce, &counters) };
  let mut poly_key = [0u8; 32];
  // SAFETY: `poly_key` is 32 writable bytes; the mask selects the first 32 of block 0.
  unsafe { _mm512_mask_storeu_epi8(poly_key.as_mut_ptr().cast(), x86_byte_mask(32), keystream[0][0]) };
  // One-pass messages are shorter than VECTOR_POLY_MIN, where `tag` takes the
  // scalar chain. This one inlines, so no call clobbers the keystream registers.
  let lengths = super::super::AeadByteLengths::from_usize(aad.len(), len).to_le_bytes_block();
  // SAFETY: `data` is valid for `len` reads and no mutable borrow is live.
  let computed = aead_tag_scalar_one_chain(
    &poly_key,
    aad,
    unsafe { core::slice::from_raw_parts(data, len) },
    &lengths,
  );
  ct::zeroize(&mut poly_key);
  let matched = ct::fixed_eq(&computed, expected).declassify();
  // Literal indices keep the keystream in registers; a loop over the nested
  // arrays leaves it in a stack temporary. Block 0 was the Poly1305 key.
  macro_rules! block {
    ($group:literal, $lane:literal, $offset:literal) => {
      if let Some(row) = keystream.get($group)
        && $offset < len
      {
        // SAFETY: AVX512BW is established; `offset < len`, and the mask
        // selects only bytes inside `data[..len]`.
        unsafe {
          let text = if matched {
            _mm512_xor_si512(source.chunk($offset), row[$lane])
          } else {
            _mm512_setzero_si512()
          };
          _mm512_mask_storeu_epi8(data.add($offset).cast(), x86_byte_mask(len.strict_sub($offset)), text);
        }
      }
    };
  }
  block!(0, 1, 0);
  block!(0, 2, 64);
  block!(0, 3, 128);
  block!(1, 0, 192);
  block!(1, 1, 256);
  block!(1, 2, 320);
  block!(1, 3, 384);
  matched
}
