//! Private SHA-1 core for the RFC 6455 accept-digest capability.

use crate::backend::cache::OnceCache;

#[cfg(all(not(miri), target_arch = "aarch64"))]
mod aarch64;
#[cfg(all(not(miri), target_arch = "x86_64"))]
mod x86_64;

const BLOCK_LEN: usize = 64;
const LENGTH_OFFSET: usize = 56;
const MAX_MESSAGE_BYTES: u64 = u64::MAX / 8;
const INITIAL_STATE: [u32; 5] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476, 0xc3d2_e1f0];
const STANDARD_WEBSOCKET_KEY_LEN: usize = 24;
const STANDARD_WEBSOCKET_MESSAGE_LEN: usize = STANDARD_WEBSOCKET_KEY_LEN.strict_add(super::WEBSOCKET_GUID.len());

type CompressFn = fn(&mut [u32; 5], &[u8; BLOCK_LEN]);

static ACTIVE_COMPRESS: OnceCache<CompressFn> = OnceCache::new();

const COMPILE_TIME_HW: bool = cfg!(all(
  not(miri),
  not(feature = "portable-only"),
  any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_feature = "sha2")
    ),
    all(
      target_arch = "x86_64",
      target_feature = "sha",
      target_feature = "ssse3",
      target_feature = "sse4.1"
    )
  )
));

struct Sha1 {
  state: [u32; 5],
  buffer: [u8; BLOCK_LEN],
  buffer_len: usize,
  message_len: u64,
  compress: CompressFn,
}

impl Sha1 {
  fn new() -> Self {
    Self::new_with(selected_compress())
  }

  const fn new_with(compress: CompressFn) -> Self {
    Self {
      state: INITIAL_STATE,
      buffer: [0; BLOCK_LEN],
      buffer_len: 0,
      message_len: 0,
      compress,
    }
  }

  fn update(&mut self, mut input: &[u8]) {
    let input_len = u64::try_from(input.len()).expect("SHA-1 input length exceeds u64");
    let message_len = self
      .message_len
      .checked_add(input_len)
      .expect("SHA-1 message byte length overflow");
    assert!(
      message_len <= MAX_MESSAGE_BYTES,
      "SHA-1 message length exceeds the 64-bit bit-length field"
    );
    self.message_len = message_len;

    if self.buffer_len != 0 {
      let available = BLOCK_LEN.strict_sub(self.buffer_len);
      let take = core::cmp::min(available, input.len());
      let end = self.buffer_len.strict_add(take);
      self.buffer[self.buffer_len..end].copy_from_slice(&input[..take]);
      self.buffer_len = end;
      input = &input[take..];

      if self.buffer_len == BLOCK_LEN {
        (self.compress)(&mut self.state, &self.buffer);
        self.buffer_len = 0;
      }
    }

    while input.len() >= BLOCK_LEN {
      let (block, rest) = input.split_at(BLOCK_LEN);
      let block = <&[u8; BLOCK_LEN]>::try_from(block).expect("split SHA-1 block has exact length");
      (self.compress)(&mut self.state, block);
      input = rest;
    }

    if !input.is_empty() {
      self.buffer[..input.len()].copy_from_slice(input);
      self.buffer_len = input.len();
    }
  }

  fn finalize(mut self) -> [u8; 20] {
    let bit_len = self
      .message_len
      .checked_mul(8)
      .expect("SHA-1 message bit length overflow");

    self.buffer[self.buffer_len] = 0x80;
    self.buffer_len = self.buffer_len.strict_add(1);

    if self.buffer_len > LENGTH_OFFSET {
      self.buffer[self.buffer_len..].fill(0);
      (self.compress)(&mut self.state, &self.buffer);
      self.buffer_len = 0;
    }

    self.buffer[self.buffer_len..LENGTH_OFFSET].fill(0);
    self.buffer[LENGTH_OFFSET..].copy_from_slice(&bit_len.to_be_bytes());
    (self.compress)(&mut self.state, &self.buffer);

    state_to_digest(&self.state)
  }
}

#[inline]
pub(super) fn digest_websocket_key(sec_websocket_key: &[u8]) -> [u8; 20] {
  if let Ok(standard_key) = <&[u8; STANDARD_WEBSOCKET_KEY_LEN]>::try_from(sec_websocket_key) {
    digest_standard_websocket_key(standard_key)
  } else {
    digest_parts(sec_websocket_key, super::WEBSOCKET_GUID)
  }
}

#[inline]
fn digest_standard_websocket_key(sec_websocket_key: &[u8; STANDARD_WEBSOCKET_KEY_LEN]) -> [u8; 20] {
  if COMPILE_TIME_HW {
    return digest_standard_websocket_key_with(sec_websocket_key, compress_compile_time);
  }

  digest_standard_websocket_key_with(sec_websocket_key, selected_compress())
}

#[inline(always)]
fn digest_standard_websocket_key_with(
  sec_websocket_key: &[u8; STANDARD_WEBSOCKET_KEY_LEN],
  mut compress: impl FnMut(&mut [u32; 5], &[u8; BLOCK_LEN]),
) -> [u8; 20] {
  let mut first_block = [0u8; BLOCK_LEN];
  first_block[..STANDARD_WEBSOCKET_KEY_LEN].copy_from_slice(sec_websocket_key);
  first_block[STANDARD_WEBSOCKET_KEY_LEN..STANDARD_WEBSOCKET_MESSAGE_LEN].copy_from_slice(super::WEBSOCKET_GUID);
  first_block[STANDARD_WEBSOCKET_MESSAGE_LEN] = 0x80;

  let mut final_block = [0u8; BLOCK_LEN];
  let bit_len = u64::try_from(STANDARD_WEBSOCKET_MESSAGE_LEN)
    .expect("fixed WebSocket message length fits u64")
    .strict_mul(8);
  final_block[LENGTH_OFFSET..].copy_from_slice(&bit_len.to_be_bytes());

  let mut state = INITIAL_STATE;
  compress(&mut state, &first_block);
  compress(&mut state, &final_block);
  state_to_digest(&state)
}

#[inline(always)]
fn compress_compile_time(state: &mut [u32; 5], block: &[u8; BLOCK_LEN]) {
  #[cfg(all(
    not(miri),
    not(feature = "portable-only"),
    target_arch = "aarch64",
    any(target_os = "macos", target_feature = "sha2")
  ))]
  {
    // SAFETY: This function is reachable only when the Apple AArch64 target
    // contract or a static `sha2` target feature guarantees SHA instructions.
    unsafe { aarch64::compress(state, block) }
  }

  #[cfg(all(
    not(miri),
    not(feature = "portable-only"),
    target_arch = "x86_64",
    target_feature = "sha",
    target_feature = "ssse3",
    target_feature = "sse4.1"
  ))]
  {
    // SAFETY: The compile-time cfg above establishes every target feature
    // required by the x86_64 SHA-NI kernel.
    unsafe { x86_64::compress(state, block) }
  }

  #[cfg(not(all(
    not(miri),
    not(feature = "portable-only"),
    any(
      all(target_arch = "aarch64", any(target_os = "macos", target_feature = "sha2")),
      all(
        target_arch = "x86_64",
        target_feature = "sha",
        target_feature = "ssse3",
        target_feature = "sse4.1"
      )
    )
  )))]
  {
    compress_portable(state, block);
  }
}

fn state_to_digest(state: &[u32; 5]) -> [u8; 20] {
  let mut digest = [0u8; 20];
  for (word, output) in state.iter().zip(digest.as_chunks_mut::<4>().0) {
    *output = word.to_be_bytes();
  }
  digest
}

pub(super) fn digest_parts(first: &[u8], second: &[u8]) -> [u8; 20] {
  let mut sha1 = Sha1::new();
  sha1.update(first);
  sha1.update(second);
  sha1.finalize()
}

#[cfg(test)]
fn digest_parts_with(first: &[u8], second: &[u8], compress: CompressFn) -> [u8; 20] {
  let mut sha1 = Sha1::new_with(compress);
  sha1.update(first);
  sha1.update(second);
  sha1.finalize()
}

#[cfg(miri)]
fn selected_compress() -> CompressFn {
  ACTIVE_COMPRESS.get_or_init(|| compress_portable)
}

#[cfg(all(not(miri), target_arch = "aarch64"))]
fn selected_compress() -> CompressFn {
  ACTIVE_COMPRESS.get_or_init(|| {
    use crate::platform::caps::aarch64;

    if crate::platform::caps().has(aarch64::SHA2) {
      compress_aarch64_sha2
    } else {
      compress_portable
    }
  })
}

#[cfg(all(not(miri), target_arch = "aarch64"))]
fn compress_aarch64_sha2(state: &mut [u32; 5], block: &[u8; BLOCK_LEN]) {
  // SAFETY: This wrapper is selected only when AArch64 SHA2 is guaranteed by
  // the macOS target contract or reported by `platform::caps()`.
  unsafe { aarch64::compress(state, block) }
}

#[cfg(all(not(miri), target_arch = "x86_64"))]
fn selected_compress() -> CompressFn {
  ACTIVE_COMPRESS.get_or_init(|| {
    use crate::platform::caps::x86;

    let required = x86::SHA | x86::SSSE3 | x86::SSE41;
    if crate::platform::caps().has(required) {
      compress_x86_sha
    } else {
      compress_portable
    }
  })
}

#[cfg(all(not(miri), target_arch = "x86_64"))]
fn compress_x86_sha(state: &mut [u32; 5], block: &[u8; BLOCK_LEN]) {
  // SAFETY: This wrapper is selected only after compile-time or runtime
  // validation of SHA, SSSE3, and SSE4.1. x86_64 supplies SSE2.
  unsafe { x86_64::compress(state, block) }
}

#[cfg(all(not(miri), not(any(target_arch = "aarch64", target_arch = "x86_64"))))]
fn selected_compress() -> CompressFn {
  ACTIVE_COMPRESS.get_or_init(|| compress_portable)
}

// The four-round portable schedule follows the optimizer-legible structure in
// RustCrypto `sha1` 0.11.0 (MIT OR Apache-2.0), adapted to rscrypto's checked
// arithmetic and single-block private capability boundary.

const ROUND_CONSTANTS: [u32; 4] = [0x5a82_7999, 0x6ed9_eba1, 0x8f1b_bcdc, 0xca62_c1d6];

#[inline(always)]
fn add4(left: [u32; 4], right: [u32; 4]) -> [u32; 4] {
  [
    left[0].wrapping_add(right[0]),
    left[1].wrapping_add(right[1]),
    left[2].wrapping_add(right[2]),
    left[3].wrapping_add(right[3]),
  ]
}

#[inline(always)]
fn xor4(left: [u32; 4], right: [u32; 4]) -> [u32; 4] {
  [
    left[0] ^ right[0],
    left[1] ^ right[1],
    left[2] ^ right[2],
    left[3] ^ right[3],
  ]
}

#[inline(always)]
fn first_add(e: u32, words: [u32; 4]) -> [u32; 4] {
  let [first, second, third, fourth] = words;
  [e.wrapping_add(first), second, third, fourth]
}

#[inline(always)]
fn schedule_first(left: [u32; 4], right: [u32; 4]) -> [u32; 4] {
  let [_, _, left_two, left_three] = left;
  let [right_zero, right_one, _, _] = right;
  [
    left[0] ^ left_two,
    left[1] ^ left_three,
    left[2] ^ right_zero,
    left[3] ^ right_one,
  ]
}

#[inline(always)]
fn schedule_second(left: [u32; 4], right: [u32; 4]) -> [u32; 4] {
  let [left_zero, left_one, left_two, left_three] = left;
  let [_, right_one, right_two, right_three] = right;

  let word_16 = (left_zero ^ right_one).rotate_left(1);
  let word_17 = (left_one ^ right_two).rotate_left(1);
  let word_18 = (left_two ^ right_three).rotate_left(1);
  let word_19 = (left_three ^ word_16).rotate_left(1);
  [word_16, word_17, word_18, word_19]
}

#[inline(always)]
fn first_half(state: [u32; 4], message: [u32; 4]) -> [u32; 4] {
  first_add(state[0].rotate_left(30), message)
}

#[inline(always)]
fn rounds_choose(state: [u32; 4], message: [u32; 4]) -> [u32; 4] {
  let [mut a, mut b, mut c, mut d] = state;
  let [first, second, third, fourth] = message;

  let mut e = a.rotate_left(5).wrapping_add(d ^ (b & (c ^ d))).wrapping_add(first);
  b = b.rotate_left(30);
  d = e
    .rotate_left(5)
    .wrapping_add(c ^ (a & (b ^ c)))
    .wrapping_add(second)
    .wrapping_add(d);
  a = a.rotate_left(30);
  c = d
    .rotate_left(5)
    .wrapping_add(b ^ (e & (a ^ b)))
    .wrapping_add(third)
    .wrapping_add(c);
  e = e.rotate_left(30);
  b = c
    .rotate_left(5)
    .wrapping_add(a ^ (d & (e ^ a)))
    .wrapping_add(fourth)
    .wrapping_add(b);
  d = d.rotate_left(30);

  [b, c, d, e]
}

#[inline(always)]
fn rounds_parity(state: [u32; 4], message: [u32; 4]) -> [u32; 4] {
  let [mut a, mut b, mut c, mut d] = state;
  let [first, second, third, fourth] = message;

  let mut e = a.rotate_left(5).wrapping_add(b ^ c ^ d).wrapping_add(first);
  b = b.rotate_left(30);
  d = e
    .rotate_left(5)
    .wrapping_add(a ^ b ^ c)
    .wrapping_add(second)
    .wrapping_add(d);
  a = a.rotate_left(30);
  c = d
    .rotate_left(5)
    .wrapping_add(e ^ a ^ b)
    .wrapping_add(third)
    .wrapping_add(c);
  e = e.rotate_left(30);
  b = c
    .rotate_left(5)
    .wrapping_add(d ^ e ^ a)
    .wrapping_add(fourth)
    .wrapping_add(b);
  d = d.rotate_left(30);

  [b, c, d, e]
}

#[inline(always)]
fn rounds_majority(state: [u32; 4], message: [u32; 4]) -> [u32; 4] {
  let [mut a, mut b, mut c, mut d] = state;
  let [first, second, third, fourth] = message;

  let mut e = a
    .rotate_left(5)
    .wrapping_add((b & c) ^ (b & d) ^ (c & d))
    .wrapping_add(first);
  b = b.rotate_left(30);
  d = e
    .rotate_left(5)
    .wrapping_add((a & b) ^ (a & c) ^ (b & c))
    .wrapping_add(second)
    .wrapping_add(d);
  a = a.rotate_left(30);
  c = d
    .rotate_left(5)
    .wrapping_add((e & a) ^ (e & b) ^ (a & b))
    .wrapping_add(third)
    .wrapping_add(c);
  e = e.rotate_left(30);
  b = c
    .rotate_left(5)
    .wrapping_add((d & e) ^ (d & a) ^ (e & a))
    .wrapping_add(fourth)
    .wrapping_add(b);
  d = d.rotate_left(30);

  [b, c, d, e]
}

#[inline(always)]
fn digest_rounds(state: [u32; 4], work: [u32; 4], phase: usize) -> [u32; 4] {
  let constant = [ROUND_CONSTANTS[phase]; 4];
  let work = add4(work, constant);
  match phase {
    0 => rounds_choose(state, work),
    2 => rounds_majority(state, work),
    // The constant-table lookup above rejects phases outside 0..4. Both
    // remaining valid phases use SHA-1's parity function.
    _ => rounds_parity(state, work),
  }
}

macro_rules! rounds4 {
  ($left:ident, $right:ident, $words:expr, $phase:expr) => {
    digest_rounds($left, first_half($right, $words), $phase)
  };
}

macro_rules! expand_schedule {
  ($first:expr, $second:expr, $third:expr, $fourth:expr) => {
    schedule_second(xor4(schedule_first($first, $second), $third), $fourth)
  };
}

macro_rules! schedule_rounds4 {
  ($left:ident, $right:ident, $first:expr, $second:expr, $third:expr, $fourth:expr, $next:expr, $phase:expr) => {
    $next = expand_schedule!($first, $second, $third, $fourth);
    $right = rounds4!($left, $right, $next, $phase);
  };
}

#[inline(always)]
fn compress_portable(state: &mut [u32; 5], block: &[u8; BLOCK_LEN]) {
  let mut words = [[0u32; 4]; 4];
  for (word_group, bytes) in words.iter_mut().zip(block.as_chunks::<16>().0) {
    for (word, chunk) in word_group.iter_mut().zip(bytes.as_chunks::<4>().0) {
      *word = u32::from_be_bytes(*chunk);
    }
  }

  let [mut words_zero, mut words_one, mut words_two, mut words_three] = words;
  let mut words_four;
  let mut state_zero = [state[0], state[1], state[2], state[3]];
  let mut state_one = first_add(state[4], words_zero);

  state_one = digest_rounds(state_zero, state_one, 0);
  state_zero = rounds4!(state_one, state_zero, words_one, 0);
  state_one = rounds4!(state_zero, state_one, words_two, 0);
  state_zero = rounds4!(state_one, state_zero, words_three, 0);
  schedule_rounds4!(
    state_zero,
    state_one,
    words_zero,
    words_one,
    words_two,
    words_three,
    words_four,
    0
  );

  schedule_rounds4!(
    state_one,
    state_zero,
    words_one,
    words_two,
    words_three,
    words_four,
    words_zero,
    1
  );
  schedule_rounds4!(
    state_zero,
    state_one,
    words_two,
    words_three,
    words_four,
    words_zero,
    words_one,
    1
  );
  schedule_rounds4!(
    state_one,
    state_zero,
    words_three,
    words_four,
    words_zero,
    words_one,
    words_two,
    1
  );
  schedule_rounds4!(
    state_zero,
    state_one,
    words_four,
    words_zero,
    words_one,
    words_two,
    words_three,
    1
  );
  schedule_rounds4!(
    state_one,
    state_zero,
    words_zero,
    words_one,
    words_two,
    words_three,
    words_four,
    1
  );

  schedule_rounds4!(
    state_zero,
    state_one,
    words_one,
    words_two,
    words_three,
    words_four,
    words_zero,
    2
  );
  schedule_rounds4!(
    state_one,
    state_zero,
    words_two,
    words_three,
    words_four,
    words_zero,
    words_one,
    2
  );
  schedule_rounds4!(
    state_zero,
    state_one,
    words_three,
    words_four,
    words_zero,
    words_one,
    words_two,
    2
  );
  schedule_rounds4!(
    state_one,
    state_zero,
    words_four,
    words_zero,
    words_one,
    words_two,
    words_three,
    2
  );
  schedule_rounds4!(
    state_zero,
    state_one,
    words_zero,
    words_one,
    words_two,
    words_three,
    words_four,
    2
  );

  schedule_rounds4!(
    state_one,
    state_zero,
    words_one,
    words_two,
    words_three,
    words_four,
    words_zero,
    3
  );
  schedule_rounds4!(
    state_zero,
    state_one,
    words_two,
    words_three,
    words_four,
    words_zero,
    words_one,
    3
  );
  schedule_rounds4!(
    state_one,
    state_zero,
    words_three,
    words_four,
    words_zero,
    words_one,
    words_two,
    3
  );
  schedule_rounds4!(
    state_zero,
    state_one,
    words_four,
    words_zero,
    words_one,
    words_two,
    words_three,
    3
  );
  schedule_rounds4!(
    state_one,
    state_zero,
    words_zero,
    words_one,
    words_two,
    words_three,
    words_four,
    3
  );

  let [a, b, c, d] = state_zero;
  let e = state_one[0].rotate_left(30);
  state[0] = state[0].wrapping_add(a);
  state[1] = state[1].wrapping_add(b);
  state[2] = state[2].wrapping_add(c);
  state[3] = state[3].wrapping_add(d);
  state[4] = state[4].wrapping_add(e);
}

#[cfg(test)]
mod tests {
  use super::{compress_portable, digest_parts, digest_parts_with, digest_websocket_key};

  #[cfg(all(not(miri), any(target_arch = "aarch64", target_arch = "x86_64")))]
  use super::CompressFn;

  fn assert_hex(message: &[u8], expected: [u8; 20]) {
    assert_eq!(digest_parts_with(message, b"", compress_portable), expected);
  }

  #[cfg(all(not(miri), any(target_arch = "aarch64", target_arch = "x86_64")))]
  fn assert_backend_matches_portable(compress: CompressFn) {
    let mut input = [0u8; 257];
    for (index, byte) in input.iter_mut().enumerate() {
      *byte = index.to_le_bytes()[0].wrapping_mul(37).wrapping_add(11);
    }

    for len in [0usize, 1, 19, 20, 27, 28, 55, 56, 63, 64, 65, 127, 128, 129, 256, 257] {
      let message = &input[..len];
      assert_eq!(
        digest_parts_with(message, b"", compress),
        digest_parts_with(message, b"", compress_portable),
        "SHA-1 backend mismatch for message length {len}"
      );
    }
  }

  #[test]
  fn sha1_known_answer_tests() {
    assert_hex(
      b"",
      [
        0xda, 0x39, 0xa3, 0xee, 0x5e, 0x6b, 0x4b, 0x0d, 0x32, 0x55, 0xbf, 0xef, 0x95, 0x60, 0x18, 0x90, 0xaf, 0xd8,
        0x07, 0x09,
      ],
    );
    assert_hex(
      b"abc",
      [
        0xa9, 0x99, 0x3e, 0x36, 0x47, 0x06, 0x81, 0x6a, 0xba, 0x3e, 0x25, 0x71, 0x78, 0x50, 0xc2, 0x6c, 0x9c, 0xd0,
        0xd8, 0x9d,
      ],
    );
    assert_hex(
      b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
      [
        0x84, 0x98, 0x3e, 0x44, 0x1c, 0x3b, 0xd2, 0x6e, 0xba, 0xae, 0x4a, 0xa1, 0xf9, 0x51, 0x29, 0xe5, 0xe5, 0x46,
        0x70, 0xf1,
      ],
    );
  }

  #[test]
  fn split_updates_match_contiguous_known_answer() {
    assert_eq!(
      digest_parts(b"abcdbcdecdefdefgefghfghi", b"ghijhijkijkljklmklmnlmnomnopnopq"),
      digest_parts(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", b"")
    );
  }

  #[test]
  fn standard_websocket_key_path_matches_general_path() {
    let key = b"dGhlIHNhbXBsZSBub25jZQ==";
    assert_eq!(
      digest_websocket_key(key),
      digest_parts(key, super::super::WEBSOCKET_GUID)
    );
  }

  #[cfg(all(not(miri), target_arch = "aarch64"))]
  #[test]
  fn aarch64_sha2_matches_portable() {
    use crate::platform::caps::aarch64;

    if crate::platform::caps().has(aarch64::SHA2) {
      assert_backend_matches_portable(super::compress_aarch64_sha2);
    }
  }

  #[cfg(all(not(miri), target_arch = "x86_64"))]
  #[test]
  fn x86_sha_matches_portable_when_available() {
    use crate::platform::caps::x86;

    if crate::platform::caps().has(x86::SHA | x86::SSSE3 | x86::SSE41) {
      assert_backend_matches_portable(super::compress_x86_sha);
    }
  }
}
