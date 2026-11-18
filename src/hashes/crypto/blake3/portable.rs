// The portable BLAKE3 compression function, with its flags, IV, and message schedule.
//
// rscrypto compiles this file as its Portable kernel. It also exports the text as
// `rscrypto::hashes::expert::BLAKE3_PORTABLE_SOURCE`, for code that cannot link rscrypto.
// An embedder compiles the same text inline, as a `mod` file, or with `include!`.
//
// Rules for this text:
// - Use only `core`. Use no item and no macro of another crate.
// - Write no `unsafe` code and no conditional compilation.
// - Write no inner attribute, no inner doc comment, and no lint attribute.
// - Declare each item `pub(crate)`.
// - Use only the language and library of Rust 1.85, edition 2024.
//
// The lints of rscrypto and the `compatibility` lane of `scripts/check/compat.py` enforce these
// rules. That lane rejects conditional compilation in this text. Then it builds the text alone
// in a `no_std` cdylib with Rust 1.85, and it denies each warning.

// Domain flags: the `flags` input of the compression function.
pub(crate) const CHUNK_START: u32 = 1 << 0;
pub(crate) const CHUNK_END: u32 = 1 << 1;
pub(crate) const PARENT: u32 = 1 << 2;
pub(crate) const ROOT: u32 = 1 << 3;
pub(crate) const KEYED_HASH: u32 = 1 << 4;
pub(crate) const DERIVE_KEY_CONTEXT: u32 = 1 << 5;
pub(crate) const DERIVE_KEY_MATERIAL: u32 = 1 << 6;

/// BLAKE3 initialization vector: the key words of the unkeyed modes.
pub(crate) const IV: [u32; 8] = [
  0x6A09_E667,
  0xBB67_AE85,
  0x3C6E_F372,
  0xA54F_F53A,
  0x510E_527F,
  0x9B05_688C,
  0x1F83_D9AB,
  0x5BE0_CD19,
];

/// BLAKE3 message schedule.
///
/// `MSG_SCHEDULE[round][i]` gives the index of the message word to use.
pub(crate) const MSG_SCHEDULE: [[usize; 16]; 7] = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
  [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
  [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
  [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
  [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
  [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

/// Decodes a 64-byte block into its 16 little-endian message words.
#[inline(always)]
pub(crate) const fn block_words(block: &[u8; 64]) -> [u32; 16] {
  let mut words = [0u32; 16];
  let mut bytes: &[u8] = block;
  let mut out: &mut [u32] = &mut words;
  while let (Some((word_bytes, rest)), Some((word, more))) = (bytes.split_first_chunk::<4>(), out.split_first_mut()) {
    *word = u32::from_le_bytes(*word_bytes);
    bytes = rest;
    out = more;
  }
  words
}

/// The seven rounds of the compression function, before its output step.
///
/// The rounds are shared, but the 8-word chaining-value output step stays separate from
/// the 16-word XOF output step. Inlining the round state lets LLVM omit the unused
/// second output half from a chaining-value caller.
#[inline(always)]
pub(crate) const fn compress_pre(
  cv: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  let m = *block_words;
  let [c0, c1, c2, c3, c4, c5, c6, c7] = counter.to_le_bytes();
  let mut v0 = cv[0];
  let mut v1 = cv[1];
  let mut v2 = cv[2];
  let mut v3 = cv[3];
  let mut v4 = cv[4];
  let mut v5 = cv[5];
  let mut v6 = cv[6];
  let mut v7 = cv[7];
  let mut v8 = IV[0];
  let mut v9 = IV[1];
  let mut v10 = IV[2];
  let mut v11 = IV[3];
  let mut v12 = u32::from_le_bytes([c0, c1, c2, c3]);
  let mut v13 = u32::from_le_bytes([c4, c5, c6, c7]);
  let mut v14 = block_len;
  let mut v15 = flags;

  macro_rules! g {
    ($a:ident, $b:ident, $c:ident, $d:ident, $mx:expr, $my:expr) => {{
      $a = $a.wrapping_add($b).wrapping_add($mx);
      $d = ($d ^ $a).rotate_right(16);
      $c = $c.wrapping_add($d);
      $b = ($b ^ $c).rotate_right(12);
      $a = $a.wrapping_add($b).wrapping_add($my);
      $d = ($d ^ $a).rotate_right(8);
      $c = $c.wrapping_add($d);
      $b = ($b ^ $c).rotate_right(7);
    }};
  }

  // One full round: four column steps, then four diagonal steps, with the message
  // words in the order of `MSG_SCHEDULE[$r]`.
  macro_rules! round {
    ($r:literal) => {{
      let s = &MSG_SCHEDULE[$r];
      g!(v0, v4, v8, v12, m[s[0]], m[s[1]]);
      g!(v1, v5, v9, v13, m[s[2]], m[s[3]]);
      g!(v2, v6, v10, v14, m[s[4]], m[s[5]]);
      g!(v3, v7, v11, v15, m[s[6]], m[s[7]]);

      g!(v0, v5, v10, v15, m[s[8]], m[s[9]]);
      g!(v1, v6, v11, v12, m[s[10]], m[s[11]]);
      g!(v2, v7, v8, v13, m[s[12]], m[s[13]]);
      g!(v3, v4, v9, v14, m[s[14]], m[s[15]]);
    }};
  }

  round!(0);
  round!(1);
  round!(2);
  round!(3);
  round!(4);
  round!(5);
  round!(6);

  [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15]
}

/// The compression function of a block that is already decoded into message words.
///
/// Returns all 16 output words. The first 8 words are the new chaining value.
#[inline]
pub(crate) fn compress_words(
  cv: &[u32; 8],
  block_words: &[u32; 16],
  counter: u64,
  block_len: u32,
  flags: u32,
) -> [u32; 16] {
  let mut state = compress_pre(cv, block_words, counter, block_len, flags);
  state[0] ^= state[8];
  state[1] ^= state[9];
  state[2] ^= state[10];
  state[3] ^= state[11];
  state[4] ^= state[12];
  state[5] ^= state[13];
  state[6] ^= state[14];
  state[7] ^= state[15];

  state[8] ^= cv[0];
  state[9] ^= cv[1];
  state[10] ^= cv[2];
  state[11] ^= cv[3];
  state[12] ^= cv[4];
  state[13] ^= cv[5];
  state[14] ^= cv[6];
  state[15] ^= cv[7];
  state
}

/// The compression function of one 64-byte block.
///
/// `block_len` is the count of input bytes in `block`. BLAKE3 pads a short block with zero
/// bytes. Returns all 16 output words. The first 8 words are the new chaining value. When
/// `flags` holds `ROOT`, the 16 words in little-endian order are the output block at `counter`.
#[inline]
pub(crate) fn compress(cv: &[u32; 8], block: &[u8; 64], counter: u64, block_len: u32, flags: u32) -> [u32; 16] {
  compress_words(cv, &block_words(block), counter, block_len, flags)
}
