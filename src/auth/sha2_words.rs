#[inline(always)]
pub(super) fn write_u32x8_be(dst: &mut [u8], words: &[u32; 8]) {
  for (chunk, word) in dst[..32].as_chunks_mut::<4>().0.iter_mut().zip(words) {
    *chunk = word.to_be_bytes();
  }
}

#[inline(always)]
pub(super) fn write_u64_be<const WORDS: usize>(dst: &mut [u8], words: &[u64; 8]) {
  for (chunk, word) in dst[..WORDS.strict_mul(8)]
    .as_chunks_mut::<8>()
    .0
    .iter_mut()
    .zip(&words[..WORDS])
  {
    *chunk = word.to_be_bytes();
  }
}
