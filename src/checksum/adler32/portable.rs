//! Portable Adler-32 kernel with deferred reduction.

/// The largest prime below 2^16.
pub(super) const MOD: u32 = 65_521;
pub(super) const MOD64: u64 = 65_521;

/// Canonical name of the portable kernel.
pub(super) const KERNEL_NAME: &str = "portable/block32";

/// Bytes per block of the inner loop, which keeps one column sum per byte position.
const BLOCK: usize = 32;
/// Blocks between reductions. A column sum stays at most `255 * CHUNK_BLOCKS`, and a sum of
/// column sums at most `255 * CHUNK_BLOCKS * (CHUNK_BLOCKS + 1) / 2`.
const CHUNK_BLOCKS: usize = 4096;
/// Every 32-bit lane sum of a chunk stays at most `u32::MAX`.
const CHUNK_SUMS_FIT: () = assert!(255 * CHUNK_BLOCKS * (CHUNK_BLOCKS + 1) / 2 <= u32::MAX as usize);
/// Blocks whose column sums fit in 16 bits, so the innermost loop widens each byte only once.
const SUB_BLOCKS: usize = 256;
/// Every 16-bit column sum of a sub-chunk stays at most `u16::MAX`.
const SUB_SUMS_FIT: () = assert!(255 * SUB_BLOCKS <= u16::MAX as usize);

/// Add `data` to the Adler-32 value `adler`, whose two sums are reduced.
///
/// Over `n` bytes `x[j]`, `a` gains `Σ x[j]` and `b` gains `n·a + Σ (n - j)·x[j]`.
/// Each chunk keeps, for every position `i` in a block, the column sum of `x` at
/// `i` and the running total of that column sum after each block. Those are
/// independent lane additions that the compiler can vectorize; the weighted sum
/// follows from them once per chunk.
pub(super) fn update(adler: u32, data: &[u8]) -> u32 {
  let mut a = adler & 0xffff;
  let mut b = adler >> 16;
  let (blocks, tail) = data.as_chunks::<BLOCK>();
  for chunk in blocks.chunks(CHUNK_BLOCKS) {
    // `SUB_SUMS_FIT` and `CHUNK_SUMS_FIT` bound every lane sum by its type's maximum, so no
    // lane addition below wraps and each wrapping add equals the exact add. Checked adds
    // would keep the compiler from vectorizing these loops.
    let ((), ()) = (SUB_SUMS_FIT, CHUNK_SUMS_FIT);
    let mut columns = [0u32; BLOCK];
    let mut running = [0u32; BLOCK];
    for sub in chunk.chunks(SUB_BLOCKS) {
      let mut sub_columns = [0u16; BLOCK];
      let mut sub_running = [0u32; BLOCK];
      for block in sub {
        for ((column, total), &byte) in sub_columns.iter_mut().zip(&mut sub_running).zip(block) {
          *column = column.wrapping_add(u16::from(byte));
          *total = total.wrapping_add(u32::from(*column));
        }
      }
      // Each block of `sub` also carried the column sums of the earlier blocks.
      let blocks = u32::try_from(sub.len()).expect("a sub-chunk holds at most SUB_BLOCKS blocks");
      for (((column, total), &sub_column), &sub_total) in
        columns.iter_mut().zip(&mut running).zip(&sub_columns).zip(&sub_running)
      {
        *total = total.wrapping_add(sub_total).wrapping_add(blocks.wrapping_mul(*column));
        *column = column.wrapping_add(u32::from(sub_column));
      }
    }
    // Byte `i` of block `t` among `k` blocks has weight `BLOCK·(k - t) - i`, and the running
    // totals hold `Σ_t (k - t)·x[t][i]`. Under `CHUNK_SUMS_FIT`, the column sums total below
    // 2^25, their index-weighted total below 2^30, and the running totals below 2^37, so these
    // folds do not wrap either and each wrapping add equals the exact add.
    let sum = columns.iter().fold(0u32, |sum, &column| sum.wrapping_add(column));
    let offset = (0u32..)
      .zip(&columns)
      .fold(0u32, |offset, (i, &column)| offset.wrapping_add(i.wrapping_mul(column)));
    let total = running
      .iter()
      .fold(0u64, |total, &running| total.wrapping_add(u64::from(running)));
    let len = (chunk.len() as u64).strict_mul(BLOCK as u64);
    let weighted = u64::from(b)
      .strict_add(len.strict_mul(u64::from(a)))
      .strict_add(total.strict_mul(BLOCK as u64))
      .strict_sub(u64::from(offset));
    a = reduce(u64::from(a).strict_add(u64::from(sum)));
    b = reduce(weighted);
  }
  // Fewer than 32 bytes remain, so neither sum can pass 2^32.
  for &byte in tail {
    a = a.strict_add(u32::from(byte));
    b = b.strict_add(a);
  }
  ((b % MOD) << 16) | (a % MOD)
}

/// Reduce `value` modulo 65521.
#[inline(always)]
pub(super) fn reduce(value: u64) -> u32 {
  let [r0, r1, r2, r3, ..] = (value % MOD64).to_le_bytes();
  u32::from_le_bytes([r0, r1, r2, r3])
}
