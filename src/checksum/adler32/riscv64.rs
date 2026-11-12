//! riscv64 vector (RVV 1.0) Adler-32 kernel.
//!
//! Rows of `lanes` bytes widen into 32-bit lanes, one per byte position. Each lane keeps the
//! column sum of its position and the running total of that column sum after each row; the
//! weighted sum follows from both once per chunk. `lanes` is the hart's vector length for
//! 32-bit elements at LMUL 4, at most `MAX_LANES`, so the kernel runs on every VLEN. The byte
//! tail goes to the portable kernel.
//!
//! Rust has no RVV intrinsics, and the safe column-sum loop that LLVM vectorizes keeps both
//! column arrays in memory: two vector loads and two vector stores per row. The inline assembly
//! keeps them in vector registers for a whole chunk.

use core::arch::asm;

use super::portable::{self, reduce};

/// Canonical name of the RVV kernel.
pub(in crate::checksum) const KERNEL_NAME: &str = "riscv64/v";

/// The most lanes per row.
const MAX_LANES: usize = 256;
/// Rows between reductions.
const ROWS: usize = 1024;
/// Over a chunk of at most `ROWS` rows, a column sum stays at most `255·ROWS`, a running total
/// at most `255·ROWS·(ROWS + 1)/2`, and a lane index times its column sum at most
/// `(MAX_LANES - 1)·255·ROWS`. Each fits a 32-bit lane.
const ROW_SUMS_FIT: () = {
  let (rows, lanes) = (ROWS as u64, MAX_LANES as u64);
  assert!(255 * rows * (rows + 1) / 2 <= u32::MAX as u64);
  assert!((lanes - 1) * 255 * rows <= u32::MAX as u64);
};

/// Lanes per row: the vector length for 32-bit elements at LMUL 4, at most `MAX_LANES`.
///
/// # Safety
///
/// The CPU must support the V extension.
#[target_feature(enable = "v")]
fn lanes() -> usize {
  let lanes: usize;
  // SAFETY: The V extension is enabled for this function. `vsetvli` writes only `lanes` and
  // the vector length and type, which no Rust code keeps across inline assembly.
  unsafe {
    asm!(
      "vsetvli {lanes}, {max}, e32, m4, ta, ma",
      lanes = lateout(reg) lanes,
      max = in(reg) MAX_LANES,
      options(nomem, nostack),
    );
  }
  lanes
}

/// Sums of the whole rows of `chunk`, each `lanes` bytes long, with `lanes` from [`lanes`]:
/// the byte sum, the total of the running column sums, and the byte sum weighted by lane index.
///
/// # Safety
///
/// The CPU must support the V extension.
#[target_feature(enable = "v")]
fn chunk_sums(chunk: &[u8], lanes: usize) -> (u64, u64, u64) {
  let rows = chunk.len().strict_div(lanes);
  let (sum, running, offset): (u64, u64, u64);
  // SAFETY: The V extension is enabled for this function. The vector length is `lanes` when
  // `lanes` comes from `lanes()`, and at most `lanes` otherwise. Row `k` reads that many
  // bytes from offset `k·lanes` of `chunk`, so the `rows = chunk.len() / lanes` rows stay in
  // `chunk`, which is borrowed for the whole block. Unit-stride loads accept any alignment.
  // The block reads no other memory, writes none, and uses no stack. It clobbers only the
  // vector registers listed below and the vector length and type.
  unsafe {
    asm!(
      "vsetvli zero, {lanes}, e32, m4, ta, ma",
      // v8: column sums. v12: running totals of the column sums.
      "vmv.v.i v8, 0",
      "vmv.v.i v12, 0",
      "beqz {rows}, 3f",
      "2:",
      "vle8.v v0, ({ptr})",
      "vzext.vf4 v4, v0",
      "vadd.vv v8, v8, v4",
      "vadd.vv v12, v12, v8",
      "add {ptr}, {ptr}, {lanes}",
      "addi {rows}, {rows}, -1",
      "bnez {rows}, 2b",
      "3:",
      // v4: lane index times column sum.
      "vid.v v4",
      "vmul.vv v4, v4, v8",
      "vsetivli zero, 1, e64, m1, ta, ma",
      "vmv.s.x v16, zero",
      "vsetvli zero, {lanes}, e32, m4, ta, ma",
      "vwredsumu.vs v17, v8, v16",
      "vwredsumu.vs v18, v12, v16",
      "vwredsumu.vs v19, v4, v16",
      "vsetivli zero, 1, e64, m1, ta, ma",
      "vmv.x.s {sum}, v17",
      "vmv.x.s {running}, v18",
      "vmv.x.s {offset}, v19",
      lanes = in(reg) lanes,
      ptr = inout(reg) chunk.as_ptr() => _,
      rows = inout(reg) rows => _,
      sum = lateout(reg) sum,
      running = lateout(reg) running,
      offset = lateout(reg) offset,
      out("v0") _,
      out("v4") _,
      out("v5") _,
      out("v6") _,
      out("v7") _,
      out("v8") _,
      out("v9") _,
      out("v10") _,
      out("v11") _,
      out("v12") _,
      out("v13") _,
      out("v14") _,
      out("v15") _,
      out("v16") _,
      out("v17") _,
      out("v18") _,
      out("v19") _,
      options(readonly, nostack),
    );
  }
  (sum, running, offset)
}

/// Add `data` to the Adler-32 value `adler`, whose two sums are reduced.
///
/// For `k` rows of `L` lanes, `a` gains the byte sum and `b` gains `k·L·a`, plus `L` times
/// the total of the running column sums, minus each column sum times its lane index.
///
/// # Safety
///
/// The CPU must support the V extension.
#[target_feature(enable = "v")]
pub(in crate::checksum) fn update(adler: u32, data: &[u8]) -> u32 {
  // Under `ROW_SUMS_FIT`, no lane of a chunk of at most `ROWS` rows wraps.
  let () = ROW_SUMS_FIT;
  let lanes = lanes();
  let (rows, tail) = data.split_at(data.len().strict_sub(data.len().strict_rem(lanes)));
  let mut a = adler & 0xffff;
  let mut b = adler >> 16;
  for chunk in rows.chunks(lanes.strict_mul(ROWS)) {
    let (sum, running, offset) = chunk_sums(chunk, lanes);
    // A running total is at least its column sum, and a lane index is below `lanes`, so the
    // subtraction does not wrap.
    let weighted = u64::from(b)
      .strict_add((chunk.len() as u64).strict_mul(u64::from(a)))
      .strict_add((lanes as u64).strict_mul(running))
      .strict_sub(offset);
    a = reduce(u64::from(a).strict_add(sum));
    b = reduce(weighted);
  }
  portable::update((b << 16) | a, tail)
}
