//! Portable slice-by-N CRC, generic over the register word and the bit order.
//!
//! Slice-by-N reads N bytes per step. The register is XORed into the first bytes of the step, and
//! each byte selects the table entry for its distance from the end of the step. The step body is
//! the same for both bit orders: only the order of the register's bytes differs.
//!
//! The kernels index input bytes directly. On cores without fast unaligned loads, such as many
//! RISC-V cores, this avoids assembling words from bytes and splitting them again.

use super::{Crc, SLICES, Word};

/// Byte `i` of the register in input order: the low byte first when reflected, the top byte first
/// when MSB-first.
#[inline(always)]
fn register_byte<W: Word>(reflected: bool, register: u64, i: usize) -> u8 {
  let position = if reflected {
    i
  } else {
    W::BYTES.strict_sub(1).strict_sub(i)
  };
  (register >> position.strict_mul(8)).to_le_bytes()[0]
}

/// One slice-by-`K` step over `block`.
#[inline(always)]
fn step<W: Word, const K: usize>(crc: &Crc<W>, register: W, block: &[u8; K]) -> W {
  let reflected = crc.spec.reflected;
  let wide = register.widen();
  let mut next = W::narrow(0);
  let mut i = 0usize;
  while i < K {
    let mut byte = block[i];
    if i < W::BYTES {
      byte ^= register_byte::<W>(reflected, wide, i);
    }
    next = next ^ crc.table[K.strict_sub(1).strict_sub(i)][usize::from(byte)];
    i = i.strict_add(1);
  }
  next
}

/// One byte at a time.
#[inline(always)]
pub(in crate::checksum) fn bytewise<W: Word>(crc: &Crc<W>, mut register: W, data: &[u8]) -> W {
  let reflected = crc.spec.reflected;
  for &byte in data {
    let wide = register.widen();
    register = if reflected {
      let index = usize::from(wide.to_le_bytes()[0] ^ byte);
      crc.table[0][index] ^ register.shift_down()
    } else {
      let index = usize::from((wide >> (W::BITS.strict_sub(8))).to_le_bytes()[0] ^ byte);
      crc.table[0][index] ^ register.shift_up()
    };
  }
  register
}

/// Updates a word-domain register with `data`.
#[inline]
pub(in crate::checksum) fn update<W: Word>(crc: &Crc<W>, mut register: W, data: &[u8]) -> W {
  let (blocks, mut rest) = data.as_chunks::<SLICES>();
  for block in blocks {
    register = step(crc, register, block);
  }
  if let Some((half, tail)) = rest.split_first_chunk::<8>() {
    register = step(crc, register, half);
    rest = tail;
  }
  bytewise(crc, register, rest)
}

/// The register for eight input bytes from a zero register: `bytes * x^width mod G`.
#[inline]
pub(in crate::checksum) fn reduce8<W: Word>(crc: &Crc<W>, bytes: &[u8; 8]) -> W {
  step(crc, W::narrow(0), bytes)
}
