//! Original scalar Montgomery kernels for the RISC-V M baseline.
//!
//! The parent module compiles this backend only with M enabled. These kernels
//! share the scalar transform schedule and canonical representation with Rust.
//! They require no vector extension, OS state, or additional scratch storage.

use super::{NEG_Q_INVERSE, Q};

/// Accept operands below 2q and return a canonical Montgomery product.
#[cfg(target_arch = "riscv64")]
#[inline]
#[expect(clippy::cast_possible_truncation, reason = "the final residue is below q")]
pub(super) fn multiply(a: u32, b: u32) -> u32 {
  let out: u64;
  // SAFETY: The module's cfg establishes RV64 with M. Inputs are zero-extended
  // u32 values below 2q. MUL forms t below 2^48. The two shifts retain m modulo
  // 2^32, and t + mq fits below 2^56. After division by 2^32, subtracting q
  // lies in (-q, q), so SRAI/AND restore q exactly on borrow. Early scratch
  // outputs preserve every live input. All operations use registers, without
  // memory, stack, condition flags, or secret-dependent control flow.
  unsafe {
    core::arch::asm!(
      "mul {t}, {a}, {b}",
      "mul {m}, {t}, {inverse}",
      "slli {m}, {m}, 32",
      "srli {m}, {m}, 32",
      "mul {m}, {m}, {q}",
      "add {t}, {t}, {m}",
      "srli {t}, {t}, 32",
      "sub {t}, {t}, {q}",
      "srai {m}, {t}, 63",
      "and {m}, {m}, {q}",
      "add {out}, {t}, {m}",
      a = in(reg) u64::from(a),
      b = in(reg) u64::from(b),
      q = in(reg) u64::from(Q),
      inverse = in(reg) u64::from(NEG_Q_INVERSE),
      t = out(reg) _,
      m = out(reg) _,
      out = lateout(reg) out,
      options(nomem, nostack, pure, preserves_flags),
    );
  }
  out as u32
}

/// Accept operands below 2q and return a canonical Montgomery product.
#[cfg(target_arch = "riscv32")]
#[inline]
pub(super) fn multiply(a: u32, b: u32) -> u32 {
  let out: u32;
  // SAFETY: The module's cfg establishes RV32 with M. MUL/MULHU split a*b.
  // m = low(a*b)*(-q^-1) mod 2^32 makes low(a*b + mq) zero; its carry is one
  // precisely when low(a*b) is nonzero. Adding both high halves and that carry
  // yields the same quotient as the Rust u64 arithmetic, below 2q. Signed
  // subtraction and SRAI/AND canonicalize it. Early outputs preserve live
  // inputs. No memory, stack, flags, or data-dependent branch is used.
  unsafe {
    core::arch::asm!(
      "mul {lo}, {a}, {b}",
      "mulhu {hi}, {a}, {b}",
      "mul {m}, {lo}, {inverse}",
      "mulhu {m}, {m}, {q}",
      "sltu {lo}, zero, {lo}",
      "add {hi}, {hi}, {m}",
      "add {hi}, {hi}, {lo}",
      "sub {hi}, {hi}, {q}",
      "srai {m}, {hi}, 31",
      "and {m}, {m}, {q}",
      "add {out}, {hi}, {m}",
      a = in(reg) a,
      b = in(reg) b,
      q = in(reg) Q,
      inverse = in(reg) NEG_Q_INVERSE,
      lo = out(reg) _,
      hi = out(reg) _,
      m = out(reg) _,
      out = lateout(reg) out,
      options(nomem, nostack, pure, preserves_flags),
    );
  }
  out
}
