//! Software prefetch helpers for hash algorithm kernels.
//!
//! Provides platform-specific `prefetch_read_l1` for optimal memory access
//! patterns in multi-chunk hash computation (e.g., Blake3 NEON hot loops).
//!
//! Prefetch instructions are CPU hints — invalid addresses are silently ignored.

// SAFETY: This module provides low-level prefetch intrinsics that require unsafe.
// Prefetch instructions are hints to the CPU and cannot cause memory unsafety;
// invalid addresses are silently ignored.
#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

/// Prefetch data for read into L1 cache (PLDL1KEEP).
///
/// Uses the ARM PRFM instruction with PLDL1KEEP hint.
///
/// # Safety
///
/// The pointer does not need to be valid or aligned. Prefetch is a hint;
/// invalid addresses are silently ignored by the CPU.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn prefetch_read_l1(ptr: *const u8) {
  core::arch::asm!(
    "prfm pldl1keep, [{ptr}]",
    ptr = in(reg) ptr,
    options(nostack, preserves_flags)
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
  use super::*;

  #[test]
  #[cfg_attr(miri, ignore)] // Miri does not support inline assembly (aarch64 PRFM)
  fn prefetch_does_not_crash_on_null() {
    // SAFETY: prefetch is a CPU hint — any pointer value is safe.
    unsafe {
      prefetch_read_l1(core::ptr::null());
    }
  }

  #[test]
  #[cfg_attr(miri, ignore)] // Miri does not support inline assembly (aarch64 PRFM)
  fn prefetch_does_not_crash_on_unaligned() {
    let data = [0u8; 256];
    // SAFETY: prefetch is a CPU hint — any pointer value is safe.
    unsafe {
      prefetch_read_l1(data.as_ptr().add(1));
      prefetch_read_l1(data.as_ptr().add(7));
      prefetch_read_l1(data.as_ptr().add(63));
    }
  }

  #[test]
  #[cfg_attr(miri, ignore)] // Miri does not support inline assembly (aarch64 PRFM)
  fn prefetch_does_not_crash_on_out_of_bounds() {
    let data = [0u8; 64];
    // SAFETY: prefetch is a CPU hint — invalid addresses are silently ignored.
    unsafe {
      prefetch_read_l1(data.as_ptr().wrapping_add(8192));
    }
  }
}
