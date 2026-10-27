//! Copy the prepared key's rho, private K, and tr without retaining copy scratch.

/// Keep the copy and its register cleanup below the returning preparation frame.
///
/// The native blocks expose only addresses to LLVM. Their final instructions
/// clear every register that loads the seed fields. This closes this named copy
/// boundary; it does not clear registers used by polynomial preparation or callers.
#[inline(never)]
pub(super) fn copy(output: &mut [u8; 128], input: &[u8; 128]) {
  #[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    any(target_os = "macos", target_os = "linux"),
    not(feature = "portable-only"),
    not(miri)
  ))]
  // SAFETY: the fixed arrays provide 128 initialized readable bytes and 128
  // exclusively writable bytes in disjoint regions. Each unaligned-capable
  // load/store pair stays within those arrays. NEON is an ABI feature on these
  // targets and is required by cfg. Only caller-saved v0/v1 carry copied bytes;
  // both are cleared before return. There is no implicit stack scratch or flags change.
  // Default asm memory effects keep the stores ordered with respect to callers.
  unsafe {
    core::arch::asm!(
      "ldp q0, q1, [{input}, #0]",
      "stp q0, q1, [{output}, #0]",
      "ldp q0, q1, [{input}, #32]",
      "stp q0, q1, [{output}, #32]",
      "ldp q0, q1, [{input}, #64]",
      "stp q0, q1, [{output}, #64]",
      "ldp q0, q1, [{input}, #96]",
      "stp q0, q1, [{output}, #96]",
      "eor v0.16b, v0.16b, v0.16b",
      "eor v1.16b, v1.16b, v1.16b",
      input = in(reg) input.as_ptr(),
      output = in(reg) output.as_mut_ptr(),
      out("v0") _,
      out("v1") _,
      options(nostack, preserves_flags),
    );
  }

  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    any(target_os = "macos", target_os = "linux"),
    not(feature = "portable-only"),
    not(miri)
  ))]
  // SAFETY: the distinct fixed-array borrows establish initialized input and
  // exclusive output for all eight 16-byte transfers. MOVDQU permits unaligned
  // addresses, and cfg establishes SSE2. Only caller-saved xmm0/xmm1 carry the
  // copied bytes; PXOR clears their full 128-bit payloads before return. Upper
  // vector lanes are never loaded by this block. There is no implicit stack scratch or flags use.
  unsafe {
    core::arch::asm!(
      "movdqu xmm0, [{input} + 0]",
      "movdqu xmm1, [{input} + 16]",
      "movdqu [{output} + 0], xmm0",
      "movdqu [{output} + 16], xmm1",
      "movdqu xmm0, [{input} + 32]",
      "movdqu xmm1, [{input} + 48]",
      "movdqu [{output} + 32], xmm0",
      "movdqu [{output} + 48], xmm1",
      "movdqu xmm0, [{input} + 64]",
      "movdqu xmm1, [{input} + 80]",
      "movdqu [{output} + 64], xmm0",
      "movdqu [{output} + 80], xmm1",
      "movdqu xmm0, [{input} + 96]",
      "movdqu xmm1, [{input} + 112]",
      "movdqu [{output} + 96], xmm0",
      "movdqu [{output} + 112], xmm1",
      "pxor xmm0, xmm0",
      "pxor xmm1, xmm1",
      input = in(reg) input.as_ptr(),
      output = in(reg) output.as_mut_ptr(),
      out("xmm0") _,
      out("xmm1") _,
      options(nostack, preserves_flags),
    );
  }

  #[cfg(not(any(
    all(
      target_arch = "aarch64",
      target_feature = "neon",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only"),
      not(miri)
    ),
    all(
      target_arch = "x86_64",
      target_feature = "sse2",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only"),
      not(miri)
    ),
  )))]
  output.copy_from_slice(input);
}
