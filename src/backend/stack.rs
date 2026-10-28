//! Dead-stack scrub for secret workers.
//!
//! A secret worker runs out of line, and its caller then calls a scrub from
//! the same frame. The scrub's buffer occupies the
//! stack the worker just released, so it overwrites register spills and other
//! compiler-created copies that no named owner can clear. `just stack-frames`
//! checks on each reviewed target that every registered worker's linked depth
//! fits within its scrub.

/// Overwrite `WORDS` eight-byte words of dead stack below the caller's frame.
#[cfg(any(feature = "ml-kem", feature = "ml-dsa"))]
#[inline(never)]
pub(crate) fn scrub_dead_stack<const WORDS: usize>() {
  const { assert!(WORDS.is_multiple_of(4)) };
  // Leave the buffer uninitialized so only the volatile stores write it.
  // Volatile stores cannot be elided, and no later access depends on their
  // order, so no fence is needed.
  let mut scratch = core::mem::MaybeUninit::<[u64; WORDS]>::uninit();
  let words = scratch.as_mut_ptr().cast::<u64>();
  // Four stores per iteration let cores with two store ports retire the
  // scrub faster than a one-store loop.
  for index in (0..WORDS).step_by(4) {
    // SAFETY: the word count is a multiple of four, so `index + 3` is below
    // the array length and every pointer stays inside this local, properly
    // aligned `[u64; N]` allocation. Writing a `u64` needs no prior
    // initialization, and the buffer is never read.
    unsafe {
      words.add(index).write_volatile(0);
      words.add(index.strict_add(1)).write_volatile(0);
      words.add(index.strict_add(2)).write_volatile(0);
      words.add(index.strict_add(3)).write_volatile(0);
    }
  }
}

/// [`scrub_dead_stack`] with 64-byte stores, for x86-64 workers that run only
/// with AVX-512.
///
/// The buffer keeps eight-byte alignment, so the frame needs no realignment
/// and the buffer starts where an eight-byte buffer would. Words before its
/// first and after its last 64-byte boundary take eight-byte stores.
///
/// # Safety
///
/// The caller must ensure that AVX512F is available.
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  any(
    feature = "aes-gcm",
    feature = "chacha20poly1305",
    feature = "xchacha20poly1305",
    feature = "chacha20-rng"
  )
))]
#[inline(never)]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn scrub_dead_stack_avx512<const WORDS: usize>() {
  use core::arch::x86_64::{__m512i, _mm512_setzero_si512};

  let mut scratch = core::mem::MaybeUninit::<[u64; WORDS]>::uninit();
  let words = scratch.as_mut_ptr().cast::<u64>();
  // `align_offset` may report that it cannot align; all words then take
  // eight-byte stores.
  let head = words.align_offset(64).min(WORDS);
  let registers = WORDS.strict_sub(head).strict_div(8);
  let middle = head.strict_add(registers.strict_mul(8));
  // SAFETY: every index is below `WORDS`, so each pointer stays inside this
  // local `[u64; WORDS]` allocation. `words + head` is 64-byte aligned when
  // `registers > 0`, and each register store covers eight words before
  // `middle`. Writing needs no prior initialization, and the buffer is never
  // read. Volatile stores cannot be elided.
  unsafe {
    for index in 0..head {
      words.add(index).write_volatile(0);
    }
    #[expect(clippy::cast_ptr_alignment, reason = "`head` aligns this pointer to 64 bytes")]
    let lanes = words.add(head).cast::<__m512i>();
    for index in 0..registers {
      lanes.add(index).write_volatile(_mm512_setzero_si512());
    }
    for index in middle..WORDS {
      words.add(index).write_volatile(0);
    }
  }
}
