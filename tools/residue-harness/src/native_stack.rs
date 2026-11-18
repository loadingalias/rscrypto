//! A private, painted stack for native residue measurements.
//!
//! `measure` runs one non-unwinding callback on a fresh 1 MiB stack between
//! inaccessible guard ranges and returns a copy of that stack after the
//! callback returns. Nothing writes the private stack after the callback.

use core::{ffi::c_void, hint::black_box};

pub(crate) const STACK_BYTES: usize = 1024 * 1024;
pub(crate) const PAINT: u8 = 0xa5;

// The wrapper preserves the platform's callee-saved registers on the original
// stack, switches to the aligned private stack for one non-unwinding call,
// then restores the original stack. The callback receives only its context.
// Nothing writes the private stack after the callback returns.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
core::arch::global_asm!(
  ".text",
  ".p2align 2",
  ".global _rscrypto_residue_call_on_stack",
  "_rscrypto_residue_call_on_stack:",
  "stp x29, x30, [sp, #-32]!",
  "stp x19, x20, [sp, #16]",
  "mov x19, sp",
  "mov x20, x0",
  "mov sp, x1",
  "mov x0, x2",
  "blr x20",
  "mov sp, x19",
  "ldp x19, x20, [sp, #16]",
  "ldp x29, x30, [sp], #32",
  "ret",
);

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
core::arch::global_asm!(
  ".text",
  ".p2align 2",
  ".global rscrypto_residue_call_on_stack",
  ".type rscrypto_residue_call_on_stack, %function",
  "rscrypto_residue_call_on_stack:",
  "stp x29, x30, [sp, #-32]!",
  "stp x19, x20, [sp, #16]",
  "mov x19, sp",
  "mov x20, x0",
  "mov sp, x1",
  "mov x0, x2",
  "blr x20",
  "mov sp, x19",
  "ldp x19, x20, [sp, #16]",
  "ldp x29, x30, [sp], #32",
  "ret",
  ".size rscrypto_residue_call_on_stack, .-rscrypto_residue_call_on_stack",
);

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
core::arch::global_asm!(
  ".text",
  ".p2align 4",
  ".global rscrypto_residue_call_on_stack",
  ".type rscrypto_residue_call_on_stack, @function",
  "rscrypto_residue_call_on_stack:",
  "push r12",
  "mov r12, rsp",
  "mov rax, rdi",
  "mov rsp, rsi",
  "mov rdi, rdx",
  "call rax",
  "mov rsp, r12",
  "pop r12",
  "ret",
  ".size rscrypto_residue_call_on_stack, .-rscrypto_residue_call_on_stack",
);

unsafe extern "C" {
  fn rscrypto_residue_call_on_stack(callback: extern "C" fn(*mut c_void), top: *mut u8, context: *mut c_void);
  fn mmap(address: *mut c_void, length: usize, protection: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
  fn mprotect(address: *mut c_void, length: usize, protection: i32) -> i32;
  fn munmap(address: *mut c_void, length: usize) -> i32;
  fn getpagesize() -> i32;
}

/// A 1 MiB writable stack between two 1 MiB inaccessible guard ranges.
/// The guards stop excessive frames before they can overwrite an adjacent
/// allocation. Production callbacks have no dynamic stack allocation.
struct GuardedStack {
  allocation: *mut c_void,
  middle: *mut u8,
}

impl GuardedStack {
  fn new() -> Self {
    // SAFETY: getpagesize has no arguments or memory effects visible to Rust.
    let page = usize::try_from(unsafe { getpagesize() }).expect("positive page size");
    assert!(
      page != 0 && STACK_BYTES.is_multiple_of(page),
      "stack and guards must be page aligned"
    );
    #[cfg(target_os = "macos")]
    const ANONYMOUS: i32 = 0x1000;
    #[cfg(target_os = "linux")]
    const ANONYMOUS: i32 = 0x20;
    // SAFETY: null requests a fresh mapping; MAP_PRIVATE|MAP_ANONYMOUS, fd=-1,
    // offset=0 reserves three aligned MiB with PROT_NONE. These constants and
    // signatures come from the macOS and Linux sys/mman.h platform ABIs.
    let allocation = unsafe {
      mmap(
        core::ptr::null_mut(),
        STACK_BYTES.strict_mul(3),
        0,
        2 | ANONYMOUS,
        -1,
        0,
      )
    };
    assert!(
      allocation.addr() != usize::MAX,
      "mmap failed: {}",
      std::io::Error::last_os_error()
    );
    // SAFETY: the middle MiB lies inside the successful three-MiB mapping.
    let middle = unsafe { allocation.cast::<u8>().add(STACK_BYTES) };
    // SAFETY: `middle` and STACK_BYTES are page aligned and wholly inside the
    // mapping. PROT_READ|PROT_WRITE changes only the active stack's protection.
    let result = unsafe { mprotect(middle.cast(), STACK_BYTES, 3) };
    assert_eq!(result, 0, "mprotect failed: {}", std::io::Error::last_os_error());
    Self { allocation, middle }
  }
}

impl Drop for GuardedStack {
  fn drop(&mut self) {
    // SAFETY: this owner uniquely retains the original mapping. All callbacks
    // have returned and all byte views ended before the owner is dropped.
    let result = unsafe { munmap(self.allocation, STACK_BYTES.strict_mul(3)) };
    assert_eq!(result, 0, "munmap failed");
  }
}

/// Test inputs are synthetic and public. This generator is not cryptography.
pub(crate) fn input(tag: u64, index: u32) -> [u8; 32] {
  let mut state = tag.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(u64::from(index));
  let mut out = [0u8; 32];
  for chunk in out.chunks_mut(8) {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    chunk.copy_from_slice(&(z ^ (z >> 31)).to_le_bytes());
  }
  out
}

pub(crate) fn hex(bytes: &[u8]) -> String {
  use core::fmt::Write;
  let mut output = String::with_capacity(bytes.len().strict_mul(2));
  for byte in bytes {
    write!(output, "{byte:02x}").expect("String formatting");
  }
  output
}

extern "C" fn invoke<F: FnMut()>(context: *mut c_void) {
  // SAFETY: `measure` supplies a unique live F for the synchronous callback.
  let function = unsafe { &mut *context.cast::<F>() };
  function();
}

/// Run `function` on a fresh painted private stack and return that stack.
pub(crate) fn measure<F: FnMut()>(mut function: F) -> Vec<u8> {
  let stack = GuardedStack::new();
  // SAFETY: the active mapping is writable, unique, and STACK_BYTES long.
  // This view ends before the callback takes over the stack allocation.
  unsafe { core::slice::from_raw_parts_mut(stack.middle, STACK_BYTES) }.fill(PAINT);
  // SAFETY: one-past the active mapping. The stack grows down from this
  // page-aligned address; the next MiB above it is also a guard range.
  let top = unsafe { stack.middle.add(STACK_BYTES) };
  // SAFETY: [top - STACK_BYTES, top) is initialized, writable, unique storage.
  // The stack grows down and is at least 16-byte aligned. Inaccessible guards
  // bound both ends; the lower painted sentinel also detects near-exhaustion.
  // `function` is live on the original stack throughout this synchronous call.
  // Both ABIs are non-unwinding and this package uses panic=abort. No references
  // into `stack` are live while the callback executes. The wrapper restores all
  // callee-saved registers and the original SP before returning to Rust.
  unsafe {
    rscrypto_residue_call_on_stack(invoke::<F>, top, (&raw mut function).cast());
  }
  // SAFETY: the callback returned before this shared view is created. The
  // allocation was painted before use and remains mapped until copying ends.
  let bytes = unsafe { core::slice::from_raw_parts(stack.middle, STACK_BYTES) };
  assert!(
    bytes[..4096].iter().all(|&byte| byte == PAINT),
    "private stack exhausted"
  );
  bytes.to_vec()
}

/// Copy `needle` into a returned frame, and clear that copy first if `clear`.
#[inline(never)]
pub(crate) fn plant(needle: &[u8], clear: bool) {
  assert!(needle.len() <= 8192, "control fixture exceeds local buffer");
  let mut copy = [0u8; 8192];
  copy[..needle.len()].copy_from_slice(black_box(needle));
  black_box(&copy);
  if clear {
    for byte in &mut copy {
      // SAFETY: each byte is an exclusive initialized element of local copy.
      unsafe { core::ptr::write_volatile(byte, 0) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  }
}
