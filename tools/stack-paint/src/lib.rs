//! Painted, guarded private stacks for native stack measurements.
//!
//! [`GuardedStack`] maps a private stack between two inaccessible 1 MiB guard
//! ranges. Each measured call first fills the stack with a paint byte, then
//! runs one non-unwinding call with the stack pointer at the stack's top. After
//! the call returns, the lowest byte that no longer holds the paint gives the
//! call's high-water mark. Only the caller's thread uses the stack, and nothing
//! writes it between the callee's return and the scan.
//!
//! A written byte that happens to equal the paint is not seen, so a result can
//! fall short of the true mark by the bytes of the deepest store that match the
//! paint. Callers report results rounded up to 16 bytes and can repeat a
//! measurement with a second paint.

#[cfg(not(target_pointer_width = "64"))]
compile_error!("painted stacks support 64-bit targets only");

#[cfg(not(any(
  all(target_arch = "x86_64", any(target_os = "linux", windows)),
  all(target_arch = "aarch64", any(target_os = "linux", target_os = "macos")),
  all(target_arch = "powerpc64", target_endian = "little", target_os = "linux"),
  all(target_arch = "s390x", target_os = "linux"),
  all(target_arch = "riscv64", target_os = "linux"),
)))]
compile_error!("no painted-stack call wrapper for this target");

use core::ffi::c_void;

/// Bytes between the private stack's top and the stack pointer at the
/// measured function's entry: the slots the wrapper's call supplies.
///
/// x86-64 pushes the return address, and Windows also reserves 32 bytes of
/// shadow space above it. POWER ELFv2 callees store the link register and TOC
/// in a 32-byte header that the caller supplies. IBM Z callees save registers
/// into a 160-byte area that the caller supplies. AArch64 and RISC-V supply none.
pub const ENTRY_OFFSET: usize = if cfg!(all(target_arch = "x86_64", windows)) {
  40
} else if cfg!(target_arch = "x86_64") {
  8
} else if cfg!(target_arch = "powerpc64") {
  32
} else if cfg!(target_arch = "s390x") {
  160
} else {
  0
};

/// Bytes below the stack pointer that a function may use without moving it.
pub const RED_ZONE: usize = if cfg!(any(
  all(target_arch = "x86_64", not(windows)),
  all(target_arch = "aarch64", target_os = "macos")
)) {
  128
} else if cfg!(target_arch = "powerpc64") {
  288
} else {
  0
};

/// The lowest bytes of every stack, which must keep their paint. A changed
/// sentinel means that the call came within this distance of the guard.
pub const SENTINEL_BYTES: usize = 4096;

const GUARD_BYTES: usize = 1024 * 1024;

/// One call for the platform wrapper; the field offsets are part of its ABI.
#[repr(C)]
struct Call {
  function: *const (),
  arguments: [usize; 3],
  result: usize,
  top: *mut u8,
  limit: *mut u8,
  base: *mut u8,
}

const _: () = {
  assert!(core::mem::offset_of!(Call, arguments) == 8);
  assert!(core::mem::offset_of!(Call, result) == 32);
  assert!(core::mem::offset_of!(Call, top) == 40);
  assert!(core::mem::offset_of!(Call, limit) == 48);
  assert!(core::mem::offset_of!(Call, base) == 56);
};

// Each wrapper preserves the registers it uses on the original stack, moves the
// stack pointer to `top` less the ABI's entry slots, calls `function` with the
// three integer arguments, stores the integer result, and restores the original
// stack pointer. It writes the private stack only through the call itself.

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
core::arch::global_asm!(
  ".text",
  ".p2align 4",
  ".globl rscrypto_stack_paint_call",
  ".type rscrypto_stack_paint_call, @function",
  "rscrypto_stack_paint_call:",
  "push rbx",
  "push r12",
  "mov rbx, rdi",
  "mov r12, rsp",
  "mov rsp, qword ptr [rbx + 40]",
  "mov rdi, qword ptr [rbx + 8]",
  "mov rsi, qword ptr [rbx + 16]",
  "mov rdx, qword ptr [rbx + 24]",
  "call qword ptr [rbx]",
  "mov qword ptr [rbx + 32], rax",
  "mov rsp, r12",
  "pop r12",
  "pop rbx",
  "ret",
  ".size rscrypto_stack_paint_call, .-rscrypto_stack_paint_call",
);

// Windows also moves the thread's stack bounds in the TEB (`StackBase` at
// gs:0x08, `StackLimit` at gs:0x10, `DeallocationStack` at gs:0x1478) to the
// private stack for the call, so that `__chkstk` and exception dispatch see the
// stack in use, and restores them afterward.
#[cfg(all(target_arch = "x86_64", windows))]
core::arch::global_asm!(
  ".text",
  ".p2align 4",
  ".globl rscrypto_stack_paint_call",
  "rscrypto_stack_paint_call:",
  "push rbx",
  "push rsi",
  "push rdi",
  "push r12",
  "push r13",
  "mov rbx, rcx",
  "mov r12, rsp",
  "mov rsi, qword ptr gs:[0x08]",
  "mov rdi, qword ptr gs:[0x10]",
  "mov r13, qword ptr gs:[0x1478]",
  "mov rax, qword ptr [rbx + 40]",
  "mov qword ptr gs:[0x08], rax",
  "mov rax, qword ptr [rbx + 48]",
  "mov qword ptr gs:[0x10], rax",
  "mov rax, qword ptr [rbx + 56]",
  "mov qword ptr gs:[0x1478], rax",
  "mov rsp, qword ptr [rbx + 40]",
  "sub rsp, 32",
  "mov rcx, qword ptr [rbx + 8]",
  "mov rdx, qword ptr [rbx + 16]",
  "mov r8, qword ptr [rbx + 24]",
  "call qword ptr [rbx]",
  "mov qword ptr [rbx + 32], rax",
  "mov rsp, r12",
  "mov qword ptr gs:[0x08], rsi",
  "mov qword ptr gs:[0x10], rdi",
  "mov qword ptr gs:[0x1478], r13",
  "pop r13",
  "pop r12",
  "pop rdi",
  "pop rsi",
  "pop rbx",
  "ret",
);

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
core::arch::global_asm!(
  ".text",
  ".p2align 2",
  ".global rscrypto_stack_paint_call",
  ".type rscrypto_stack_paint_call, %function",
  "rscrypto_stack_paint_call:",
  "stp x29, x30, [sp, #-32]!",
  "stp x19, x20, [sp, #16]",
  "mov x29, sp",
  "mov x19, x0",
  "mov x20, sp",
  "ldr x9, [x19, #40]",
  "mov sp, x9",
  "ldr x0, [x19, #8]",
  "ldr x1, [x19, #16]",
  "ldr x2, [x19, #24]",
  "ldr x9, [x19]",
  "blr x9",
  "str x0, [x19, #32]",
  "mov sp, x20",
  "ldp x19, x20, [sp, #16]",
  "ldp x29, x30, [sp], #32",
  "ret",
  ".size rscrypto_stack_paint_call, .-rscrypto_stack_paint_call",
);

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
core::arch::global_asm!(
  ".text",
  ".p2align 2",
  ".global _rscrypto_stack_paint_call",
  "_rscrypto_stack_paint_call:",
  "stp x29, x30, [sp, #-32]!",
  "stp x19, x20, [sp, #16]",
  "mov x29, sp",
  "mov x19, x0",
  "mov x20, sp",
  "ldr x9, [x19, #40]",
  "mov sp, x9",
  "ldr x0, [x19, #8]",
  "ldr x1, [x19, #16]",
  "ldr x2, [x19, #24]",
  "ldr x9, [x19]",
  "blr x9",
  "str x0, [x19, #32]",
  "mov sp, x20",
  "ldp x19, x20, [sp, #16]",
  "ldp x29, x30, [sp], #32",
  "ret",
);

// ELFv2: the wrapper saves its caller's TOC in the caller's TOC slot and
// restores it after the call; the callee's global entry derives its own TOC
// from r12. The private stack's 32-byte header carries a back chain to the
// wrapper's frame.
#[cfg(all(target_arch = "powerpc64", target_endian = "little", target_os = "linux"))]
core::arch::global_asm!(
  ".text",
  ".p2align 4",
  ".globl rscrypto_stack_paint_call",
  ".type rscrypto_stack_paint_call, @function",
  "rscrypto_stack_paint_call:",
  "mflr 0",
  "std 0, 16(1)",
  "std 2, 24(1)",
  "stdu 1, -48(1)",
  "std 30, 32(1)",
  "std 31, 40(1)",
  "mr 31, 3",
  "mr 30, 1",
  "ld 12, 0(31)",
  "ld 4, 40(31)",
  "addi 1, 4, -32",
  "std 30, 0(1)",
  "ld 3, 8(31)",
  "ld 4, 16(31)",
  "ld 5, 24(31)",
  "mtctr 12",
  "bctrl",
  "std 3, 32(31)",
  "mr 1, 30",
  "ld 30, 32(1)",
  "ld 31, 40(1)",
  "addi 1, 1, 48",
  "ld 2, 24(1)",
  "ld 0, 16(1)",
  "mtlr 0",
  "blr",
  ".size rscrypto_stack_paint_call, .-rscrypto_stack_paint_call",
);

// The caller of an IBM Z function supplies its 160-byte register save area.
#[cfg(all(target_arch = "s390x", target_os = "linux"))]
core::arch::global_asm!(
  ".text",
  ".p2align 4",
  ".globl rscrypto_stack_paint_call",
  ".type rscrypto_stack_paint_call, @function",
  "rscrypto_stack_paint_call:",
  "stmg %r6, %r15, 48(%r15)",
  "lgr %r6, %r2",
  "lgr %r7, %r15",
  "lg %r15, 40(%r6)",
  "aghi %r15, -160",
  "lg %r1, 0(%r6)",
  "lg %r2, 8(%r6)",
  "lg %r3, 16(%r6)",
  "lg %r4, 24(%r6)",
  "basr %r14, %r1",
  "stg %r2, 32(%r6)",
  "lgr %r15, %r7",
  "lmg %r6, %r15, 48(%r15)",
  "br %r14",
  ".size rscrypto_stack_paint_call, .-rscrypto_stack_paint_call",
);

#[cfg(all(target_arch = "riscv64", target_os = "linux"))]
core::arch::global_asm!(
  ".text",
  ".p2align 2",
  ".globl rscrypto_stack_paint_call",
  ".type rscrypto_stack_paint_call, @function",
  "rscrypto_stack_paint_call:",
  "addi sp, sp, -32",
  "sd ra, 24(sp)",
  "sd s0, 16(sp)",
  "sd s1, 8(sp)",
  "mv s0, a0",
  "mv s1, sp",
  "ld sp, 40(s0)",
  "ld t0, 0(s0)",
  "ld a0, 8(s0)",
  "ld a1, 16(s0)",
  "ld a2, 24(s0)",
  "jalr t0",
  "sd a0, 32(s0)",
  "mv sp, s1",
  "ld ra, 24(sp)",
  "ld s0, 16(sp)",
  "ld s1, 8(sp)",
  "addi sp, sp, 32",
  "ret",
  ".size rscrypto_stack_paint_call, .-rscrypto_stack_paint_call",
);

unsafe extern "C" {
  fn rscrypto_stack_paint_call(call: *mut Call);
}

#[cfg(unix)]
mod os {
  use core::ffi::c_void;

  pub(crate) fn page_size() -> usize {
    // SAFETY: sysconf reads a process constant and has no memory effects.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(page).expect("positive page size")
  }

  /// An inaccessible private anonymous mapping of `bytes`.
  pub(crate) fn reserve(bytes: usize) -> *mut c_void {
    // SAFETY: a null hint requests a fresh private anonymous mapping; fd -1
    // and offset 0 are required for MAP_ANON. No existing memory is affected.
    let allocation = unsafe {
      libc::mmap(
        core::ptr::null_mut(),
        bytes,
        libc::PROT_NONE,
        libc::MAP_PRIVATE | libc::MAP_ANON,
        -1,
        0,
      )
    };
    assert!(
      allocation != libc::MAP_FAILED,
      "mmap failed: {}",
      std::io::Error::last_os_error()
    );
    allocation
  }

  /// Makes `[address, address + bytes)`, inside a reservation, readable and writable.
  pub(crate) fn commit(address: *mut c_void, bytes: usize) {
    // SAFETY: the caller passes a page-aligned range inside its own mapping;
    // changing its protection affects no other memory.
    let result = unsafe { libc::mprotect(address, bytes, libc::PROT_READ | libc::PROT_WRITE) };
    assert_eq!(result, 0, "mprotect failed: {}", std::io::Error::last_os_error());
  }

  pub(crate) fn release(allocation: *mut c_void, bytes: usize) {
    // SAFETY: the owner passes back its whole mapping exactly once, after all
    // calls returned and all views of it ended.
    let result = unsafe { libc::munmap(allocation, bytes) };
    assert_eq!(result, 0, "munmap failed: {}", std::io::Error::last_os_error());
  }
}

#[cfg(windows)]
mod os {
  use core::ffi::c_void;

  const MEM_COMMIT: u32 = 0x1000;
  const MEM_RESERVE: u32 = 0x2000;
  const MEM_RELEASE: u32 = 0x8000;
  const PAGE_NOACCESS: u32 = 0x01;
  const PAGE_READWRITE: u32 = 0x04;

  #[link(name = "kernel32")]
  unsafe extern "system" {
    fn VirtualAlloc(address: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
    fn VirtualFree(address: *mut c_void, size: usize, kind: u32) -> i32;
  }

  pub(crate) fn page_size() -> usize {
    4096
  }

  pub(crate) fn reserve(bytes: usize) -> *mut c_void {
    // SAFETY: a null address reserves a fresh range; no existing memory is affected.
    let allocation = unsafe { VirtualAlloc(core::ptr::null_mut(), bytes, MEM_RESERVE, PAGE_NOACCESS) };
    assert!(
      !allocation.is_null(),
      "VirtualAlloc reserve failed: {}",
      std::io::Error::last_os_error()
    );
    allocation
  }

  pub(crate) fn commit(address: *mut c_void, bytes: usize) {
    // SAFETY: the caller passes a page-aligned range inside its own reservation.
    let result = unsafe { VirtualAlloc(address, bytes, MEM_COMMIT, PAGE_READWRITE) };
    assert_eq!(
      result,
      address,
      "VirtualAlloc commit failed: {}",
      std::io::Error::last_os_error()
    );
  }

  pub(crate) fn release(allocation: *mut c_void, _bytes: usize) {
    // SAFETY: the owner releases its whole reservation exactly once, after all
    // calls returned and all views of it ended.
    let result = unsafe { VirtualFree(allocation, 0, MEM_RELEASE) };
    assert_ne!(result, 0, "VirtualFree failed: {}", std::io::Error::last_os_error());
  }
}

/// A private, writable stack between two inaccessible guard ranges.
#[derive(Debug)]
pub struct GuardedStack {
  allocation: *mut c_void,
  total: usize,
  stack: *mut u8,
  bytes: usize,
}

/// The result of one painted call.
#[derive(Clone, Copy, Debug)]
pub struct Painted {
  /// The callee's integer or pointer result.
  pub result: usize,
  /// Bytes from the stack's top down to the lowest byte that lost its paint.
  pub used_bytes: usize,
}

impl Painted {
  /// Bytes the callee used below its entry stack pointer: [`Painted::used_bytes`]
  /// less [`ENTRY_OFFSET`]. The wrapper's entry slots belong to the caller, so a
  /// callee that changed nothing below its entry pointer used zero bytes.
  #[must_use]
  pub const fn below_entry_bytes(&self) -> usize {
    match self.used_bytes.checked_sub(ENTRY_OFFSET) {
      Some(bytes) => bytes,
      None => 0,
    }
  }
}

impl GuardedStack {
  /// Maps `bytes` of private stack, a multiple of the page size, between guards.
  #[must_use]
  pub fn new(bytes: usize) -> Self {
    let page = os::page_size();
    assert!(
      page != 0 && bytes.is_multiple_of(page) && GUARD_BYTES.is_multiple_of(page),
      "stack and guards must be page aligned"
    );
    assert!(bytes > SENTINEL_BYTES, "the stack must exceed its sentinel");
    let total = bytes.strict_add(GUARD_BYTES.strict_mul(2));
    let allocation = os::reserve(total);
    // SAFETY: the guard offset lies inside the `total`-byte reservation.
    let stack = unsafe { allocation.cast::<u8>().add(GUARD_BYTES) };
    os::commit(stack.cast(), bytes);
    Self {
      allocation,
      total,
      stack,
      bytes,
    }
  }

  /// The usable stack size in bytes.
  #[must_use]
  pub const fn bytes(&self) -> usize {
    self.bytes
  }

  /// Calls the C-ABI `function` with `arguments` on this stack, painted with `paint`.
  ///
  /// # Safety
  ///
  /// `function` must be a C-ABI function of up to three integer or pointer
  /// arguments that returns an integer, a pointer, or nothing. Every argument
  /// must be valid for it. It must not unwind, jump out of the call, retain an
  /// address of the private stack, or need more stack than [`Self::bytes`] less
  /// [`SENTINEL_BYTES`].
  pub unsafe fn call(&mut self, paint: u8, function: *const (), arguments: [usize; 3]) -> Painted {
    // SAFETY: `self.stack` is the start of `self.bytes` committed, writable
    // bytes that only this owner addresses; no view of them is live.
    unsafe { core::ptr::write_bytes(self.stack, paint, self.bytes) };
    // SAFETY: one past the committed range; the upper guard follows it.
    let top = unsafe { self.stack.add(self.bytes) };
    let mut call = Call {
      function,
      arguments,
      result: 0,
      top,
      limit: self.stack,
      base: self.allocation.cast(),
    };
    // SAFETY: `top` is page aligned, so the ABI's 16-byte alignment holds, and
    // the stack below it is committed and unique to this call. The wrapper
    // restores the original stack pointer and every register the ABI preserves.
    // The caller guarantees the callee's contract above; no Rust reference into
    // the stack is live while it runs.
    unsafe { rscrypto_stack_paint_call(&raw mut call) };
    // SAFETY: the call returned; the committed range stays mapped and no
    // mutable view of it exists while this shared view is live.
    let bytes = unsafe { core::slice::from_raw_parts(self.stack, self.bytes) };
    let lowest = bytes.iter().position(|&byte| byte != paint);
    let used_bytes = lowest.map_or(0, |index| {
      assert!(index >= SENTINEL_BYTES, "private stack exhausted");
      self.bytes.strict_sub(index)
    });
    Painted {
      result: call.result,
      used_bytes,
    }
  }

  /// Runs `function` on this stack, painted with `paint`, and returns the bytes used.
  ///
  /// `function` must not unwind; the measurement binaries abort on panic.
  pub fn run<F: FnMut()>(&mut self, paint: u8, mut function: F) -> usize {
    extern "C" fn invoke<F: FnMut()>(context: usize, _: usize, _: usize) -> usize {
      // SAFETY: `run` passes the exposed address of its unique, live `F`, which
      // outlives this synchronous call.
      let function = unsafe { &mut *core::ptr::with_exposed_provenance_mut::<F>(context) };
      function();
      0
    }
    let context = (&raw mut function).expose_provenance();
    let trampoline: extern "C" fn(usize, usize, usize) -> usize = invoke::<F>;
    // SAFETY: the trampoline is a non-unwinding C-ABI function of three
    // integers (these binaries abort on panic). Its context is a live unique
    // `F`. Rust code needs no more than the stack's size, which every caller
    // chooses above the measured scenario's use.
    unsafe { self.call(paint, trampoline as *const (), [context, 0, 0]) }.used_bytes
  }
}

impl Drop for GuardedStack {
  fn drop(&mut self) {
    os::release(self.allocation, self.total);
  }
}

/// Rounds `bytes` up to the next multiple of 16, the stack slot alignment.
#[must_use]
pub const fn round16(bytes: usize) -> usize {
  bytes.strict_add(15) & !15
}

/// Writes `value` as a JSON string literal.
#[must_use]
pub fn json_string(value: &str) -> String {
  let mut out = String::with_capacity(value.len().strict_add(2));
  out.push('"');
  for character in value.chars() {
    match character {
      '"' => out.push_str("\\\""),
      '\\' => out.push_str("\\\\"),
      '\n' => out.push_str("\\n"),
      character if u32::from(character) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(character))),
      character => out.push(character),
    }
  }
  out.push('"');
  out
}
