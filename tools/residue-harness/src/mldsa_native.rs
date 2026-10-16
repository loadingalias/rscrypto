//! Native ML-DSA residue measurement on a private, painted stack.
//!
//! The capture build uses a retained patch in an isolated source copy to emit
//! production intermediate values. The measured build has no capture hooks.
//! Signatures bind each capture to its measured call. This is a functional
//! cleanup experiment; it records no timings.

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
mod native {
  extern crate alloc;

  use core::{ffi::c_void, hint::black_box};
  use std::{fs, path::Path};

  use rscrypto::{MlDsa44, MlDsa65, MlDsa87};

  const STACK_BYTES: usize = 1024 * 1024;
  const PAINT: u8 = 0xa5;

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
  fn input(tag: u64, index: u32) -> [u8; 32] {
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

  fn hex(bytes: &[u8]) -> String {
    use core::fmt::Write;
    let mut output = String::with_capacity(bytes.len().strict_mul(2));
    for byte in bytes {
      write!(output, "{byte:02x}").expect("String formatting");
    }
    output
  }

  #[cfg(feature = "capture-mldsa")]
  #[unsafe(no_mangle)]
  extern "C" fn rscrypto_residue_observe(attempt: usize, stage: u32, index: usize, data: *const u8, len: usize) {
    // SAFETY: the isolated capture patch calls synchronously with initialized
    // production scratch, borrowing `len` bytes until this function returns.
    let bytes = unsafe { core::slice::from_raw_parts(data, len) };
    println!("capture {attempt} {stage} {index} {}", hex(bytes));
  }

  extern "C" fn invoke<F: FnMut()>(context: *mut c_void) {
    // SAFETY: `measure` supplies a unique live F for the synchronous callback.
    let function = unsafe { &mut *context.cast::<F>() };
    function();
  }

  fn measure<F: FnMut()>(mut function: F) -> Vec<u8> {
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

  #[inline(never)]
  fn plant(needle: &[u8], clear: bool) {
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

  macro_rules! run_set {
    ($directory:expr, $start:expr, $count:expr, $name:literal, $set:ident) => {{
      let (public, secret) = $set::keypair_from_seed(&input(0, 0)).expect("key generation");
      let prepared = secret.prepare_in(alloc::alloc::Global).expect("preparation");
      for index in $start..$start.strict_add($count) {
        let message = input(1, index);
        let random = input(2, index);
        for hedged in [false, true] {
          for prepared_form in [false, true] {
            let name = format!(
              "{}-{}-{}-{index}",
              $name,
              if prepared_form { "prepared" } else { "compact" },
              if hedged { "hedged" } else { "deterministic" }
            );
            println!("case {name}");
            let mut signature = None;
            let stack = measure(|| {
              let fill = |out: &mut [u8]| {
                out.copy_from_slice(black_box(&random));
                Ok::<(), rscrypto::MlDsaError>(())
              };
              signature = Some(
                match (prepared_form, hedged) {
                  (false, false) => secret.sign_deterministic(black_box(&message), b"residue"),
                  (false, true) => secret.sign_with(black_box(&message), b"residue", fill),
                  (true, false) => prepared.sign_deterministic(black_box(&message), b"residue"),
                  (true, true) => prepared.sign_with(black_box(&message), b"residue", fill),
                }
                .expect("signature"),
              );
            });
            let signature = signature.expect("measured signature");
            public
              .verify_with_context(&message, b"residue", &signature)
              .expect("signature verifies");
            println!("signature {}", hex(signature.as_bytes()));
            if !cfg!(feature = "capture-mldsa") {
              fs::write($directory.join(format!("{name}.stack")), &stack).expect("write stack snapshot");
            }
            println!("end");
          }
        }
      }
    }};
  }

  pub(super) fn run() {
    let arguments: Vec<String> = std::env::args().collect();
    let directory = Path::new(arguments.get(1).expect("output directory"));
    fs::create_dir_all(directory).expect("output directory");
    if let Some(path) = arguments.get(2).and_then(|value| value.strip_prefix("control=")) {
      let needle = fs::read(path).expect("control needle");
      for clear in [false, true] {
        let stack = measure(|| plant(&needle, clear));
        fs::write(
          directory.join(if clear { "cleared.stack" } else { "planted.stack" }),
          stack,
        )
        .expect("control stack snapshot");
      }
      return;
    }
    let count: u32 = arguments.get(2).map_or(1, |value| value.parse().expect("case count"));
    let start: u32 = arguments
      .get(3)
      .map_or(0, |value| value.parse().expect("first message index"));
    assert!((1..=128).contains(&count), "case count 1..=128");
    assert!(start <= 4096, "first message index 0..=4096");
    run_set!(directory, start, count, "mldsa44", MlDsa44);
    run_set!(directory, start, count, "mldsa65", MlDsa65);
    run_set!(directory, start, count, "mldsa87", MlDsa87);
  }
}

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
fn main() {
  native::run();
}

#[cfg(not(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
)))]
fn main() {
  eprintln!("native ML-DSA residue measurement requires macOS AArch64 or Linux AArch64/x86-64");
  std::process::exit(2);
}
