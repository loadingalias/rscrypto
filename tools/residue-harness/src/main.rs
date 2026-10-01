//! Bare-metal moved-copy residue harness for QEMU.
//!
//! `scripts/stack/residue.py` boots this program on the RV32 `virt` and the
//! Cortex-M3 `mps2-an385` boards. Each scenario runs once on a painted stack
//! and painted allocator arenas. When it returns, the dead stack below the
//! caller and both arenas are copied into a snapshot with call-free volatile
//! loops, before any secret is derived again. The harness then derives the
//! scenario's secret byte strings ("needles") and writes the snapshot and the
//! needles to the UART. The host searches the snapshot for the needles.
//!
//! Controls run alongside the measurements: a needle planted in a returned
//! frame and in a freed allocation must be found in full, and a needle that is
//! cleared before return must not be found.

#![no_std]
#![no_main]

extern crate alloc;

use alloc::boxed::Box;
use core::{
  alloc::{AllocError, Allocator, GlobalAlloc, Layout},
  cell::UnsafeCell,
  fmt::Write,
  hint::black_box,
  ptr::NonNull,
};

use rscrypto::{
  Kem, MlDsa44, MlDsa65, MlDsa87, MlKem512, MlKem512DecapsulationKey, MlKem768, MlKem768DecapsulationKey, MlKem1024,
  MlKem1024DecapsulationKey, MlKemError,
};

const PAINT: u32 = 0xA55A_5AA5;
const ARENA_BYTES: usize = 64 * 1024;

#[cfg(target_arch = "riscv32")]
mod board {
  //! QEMU `virt`: NS16550A UART at 0x1000_0000. QEMU accepts every write.
  const UART: *mut u8 = 0x1000_0000 as *mut u8;

  core::arch::global_asm!(
    ".section .text.start, \"ax\"",
    ".global _start",
    "_start:",
    "  la sp, __stack_top",
    "  j {main}",
    main = sym crate::main,
  );

  pub(crate) fn init() {}

  pub(crate) fn write(byte: u8) {
    // SAFETY: the transmit register of the board's MMIO UART accepts byte
    // writes at any time under QEMU.
    unsafe { UART.write_volatile(byte) }
  }

  #[inline(always)]
  pub(crate) fn stack_pointer() -> usize {
    let sp: usize;
    // SAFETY: copies the stack pointer into a register; no memory access.
    unsafe { core::arch::asm!("mv {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags)) };
    sp
  }

  pub(crate) const NAME: &str = "riscv32-virt";
}

#[cfg(target_arch = "arm")]
mod board {
  //! QEMU `mps2-an385`: CMSDK APB UART0 at 0x4000_4000.
  const UART_DATA: *mut u32 = 0x4000_4000 as *mut u32;
  const UART_CTRL: *mut u32 = 0x4000_4008 as *mut u32;
  const UART_BAUDDIV: *mut u32 = 0x4000_4010 as *mut u32;

  #[repr(C)]
  struct Vectors {
    stack: *const u32,
    reset: unsafe extern "C" fn() -> !,
  }

  // SAFETY: the table is immutable and read only by the core at reset.
  unsafe impl Sync for Vectors {}

  unsafe extern "C" {
    static __stack_top: u32;
  }

  #[unsafe(link_section = ".vectors")]
  #[used]
  static VECTORS: Vectors = Vectors {
    stack: &raw const __stack_top,
    reset,
  };

  #[unsafe(no_mangle)]
  unsafe extern "C" fn reset() -> ! {
    crate::main()
  }

  pub(crate) fn init() {
    // SAFETY: board MMIO registers; enabling the transmitter is required
    // before QEMU accepts data writes.
    unsafe {
      UART_BAUDDIV.write_volatile(16);
      UART_CTRL.write_volatile(1);
    }
  }

  pub(crate) fn write(byte: u8) {
    // SAFETY: the transmitter was enabled in `init`; QEMU sends each write.
    unsafe { UART_DATA.write_volatile(u32::from(byte)) }
  }

  #[inline(always)]
  pub(crate) fn stack_pointer() -> usize {
    let sp: usize;
    // SAFETY: copies the stack pointer into a register; no memory access.
    unsafe { core::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags)) };
    sp
  }

  pub(crate) const NAME: &str = "thumb-mps2-an385";
}

#[cfg(not(any(target_arch = "riscv32", target_arch = "arm")))]
mod board {
  //! Host stand-in so workspace lints can type-check the harness.
  pub(crate) fn init() {}
  pub(crate) fn write(_: u8) {}
  pub(crate) fn stack_pointer() -> usize {
    0
  }
  pub(crate) const NAME: &str = "host";
}

struct Uart;

impl Write for Uart {
  fn write_str(&mut self, text: &str) -> core::fmt::Result {
    text.bytes().for_each(board::write);
    Ok(())
  }
}

macro_rules! say {
  ($($arg:tt)*) => {
    writeln!(Uart, $($arg)*).expect("UART writes cannot fail")
  };
}

fn hex(bytes: &[u8]) {
  for chunk in bytes.chunks(32) {
    for byte in chunk {
      write!(Uart, "{byte:02x}").expect("UART writes cannot fail");
    }
    say!();
  }
}

/// A bump arena that never reuses memory, so a freed block keeps whatever the
/// owner left in it until the next scenario repaints the arena.
struct Arena {
  memory: UnsafeCell<[u32; ARENA_BYTES / 4]>,
  used: UnsafeCell<usize>,
}

// SAFETY: the harness runs on one core with no interrupts.
unsafe impl Sync for Arena {}

impl Arena {
  const fn new() -> Self {
    Self {
      memory: UnsafeCell::new([0; ARENA_BYTES / 4]),
      used: UnsafeCell::new(0),
    }
  }

  fn base(&self) -> *mut u32 {
    self.memory.get().cast()
  }

  /// Paints the whole arena and forgets every allocation.
  fn reset(&self) {
    for index in 0..ARENA_BYTES / 4 {
      // SAFETY: `index` is inside the arena, and no allocation is live
      // between scenarios.
      unsafe { self.base().add(index).write_volatile(PAINT) };
    }
    // SAFETY: single-threaded; no other reference to `used` exists.
    unsafe { *self.used.get() = 0 };
  }

  fn take(&self, layout: Layout) -> Option<NonNull<u8>> {
    // SAFETY: single-threaded; no other reference to `used` exists.
    let used = unsafe { &mut *self.used.get() };
    let base = self.base() as usize;
    let start = base.strict_add(*used).next_multiple_of(layout.align());
    let end = start.checked_add(layout.size())?;
    if end > base.strict_add(ARENA_BYTES) {
      return None;
    }
    *used = end.strict_sub(base);
    NonNull::new(start as *mut u8)
  }
}

// SAFETY: blocks are disjoint, stay valid until the next reset, and respect
// the requested alignment.
unsafe impl Allocator for &Arena {
  fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    let block = self.take(layout).ok_or(AllocError)?;
    Ok(NonNull::slice_from_raw_parts(block, layout.size()))
  }

  unsafe fn deallocate(&self, _: NonNull<u8>, _: Layout) {}
}

// SAFETY: as for the `Allocator` implementation.
unsafe impl GlobalAlloc for Arena {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    self.take(layout).map_or(core::ptr::null_mut(), NonNull::as_ptr)
  }

  unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}

#[global_allocator]
static HEAP: Arena = Arena::new();
static ARENA: Arena = Arena::new();

/// Copies of the stack and arenas taken right after a scenario returns.
struct Snapshot {
  stack: UnsafeCell<[u32; 1 << 18]>,
  heap: UnsafeCell<[u32; ARENA_BYTES / 4]>,
  arena: UnsafeCell<[u32; ARENA_BYTES / 4]>,
  stack_words: UnsafeCell<usize>,
  stack_base: UnsafeCell<usize>,
}

// SAFETY: the harness runs on one core with no interrupts.
unsafe impl Sync for Snapshot {}

static SNAPSHOT: Snapshot = Snapshot {
  stack: UnsafeCell::new([0; 1 << 18]),
  heap: UnsafeCell::new([0; ARENA_BYTES / 4]),
  arena: UnsafeCell::new([0; ARENA_BYTES / 4]),
  stack_words: UnsafeCell::new(0),
  stack_base: UnsafeCell::new(0),
};

unsafe extern "C" {
  static __stack_bottom: u32;
}

/// Runs `scenario` on freshly painted memory, then snapshots it.
///
/// Between reading the stack pointer and finishing the snapshot, this frame
/// makes no call except `scenario`: every loop uses volatile word accesses,
/// which the compiler cannot turn into `memset` or `memcpy` calls that would
/// write below the stack pointer.
#[inline(never)]
fn measure(scenario: fn()) {
  HEAP.reset();
  ARENA.reset();
  let bottom = (&raw const __stack_bottom) as usize;
  let top = board::stack_pointer() & !3;
  let words = top.strict_sub(bottom) / 4;
  // SAFETY: SNAPSHOT is touched only here and in `report`, never concurrently.
  let capacity = unsafe { (*SNAPSHOT.stack.get()).len() };
  assert!(words <= capacity, "stack snapshot too small");
  let stack = bottom as *mut u32;
  for index in 0..words {
    // SAFETY: the linker reserves [__stack_bottom, __stack_top) for the stack,
    // and every word below the current stack pointer is unused: no frame
    // lives there, and these targets have no red zone or interrupts.
    unsafe { stack.add(index).write_volatile(PAINT) };
  }

  scenario();

  // SAFETY: as above; the scenario's frames below `top` are now dead.
  unsafe {
    let copy = SNAPSHOT.stack.get().cast::<u32>();
    for index in 0..words {
      copy.add(index).write_volatile(stack.add(index).read_volatile());
    }
    for (source, destination) in [(&HEAP, SNAPSHOT.heap.get()), (&ARENA, SNAPSHOT.arena.get())] {
      let destination = destination.cast::<u32>();
      for index in 0..ARENA_BYTES / 4 {
        destination.add(index).write_volatile(source.base().add(index).read_volatile());
      }
    }
    *SNAPSHOT.stack_words.get() = words;
    *SNAPSHOT.stack_base.get() = bottom;
  }
}

fn dump(name: &str, base: usize, words: &[u32], keep_end: bool) {
  // Omit painted words outside the touched span. The stack keeps its top,
  // where the scenario's frames begin, so its length is the stack used.
  let first = words.iter().position(|&word| word != PAINT).unwrap_or(words.len()) & !7;
  let end = if keep_end {
    words.len()
  } else {
    let last = words.iter().rposition(|&word| word != PAINT);
    last.map_or(first, |last| last.strict_add(8) & !7).min(words.len())
  };
  say!("region {name} {:#x} {:#x}", base.strict_add(first.strict_mul(4)), end.strict_sub(first).strict_mul(4));
  let mut line = [0u8; 32];
  for chunk in words[first..end].chunks(8) {
    for (slot, word) in line.chunks_mut(4).zip(chunk) {
      slot.copy_from_slice(&word.to_le_bytes());
    }
    hex(&line[..chunk.len().strict_mul(4)]);
  }
  say!("endregion");
}

fn report() {
  // SAFETY: SNAPSHOT is touched only here and in `measure`, never concurrently.
  unsafe {
    let words = *SNAPSHOT.stack_words.get();
    let stack = &(&*SNAPSHOT.stack.get())[..words];
    dump("stack", *SNAPSHOT.stack_base.get(), stack, true);
    dump("heap", HEAP.base() as usize, &*SNAPSHOT.heap.get(), false);
    dump("arena", ARENA.base() as usize, &*SNAPSHOT.arena.get(), false);
  }
}

fn needle(name: &str, bytes: &[u8]) {
  say!("needle {name} {}", bytes.len());
  hex(bytes);
}

/// One measured operation and the secrets it must not leave behind.
struct Scenario {
  name: &'static str,
  /// `none`: no needle byte may remain; `report`: the residue is measured
  /// only; `stack` or `arena`: a control needle must be found there in full.
  expect: &'static str,
  /// Runs before the memory is painted, so its own traces are erased.
  prepare: fn(),
  run: fn(),
  /// Runs after the snapshot, so its copies of the secrets are not measured.
  needles: fn(),
}

const CONTROL: [u8; 96] = {
  let mut bytes = [0u8; 96];
  let mut state = 0x9E37_79B9_u32;
  let mut index = 0;
  while index < bytes.len() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    bytes[index] = state.to_le_bytes()[0];
    index += 1;
  }
  bytes
};

#[inline(never)]
fn planted_stack() {
  let copy = black_box(CONTROL);
  black_box(&copy);
}

#[inline(never)]
fn planted_arena() {
  let block = Box::new_in(black_box(CONTROL), &ARENA);
  black_box(&block);
}

#[inline(never)]
fn cleared_stack() {
  let mut copy = black_box(CONTROL);
  black_box(&copy);
  for byte in &mut copy {
    // SAFETY: `byte` is a valid, aligned reference into a local array.
    unsafe { core::ptr::write_volatile(byte, 0) };
  }
  black_box(&copy);
}

fn control_needles() {
  needle("control", &CONTROL);
}

fn nothing() {}

fn fill(byte: u8) -> impl FnMut(&mut [u8]) -> Result<(), MlKemError> {
  move |out| {
    out.fill(black_box(byte));
    Ok(())
  }
}

/// Encoded key input for import scenarios, outside every scanned region.
struct KeyInput(UnsafeCell<[u8; 3168]>);

// SAFETY: the harness runs on one core with no interrupts.
unsafe impl Sync for KeyInput {}

static KEY_INPUT: KeyInput = KeyInput(UnsafeCell::new([0; 3168]));

fn key_input(len: usize) -> &'static [u8] {
  // SAFETY: written only by a `prepare` function, never while borrowed.
  unsafe { &(&*KEY_INPUT.0.get())[..len] }
}

macro_rules! mlkem {
  ($module:ident, $profile:ty, $key:ty, $byte:literal) => {
    mod $module {
      use super::*;

      pub(super) fn keygen() {
        let keys = <$profile>::generate_keypair(fill($byte)).expect("key generation");
        black_box(&keys);
      }

      pub(super) fn keygen_in() {
        let keys = <$profile>::generate_keypair_in(fill($byte), &ARENA).expect("key generation");
        black_box(&keys);
      }

      pub(super) fn import() {
        let key = <$key>::try_from_slice(key_input(<$key>::LENGTH)).expect("import");
        black_box(&key);
      }

      pub(super) fn import_in() {
        let key = <$key>::try_from_slice_in(key_input(<$key>::LENGTH), &ARENA).expect("import");
        black_box(&key);
      }

      pub(super) fn prepare() {
        let (_, key) = <$profile>::generate_keypair_in(fill($byte), &HEAP).expect("key generation");
        let secret = key.expose_secret();
        // SAFETY: no reference into KEY_INPUT is live while it is written.
        unsafe { (&mut *KEY_INPUT.0.get())[..<$key>::LENGTH].copy_from_slice(secret.as_bytes()) };
      }

      pub(super) fn needles() {
        let (_, key) = <$profile>::generate_keypair_in(fill($byte), &HEAP).expect("key generation");
        let secret = key.expose_secret();
        let bytes = secret.as_bytes();
        // dk = dk_pke || ek || H(ek) || z, where ek = 384k + 32 bytes and
        // dk_pke = 384k bytes.
        let pke = bytes.len().strict_sub(64 + 32) / 2;
        needle("dk_pke", &bytes[..pke]);
        needle("z", &bytes[bytes.len().strict_sub(32)..]);
      }
    }
  };
}

mlkem!(mlkem512, MlKem512, MlKem512DecapsulationKey, 0x51);
mlkem!(mlkem768, MlKem768, MlKem768DecapsulationKey, 0x76);
mlkem!(mlkem1024, MlKem1024, MlKem1024DecapsulationKey, 0x10);

const MLDSA_SEED: [u8; 32] = [0x3d; 32];

macro_rules! mldsa {
  ($module:ident, $profile:ty) => {
    mod $module {
      use super::*;

      pub(super) fn keygen() {
        let keys = <$profile>::keypair_from_seed(black_box(&MLDSA_SEED)).expect("key generation");
        black_box(&keys);
      }

      pub(super) fn keygen_in() {
        let keys = <$profile>::keypair_from_seed_in(black_box(&MLDSA_SEED), &ARENA).expect("key generation");
        black_box(&keys);
      }

      pub(super) fn needles() {
        let (_, key) = <$profile>::keypair_from_seed_in(&MLDSA_SEED, &HEAP).expect("key generation");
        let secret = key.expose_secret();
        let bytes = secret.as_bytes();
        // sk = rho || K || tr || s1 || s2 || t0; rho and tr are public.
        needle("K", &bytes[32..64]);
        needle("s1_s2_t0", &bytes[128..]);
      }
    }
  };
}

mldsa!(mldsa44, MlDsa44);
mldsa!(mldsa65, MlDsa65);
mldsa!(mldsa87, MlDsa87);

macro_rules! scenario {
  ($name:literal, $expect:literal, $run:path, $needles:path) => {
    scenario!($name, $expect, nothing, $run, $needles)
  };
  ($name:literal, $expect:literal, $prepare:path, $run:path, $needles:path) => {
    Scenario { name: $name, expect: $expect, prepare: $prepare, run: $run, needles: $needles }
  };
}

const SCENARIOS: &[Scenario] = &[
  scenario!("control-planted-stack", "stack", planted_stack, control_needles),
  scenario!("control-planted-arena", "arena", planted_arena, control_needles),
  scenario!("control-cleared-stack", "none", cleared_stack, control_needles),
  scenario!("ml-kem-512-keygen", "report", mlkem512::keygen, mlkem512::needles),
  scenario!("ml-kem-512-keygen-in", "none", mlkem512::keygen_in, mlkem512::needles),
  scenario!("ml-kem-512-import", "report", mlkem512::prepare, mlkem512::import, mlkem512::needles),
  scenario!("ml-kem-512-import-in", "none", mlkem512::prepare, mlkem512::import_in, mlkem512::needles),
  scenario!("ml-kem-768-keygen", "report", mlkem768::keygen, mlkem768::needles),
  scenario!("ml-kem-768-keygen-in", "none", mlkem768::keygen_in, mlkem768::needles),
  scenario!("ml-kem-768-import", "report", mlkem768::prepare, mlkem768::import, mlkem768::needles),
  scenario!("ml-kem-768-import-in", "none", mlkem768::prepare, mlkem768::import_in, mlkem768::needles),
  scenario!("ml-kem-1024-keygen", "report", mlkem1024::keygen, mlkem1024::needles),
  scenario!("ml-kem-1024-keygen-in", "none", mlkem1024::keygen_in, mlkem1024::needles),
  scenario!("ml-kem-1024-import", "report", mlkem1024::prepare, mlkem1024::import, mlkem1024::needles),
  scenario!("ml-kem-1024-import-in", "none", mlkem1024::prepare, mlkem1024::import_in, mlkem1024::needles),
  scenario!("ml-dsa-44-keygen", "report", mldsa44::keygen, mldsa44::needles),
  scenario!("ml-dsa-44-keygen-in", "none", mldsa44::keygen_in, mldsa44::needles),
  scenario!("ml-dsa-65-keygen", "report", mldsa65::keygen, mldsa65::needles),
  scenario!("ml-dsa-65-keygen-in", "none", mldsa65::keygen_in, mldsa65::needles),
  scenario!("ml-dsa-87-keygen", "report", mldsa87::keygen, mldsa87::needles),
  scenario!("ml-dsa-87-keygen-in", "none", mldsa87::keygen_in, mldsa87::needles),
];

#[unsafe(no_mangle)]
extern "C" fn main() -> ! {
  board::init();
  let backend = if cfg!(feature = "portable-only") { "portable" } else { "native" };
  say!("residue 1 board={} backend={backend} scenarios={}", board::NAME, SCENARIOS.len());
  for scenario in SCENARIOS {
    (scenario.prepare)();
    measure(scenario.run);
    say!("scenario {} expect={}", scenario.name, scenario.expect);
    report();
    (scenario.needles)();
    say!("end");
  }
  say!("done");
  loop {
    core::hint::spin_loop();
  }
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
  say!("panic {info}");
  loop {
    core::hint::spin_loop();
  }
}
