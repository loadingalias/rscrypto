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
  Kem, MlDsa44, MlDsa44SecretKey, MlDsa44Seed, MlDsa65, MlDsa65SecretKey, MlDsa65Seed, MlDsa87, MlDsa87SecretKey,
  MlDsa87Seed, MlKem512, MlKem512DecapsulationKey, MlKem512Seed, MlKem768, MlKem768DecapsulationKey, MlKem768Seed,
  MlKem1024, MlKem1024DecapsulationKey, MlKem1024Seed, MlKemError, Shake256, XWing, XWingCiphertext,
  XWingDecapsulationKey, XWingError,
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
        destination
          .add(index)
          .write_volatile(source.base().add(index).read_volatile());
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
  say!(
    "region {name} {:#x} {:#x}",
    base.strict_add(first.strict_mul(4)),
    end.strict_sub(first).strict_mul(4)
  );
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

/// Key generation randomness from a xorshift stream: distinct, non-repeating
/// bytes, so every needle derived from it is specific. A repeated byte would
/// equal its own limb form and match any run of that byte.
fn stream_fill<E>(seed: u32) -> impl FnMut(&mut [u8]) -> Result<(), E> {
  let mut state = seed;
  move |out| {
    for byte in out.iter_mut() {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      *byte = black_box(state.to_le_bytes()[1]);
    }
    Ok(())
  }
}

/// The 64-byte d || z that `stream_fill(seed)` gives ML-KEM key generation.
fn mlkem_seed(seed: u32) -> [u8; 64] {
  let mut bytes = [0; 64];
  stream_fill::<MlKemError>(seed)(&mut bytes).expect("the stream cannot fail");
  bytes
}

/// Encoded key input for import scenarios, outside every scanned region.
struct KeyInput(UnsafeCell<[u8; 4924]>);

// SAFETY: the harness runs on one core with no interrupts.
unsafe impl Sync for KeyInput {}

// The largest input is the 4,924-byte ML-DSA-87 expanded-form PKCS #8 key.
static KEY_INPUT: KeyInput = KeyInput(UnsafeCell::new([0; 4924]));

fn key_input(len: usize) -> &'static [u8] {
  // SAFETY: written only by a `prepare` function, never while borrowed.
  unsafe { &(&*KEY_INPUT.0.get())[..len] }
}

/// Fixed-size view of the input that a `prepare` function writes.
fn key_input_mut<const N: usize>() -> &'static mut [u8; N] {
  // SAFETY: called only by a `prepare` function, while no other reference
  // into KEY_INPUT is live.
  let input = unsafe { &mut *KEY_INPUT.0.get() };
  input.first_chunk_mut().expect("key input holds every encoding")
}

macro_rules! mlkem {
  ($module:ident, $profile:ty, $key:ty, $seed:ty, $stream:literal) => {
    mod $module {
      use super::*;

      pub(super) fn keygen() {
        let keys = <$profile>::generate_keypair(stream_fill($stream)).expect("key generation");
        black_box(&keys);
      }

      pub(super) fn keygen_in() {
        let keys = <$profile>::generate_keypair_in(stream_fill($stream), &ARENA).expect("key generation");
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
        let (_, key) = <$profile>::generate_keypair_in(stream_fill($stream), &HEAP).expect("key generation");
        let secret = key.expose_secret();
        // SAFETY: no reference into KEY_INPUT is live while it is written.
        unsafe { (&mut *KEY_INPUT.0.get())[..<$key>::LENGTH].copy_from_slice(secret.as_bytes()) };
      }

      pub(super) fn prepare_pkcs8_seed() {
        <$seed>::from_bytes(mlkem_seed($stream)).to_pkcs8_der_into(key_input_mut());
      }

      pub(super) fn prepare_pkcs8_expanded() {
        let (_, key) = <$profile>::generate_keypair_in(stream_fill($stream), &HEAP).expect("key generation");
        key.to_pkcs8_der_into(key_input_mut());
      }

      pub(super) fn pkcs8_seed_in() {
        let key = <$key>::from_pkcs8_der_in(key_input(<$seed>::PKCS8_DER_LENGTH), &ARENA).expect("import");
        black_box(&key);
      }

      pub(super) fn pkcs8_expanded() {
        let key = <$key>::from_pkcs8_der(key_input(<$key>::PKCS8_DER_LENGTH)).expect("import");
        black_box(&key);
      }

      pub(super) fn pkcs8_expanded_in() {
        let key = <$key>::from_pkcs8_der_in(key_input(<$key>::PKCS8_DER_LENGTH), &ARENA).expect("import");
        black_box(&key);
      }

      pub(super) fn seed_owner() {
        let seed = <$seed>::from_pkcs8_der(key_input(<$seed>::PKCS8_DER_LENGTH)).expect("import");
        black_box(&seed);
      }

      pub(super) fn needles() {
        let (_, key) = <$profile>::generate_keypair_in(stream_fill($stream), &HEAP).expect("key generation");
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

mlkem!(mlkem512, MlKem512, MlKem512DecapsulationKey, MlKem512Seed, 0x0512_3A5D);
mlkem!(mlkem768, MlKem768, MlKem768DecapsulationKey, MlKem768Seed, 0x0768_C1E7);
mlkem!(
  mlkem1024,
  MlKem1024,
  MlKem1024DecapsulationKey,
  MlKem1024Seed,
  0x1024_7F3B
);

mod xwing {
  use super::*;

  const STREAM: u32 = 0x5857_1D6E;
  const ENCAPSULATION_STREAM: u32 = 0x5857_E2C4;

  fn decapsulation_key() -> XWingDecapsulationKey {
    XWingDecapsulationKey::generate(stream_fill(STREAM)).expect("key generation")
  }

  pub(super) fn keygen() {
    let keys = XWing::generate_keypair(stream_fill(STREAM)).expect("key generation");
    black_box(&keys);
  }

  /// Writes the decapsulation key, then a ciphertext for it.
  pub(super) fn prepare() {
    let key = decapsulation_key();
    let (ciphertext, _) =
      XWing::encapsulate(&key.encapsulation_key(), stream_fill(ENCAPSULATION_STREAM)).expect("encapsulation");
    let input: &mut [u8; XWing::DECAPSULATION_KEY_SIZE + XWing::CIPHERTEXT_SIZE] = key_input_mut();
    let (key_part, ciphertext_part) = input.split_at_mut(XWing::DECAPSULATION_KEY_SIZE);
    key_part.copy_from_slice(key.as_bytes());
    ciphertext_part.copy_from_slice(ciphertext.as_bytes());
  }

  fn imported() -> XWingDecapsulationKey {
    XWingDecapsulationKey::try_from_slice(key_input(XWing::DECAPSULATION_KEY_SIZE)).expect("import")
  }

  pub(super) fn prepare_in() {
    let key = imported().prepare_in(&ARENA);
    black_box(&key);
  }

  pub(super) fn decapsulate() {
    let ciphertext = XWingCiphertext::try_from_slice(
      &key_input(XWing::DECAPSULATION_KEY_SIZE + XWing::CIPHERTEXT_SIZE)[XWing::DECAPSULATION_KEY_SIZE..],
    )
    .expect("ciphertext");
    let shared = XWing::decapsulate(&imported(), &ciphertext).expect("decapsulation");
    black_box(&shared);
  }

  pub(super) fn encapsulate() {
    let key = XWingDecapsulationKey::from_bytes([0x5a; XWing::DECAPSULATION_KEY_SIZE]).encapsulation_key();
    let output = XWing::encapsulate(black_box(&key), stream_fill(ENCAPSULATION_STREAM)).expect("encapsulation");
    black_box(&output);
  }

  /// The expanded secrets: the ML-KEM-768 secret vector and `z`, and the
  /// X25519 secret key.
  pub(super) fn needles() {
    expanded_needles(decapsulation_key().as_bytes());
  }

  /// The expanded secrets of `seed`.
  pub(super) fn expanded_needles(seed: &[u8; XWing::DECAPSULATION_KEY_SIZE]) {
    let mut expanded = [0u8; 96];
    Shake256::hash_into(seed, &mut expanded);
    let mut ml_kem_seed = [0u8; 64];
    ml_kem_seed.copy_from_slice(&expanded[..64]);
    let (_, ml_kem) = MlKem768Seed::from_bytes(ml_kem_seed).keypair_in(&HEAP);
    needle("dk_pke", &ml_kem.expose_secret().as_bytes()[..1152]);
    needle("z", &expanded[32..64]);
    needle("x25519_secret", &expanded[64..]);
  }

  /// The encapsulation randomness: the ML-KEM-768 coins and the X25519
  /// ephemeral secret key.
  pub(super) fn encapsulation_needles() {
    let mut random = [0u8; XWing::ENCAPSULATION_RANDOM_SIZE];
    stream_fill::<XWingError>(ENCAPSULATION_STREAM)(&mut random).expect("the stream cannot fail");
    needle("coins", &random[..32]);
    needle("x25519_ephemeral", &random[32..]);
  }

  /// The shared secret that `encapsulate` returns by value.
  pub(super) fn encapsulated_secret_needle() {
    let key = XWingDecapsulationKey::from_bytes([0x5a; XWing::DECAPSULATION_KEY_SIZE]).encapsulation_key();
    let (_, shared) = XWing::encapsulate(&key, stream_fill(ENCAPSULATION_STREAM)).expect("encapsulation");
    needle("shared_secret", shared.as_bytes());
  }

  /// The shared secret that `decapsulate` returns by value for the prepared
  /// ciphertext.
  pub(super) fn decapsulated_secret_needle() {
    let ciphertext = XWingCiphertext::try_from_slice(
      &key_input(XWing::DECAPSULATION_KEY_SIZE + XWing::CIPHERTEXT_SIZE)[XWing::DECAPSULATION_KEY_SIZE..],
    )
    .expect("ciphertext");
    let shared = XWing::decapsulate(&imported(), &ciphertext).expect("decapsulation");
    needle("shared_secret", shared.as_bytes());
  }
}

// RFC 9180 A.1.1 and draft-ietf-hpke-pq-05 A.5. The key schedule outputs of
// each vector are needles, so no needle comes from the code under test. The
// inputs live in KEY_INPUT, outside every scanned region.
mod hpke {
  use super::*;
  use rscrypto::{
    HpkeError, HpkeReceiver, HpkeSender, HpkeSuite, HpkeX25519HkdfSha256Aes128Gcm as X25519Suite, HpkeX25519PrivateKey,
    HpkeXWingHkdfSha256ChaCha20Poly1305 as XWingSuite, X25519PublicKey, X25519SecretKey, X25519SharedSecret,
    XWingEncapsulationKey,
  };

  const X25519_VECTOR: &str = include_str!("../../../testdata/hpke/hpke-x25519-hkdfsha256-aes128gcm.json");
  const XWING_VECTORS: &str = include_str!("../../../testdata/hpke/hpke-pq-xwing.json");
  const MESSAGE: &[u8] = b"rscrypto HPKE residue";
  const NAMES: [&str; 8] = ["ikmR", "ikmE", "pkRm", "skRm", "enc", "info", "aad", "ct"];

  /// One vector's fields in KEY_INPUT, in `NAMES` order: `[offset, length]`.
  struct Layout([[usize; 2]; 8]);

  impl Layout {
    const fn new(lengths: [usize; 8]) -> Self {
      let mut fields = [[0; 2]; 8];
      let mut index = 0;
      let mut offset = 0;
      while index < 8 {
        fields[index] = [offset, lengths[index]];
        offset = offset.strict_add(lengths[index]);
        index = index.strict_add(1);
      }
      Self(fields)
    }

    const fn end(&self) -> usize {
      self.0[7][0].strict_add(self.0[7][1])
    }

    fn input(&self, name: &str) -> &'static [u8] {
      let index = NAMES.iter().position(|field| *field == name).expect("field name");
      let [offset, len] = self.0[index];
      &key_input(offset.strict_add(len))[offset..]
    }

    fn array<const N: usize>(&self, name: &str) -> [u8; N] {
      self.input(name).try_into().expect("field length")
    }
  }

  const X25519: Layout = Layout::new([32, 32, 32, 32, 32, 20, 7, 45]);
  const XWING: Layout = Layout::new([32, 64, 1216, 32, 1120, 40, 7, 74]);
  const INPUT_SIZE: usize = XWING.end();

  fn xwing_vector() -> &'static str {
    XWING_VECTORS
      .split("{\"mode\":")
      .find(|vector| vector.contains("\"kdf_id\":1,"))
      .expect("the A.5 vector")
  }

  /// The first hexadecimal field `name` of `vector`, decoded.
  fn decode(vector: &str, name: &str) -> alloc::vec::Vec<u8> {
    let (_, value) = vector
      .split_once(&alloc::format!("\"{name}\":\""))
      .expect("HPKE vector field");
    let (value, _) = value.split_once('"').expect("HPKE vector field end");
    value
      .as_bytes()
      .chunks(2)
      .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).expect("ASCII"), 16).expect("hex"))
      .collect()
  }

  fn write(vector: &str, layout: &Layout) {
    let input: &mut [u8; INPUT_SIZE] = key_input_mut();
    for (name, [offset, len]) in NAMES.iter().zip(layout.0) {
      let bytes = decode(vector, name);
      assert_eq!(bytes.len(), len, "{name}");
      input[offset..offset.strict_add(len)].copy_from_slice(&bytes);
    }
  }

  pub(super) fn prepare_x25519() {
    write(X25519_VECTOR, &X25519);
  }

  pub(super) fn prepare_xwing() {
    write(xwing_vector(), &XWING);
  }

  /// Run the sender with the vector's `ikmE` and seal one message.
  fn send<S: HpkeSuite>(layout: &Layout, public_key: &S::PublicKey) {
    let random = layout.input("ikmE");
    let (encapsulation, mut sender) = HpkeSender::<S>::setup(public_key, layout.input("info"), |out| {
      out.copy_from_slice(random);
      Ok::<(), HpkeError>(())
    })
    .expect("setup");
    let mut ciphertext = [0u8; MESSAGE.len() + 16];
    sender.seal(b"", MESSAGE, &mut ciphertext).expect("seal");
    black_box((&encapsulation, &sender, &ciphertext));
  }

  /// Run the receiver and open the vector's first ciphertext.
  fn receive<S: HpkeSuite>(layout: &Layout, encapsulation: &S::Encapsulation, private_key: &S::PrivateKey) {
    let mut receiver = HpkeReceiver::<S>::setup(encapsulation, private_key, layout.input("info")).expect("setup");
    let ciphertext = layout.input("ct");
    let mut plaintext = [0u8; 64];
    receiver
      .open(
        layout.input("aad"),
        ciphertext,
        &mut plaintext[..ciphertext.len().strict_sub(16)],
      )
      .expect("open");
    black_box((&receiver, &plaintext));
  }

  pub(super) fn x25519_derive() {
    let key = X25519Suite::derive_keypair(X25519.input("ikmR"));
    black_box(&key);
  }

  pub(super) fn x25519_sender() {
    send::<X25519Suite>(&X25519, &X25519PublicKey::from_bytes(X25519.array("pkRm")));
  }

  /// The recipient key is imported in this frame, so its bytes are not a
  /// needle: the import is the caller's copy, not HPKE's.
  pub(super) fn x25519_receiver() {
    let recipient = HpkeX25519PrivateKey::new(X25519SecretKey::from_bytes(X25519.array("skRm")));
    receive::<X25519Suite>(&X25519, &X25519PublicKey::from_bytes(X25519.array("enc")), &recipient);
  }

  pub(super) fn xwing_derive() {
    let key = XWingSuite::derive_keypair(XWING.input("ikmR"));
    black_box(&key);
  }

  pub(super) fn xwing_sender() {
    let public_key = XWingEncapsulationKey::try_from_slice(XWING.input("pkRm"))
      .expect("encapsulation key")
      .prepare()
      .expect("valid key");
    send::<XWingSuite>(&XWING, &public_key);
  }

  /// Run the sender with its context in the arena.
  fn send_in<S: HpkeSuite>(layout: &Layout, public_key: &S::PublicKey) {
    let random = layout.input("ikmE");
    let (encapsulation, mut sender) = HpkeSender::<S>::setup_in(
      public_key,
      layout.input("info"),
      |out| {
        out.copy_from_slice(random);
        Ok::<(), HpkeError>(())
      },
      &ARENA,
    )
    .expect("setup");
    let mut ciphertext = [0u8; MESSAGE.len() + 16];
    sender.seal(b"", MESSAGE, &mut ciphertext).expect("seal");
    black_box((&encapsulation, &sender, &ciphertext));
  }

  /// Run the receiver with its context in the arena.
  fn receive_in<S: HpkeSuite>(layout: &Layout, encapsulation: &S::Encapsulation, private_key: &S::PrivateKey) {
    let mut receiver =
      HpkeReceiver::<S>::setup_in(encapsulation, private_key, layout.input("info"), &ARENA).expect("setup");
    let ciphertext = layout.input("ct");
    let mut plaintext = [0u8; 64];
    receiver
      .open(
        layout.input("aad"),
        ciphertext,
        &mut plaintext[..ciphertext.len().strict_sub(16)],
      )
      .expect("open");
    black_box((&receiver, &plaintext));
  }

  pub(super) fn x25519_derive_in() {
    let key = X25519Suite::derive_keypair_in(X25519.input("ikmR"), &ARENA);
    black_box(&key);
  }

  pub(super) fn x25519_sender_in() {
    send_in::<X25519Suite>(&X25519, &X25519PublicKey::from_bytes(X25519.array("pkRm")));
  }

  pub(super) fn x25519_receiver_in() {
    let recipient = X25519Suite::derive_keypair_in(X25519.input("ikmR"), &ARENA);
    receive_in::<X25519Suite>(&X25519, &X25519PublicKey::from_bytes(X25519.array("enc")), &recipient);
  }

  pub(super) fn xwing_derive_in() {
    let key = XWingSuite::derive_keypair_in(XWING.input("ikmR"), &ARENA);
    black_box(&key);
  }

  pub(super) fn xwing_sender_in() {
    let public_key = XWingEncapsulationKey::try_from_slice(XWING.input("pkRm"))
      .expect("encapsulation key")
      .prepare()
      .expect("valid key");
    send_in::<XWingSuite>(&XWING, &public_key);
  }

  pub(super) fn xwing_receiver_in() {
    let key = XWingSuite::derive_keypair_in(XWING.input("ikmR"), &ARENA);
    let encapsulation = XWingCiphertext::try_from_slice(XWING.input("enc")).expect("encapsulation");
    receive_in::<XWingSuite>(&XWING, &encapsulation, &key);
  }

  pub(super) fn xwing_receiver() {
    let key = XWingDecapsulationKey::try_from_slice(XWING.input("skRm"))
      .expect("decapsulation key")
      .prepare_in(&ARENA);
    let encapsulation = XWingCiphertext::try_from_slice(XWING.input("enc")).expect("encapsulation");
    receive::<XWingSuite>(&XWING, &encapsulation, &key);
  }

  /// The key schedule outputs that both roles hold, and the HMAC inner and
  /// outer midstates of the exporter secret, which the context stores and
  /// `sha2` derives independently. The 12-byte base nonce is shorter than the
  /// scanner's 16-byte window, so it is not a needle.
  fn schedule_needles(vector: &str) {
    for name in ["shared_secret", "key", "exporter_secret"] {
      needle(name, &decode(vector, name));
    }
    let exporter_secret = decode(vector, "exporter_secret");
    needle("exporter_inner", &hkdf_sha256::prefix(&exporter_secret, 0x36));
    needle("exporter_outer", &hkdf_sha256::prefix(&exporter_secret, 0x5c));
  }

  /// The X25519 shared secret of the vector, from the recipient side.
  fn x25519_dh() -> X25519SharedSecret {
    let secret = X25519SecretKey::from_bytes(decode(X25519_VECTOR, "skRm").try_into().expect("32 bytes"));
    let enc = X25519PublicKey::from_bytes(decode(X25519_VECTOR, "enc").try_into().expect("32 bytes"));
    secret.diffie_hellman(&enc).expect("vector DH")
  }

  pub(super) fn x25519_derive_needles() {
    needle("skRm", &decode(X25519_VECTOR, "skRm"));
  }

  pub(super) fn x25519_sender_needles() {
    needle("ikmE", &decode(X25519_VECTOR, "ikmE"));
    needle("skEm", &decode(X25519_VECTOR, "skEm"));
    needle("dh", x25519_dh().as_bytes());
    schedule_needles(X25519_VECTOR);
  }

  pub(super) fn x25519_receiver_needles() {
    needle("dh", x25519_dh().as_bytes());
    schedule_needles(X25519_VECTOR);
  }

  pub(super) fn xwing_derive_needles() {
    let seed = decode(xwing_vector(), "skRm");
    needle("seed", &seed);
    xwing::expanded_needles(&seed.try_into().expect("32-byte seed"));
  }

  pub(super) fn xwing_sender_needles() {
    let random = decode(xwing_vector(), "ikmE");
    needle("coins", &random[..32]);
    needle("x25519_ephemeral", &random[32..]);
    schedule_needles(xwing_vector());
  }

  pub(super) fn xwing_receiver_needles() {
    xwing::expanded_needles(&decode(xwing_vector(), "skRm").try_into().expect("32-byte seed"));
    schedule_needles(xwing_vector());
  }
}

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

/// A seed with distinct bytes, so the seed itself is a usable needle.
const MLDSA_PKCS8_SEED: [u8; 32] = [
  0x9e, 0x37, 0x79, 0xb9, 0x7f, 0x4a, 0x7c, 0x15, 0xf3, 0x9c, 0xc0, 0x60, 0x5c, 0xed, 0xc8, 0x34, 0x10, 0x82, 0x27,
  0x6b, 0xf3, 0xa2, 0x72, 0x39, 0x5a, 0x6c, 0xd4, 0x1e, 0x8b, 0x0f, 0x31, 0xc7,
];

macro_rules! mldsa_pkcs8 {
  ($module:ident, $profile:ty, $secret:ty, $seed:ty) => {
    mod $module {
      use super::*;

      pub(super) fn prepare_seed() {
        <$seed>::from_bytes(MLDSA_PKCS8_SEED).to_pkcs8_der_into(key_input_mut());
      }

      pub(super) fn prepare_expanded() {
        let (_, key) = <$profile>::keypair_from_seed_in(&MLDSA_PKCS8_SEED, &HEAP).expect("key generation");
        key.to_pkcs8_der_into(key_input_mut());
      }

      pub(super) fn seed_in() {
        let key = <$secret>::from_pkcs8_der_in(key_input(<$seed>::PKCS8_DER_LENGTH), &ARENA).expect("import");
        black_box(&key);
      }

      pub(super) fn expanded() {
        let key = <$secret>::from_pkcs8_der(key_input(<$secret>::PKCS8_DER_LENGTH)).expect("import");
        black_box(&key);
      }

      pub(super) fn expanded_in() {
        let key = <$secret>::from_pkcs8_der_in(key_input(<$secret>::PKCS8_DER_LENGTH), &ARENA).expect("import");
        black_box(&key);
      }

      pub(super) fn seed_owner() {
        let seed = <$seed>::from_pkcs8_der(key_input(<$seed>::PKCS8_DER_LENGTH)).expect("import");
        black_box(&seed);
      }

      pub(super) fn needles() {
        let (_, key) = <$profile>::keypair_from_seed_in(&MLDSA_PKCS8_SEED, &HEAP).expect("key generation");
        let secret = key.expose_secret();
        let bytes = secret.as_bytes();
        needle("seed", &MLDSA_PKCS8_SEED);
        // sk = rho || K || tr || s1 || s2 || t0; rho and tr are public.
        needle("K", &bytes[32..64]);
        needle("s1_s2_t0", &bytes[128..]);
      }
    }
  };
}

mldsa_pkcs8!(mldsa44_pkcs8, MlDsa44, MlDsa44SecretKey, MlDsa44Seed);
mldsa_pkcs8!(mldsa65_pkcs8, MlDsa65, MlDsa65SecretKey, MlDsa65Seed);
mldsa_pkcs8!(mldsa87_pkcs8, MlDsa87, MlDsa87SecretKey, MlDsa87Seed);

/// Distinct SK.seed || SK.prf || PK.seed bytes, so each field is a usable needle.
const SLHDSA_SEEDS: [u8; 96] = {
  let mut bytes = [0u8; 96];
  let mut state = 0x7F4A_7C15_u32;
  let mut index = 0;
  while index < bytes.len() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    bytes[index] = state.to_le_bytes()[1];
    index += 1;
  }
  bytes
};
/// Hedged-signing randomness, also a needle.
const SLHDSA_ADDRND: [u8; 32] = [
  0x5c, 0xed, 0xc8, 0x34, 0x10, 0x82, 0x27, 0x6b, 0xf3, 0xa2, 0x72, 0x39, 0x5a, 0x6c, 0xd4, 0x1e, 0x8b, 0x0f, 0x31,
  0xc7, 0x9e, 0x37, 0x79, 0xb9, 0x7f, 0x4a, 0x7c, 0x15, 0xf3, 0x9c, 0xc0, 0x60,
];
const SLHDSA_MESSAGE: &[u8] = b"rscrypto SLH-DSA residue";

/// Signature output outside every scanned region, so the caller's buffer is
/// not counted as operation stack. The largest signature is SLH-DSA-*-256f.
struct SignatureBuffer(UnsafeCell<[u8; 49_856]>);

// SAFETY: the harness runs on one core with no interrupts.
unsafe impl Sync for SignatureBuffer {}

static SLHDSA_SIGNATURE: SignatureBuffer = SignatureBuffer(UnsafeCell::new([0; 49_856]));

fn slhdsa_signature<const N: usize>() -> &'static mut [u8; N] {
  // SAFETY: used by one scenario function at a time, with no other live
  // reference into the buffer.
  let buffer = unsafe { &mut *SLHDSA_SIGNATURE.0.get() };
  buffer.first_chunk_mut().expect("the buffer holds every signature")
}

/// A key built by `prepare`, outside every scanned region.
struct KeyCell<T>(UnsafeCell<Option<T>>);

// SAFETY: the harness runs on one core with no interrupts.
unsafe impl<T> Sync for KeyCell<T> {}

impl<T> KeyCell<T> {
  const fn new() -> Self {
    Self(UnsafeCell::new(None))
  }

  fn set(&self, value: T) {
    // SAFETY: written only by a `prepare` function, never while borrowed.
    unsafe { *self.0.get() = Some(value) };
  }

  fn get(&self) -> &T {
    // SAFETY: read only after `prepare` has set it, never while written.
    unsafe { (*self.0.get()).as_ref().expect("prepared key") }
  }
}

macro_rules! slhdsa {
  ($module:ident, $profile:ident, $secret:ident, $hash_secret:ident, $n:literal, $digest:literal) => {
    mod $module {
      use super::*;
      use rscrypto::{$hash_secret, $profile, $secret};

      static SECRET: KeyCell<$secret> = KeyCell::new();
      static HASH_SECRET: KeyCell<$hash_secret> = KeyCell::new();

      fn seeds(out: &mut [u8]) -> Result<(), rscrypto::SlhDsaError> {
        out.copy_from_slice(black_box(&SLHDSA_SEEDS[..3 * $n]));
        Ok(())
      }

      pub(super) fn keygen() {
        let keys = $profile::generate_keypair(seeds).expect("key generation");
        black_box(&keys);
      }

      pub(super) fn prepare_import() {
        let (_, key) = $profile::generate_keypair(seeds).expect("key generation");
        // SAFETY: no reference into KEY_INPUT is live while it is written.
        unsafe { (&mut *KEY_INPUT.0.get())[..$secret::LENGTH].copy_from_slice(key.expose_secret().as_bytes()) };
      }

      pub(super) fn import() {
        let key = $secret::try_from_slice(key_input($secret::LENGTH)).expect("import");
        black_box(&key);
      }

      pub(super) fn prepare_pkcs8() {
        let (_, key) = $profile::generate_keypair(seeds).expect("key generation");
        key.to_pkcs8_der_into(key_input_mut());
      }

      pub(super) fn pkcs8() {
        let key = $secret::from_pkcs8_der(key_input($secret::PKCS8_DER_LENGTH)).expect("import");
        black_box(&key);
      }

      pub(super) fn prepare_keys() {
        let (_, key) = $profile::generate_keypair(seeds).expect("key generation");
        HASH_SECRET.set($hash_secret::try_from_slice(key.expose_secret().as_bytes()).expect("import"));
        SECRET.set(key);
      }

      pub(super) fn sign() {
        let signature = slhdsa_signature::<{ $profile::SIGNATURE_LENGTH }>();
        let result = SECRET
          .get()
          .sign_deterministic(black_box(SLHDSA_MESSAGE), b"", signature);
        black_box(&result);
      }

      pub(super) fn sign_hedged() {
        let signature = slhdsa_signature::<{ $profile::SIGNATURE_LENGTH }>();
        let result = SECRET.get().sign_with(
          black_box(SLHDSA_MESSAGE),
          b"",
          |out| {
            out.copy_from_slice(black_box(&SLHDSA_ADDRND[..$n]));
            Ok(())
          },
          signature,
        );
        black_box(&result);
      }

      pub(super) fn sign_prehash() {
        let signature = slhdsa_signature::<{ $profile::SIGNATURE_LENGTH }>();
        let result = HASH_SECRET
          .get()
          .sign_prehash_deterministic(black_box(&[0x42; $digest]), b"", signature);
        black_box(&result);
      }

      pub(super) fn prepare_verify() {
        prepare_keys();
        sign();
      }

      pub(super) fn verify() {
        let signature = slhdsa_signature::<{ $profile::SIGNATURE_LENGTH }>();
        let result = SECRET.get().public_key().verify(black_box(SLHDSA_MESSAGE), signature);
        black_box(&result).expect("verification");
      }

      pub(super) fn needles() {
        needle("sk_seed", &SLHDSA_SEEDS[..$n]);
        needle("sk_prf", &SLHDSA_SEEDS[$n..2 * $n]);
      }

      pub(super) fn hedged_needles() {
        needles();
        needle("addrnd", &SLHDSA_ADDRND[..$n]);
      }
    }
  };
}

slhdsa!(
  slhdsa_sha2_128s,
  SlhDsaSha2_128s,
  SlhDsaSha2_128sSecretKey,
  HashSlhDsaSha2_128sWithSha256SecretKey,
  16,
  32
);
slhdsa!(
  slhdsa_sha2_128f,
  SlhDsaSha2_128f,
  SlhDsaSha2_128fSecretKey,
  HashSlhDsaSha2_128fWithSha256SecretKey,
  16,
  32
);
slhdsa!(
  slhdsa_sha2_192f,
  SlhDsaSha2_192f,
  SlhDsaSha2_192fSecretKey,
  HashSlhDsaSha2_192fWithSha512SecretKey,
  24,
  64
);
slhdsa!(
  slhdsa_sha2_256f,
  SlhDsaSha2_256f,
  SlhDsaSha2_256fSecretKey,
  HashSlhDsaSha2_256fWithSha512SecretKey,
  32,
  64
);
slhdsa!(
  slhdsa_shake_128f,
  SlhDsaShake128f,
  SlhDsaShake128fSecretKey,
  HashSlhDsaShake128fWithShake128SecretKey,
  16,
  32
);
slhdsa!(
  slhdsa_shake_192f,
  SlhDsaShake192f,
  SlhDsaShake192fSecretKey,
  HashSlhDsaShake192fWithShake256SecretKey,
  24,
  64
);
slhdsa!(
  slhdsa_shake_256f,
  SlhDsaShake256f,
  SlhDsaShake256fSecretKey,
  HashSlhDsaShake256fWithShake256SecretKey,
  32,
  64
);

// RFC 5869 A.1 extract inputs. The input key material is caller-owned and
// lives outside every scanned region. The PRK and its HMAC prefix states are
// the owned secrets; `sha2` derives the prefix states independently.
const HKDF_SALT: [u8; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
static HKDF_IKM: [u8; 22] = [0x0b; 22];

const SHA256_H0: [u32; 8] = [
  0x6a09_e667,
  0xbb67_ae85,
  0x3c6e_f372,
  0xa54f_f53a,
  0x510e_527f,
  0x9b05_688c,
  0x1f83_d9ab,
  0x5be0_cd19,
];
const SHA384_H0: [u64; 8] = [
  0xcbbb_9d5d_c105_9ed8,
  0x629a_292a_367c_d507,
  0x9159_015a_3070_dd17,
  0x152f_ecd8_f70e_5939,
  0x6733_2667_ffc0_0b31,
  0x8eb4_4a87_6858_1511,
  0xdb0c_2e0d_64f9_8fa7,
  0x47b5_481d_befa_4fa4,
];
const SHA512_H0: [u64; 8] = [
  0x6a09_e667_f3bc_c908,
  0xbb67_ae85_84ca_a73b,
  0x3c6e_f372_fe94_f82b,
  0xa54f_f53a_5f1d_36f1,
  0x510e_527f_ade6_82d1,
  0x9b05_688c_2b3e_6c1f,
  0x1f83_d9ab_fb41_bd6b,
  0x5be0_cd19_137e_2179,
];

macro_rules! hkdf {
  ($module:ident, $hkdf:ident, $word:ty, $block:literal, $h0:expr, $compress:ident) => {
    mod $module {
      use alloc::vec::Vec;

      use super::*;
      use rscrypto::$hkdf;

      pub(super) fn extract() {
        let hkdf = $hkdf::extract(black_box(&HKDF_SALT), black_box(&HKDF_IKM));
        black_box(&hkdf);
      }

      /// Chaining state after one pad block, as the owner stores its words.
      pub(super) fn prefix(prk: &[u8], pad: u8) -> Vec<u8> {
        let mut block = [pad; $block];
        for (byte, key) in block.iter_mut().zip(prk) {
          *byte ^= key;
        }
        let mut state: [$word; 8] = $h0;
        sha2::block_api::$compress(&mut state, &[block]);
        state.iter().flat_map(|word| word.to_ne_bytes()).collect()
      }

      pub(super) fn needles() {
        let hkdf = $hkdf::extract(&HKDF_SALT, &HKDF_IKM);
        needle("prk", hkdf.prk());
        needle("inner_init", &prefix(hkdf.prk(), 0x36));
        needle("outer_init", &prefix(hkdf.prk(), 0x5c));
      }
    }
  };
}

hkdf!(hkdf_sha256, HkdfSha256, u32, 64, SHA256_H0, compress256);
hkdf!(hkdf_sha384, HkdfSha384, u64, 128, SHA384_H0, compress512);
hkdf!(hkdf_sha512, HkdfSha512, u64, 128, SHA512_H0, compress512);

/// Hedged composite signing randomness (32 bytes with Ed25519, 96 with P-256),
/// also a needle.
const COMPOSITE_RANDOM: [u8; 96] = {
  let mut bytes = [0u8; 96];
  let mut state = 0x2545_F491_u32;
  let mut index = 0;
  while index < bytes.len() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    bytes[index] = state.to_le_bytes()[2];
    index += 1;
  }
  bytes
};
const COMPOSITE_MESSAGE: &[u8] = b"rscrypto composite ML-DSA residue";

/// The Ed25519 seed and its SHA-512 expansion: the clamped scalar and the nonce prefix.
fn ed25519_needles(seed: &[u8]) {
  use sha2::Digest as _;
  let digest = sha2::Sha512::digest(seed);
  let mut scalar = [0u8; 32];
  scalar.copy_from_slice(&digest[..32]);
  scalar[0] &= 248;
  scalar[31] &= 127;
  scalar[31] |= 64;
  needle("ed25519_seed", seed);
  needle("ed25519_scalar", &scalar);
  needle("ed25519_prefix", &digest[32..]);
}

/// The P-256 scalar of the draft's `ECPrivateKey` form, after its 7-byte
/// header, and the scalar as signing holds it: little-endian 64-bit limbs,
/// which on these little-endian boards are the reversed big-endian bytes.
fn p256_needles(ec_private_key: &[u8]) {
  let scalar = &ec_private_key[7..39];
  let mut limbs = [0u8; 32];
  limbs.copy_from_slice(scalar);
  limbs.reverse();
  needle("p256_scalar", scalar);
  needle("p256_scalar_limbs", &limbs);
}

macro_rules! composite {
  ($module:ident, $profile:ident, $secret:ident, $seed:literal, $random:literal, $traditional:path) => {
    mod $module {
      use super::*;
      use rscrypto::{$profile, $secret};

      static SECRET: KeyCell<$secret> = KeyCell::new();

      pub(super) fn keygen() {
        let keys = $profile::generate_keypair(stream_fill($seed)).expect("key generation");
        black_box(&keys);
      }

      pub(super) fn keygen_in() {
        let keys = $profile::generate_keypair_in(stream_fill($seed), &ARENA).expect("key generation");
        black_box(&keys);
      }

      pub(super) fn prepare_import() {
        let (_, key) = $profile::generate_keypair_in(stream_fill($seed), &HEAP).expect("key generation");
        key_input_mut::<{ $secret::LENGTH }>().copy_from_slice(key.expose_secret().as_bytes());
      }

      pub(super) fn import() {
        let key = $secret::try_from_slice(key_input($secret::LENGTH)).expect("import");
        black_box(&key);
      }

      pub(super) fn import_in() {
        let key = $secret::try_from_slice_in(key_input($secret::LENGTH), &ARENA).expect("import");
        black_box(&key);
      }

      pub(super) fn prepare_pkcs8() {
        let (_, key) = $profile::generate_keypair_in(stream_fill($seed), &HEAP).expect("key generation");
        key.to_pkcs8_der_into(key_input_mut());
      }

      pub(super) fn pkcs8() {
        let key = $secret::from_pkcs8_der(key_input($secret::PKCS8_DER_LENGTH)).expect("import");
        black_box(&key);
      }

      pub(super) fn pkcs8_in() {
        let key = $secret::from_pkcs8_der_in(key_input($secret::PKCS8_DER_LENGTH), &ARENA).expect("import");
        black_box(&key);
      }

      /// The key lives outside every scanned region, as a boxed key does.
      pub(super) fn prepare_sign() {
        let (_, key) = $profile::generate_keypair(stream_fill($seed)).expect("key generation");
        SECRET.set(key);
      }

      pub(super) fn sign() {
        let signature = SECRET.get().sign_deterministic(black_box(COMPOSITE_MESSAGE), b"");
        black_box(&signature).as_ref().expect("signature");
      }

      pub(super) fn sign_hedged() {
        let signature = SECRET.get().sign_with(black_box(COMPOSITE_MESSAGE), b"", |out| {
          out.copy_from_slice(black_box(&COMPOSITE_RANDOM[..$random]));
          Ok(())
        });
        black_box(&signature).as_ref().expect("signature");
      }

      pub(super) fn needles() {
        let (_, key) = $profile::generate_keypair_in(stream_fill($seed), &HEAP).expect("key generation");
        let raw = key.expose_secret();
        let (seed, traditional) = raw.as_bytes().split_at(32);
        needle("mldsa_seed", seed);
        let (_, expanded) =
          MlDsa65::keypair_from_seed_in(seed.try_into().expect("seed"), &HEAP).expect("key generation");
        let expanded = expanded.expose_secret();
        // sk = rho || K || tr || s1 || s2 || t0; rho and tr are public.
        needle("mldsa_K", &expanded.as_bytes()[32..64]);
        needle("mldsa_s1_s2_t0", &expanded.as_bytes()[128..]);
        $traditional(traditional);
      }

      pub(super) fn hedged_needles() {
        needles();
        needle("random", &COMPOSITE_RANDOM[..$random]);
      }
    }
  };
}

composite!(
  composite_ed25519,
  MlDsa65Ed25519,
  MlDsa65Ed25519SecretKey,
  0x6E25_5190,
  32,
  ed25519_needles
);
composite!(
  composite_p256,
  MlDsa65EcdsaP256,
  MlDsa65EcdsaP256SecretKey,
  0x0256_ECD5,
  96,
  p256_needles
);

/// `N` distinct, non-repeating bytes from a xorshift stream, for needles that
/// live outside every scanned region.
const fn stream_bytes<const N: usize>(seed: u32) -> [u8; N] {
  let mut bytes = [0u8; N];
  let mut state = seed;
  let mut index = 0;
  while index < N {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    bytes[index] = state.to_le_bytes()[1];
    index = index.strict_add(1);
  }
  bytes
}

/// BLAKE3 key and four 1-KiB chunks of input, so hashing runs two parent
/// compressions below the root. The derive-key scenario treats the input as
/// key material.
const BLAKE3_KEY: [u8; 32] = stream_bytes(0xB1A3_E001);
static BLAKE3_INPUT: [u8; 4096] = stream_bytes(0xB1A3_E002);
const BLAKE3_CONTEXT: &str = "rscrypto residue harness BLAKE3 derive_key";

mod blake3_keyed {
  use super::*;
  use blake3::hazmat::HasherExt as _;

  pub(super) fn keyed_hash() {
    let hash = rscrypto::Blake3::keyed_digest(black_box(&BLAKE3_KEY), black_box(&BLAKE3_INPUT));
    black_box(&hash);
  }

  pub(super) fn derive_key() {
    let key = rscrypto::Blake3::derive_key(black_box(BLAKE3_CONTEXT), black_box(&BLAKE3_INPUT));
    black_box(&key);
  }

  /// The first chunk's chaining value, from the reference implementation.
  fn first_chunk_cv(mut hasher: blake3::Hasher) -> [u8; 32] {
    hasher.set_input_offset(0);
    hasher.update(&BLAKE3_INPUT[..1024]);
    hasher.finalize_non_root()
  }

  pub(super) fn keyed_needles() {
    needle("key", &BLAKE3_KEY);
    needle("chunk0_cv", &first_chunk_cv(blake3::Hasher::new_keyed(&BLAKE3_KEY)));
  }

  pub(super) fn derive_key_needles() {
    let context_key = blake3::hazmat::hash_derive_key_context(BLAKE3_CONTEXT);
    needle("key_material_block0", &BLAKE3_INPUT[..64]);
    needle(
      "chunk0_cv",
      &first_chunk_cv(blake3::Hasher::new_from_context_key(&context_key)),
    );
  }
}

/// HMAC keys longer than the block (64 bytes for SHA-256, 128 for SHA-512),
/// which HMAC hashes into its key block first.
static HMAC_LONG_KEY: [u8; 200] = stream_bytes(0x4D4A_C001);
const HMAC_MESSAGE: &[u8] = b"rscrypto HMAC long-key residue";

mod hmac_long_key {
  use super::*;
  use sha2::Digest as _;

  pub(super) fn sha256() {
    let tag = rscrypto::HmacSha256::mac(black_box(&HMAC_LONG_KEY[..100]), black_box(HMAC_MESSAGE));
    black_box(&tag);
  }

  pub(super) fn sha512() {
    let tag = rscrypto::HmacSha512::mac(black_box(&HMAC_LONG_KEY), black_box(HMAC_MESSAGE));
    black_box(&tag);
  }

  pub(super) fn sha256_needles() {
    needle("key", &HMAC_LONG_KEY[..100]);
    needle("key_digest", &sha2::Sha256::digest(&HMAC_LONG_KEY[..100]));
  }

  pub(super) fn sha512_needles() {
    needle("key", &HMAC_LONG_KEY);
    needle("key_digest", &sha2::Sha512::digest(HMAC_LONG_KEY));
  }
}

macro_rules! scenario {
  ($name:literal, $expect:literal, $run:path, $needles:path) => {
    scenario!($name, $expect, nothing, $run, $needles)
  };
  ($name:literal, $expect:literal, $prepare:path, $run:path, $needles:path) => {
    Scenario {
      name: $name,
      expect: $expect,
      prepare: $prepare,
      run: $run,
      needles: $needles,
    }
  };
}

const SCENARIOS: &[Scenario] = &[
  scenario!("control-planted-stack", "stack", planted_stack, control_needles),
  scenario!("control-planted-arena", "arena", planted_arena, control_needles),
  scenario!("control-cleared-stack", "none", cleared_stack, control_needles),
  scenario!("ml-kem-512-keygen", "report", mlkem512::keygen, mlkem512::needles),
  scenario!("ml-kem-512-keygen-in", "none", mlkem512::keygen_in, mlkem512::needles),
  scenario!(
    "ml-kem-512-import",
    "report",
    mlkem512::prepare,
    mlkem512::import,
    mlkem512::needles
  ),
  scenario!(
    "ml-kem-512-import-in",
    "none",
    mlkem512::prepare,
    mlkem512::import_in,
    mlkem512::needles
  ),
  scenario!("ml-kem-768-keygen", "report", mlkem768::keygen, mlkem768::needles),
  scenario!("ml-kem-768-keygen-in", "none", mlkem768::keygen_in, mlkem768::needles),
  scenario!(
    "ml-kem-768-import",
    "report",
    mlkem768::prepare,
    mlkem768::import,
    mlkem768::needles
  ),
  scenario!(
    "ml-kem-768-import-in",
    "none",
    mlkem768::prepare,
    mlkem768::import_in,
    mlkem768::needles
  ),
  scenario!("ml-kem-1024-keygen", "report", mlkem1024::keygen, mlkem1024::needles),
  scenario!(
    "ml-kem-1024-keygen-in",
    "none",
    mlkem1024::keygen_in,
    mlkem1024::needles
  ),
  scenario!(
    "ml-kem-1024-import",
    "report",
    mlkem1024::prepare,
    mlkem1024::import,
    mlkem1024::needles
  ),
  scenario!(
    "ml-kem-1024-import-in",
    "none",
    mlkem1024::prepare,
    mlkem1024::import_in,
    mlkem1024::needles
  ),
  scenario!(
    "ml-kem-512-pkcs8-seed-in",
    "none",
    mlkem512::prepare_pkcs8_seed,
    mlkem512::pkcs8_seed_in,
    mlkem512::needles
  ),
  scenario!(
    "ml-kem-512-pkcs8-expanded",
    "report",
    mlkem512::prepare_pkcs8_expanded,
    mlkem512::pkcs8_expanded,
    mlkem512::needles
  ),
  scenario!(
    "ml-kem-512-pkcs8-expanded-in",
    "none",
    mlkem512::prepare_pkcs8_expanded,
    mlkem512::pkcs8_expanded_in,
    mlkem512::needles
  ),
  scenario!(
    "ml-kem-512-pkcs8-seed-owner",
    "report",
    mlkem512::prepare_pkcs8_seed,
    mlkem512::seed_owner,
    mlkem512::needles
  ),
  scenario!(
    "ml-kem-768-pkcs8-seed-in",
    "none",
    mlkem768::prepare_pkcs8_seed,
    mlkem768::pkcs8_seed_in,
    mlkem768::needles
  ),
  scenario!(
    "ml-kem-768-pkcs8-expanded",
    "report",
    mlkem768::prepare_pkcs8_expanded,
    mlkem768::pkcs8_expanded,
    mlkem768::needles
  ),
  scenario!(
    "ml-kem-768-pkcs8-expanded-in",
    "none",
    mlkem768::prepare_pkcs8_expanded,
    mlkem768::pkcs8_expanded_in,
    mlkem768::needles
  ),
  scenario!(
    "ml-kem-768-pkcs8-seed-owner",
    "report",
    mlkem768::prepare_pkcs8_seed,
    mlkem768::seed_owner,
    mlkem768::needles
  ),
  scenario!(
    "ml-kem-1024-pkcs8-seed-in",
    "none",
    mlkem1024::prepare_pkcs8_seed,
    mlkem1024::pkcs8_seed_in,
    mlkem1024::needles
  ),
  scenario!(
    "ml-kem-1024-pkcs8-expanded",
    "report",
    mlkem1024::prepare_pkcs8_expanded,
    mlkem1024::pkcs8_expanded,
    mlkem1024::needles
  ),
  scenario!(
    "ml-kem-1024-pkcs8-expanded-in",
    "none",
    mlkem1024::prepare_pkcs8_expanded,
    mlkem1024::pkcs8_expanded_in,
    mlkem1024::needles
  ),
  scenario!(
    "ml-kem-1024-pkcs8-seed-owner",
    "report",
    mlkem1024::prepare_pkcs8_seed,
    mlkem1024::seed_owner,
    mlkem1024::needles
  ),
  scenario!("x-wing-keygen", "none", xwing::keygen, xwing::needles),
  scenario!(
    "x-wing-prepare-in",
    "none",
    xwing::prepare,
    xwing::prepare_in,
    xwing::needles
  ),
  scenario!(
    "x-wing-decapsulate",
    "none",
    xwing::prepare,
    xwing::decapsulate,
    xwing::needles
  ),
  scenario!(
    "x-wing-decapsulate-shared-secret",
    "report",
    xwing::prepare,
    xwing::decapsulate,
    xwing::decapsulated_secret_needle
  ),
  scenario!(
    "x-wing-encapsulate",
    "none",
    xwing::encapsulate,
    xwing::encapsulation_needles
  ),
  scenario!(
    "x-wing-encapsulate-shared-secret",
    "report",
    xwing::encapsulate,
    xwing::encapsulated_secret_needle
  ),
  scenario!(
    "hpke-x25519-derive",
    "report",
    hpke::prepare_x25519,
    hpke::x25519_derive,
    hpke::x25519_derive_needles
  ),
  scenario!(
    "hpke-x25519-derive-in",
    "none",
    hpke::prepare_x25519,
    hpke::x25519_derive_in,
    hpke::x25519_derive_needles
  ),
  scenario!(
    "hpke-x25519-sender",
    "report",
    hpke::prepare_x25519,
    hpke::x25519_sender,
    hpke::x25519_sender_needles
  ),
  scenario!(
    "hpke-x25519-sender-in",
    "none",
    hpke::prepare_x25519,
    hpke::x25519_sender_in,
    hpke::x25519_sender_needles
  ),
  scenario!(
    "hpke-x25519-receiver",
    "report",
    hpke::prepare_x25519,
    hpke::x25519_receiver,
    hpke::x25519_receiver_needles
  ),
  scenario!(
    "hpke-x25519-receiver-in",
    "none",
    hpke::prepare_x25519,
    hpke::x25519_receiver_in,
    hpke::x25519_receiver_needles
  ),
  scenario!(
    "hpke-xwing-derive",
    "report",
    hpke::prepare_xwing,
    hpke::xwing_derive,
    hpke::xwing_derive_needles
  ),
  scenario!(
    "hpke-xwing-derive-in",
    "none",
    hpke::prepare_xwing,
    hpke::xwing_derive_in,
    hpke::xwing_derive_needles
  ),
  scenario!(
    "hpke-xwing-sender",
    "report",
    hpke::prepare_xwing,
    hpke::xwing_sender,
    hpke::xwing_sender_needles
  ),
  scenario!(
    "hpke-xwing-sender-in",
    "none",
    hpke::prepare_xwing,
    hpke::xwing_sender_in,
    hpke::xwing_sender_needles
  ),
  scenario!(
    "hpke-xwing-receiver",
    "report",
    hpke::prepare_xwing,
    hpke::xwing_receiver,
    hpke::xwing_receiver_needles
  ),
  scenario!(
    "hpke-xwing-receiver-in",
    "none",
    hpke::prepare_xwing,
    hpke::xwing_receiver_in,
    hpke::xwing_receiver_needles
  ),
  scenario!("ml-dsa-44-keygen", "report", mldsa44::keygen, mldsa44::needles),
  scenario!("ml-dsa-44-keygen-in", "none", mldsa44::keygen_in, mldsa44::needles),
  scenario!("ml-dsa-65-keygen", "report", mldsa65::keygen, mldsa65::needles),
  scenario!("ml-dsa-65-keygen-in", "none", mldsa65::keygen_in, mldsa65::needles),
  scenario!("ml-dsa-87-keygen", "report", mldsa87::keygen, mldsa87::needles),
  scenario!("ml-dsa-87-keygen-in", "none", mldsa87::keygen_in, mldsa87::needles),
  scenario!(
    "ml-dsa-44-pkcs8-seed-in",
    "none",
    mldsa44_pkcs8::prepare_seed,
    mldsa44_pkcs8::seed_in,
    mldsa44_pkcs8::needles
  ),
  scenario!(
    "ml-dsa-44-pkcs8-expanded",
    "report",
    mldsa44_pkcs8::prepare_expanded,
    mldsa44_pkcs8::expanded,
    mldsa44_pkcs8::needles
  ),
  scenario!(
    "ml-dsa-44-pkcs8-expanded-in",
    "none",
    mldsa44_pkcs8::prepare_expanded,
    mldsa44_pkcs8::expanded_in,
    mldsa44_pkcs8::needles
  ),
  scenario!(
    "ml-dsa-44-pkcs8-seed-owner",
    "report",
    mldsa44_pkcs8::prepare_seed,
    mldsa44_pkcs8::seed_owner,
    mldsa44_pkcs8::needles
  ),
  scenario!(
    "ml-dsa-65-pkcs8-seed-in",
    "none",
    mldsa65_pkcs8::prepare_seed,
    mldsa65_pkcs8::seed_in,
    mldsa65_pkcs8::needles
  ),
  scenario!(
    "ml-dsa-65-pkcs8-expanded",
    "report",
    mldsa65_pkcs8::prepare_expanded,
    mldsa65_pkcs8::expanded,
    mldsa65_pkcs8::needles
  ),
  scenario!(
    "ml-dsa-65-pkcs8-expanded-in",
    "none",
    mldsa65_pkcs8::prepare_expanded,
    mldsa65_pkcs8::expanded_in,
    mldsa65_pkcs8::needles
  ),
  scenario!(
    "ml-dsa-65-pkcs8-seed-owner",
    "report",
    mldsa65_pkcs8::prepare_seed,
    mldsa65_pkcs8::seed_owner,
    mldsa65_pkcs8::needles
  ),
  scenario!(
    "ml-dsa-87-pkcs8-seed-in",
    "none",
    mldsa87_pkcs8::prepare_seed,
    mldsa87_pkcs8::seed_in,
    mldsa87_pkcs8::needles
  ),
  scenario!(
    "ml-dsa-87-pkcs8-expanded",
    "report",
    mldsa87_pkcs8::prepare_expanded,
    mldsa87_pkcs8::expanded,
    mldsa87_pkcs8::needles
  ),
  scenario!(
    "ml-dsa-87-pkcs8-expanded-in",
    "none",
    mldsa87_pkcs8::prepare_expanded,
    mldsa87_pkcs8::expanded_in,
    mldsa87_pkcs8::needles
  ),
  scenario!(
    "ml-dsa-87-pkcs8-seed-owner",
    "report",
    mldsa87_pkcs8::prepare_seed,
    mldsa87_pkcs8::seed_owner,
    mldsa87_pkcs8::needles
  ),
  scenario!(
    "slh-dsa-sha2-128s-keygen",
    "report",
    slhdsa_sha2_128s::keygen,
    slhdsa_sha2_128s::needles
  ),
  scenario!(
    "slh-dsa-sha2-128s-import",
    "report",
    slhdsa_sha2_128s::prepare_import,
    slhdsa_sha2_128s::import,
    slhdsa_sha2_128s::needles
  ),
  scenario!(
    "slh-dsa-sha2-128s-pkcs8",
    "report",
    slhdsa_sha2_128s::prepare_pkcs8,
    slhdsa_sha2_128s::pkcs8,
    slhdsa_sha2_128s::needles
  ),
  scenario!(
    "slh-dsa-sha2-128s-sign",
    "none",
    slhdsa_sha2_128s::prepare_keys,
    slhdsa_sha2_128s::sign,
    slhdsa_sha2_128s::needles
  ),
  scenario!(
    "slh-dsa-sha2-128s-sign-hedged",
    "none",
    slhdsa_sha2_128s::prepare_keys,
    slhdsa_sha2_128s::sign_hedged,
    slhdsa_sha2_128s::hedged_needles
  ),
  scenario!(
    "slh-dsa-sha2-128s-sign-prehash",
    "none",
    slhdsa_sha2_128s::prepare_keys,
    slhdsa_sha2_128s::sign_prehash,
    slhdsa_sha2_128s::needles
  ),
  scenario!(
    "slh-dsa-sha2-128s-verify",
    "none",
    slhdsa_sha2_128s::prepare_verify,
    slhdsa_sha2_128s::verify,
    slhdsa_sha2_128s::needles
  ),
  scenario!(
    "slh-dsa-sha2-128f-keygen",
    "report",
    slhdsa_sha2_128f::keygen,
    slhdsa_sha2_128f::needles
  ),
  scenario!(
    "slh-dsa-sha2-128f-import",
    "report",
    slhdsa_sha2_128f::prepare_import,
    slhdsa_sha2_128f::import,
    slhdsa_sha2_128f::needles
  ),
  scenario!(
    "slh-dsa-sha2-128f-pkcs8",
    "report",
    slhdsa_sha2_128f::prepare_pkcs8,
    slhdsa_sha2_128f::pkcs8,
    slhdsa_sha2_128f::needles
  ),
  scenario!(
    "slh-dsa-sha2-128f-sign",
    "none",
    slhdsa_sha2_128f::prepare_keys,
    slhdsa_sha2_128f::sign,
    slhdsa_sha2_128f::needles
  ),
  scenario!(
    "slh-dsa-sha2-128f-sign-hedged",
    "none",
    slhdsa_sha2_128f::prepare_keys,
    slhdsa_sha2_128f::sign_hedged,
    slhdsa_sha2_128f::hedged_needles
  ),
  scenario!(
    "slh-dsa-sha2-128f-sign-prehash",
    "none",
    slhdsa_sha2_128f::prepare_keys,
    slhdsa_sha2_128f::sign_prehash,
    slhdsa_sha2_128f::needles
  ),
  scenario!(
    "slh-dsa-sha2-128f-verify",
    "none",
    slhdsa_sha2_128f::prepare_verify,
    slhdsa_sha2_128f::verify,
    slhdsa_sha2_128f::needles
  ),
  scenario!(
    "slh-dsa-sha2-192f-keygen",
    "report",
    slhdsa_sha2_192f::keygen,
    slhdsa_sha2_192f::needles
  ),
  scenario!(
    "slh-dsa-sha2-192f-import",
    "report",
    slhdsa_sha2_192f::prepare_import,
    slhdsa_sha2_192f::import,
    slhdsa_sha2_192f::needles
  ),
  scenario!(
    "slh-dsa-sha2-192f-pkcs8",
    "report",
    slhdsa_sha2_192f::prepare_pkcs8,
    slhdsa_sha2_192f::pkcs8,
    slhdsa_sha2_192f::needles
  ),
  scenario!(
    "slh-dsa-sha2-192f-sign",
    "none",
    slhdsa_sha2_192f::prepare_keys,
    slhdsa_sha2_192f::sign,
    slhdsa_sha2_192f::needles
  ),
  scenario!(
    "slh-dsa-sha2-192f-sign-hedged",
    "none",
    slhdsa_sha2_192f::prepare_keys,
    slhdsa_sha2_192f::sign_hedged,
    slhdsa_sha2_192f::hedged_needles
  ),
  scenario!(
    "slh-dsa-sha2-192f-sign-prehash",
    "none",
    slhdsa_sha2_192f::prepare_keys,
    slhdsa_sha2_192f::sign_prehash,
    slhdsa_sha2_192f::needles
  ),
  scenario!(
    "slh-dsa-sha2-192f-verify",
    "none",
    slhdsa_sha2_192f::prepare_verify,
    slhdsa_sha2_192f::verify,
    slhdsa_sha2_192f::needles
  ),
  scenario!(
    "slh-dsa-sha2-256f-keygen",
    "report",
    slhdsa_sha2_256f::keygen,
    slhdsa_sha2_256f::needles
  ),
  scenario!(
    "slh-dsa-sha2-256f-import",
    "report",
    slhdsa_sha2_256f::prepare_import,
    slhdsa_sha2_256f::import,
    slhdsa_sha2_256f::needles
  ),
  scenario!(
    "slh-dsa-sha2-256f-pkcs8",
    "report",
    slhdsa_sha2_256f::prepare_pkcs8,
    slhdsa_sha2_256f::pkcs8,
    slhdsa_sha2_256f::needles
  ),
  scenario!(
    "slh-dsa-sha2-256f-sign",
    "none",
    slhdsa_sha2_256f::prepare_keys,
    slhdsa_sha2_256f::sign,
    slhdsa_sha2_256f::needles
  ),
  scenario!(
    "slh-dsa-sha2-256f-sign-hedged",
    "none",
    slhdsa_sha2_256f::prepare_keys,
    slhdsa_sha2_256f::sign_hedged,
    slhdsa_sha2_256f::hedged_needles
  ),
  scenario!(
    "slh-dsa-sha2-256f-sign-prehash",
    "none",
    slhdsa_sha2_256f::prepare_keys,
    slhdsa_sha2_256f::sign_prehash,
    slhdsa_sha2_256f::needles
  ),
  scenario!(
    "slh-dsa-sha2-256f-verify",
    "none",
    slhdsa_sha2_256f::prepare_verify,
    slhdsa_sha2_256f::verify,
    slhdsa_sha2_256f::needles
  ),
  scenario!(
    "slh-dsa-shake-128f-keygen",
    "report",
    slhdsa_shake_128f::keygen,
    slhdsa_shake_128f::needles
  ),
  scenario!(
    "slh-dsa-shake-128f-import",
    "report",
    slhdsa_shake_128f::prepare_import,
    slhdsa_shake_128f::import,
    slhdsa_shake_128f::needles
  ),
  scenario!(
    "slh-dsa-shake-128f-pkcs8",
    "report",
    slhdsa_shake_128f::prepare_pkcs8,
    slhdsa_shake_128f::pkcs8,
    slhdsa_shake_128f::needles
  ),
  scenario!(
    "slh-dsa-shake-128f-sign",
    "none",
    slhdsa_shake_128f::prepare_keys,
    slhdsa_shake_128f::sign,
    slhdsa_shake_128f::needles
  ),
  scenario!(
    "slh-dsa-shake-128f-sign-hedged",
    "none",
    slhdsa_shake_128f::prepare_keys,
    slhdsa_shake_128f::sign_hedged,
    slhdsa_shake_128f::hedged_needles
  ),
  scenario!(
    "slh-dsa-shake-128f-sign-prehash",
    "none",
    slhdsa_shake_128f::prepare_keys,
    slhdsa_shake_128f::sign_prehash,
    slhdsa_shake_128f::needles
  ),
  scenario!(
    "slh-dsa-shake-128f-verify",
    "none",
    slhdsa_shake_128f::prepare_verify,
    slhdsa_shake_128f::verify,
    slhdsa_shake_128f::needles
  ),
  scenario!(
    "slh-dsa-shake-192f-keygen",
    "report",
    slhdsa_shake_192f::keygen,
    slhdsa_shake_192f::needles
  ),
  scenario!(
    "slh-dsa-shake-192f-import",
    "report",
    slhdsa_shake_192f::prepare_import,
    slhdsa_shake_192f::import,
    slhdsa_shake_192f::needles
  ),
  scenario!(
    "slh-dsa-shake-192f-pkcs8",
    "report",
    slhdsa_shake_192f::prepare_pkcs8,
    slhdsa_shake_192f::pkcs8,
    slhdsa_shake_192f::needles
  ),
  scenario!(
    "slh-dsa-shake-192f-sign",
    "none",
    slhdsa_shake_192f::prepare_keys,
    slhdsa_shake_192f::sign,
    slhdsa_shake_192f::needles
  ),
  scenario!(
    "slh-dsa-shake-192f-sign-hedged",
    "none",
    slhdsa_shake_192f::prepare_keys,
    slhdsa_shake_192f::sign_hedged,
    slhdsa_shake_192f::hedged_needles
  ),
  scenario!(
    "slh-dsa-shake-192f-sign-prehash",
    "none",
    slhdsa_shake_192f::prepare_keys,
    slhdsa_shake_192f::sign_prehash,
    slhdsa_shake_192f::needles
  ),
  scenario!(
    "slh-dsa-shake-192f-verify",
    "none",
    slhdsa_shake_192f::prepare_verify,
    slhdsa_shake_192f::verify,
    slhdsa_shake_192f::needles
  ),
  scenario!(
    "slh-dsa-shake-256f-keygen",
    "report",
    slhdsa_shake_256f::keygen,
    slhdsa_shake_256f::needles
  ),
  scenario!(
    "slh-dsa-shake-256f-import",
    "report",
    slhdsa_shake_256f::prepare_import,
    slhdsa_shake_256f::import,
    slhdsa_shake_256f::needles
  ),
  scenario!(
    "slh-dsa-shake-256f-pkcs8",
    "report",
    slhdsa_shake_256f::prepare_pkcs8,
    slhdsa_shake_256f::pkcs8,
    slhdsa_shake_256f::needles
  ),
  scenario!(
    "slh-dsa-shake-256f-sign",
    "none",
    slhdsa_shake_256f::prepare_keys,
    slhdsa_shake_256f::sign,
    slhdsa_shake_256f::needles
  ),
  scenario!(
    "slh-dsa-shake-256f-sign-hedged",
    "none",
    slhdsa_shake_256f::prepare_keys,
    slhdsa_shake_256f::sign_hedged,
    slhdsa_shake_256f::hedged_needles
  ),
  scenario!(
    "slh-dsa-shake-256f-sign-prehash",
    "none",
    slhdsa_shake_256f::prepare_keys,
    slhdsa_shake_256f::sign_prehash,
    slhdsa_shake_256f::needles
  ),
  scenario!(
    "slh-dsa-shake-256f-verify",
    "none",
    slhdsa_shake_256f::prepare_verify,
    slhdsa_shake_256f::verify,
    slhdsa_shake_256f::needles
  ),
  scenario!(
    "hkdf-sha256-extract",
    "none",
    hkdf_sha256::extract,
    hkdf_sha256::needles
  ),
  scenario!(
    "hkdf-sha384-extract",
    "none",
    hkdf_sha384::extract,
    hkdf_sha384::needles
  ),
  scenario!(
    "hkdf-sha512-extract",
    "none",
    hkdf_sha512::extract,
    hkdf_sha512::needles
  ),
  scenario!(
    "blake3-keyed-hash",
    "none",
    blake3_keyed::keyed_hash,
    blake3_keyed::keyed_needles
  ),
  scenario!(
    "blake3-derive-key",
    "none",
    blake3_keyed::derive_key,
    blake3_keyed::derive_key_needles
  ),
  scenario!(
    "hmac-sha256-long-key",
    "none",
    hmac_long_key::sha256,
    hmac_long_key::sha256_needles
  ),
  scenario!(
    "hmac-sha512-long-key",
    "none",
    hmac_long_key::sha512,
    hmac_long_key::sha512_needles
  ),
  scenario!(
    "mldsa65-ed25519-keygen",
    "report",
    composite_ed25519::keygen,
    composite_ed25519::needles
  ),
  scenario!(
    "mldsa65-ed25519-keygen-in",
    "none",
    composite_ed25519::keygen_in,
    composite_ed25519::needles
  ),
  scenario!(
    "mldsa65-ed25519-import",
    "report",
    composite_ed25519::prepare_import,
    composite_ed25519::import,
    composite_ed25519::needles
  ),
  scenario!(
    "mldsa65-ed25519-import-in",
    "none",
    composite_ed25519::prepare_import,
    composite_ed25519::import_in,
    composite_ed25519::needles
  ),
  scenario!(
    "mldsa65-ed25519-pkcs8",
    "report",
    composite_ed25519::prepare_pkcs8,
    composite_ed25519::pkcs8,
    composite_ed25519::needles
  ),
  scenario!(
    "mldsa65-ed25519-pkcs8-in",
    "none",
    composite_ed25519::prepare_pkcs8,
    composite_ed25519::pkcs8_in,
    composite_ed25519::needles
  ),
  scenario!(
    "mldsa65-ed25519-sign",
    "none",
    composite_ed25519::prepare_sign,
    composite_ed25519::sign,
    composite_ed25519::needles
  ),
  scenario!(
    "mldsa65-ed25519-sign-hedged",
    "none",
    composite_ed25519::prepare_sign,
    composite_ed25519::sign_hedged,
    composite_ed25519::hedged_needles
  ),
  scenario!(
    "mldsa65-ecdsa-p256-keygen",
    "report",
    composite_p256::keygen,
    composite_p256::needles
  ),
  scenario!(
    "mldsa65-ecdsa-p256-keygen-in",
    "none",
    composite_p256::keygen_in,
    composite_p256::needles
  ),
  scenario!(
    "mldsa65-ecdsa-p256-import",
    "report",
    composite_p256::prepare_import,
    composite_p256::import,
    composite_p256::needles
  ),
  scenario!(
    "mldsa65-ecdsa-p256-import-in",
    "none",
    composite_p256::prepare_import,
    composite_p256::import_in,
    composite_p256::needles
  ),
  scenario!(
    "mldsa65-ecdsa-p256-pkcs8",
    "report",
    composite_p256::prepare_pkcs8,
    composite_p256::pkcs8,
    composite_p256::needles
  ),
  scenario!(
    "mldsa65-ecdsa-p256-pkcs8-in",
    "none",
    composite_p256::prepare_pkcs8,
    composite_p256::pkcs8_in,
    composite_p256::needles
  ),
  scenario!(
    "mldsa65-ecdsa-p256-sign",
    "none",
    composite_p256::prepare_sign,
    composite_p256::sign,
    composite_p256::needles
  ),
  scenario!(
    "mldsa65-ecdsa-p256-sign-hedged",
    "none",
    composite_p256::prepare_sign,
    composite_p256::sign_hedged,
    composite_p256::hedged_needles
  ),
];

#[unsafe(no_mangle)]
extern "C" fn main() -> ! {
  board::init();
  let backend = if cfg!(feature = "portable-only") {
    "portable"
  } else {
    "native"
  };
  say!(
    "residue 1 board={} backend={backend} scenarios={}",
    board::NAME,
    SCENARIOS.len()
  );
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
#[cfg(not(feature = "native-mldsa"))]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
  say!("panic {info}");
  loop {
    core::hint::spin_loop();
  }
}
