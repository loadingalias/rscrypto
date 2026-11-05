//! Measures the stack that each declared libc edge uses below its entry stack pointer.
//!
//! `scripts/stack/frames.py` stops its owned bounds at the libc imports
//! `calloc`, `malloc`, `free`, `memcpy`, `memmove`, `memset`, `memcmp`, `bcmp`,
//! and `getauxval`. This program calls each one directly on a painted private
//! stack, through the same dynamically resolved symbol a Rust executable
//! binds, across sizes, alignments, allocator states, threads, and failure
//! returns. It prints the deepest use for each edge as JSON. The driver,
//! `scripts/stack/painted.py`, adds the host and libc identity.
//!
//! Every allocator call in the workload runs on the painted stack, so each
//! state transition the workload reaches is measured on its first occurrence.
//! A raw pthread makes the first allocator calls of a fresh thread, before any
//! other code on that thread allocates.

#[cfg(target_os = "linux")]
mod linux {
  use core::ffi::{CStr, c_char, c_void};

  use rscrypto_stack_paint::{GuardedStack, Painted, json_string};

  const STACK_BYTES: usize = 128 * 1024;
  const PAINTS: [u8; 2] = [0xa5, 0x5a];

  unsafe extern "C" {
    // The libc crate does not declare bcmp; glibc exports it.
    fn bcmp(left: *const c_void, right: *const c_void, length: usize) -> i32;
  }

  #[derive(Clone, Copy, PartialEq, Eq)]
  enum Edge {
    Calloc,
    Malloc,
    Free,
    Memcpy,
    Memmove,
    Memset,
    Memcmp,
    Bcmp,
    Getauxval,
  }

  const EDGES: [Edge; 9] = [
    Edge::Calloc,
    Edge::Malloc,
    Edge::Free,
    Edge::Memcpy,
    Edge::Memmove,
    Edge::Memset,
    Edge::Memcmp,
    Edge::Bcmp,
    Edge::Getauxval,
  ];

  impl Edge {
    const fn name(self) -> &'static str {
      match self {
        Self::Calloc => "calloc",
        Self::Malloc => "malloc",
        Self::Free => "free",
        Self::Memcpy => "memcpy",
        Self::Memmove => "memmove",
        Self::Memset => "memset",
        Self::Memcmp => "memcmp",
        Self::Bcmp => "bcmp",
        Self::Getauxval => "getauxval",
      }
    }

    fn function(self) -> *const () {
      let calloc: unsafe extern "C" fn(usize, usize) -> *mut c_void = libc::calloc;
      let malloc: unsafe extern "C" fn(usize) -> *mut c_void = libc::malloc;
      let free: unsafe extern "C" fn(*mut c_void) = libc::free;
      let memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> *mut c_void = libc::memcpy;
      let memmove: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> *mut c_void = libc::memmove;
      let memset: unsafe extern "C" fn(*mut c_void, i32, usize) -> *mut c_void = libc::memset;
      let memcmp: unsafe extern "C" fn(*const c_void, *const c_void, usize) -> i32 = libc::memcmp;
      let compare: unsafe extern "C" fn(*const c_void, *const c_void, usize) -> i32 = bcmp;
      let getauxval: unsafe extern "C" fn(libc::c_ulong) -> libc::c_ulong = libc::getauxval;
      match self {
        Self::Calloc => calloc as *const (),
        Self::Malloc => malloc as *const (),
        Self::Free => free as *const (),
        Self::Memcpy => memcpy as *const (),
        Self::Memmove => memmove as *const (),
        Self::Memset => memset as *const (),
        Self::Memcmp => memcmp as *const (),
        Self::Bcmp => compare as *const (),
        Self::Getauxval => getauxval as *const (),
      }
    }

    fn index(self) -> usize {
      EDGES.iter().position(|&edge| edge == self).expect("listed edge")
    }
  }

  #[derive(Clone, Copy)]
  struct Deepest {
    bytes: usize,
    used_bytes: usize,
    case: &'static str,
    size: usize,
    calls: u64,
  }

  const NONE: Deepest = Deepest {
    bytes: 0,
    used_bytes: 0,
    case: "",
    size: 0,
    calls: 0,
  };

  /// Painted calls and their per-edge maxima on one thread's private stack.
  struct Meter {
    stack: GuardedStack,
    deepest: [Deepest; 9],
    paint: usize,
  }

  impl Meter {
    fn new() -> Self {
      Self {
        stack: GuardedStack::new(STACK_BYTES),
        deepest: [NONE; 9],
        paint: 0,
      }
    }

    /// Calls `edge` with `arguments` on the painted stack and records its depth.
    ///
    /// # Safety
    ///
    /// The arguments must satisfy the edge's C contract.
    unsafe fn call(&mut self, edge: Edge, case: &'static str, size: usize, arguments: [usize; 3]) -> usize {
      let paint = PAINTS[self.paint];
      // Alternate the two paints.
      self.paint = usize::from(self.paint == 0);
      // SAFETY: each edge is a non-unwinding libc function of up to three
      // integer or pointer arguments; the caller supplies valid arguments. None
      // retains a stack address, and none needs a large stack.
      let painted: Painted = unsafe { self.stack.call(paint, edge.function(), arguments) };
      let entry = &mut self.deepest[edge.index()];
      entry.calls = entry.calls.strict_add(1);
      let bytes = painted.below_entry_bytes();
      if bytes > entry.bytes || entry.case.is_empty() {
        *entry = Deepest {
          bytes,
          used_bytes: painted.used_bytes,
          case,
          size,
          calls: entry.calls,
        };
      }
      painted.result
    }

    fn malloc(&mut self, case: &'static str, size: usize) -> *mut c_void {
      // SAFETY: malloc accepts any size and returns null or a fresh allocation.
      let address = unsafe { self.call(Edge::Malloc, case, size, [size, 0, 0]) };
      core::ptr::with_exposed_provenance_mut(address)
    }

    fn calloc(&mut self, case: &'static str, count: usize, size: usize) -> *mut c_void {
      // SAFETY: calloc accepts any count and size, and returns null on overflow.
      let address = unsafe { self.call(Edge::Calloc, case, count.saturating_mul(size), [count, size, 0]) };
      core::ptr::with_exposed_provenance_mut(address)
    }

    fn free(&mut self, case: &'static str, pointer: *mut c_void) {
      // SAFETY: every pointer passed here is null or a live allocation that this
      // program obtained from malloc or calloc and frees exactly once.
      unsafe { self.call(Edge::Free, case, 0, [pointer.expose_provenance(), 0, 0]) };
    }

    fn merge(&mut self, other: &[Deepest; 9]) {
      for (mine, theirs) in self.deepest.iter_mut().zip(other) {
        let calls = mine.calls.strict_add(theirs.calls);
        if theirs.bytes > mine.bytes || mine.case.is_empty() {
          *mine = *theirs;
        }
        mine.calls = calls;
      }
    }
  }

  /// A deterministic xorshift generator; the workload must be reproducible.
  struct Rng(u64);

  impl Rng {
    fn next(&mut self) -> u64 {
      let mut value = self.0;
      value ^= value << 13;
      value ^= value >> 7;
      value ^= value << 17;
      self.0 = value;
      value
    }

    fn below(&mut self, bound: usize) -> usize {
      let bound = u64::try_from(bound).expect("bound fits u64");
      usize::try_from(self.next().checked_rem(bound).expect("nonzero bound")).expect("index fits usize")
    }

    fn between(&mut self, low: usize, high: usize) -> usize {
      low.strict_add(self.below(high.strict_sub(low).strict_add(1)))
    }
  }

  fn memory(meter: &mut Meter) {
    const SLACK: usize = 4096;
    const LARGEST: usize = 64 << 20;
    let mut source = vec![0u8; LARGEST.strict_add(SLACK)];
    let mut destination = vec![0u8; LARGEST.strict_add(SLACK)];
    let mut sizes: Vec<usize> = (0..=512).collect();
    for shift in 10..=26 {
      let power = 1usize << shift;
      sizes.extend([
        power.strict_sub(1),
        power,
        power.strict_add(1),
        power.strict_add(power / 2),
      ]);
    }
    sizes.retain(|&size| size <= LARGEST);
    let small_offsets = [0usize, 1, 7, 8, 15, 16, 31, 63];
    let large_offsets = [(0usize, 0usize), (1, 0), (0, 1), (17, 33), (63, 1)];
    for &size in &sizes {
      let pairs: Vec<(usize, usize)> = if size <= 128 {
        small_offsets
          .iter()
          .flat_map(|&left| small_offsets.iter().map(move |&right| (left, right)))
          .collect()
      } else {
        large_offsets.to_vec()
      };
      for (source_offset, destination_offset) in pairs {
        let from = source[source_offset..].as_ptr().expose_provenance();
        let to = destination[destination_offset..].as_mut_ptr().expose_provenance();
        // SAFETY: both regions hold `size` bytes inside distinct buffers with
        // `SLACK` bytes beyond every offset; neither overlaps the other.
        unsafe {
          meter.call(Edge::Memcpy, "copy", size, [to, from, size]);
          meter.call(Edge::Memmove, "move, disjoint", size, [to, from, size]);
          meter.call(Edge::Memset, "set, zero", size, [to, 0, size]);
          meter.call(Edge::Memset, "set, nonzero", size, [to, 0xa5, size]);
          meter.call(Edge::Memset, "set, zero", size, [from, 0, size]);
          // Both regions now hold zeros: an equal comparison scans all bytes.
          meter.call(Edge::Memcmp, "compare, equal", size, [to, from, size]);
          meter.call(Edge::Bcmp, "compare, equal", size, [to, from, size]);
        }
        for position in [0, size / 2, size.saturating_sub(1)] {
          if size == 0 {
            break;
          }
          destination[destination_offset.strict_add(position)] = 1;
          // SAFETY: as above; the regions differ at `position`.
          unsafe {
            meter.call(Edge::Memcmp, "compare, different", size, [to, from, size]);
            meter.call(Edge::Bcmp, "compare, different", size, [to, from, size]);
          }
          destination[destination_offset.strict_add(position)] = 0;
        }
      }
      // Overlapping moves in both directions within one buffer.
      for shift in [1usize, 8, 64, size / 2] {
        if size == 0 || shift == 0 {
          continue;
        }
        let low = destination.as_mut_ptr().expose_provenance();
        let high = destination[shift..].as_mut_ptr().expose_provenance();
        if shift <= SLACK {
          // SAFETY: both regions lie inside `destination`, which holds at least
          // `size + SLACK` bytes and so covers `shift + size` when shift <= SLACK.
          unsafe {
            meter.call(Edge::Memmove, "move, overlapping up", size, [high, low, size]);
            meter.call(Edge::Memmove, "move, overlapping down", size, [low, high, size]);
          }
        }
      }
    }
    source.fill(0);
  }

  fn auxiliary(meter: &mut Meter) {
    // Every defined type through AT_MINSIGSTKSZ and beyond, plus absent types,
    // which take the ENOENT path.
    for kind in (0..=64usize).chain([0x7fff, usize::MAX]) {
      // SAFETY: getauxval accepts any type and reads only the auxiliary vector.
      unsafe { meter.call(Edge::Getauxval, "type", kind, [kind, 0, 0]) };
    }
  }

  /// Allocates and frees a pseudo-random mix of sizes, every call painted.
  fn workload(meter: &mut Meter, seed: u64, operations: usize, keep: usize) -> Vec<usize> {
    let mut random = Rng(seed);
    // At most 1,024 live allocations bound the workload's memory to well under
    // 1 GiB of address space, most of it never touched.
    let mut live: Vec<*mut c_void> = Vec::with_capacity(1024);
    for operation in 0..operations {
      if operation % 5000 == 4999 {
        // Free everything: consolidation and trimming of the top chunk.
        while !live.is_empty() {
          let pointer = live.swap_remove(random.below(live.len()));
          meter.free("free, bulk release", pointer);
        }
        continue;
      }
      if live.is_empty() || (live.len() < 1024 && random.below(3) != 0) {
        // Tcache and fast bins, small and large bins, around the dynamic mmap
        // threshold, and mmap-only sizes.
        let size = match random.below(100) {
          0..45 => random.between(1, 1032),
          45..75 => random.between(1033, 64 << 10),
          75..90 => random.between((64 << 10) + 1, 1 << 20),
          90..98 => random.between((1 << 20) + 1, 4 << 20),
          _ => random.between((4 << 20) + 1, 16 << 20),
        };
        let pointer = if random.below(10) < 6 {
          meter.malloc("malloc, workload", size)
        } else if random.below(2) == 0 {
          meter.calloc("calloc, workload", 1, size)
        } else {
          let element = [1usize, 2, 4, 8, 16, 24][random.below(6)];
          meter.calloc("calloc, workload elements", size.div_ceil(element), element)
        };
        assert!(!pointer.is_null(), "workload allocation failed");
        // Touch the first byte so that the allocation is backed, as in use.
        // SAFETY: a successful allocation of at least one byte.
        unsafe { pointer.cast::<u8>().write_volatile(1) };
        live.push(pointer);
      } else {
        let pointer = live.swap_remove(random.below(live.len()));
        meter.free("free, workload", pointer);
      }
    }
    // Return some allocations for another thread to free.
    let kept: Vec<usize> = live
      .iter()
      .take(keep)
      .map(|pointer| pointer.expose_provenance())
      .collect();
    for pointer in live.into_iter().skip(keep) {
      meter.free("free, workload end", pointer);
    }
    kept
  }

  fn failures(meter: &mut Meter) {
    // Recoverable failure returns: each returns null and sets errno.
    let huge = meter.malloc("malloc, impossible size", usize::MAX / 2);
    assert!(huge.is_null(), "an impossible allocation succeeded");
    let overflow = meter.calloc("calloc, size overflow", usize::MAX / 2, 4);
    assert!(overflow.is_null(), "an overflowing calloc succeeded");
    let address_space = meter.malloc("malloc, beyond the address space", 1 << 62);
    assert!(address_space.is_null(), "a 4 EiB allocation succeeded");
    meter.free("free, null", core::ptr::null_mut());
  }

  /// The context of one raw pthread; the thread allocates nothing before its calls.
  struct Thread {
    meter: Meter,
    first: Edge,
    size: usize,
    seed: u64,
    operations: usize,
    kept: Vec<usize>,
  }

  extern "C" fn thread_main(context: *mut c_void) -> *mut c_void {
    // SAFETY: `spawn` passes a unique, live `Thread` and joins before using it.
    let thread = unsafe { &mut *context.cast::<Thread>() };
    let first = match thread.first {
      Edge::Malloc => thread.meter.malloc("malloc, first call on a fresh thread", thread.size),
      _ => thread
        .meter
        .calloc("calloc, first call on a fresh thread", 1, thread.size),
    };
    assert!(!first.is_null(), "first thread allocation failed");
    let second = thread
      .meter
      .malloc("malloc, second call on a fresh thread", thread.size);
    thread.meter.free("free, first call on a fresh thread", second);
    thread.meter.free("free, fresh thread", first);
    if thread.operations != 0 {
      thread.kept = workload(&mut thread.meter, thread.seed, thread.operations, 64);
    }
    core::ptr::null_mut()
  }

  fn spawn(thread: &mut Thread) {
    let mut handle: libc::pthread_t = 0;
    // SAFETY: default attributes; `thread_main` receives a unique pointer to
    // `thread`, which outlives the join below.
    let created = unsafe {
      libc::pthread_create(
        &raw mut handle,
        core::ptr::null(),
        thread_main,
        core::ptr::from_mut(thread).cast(),
      )
    };
    assert_eq!(created, 0, "pthread_create failed");
    // SAFETY: `handle` is a joinable thread created above and joined once.
    let joined = unsafe { libc::pthread_join(handle, core::ptr::null_mut()) };
    assert_eq!(joined, 0, "pthread_join failed");
  }

  fn text(pointer: *const c_char) -> String {
    if pointer.is_null() {
      return String::new();
    }
    // SAFETY: libc returns a NUL-terminated static string or a loader-owned
    // object or symbol name that stays valid while the object is loaded.
    unsafe { CStr::from_ptr(pointer) }.to_string_lossy().into_owned()
  }

  fn implementation(edge: Edge) -> String {
    let mut info = libc::Dl_info {
      dli_fname: core::ptr::null(),
      dli_fbase: core::ptr::null_mut(),
      dli_sname: core::ptr::null(),
      dli_saddr: core::ptr::null_mut(),
    };
    // SAFETY: dladdr reads loader state for the code address and writes `info`.
    let found = unsafe { libc::dladdr(edge.function().cast(), &raw mut info) };
    if found == 0 {
      return "null".to_owned();
    }
    let address = edge.function().addr();
    let offset = address.strict_sub(info.dli_fbase.addr());
    let symbol_offset = if info.dli_saddr.is_null() {
      None
    } else {
      address.checked_sub(info.dli_saddr.addr())
    };
    format!(
      "{{\"object\": {}, \"offset\": \"{offset:#x}\", \"nearest_symbol\": {}, \"symbol_offset\": {}}}",
      json_string(&text(info.dli_fname)),
      json_string(&text(info.dli_sname)),
      symbol_offset.map_or_else(|| "null".to_owned(), |value| value.to_string())
    )
  }

  pub(super) fn run() {
    let mut meter = Meter::new();
    // The fresh-thread calls come first, so that no other thread has touched
    // an arena they could reuse.
    for first in [Edge::Malloc, Edge::Calloc] {
      for size in [16usize, 1032, 4096, 200_000, 4 << 20] {
        let mut thread = Thread {
          meter: Meter::new(),
          first,
          size,
          seed: 0,
          operations: 0,
          kept: Vec::new(),
        };
        spawn(&mut thread);
        meter.merge(&thread.meter.deepest);
      }
    }
    memory(&mut meter);
    auxiliary(&mut meter);
    failures(&mut meter);
    let mut kept = workload(&mut meter, 0x9e37_79b9_7f4a_7c15, 100_000, 0);
    for index in 0..4u64 {
      let mut thread = Thread {
        meter: Meter::new(),
        first: Edge::Malloc,
        size: 64,
        seed: 0x2545_f491_4f6c_dd1d_u64.strict_add(index),
        operations: 25_000,
        kept: Vec::new(),
      };
      spawn(&mut thread);
      meter.merge(&thread.meter.deepest);
      kept.append(&mut thread.kept);
    }
    // Free chunks of the exited threads' arenas from the main thread.
    for pointer in kept {
      meter.free(
        "free, chunk of another thread's arena",
        core::ptr::with_exposed_provenance_mut(pointer),
      );
    }
    let edges: Vec<String> = EDGES
      .iter()
      .zip(&meter.deepest)
      .map(|(edge, deepest)| {
        format!(
          "    {}: {{\"max_below_entry_bytes\": {}, \"max_used_bytes\": {}, \"calls\": {}, \"deepest_case\": {}, \"deepest_size\": {}, \"implementation\": {}}}",
          json_string(edge.name()),
          deepest.bytes,
          deepest.used_bytes,
          deepest.calls,
          json_string(deepest.case),
          deepest.size,
          implementation(*edge)
        )
      })
      .collect();
    // SAFETY: both glibc functions return static NUL-terminated strings.
    let (version, release) = (
      text(unsafe { libc::gnu_get_libc_version() }),
      text(unsafe { libc::gnu_get_libc_release() }),
    );
    println!(
      "{{\n  \"glibc_version\": {},\n  \"glibc_release\": {},\n  \"entry_offset_bytes\": {},\n  \"red_zone_bytes\": {},\n  \"stack_bytes\": {},\n  \"edges\": {{\n{}\n  }}\n}}",
      json_string(&version),
      json_string(&release),
      rscrypto_stack_paint::ENTRY_OFFSET,
      rscrypto_stack_paint::RED_ZONE,
      STACK_BYTES,
      edges.join(",\n")
    );
  }
}

#[cfg(target_os = "linux")]
fn main() {
  linux::run();
}

#[cfg(not(target_os = "linux"))]
fn main() {
  eprintln!("declared libc edges are measured on Linux GNU targets only");
  std::process::exit(2);
}
