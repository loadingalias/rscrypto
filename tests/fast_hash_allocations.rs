#![cfg(all(feature = "xxh3", feature = "rapidhash"))]

use core::{
  alloc::{GlobalAlloc, Layout},
  cell::Cell,
  hash::{BuildHasher, Hasher},
};
use std::{alloc::System, collections::HashMap};

use rscrypto::{RapidBuildHasher, RapidStreamHasher, Xxh3_128Hasher, Xxh3BuildHasher};

struct CountingAllocator;

const MEASUREMENT_DISABLED: usize = usize::MAX;

std::thread_local! {
  static ALLOCATIONS: Cell<usize> = const { Cell::new(MEASUREMENT_DISABLED) };
}

fn record_allocation() {
  let _ = ALLOCATIONS.try_with(|allocations| {
    let count = allocations.get();
    if count != MEASUREMENT_DISABLED {
      allocations.set(count.strict_add(1));
    }
  });
}

// SAFETY: Delegating allocation to `System` because:
// 1. Every operation forwards the original pointer and layout unchanged.
// 2. `System` defines and upholds the global allocator contract.
// 3. Thread-local measurement only updates a `Cell<usize>` and cannot affect the allocated memory.
unsafe impl GlobalAlloc for CountingAllocator {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    record_allocation();
    // SAFETY: Delegating allocation because:
    // 1. `layout` is forwarded unchanged to the system allocator.
    unsafe { System.alloc(layout) }
  }

  unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
    record_allocation();
    // SAFETY: Delegating zeroed allocation because:
    // 1. `layout` is forwarded unchanged to the system allocator.
    unsafe { System.alloc_zeroed(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    // SAFETY: Delegating deallocation because:
    // 1. `ptr` and `layout` originated from the delegated system allocation path.
    unsafe { System.dealloc(ptr, layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    record_allocation();
    // SAFETY: Delegating reallocation because:
    // 1. `ptr`, `layout`, and `new_size` are forwarded unchanged to `System`.
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

struct AllocationMeasurement;

impl Drop for AllocationMeasurement {
  fn drop(&mut self) {
    let _ = ALLOCATIONS.try_with(|allocations| allocations.set(MEASUREMENT_DISABLED));
  }
}

fn measure_allocations(f: impl FnOnce()) -> usize {
  ALLOCATIONS.with(|allocations| {
    assert_eq!(
      allocations.replace(0),
      MEASUREMENT_DISABLED,
      "allocation measurements must not be nested"
    );
  });

  let measurement = AllocationMeasurement;
  f();
  let count = ALLOCATIONS.with(|allocations| allocations.replace(MEASUREMENT_DISABLED));
  drop(measurement);
  count
}

#[test]
fn fast_hashers_and_preallocated_maps_hash_without_allocating() {
  let xxh3_builder = Xxh3BuildHasher::with_seed(42);
  let rapid_builder = RapidBuildHasher::with_seed(42);
  let mut xxh3_map = HashMap::with_capacity_and_hasher(8, xxh3_builder);
  let mut rapid_map = HashMap::with_capacity_and_hasher(8, rapid_builder);
  let long_input = [0x5au8; 4096];

  let xxh3_direct = measure_allocations(|| {
    let mut xxh3 = xxh3_builder.build_hasher();
    for chunk in long_input.chunks(127) {
      xxh3.write(chunk);
    }
    let _ = xxh3.finish();
  });

  let xxh3_128_direct = measure_allocations(|| {
    let mut xxh3_128 = Xxh3_128Hasher::with_seed(42);
    for chunk in long_input.chunks(127) {
      xxh3_128.write(chunk);
    }
    let _ = xxh3_128.finish();
  });

  let rapid_direct = measure_allocations(|| {
    let mut rapid = rapid_builder.build_hasher();
    rapid.write(&long_input);
    let _ = rapid.finish();
  });

  let rapid_stream_direct = measure_allocations(|| {
    let mut rapid_stream = RapidStreamHasher::new();
    for chunk in long_input.chunks(127) {
      rapid_stream.write(chunk);
    }
    let _ = rapid_stream.finish();
  });

  let xxh3_map_ops = measure_allocations(|| {
    xxh3_map.insert("allocation-free", 1);
    let _ = xxh3_map.get("allocation-free");
  });

  let rapid_map_ops = measure_allocations(|| {
    rapid_map.insert("allocation-free", 1);
    let _ = rapid_map.get("allocation-free");
  });

  assert_eq!(xxh3_direct, 0, "XXH3 Hasher must not allocate");
  assert_eq!(xxh3_128_direct, 0, "XXH3-128 Hasher must not allocate");
  assert_eq!(rapid_direct, 0, "RapidHash Hasher must not allocate");
  assert_eq!(rapid_stream_direct, 0, "RapidHash streaming Hasher must not allocate");
  assert_eq!(
    xxh3_map_ops, 0,
    "preallocated XXH3 HashMap operations must not allocate"
  );
  assert_eq!(
    rapid_map_ops, 0,
    "preallocated RapidHash HashMap operations must not allocate"
  );
}
