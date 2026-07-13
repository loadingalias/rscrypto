#![cfg(all(feature = "xxh3", feature = "rapidhash"))]

use core::{
  alloc::{GlobalAlloc, Layout},
  hash::{BuildHasher, Hasher},
  sync::atomic::{AtomicUsize, Ordering},
};
use std::{alloc::System, collections::HashMap};

use rscrypto::{RapidBuildHasher, RapidStreamHasher, Xxh3_128Hasher, Xxh3BuildHasher};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Delegating allocation to `System` because:
// 1. Every operation forwards the original pointer and layout unchanged.
// 2. `System` defines and upholds the global allocator contract.
// 3. The relaxed counter is independent of the allocated memory and cannot affect its validity.
unsafe impl GlobalAlloc for CountingAllocator {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: Delegating allocation because:
    // 1. `layout` is forwarded unchanged to the system allocator.
    unsafe { System.alloc(layout) }
  }

  unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
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
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: Delegating reallocation because:
    // 1. `ptr`, `layout`, and `new_size` are forwarded unchanged to `System`.
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn reset_allocations() {
  ALLOCATIONS.store(0, Ordering::Relaxed);
}

fn allocation_count() -> usize {
  ALLOCATIONS.load(Ordering::Relaxed)
}

#[test]
fn fast_hashers_and_preallocated_maps_hash_without_allocating() {
  let xxh3_builder = Xxh3BuildHasher::with_seed(42);
  let rapid_builder = RapidBuildHasher::with_seed(42);
  let mut xxh3_map = HashMap::with_capacity_and_hasher(8, xxh3_builder);
  let mut rapid_map = HashMap::with_capacity_and_hasher(8, rapid_builder);
  let long_input = [0x5au8; 4096];

  reset_allocations();
  let mut xxh3 = xxh3_builder.build_hasher();
  for chunk in long_input.chunks(127) {
    xxh3.write(chunk);
  }
  let _ = xxh3.finish();
  let xxh3_direct = allocation_count();

  reset_allocations();
  let mut xxh3_128 = Xxh3_128Hasher::with_seed(42);
  for chunk in long_input.chunks(127) {
    xxh3_128.write(chunk);
  }
  let _ = xxh3_128.finish();
  let xxh3_128_direct = allocation_count();

  reset_allocations();
  let mut rapid = rapid_builder.build_hasher();
  rapid.write(&long_input);
  let _ = rapid.finish();
  let rapid_direct = allocation_count();

  reset_allocations();
  let mut rapid_stream = RapidStreamHasher::new();
  for chunk in long_input.chunks(127) {
    rapid_stream.write(chunk);
  }
  let _ = rapid_stream.finish();
  let rapid_stream_direct = allocation_count();

  reset_allocations();
  xxh3_map.insert("allocation-free", 1);
  let _ = xxh3_map.get("allocation-free");
  let xxh3_map_ops = allocation_count();

  reset_allocations();
  rapid_map.insert("allocation-free", 1);
  let _ = rapid_map.get("allocation-free");
  let rapid_map_ops = allocation_count();

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
