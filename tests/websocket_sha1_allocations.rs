#![cfg(feature = "websocket-sha1")]

use core::{
  alloc::{GlobalAlloc, Layout},
  cell::Cell,
};
use std::alloc::System;

use rscrypto::hashes::legacy::WebSocketAcceptDigest;

const MEASUREMENT_DISABLED: usize = usize::MAX;

struct CountingAllocator;

std::thread_local! {
  static ALLOCATIONS: Cell<usize> = const { Cell::new(MEASUREMENT_DISABLED) };
}

fn record_allocation() {
  discard_measurement_result(ALLOCATIONS.try_with(|allocations| {
    let count = allocations.get();
    if count != MEASUREMENT_DISABLED {
      allocations.set(count.strict_add(1));
    }
  }));
}

fn discard_measurement_result(_result: Result<(), std::thread::AccessError>) {}

// SAFETY: Every operation delegates the original pointer and layout unchanged
// to `System`; thread-local accounting does not touch allocated memory.
unsafe impl GlobalAlloc for CountingAllocator {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    record_allocation();
    // SAFETY: `layout` is forwarded unchanged to the system allocator.
    unsafe { System.alloc(layout) }
  }

  unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
    record_allocation();
    // SAFETY: `layout` is forwarded unchanged to the system allocator.
    unsafe { System.alloc_zeroed(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    // SAFETY: `ptr` and `layout` originated from the delegated system allocator.
    unsafe { System.dealloc(ptr, layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    record_allocation();
    // SAFETY: All arguments are forwarded unchanged to the system allocator.
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

struct AllocationMeasurement;

impl Drop for AllocationMeasurement {
  fn drop(&mut self) {
    discard_measurement_result(ALLOCATIONS.try_with(|allocations| allocations.set(MEASUREMENT_DISABLED)));
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
fn websocket_accept_digest_does_not_allocate() {
  let key = b"dGhlIHNhbXBsZSBub25jZQ==";
  let allocations = measure_allocations(|| {
    let digest = WebSocketAcceptDigest::compute(key);
    assert_eq!(digest.as_ref().len(), 20);
  });

  assert_eq!(allocations, 0, "WebSocket accept digest must not allocate");
}
