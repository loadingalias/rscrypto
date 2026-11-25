use core::{
  alloc::{GlobalAlloc, Layout},
  cell::Cell,
};
use std::alloc::System;

struct CountingAllocator;

const MEASUREMENT_DISABLED: usize = usize::MAX;

std::thread_local! {
  static ALLOCATIONS: Cell<usize> = const { Cell::new(MEASUREMENT_DISABLED) };
}

fn record_allocation() {
  let _counter_result = ALLOCATIONS.try_with(|allocations| {
    let count = allocations.get();
    if count != MEASUREMENT_DISABLED {
      allocations.set(count.strict_add(1));
    }
  });
}

// SAFETY: This allocator delegates every memory operation to `System` with the
// original pointer and layout. The thread-local counter observes allocation
// calls without changing the delegated allocation contract.
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
    // SAFETY: `ptr` and `layout` came from the delegated system allocator.
    unsafe { System.dealloc(ptr, layout) }
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    record_allocation();
    // SAFETY: The original pointer, layout, and requested size are forwarded
    // unchanged to the system allocator.
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

pub(super) struct Measurement;

impl Drop for Measurement {
  fn drop(&mut self) {
    let _counter_result = ALLOCATIONS.try_with(|allocations| allocations.set(MEASUREMENT_DISABLED));
  }
}

impl Measurement {
  pub(super) fn start() -> Self {
    ALLOCATIONS.with(|count| {
      assert_eq!(
        count.get(),
        MEASUREMENT_DISABLED,
        "allocation measurements must not be nested"
      );
      count.set(0);
    });
    Self
  }

  pub(super) fn finish(self) -> usize {
    ALLOCATIONS.get()
  }
}

pub(super) fn measure_allocations(operation: impl FnOnce()) -> usize {
  let measurement = Measurement::start();
  operation();
  measurement.finish()
}
