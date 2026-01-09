//! Kernel introspection: verify which optimizations are active.
//!
//! This example shows how to inspect which kernels are selected for
//! your platform, useful for verifying hardware acceleration is enabled.
//!
//! Run with: `cargo run --example introspect -p checksum`

use checksum::{Crc16Ccitt, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme, DispatchInfo, KernelIntrospect, kernel_for};

fn main() {
  println!("=== Checksum Kernel Introspection ===\n");

  platform_info();
  algorithm_backends();
  size_based_dispatch();
  generic_introspection();
}

/// Display detected platform capabilities.
fn platform_info() {
  println!("--- Platform Detection ---\n");

  let info = DispatchInfo::current();

  // Full platform description with CPU features
  println!("Platform: {info}");
  println!();

  // The platform field provides the raw Description
  let platform = info.platform();
  println!("Platform Debug: {platform:?}");
  println!();
}

/// Show which backend is selected for each algorithm.
fn algorithm_backends() {
  println!("--- Algorithm Backends ---\n");

  // Each CRC type can report its selected backend
  println!("CRC-16 (CCITT):    {}", Crc16Ccitt::backend_name());
  println!("CRC-24 (OpenPGP):  {}", Crc24OpenPgp::backend_name());
  println!("CRC-32 (IEEE):     {}", Crc32::backend_name());
  println!("CRC-32C:           {}", Crc32C::backend_name());
  println!("CRC-64 (XZ):       {}", Crc64::backend_name());
  println!("CRC-64 (NVMe):     {}", Crc64Nvme::backend_name());
  println!();
}

/// Kernels may vary based on buffer size.
fn size_based_dispatch() {
  println!("--- Size-Based Kernel Selection ---\n");

  // Different buffer sizes may use different kernels
  let sizes = [64, 256, 1024, 4096, 65536, 1_048_576];

  println!("CRC-64 (XZ) kernel by buffer size:");
  for size in sizes {
    let kernel = Crc64::kernel_name_for_len(size);
    println!("  {:>10} bytes: {kernel}", size);
  }
  println!();

  println!("CRC-32C kernel by buffer size:");
  for size in sizes {
    let kernel = Crc32C::kernel_name_for_len(size);
    println!("  {:>10} bytes: {kernel}", size);
  }
  println!();
}

/// Generic introspection using the kernel_for function.
fn generic_introspection() {
  println!("--- Generic Introspection ---\n");

  // The kernel_for::<T>(len) function works with any KernelIntrospect type
  fn report<T: KernelIntrospect>(name: &str, sizes: &[usize]) {
    println!("{name}:");
    for &size in sizes {
      println!("  {:>8} B: {}", size, kernel_for::<T>(size));
    }
    println!();
  }

  let sizes = [128, 4096, 1_000_000];

  report::<Crc32>("CRC-32", &sizes);
  report::<Crc64>("CRC-64 (XZ)", &sizes);
  report::<Crc64Nvme>("CRC-64 (NVMe)", &sizes);

  // Useful for runtime decisions or logging
  let len = 8192;
  println!("For {len} byte buffers, CRC-64/XZ uses: {}", kernel_for::<Crc64>(len));
}
