//! Advanced checksum and hash kernel introspection.
//!
//! This example shows how to inspect which kernels are selected for your
//! platform, useful for verifying hardware acceleration is enabled.
//!
//! Run with: `cargo run --example introspect`

use rscrypto::{
  Blake3, Crc16Ccitt, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme, RapidHash, Sha256, Shake256, Xxh3,
  checksum::introspect::{DispatchInfo, KernelIntrospect, kernel_for},
  hashes::introspect::{HashKernelIntrospect, kernel_for as hash_kernel_for},
};

fn main() {
  println!("=== Advanced Dispatch Introspection ===\n");

  platform_info();
  checksum_kernel_probes();
  checksum_size_based_dispatch();
  generic_checksum_introspection();
  hash_kernel_probes();
  hash_size_based_dispatch();
  generic_hash_introspection();
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

/// Show a representative 1KB kernel probe for each checksum algorithm.
fn checksum_kernel_probes() {
  println!("--- Checksum Kernels @ 1KB ---\n");

  println!("CRC-16 (CCITT):    {}", kernel_for::<Crc16Ccitt>(1024));
  println!("CRC-24 (OpenPGP):  {}", kernel_for::<Crc24OpenPgp>(1024));
  println!("CRC-32 (IEEE):     {}", kernel_for::<Crc32>(1024));
  println!("CRC-32C:           {}", kernel_for::<Crc32C>(1024));
  println!("CRC-64 (XZ):       {}", kernel_for::<Crc64>(1024));
  println!("CRC-64 (NVMe):     {}", kernel_for::<Crc64Nvme>(1024));
  println!();
}

/// Checksum kernels may vary based on buffer size.
fn checksum_size_based_dispatch() {
  println!("--- Checksum Size-Based Kernel Selection ---\n");

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

/// Generic checksum introspection using the kernel_for function.
fn generic_checksum_introspection() {
  println!("--- Generic Checksum Introspection ---\n");

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

/// Show a representative 1KB kernel probe for each hash family.
fn hash_kernel_probes() {
  println!("\n--- Hash Kernels @ 1KB ---\n");

  println!("SHA-256:           {}", hash_kernel_for::<Sha256>(1024));
  println!("SHAKE-256:         {}", hash_kernel_for::<Shake256>(1024));
  println!("BLAKE3:            {}", hash_kernel_for::<Blake3>(1024));
  println!("XXH3:              {}", hash_kernel_for::<Xxh3>(1024));
  println!("RapidHash:         {}", hash_kernel_for::<RapidHash>(1024));
  println!();
}

/// Hash kernels may vary based on input size.
fn hash_size_based_dispatch() {
  println!("--- Hash Size-Based Kernel Selection ---\n");

  let sizes = [64, 256, 1024, 4096, 65536, 1_048_576];

  println!("BLAKE3 kernel by buffer size:");
  for size in sizes {
    let kernel = hash_kernel_for::<Blake3>(size);
    println!("  {:>10} bytes: {kernel}", size);
  }
  println!();

  println!("SHA-256 kernel by buffer size:");
  for size in sizes {
    let kernel = hash_kernel_for::<Sha256>(size);
    println!("  {:>10} bytes: {kernel}", size);
  }
  println!();
}

/// Generic hash introspection using the hash kernel_for function.
fn generic_hash_introspection() {
  println!("--- Generic Hash Introspection ---\n");

  fn report<T: HashKernelIntrospect>(name: &str, sizes: &[usize]) {
    println!("{name}:");
    for &size in sizes {
      println!("  {:>8} B: {}", size, hash_kernel_for::<T>(size));
    }
    println!();
  }

  let sizes = [128, 4096, 1_000_000];

  report::<Sha256>("SHA-256", &sizes);
  report::<Shake256>("SHAKE-256", &sizes);
  report::<Blake3>("BLAKE3", &sizes);
  report::<Xxh3>("XXH3", &sizes);
}
