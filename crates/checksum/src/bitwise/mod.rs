//! Table-less CRC implementations using branchless bitwise computation.
//!
//! This module provides "zero table" CRC computation for environments where
//! embedding lookup tables is undesirable:
//!
//! - **Embedded systems**: Limited flash/RAM
//! - **Bootloaders**: Minimal code size requirements
//! - **WebAssembly**: Reducing binary size
//!
//! # Algorithm
//!
//! These implementations use branchless bitwise polynomial reduction:
//!
//! 1. For each bit, create a conditional mask using `wrapping_sub`:
//!    - `0u32.wrapping_sub(0)` = `0x00000000`
//!    - `0u32.wrapping_sub(1)` = `0xFFFFFFFF`
//! 2. Conditionally XOR with the polynomial using `polynomial & mask`
//! 3. Shift the CRC register
//!
//! This avoids branch mispredictions, enabling better CPU pipelining.
//!
//! # Performance
//!
//! | Implementation | Throughput | Memory |
//! |----------------|------------|--------|
//! | SIMD (PMULL/PCLMUL) | 25-100 GB/s | 0 bytes |
//! | Slicing-by-8 | ~500 MB/s | 8 KB |
//! | **Bitwise (this module)** | ~200 MB/s | **0 bytes** |
//! | Naive bit-by-bit | ~100 MB/s | 0 bytes |
//!
//! # When to Use
//!
//! Use these implementations when:
//! - Memory is extremely constrained
//! - Code size matters more than throughput
//! - You're on a platform without SIMD and can't afford 8 KB for tables
//!
//! For most applications, prefer the main [`Crc32c`](crate::Crc32c) and
//! [`Crc32`](crate::Crc32) APIs which automatically select the fastest
//! implementation available.

pub mod crc32;
pub mod crc32c;
