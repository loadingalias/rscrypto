//! CRC kernel function type definitions.
//!
//! This module provides the function signature types used by kernel dispatch tables.

// ─────────────────────────────────────────────────────────────────────────────
// CRC-16 Kernel Function Type
// ─────────────────────────────────────────────────────────────────────────────

/// Function signature for CRC-16 kernels.
///
/// Used by CRC-16-CCITT, CRC-16-IBM, CRC-16-USB, and other 16-bit CRC variants.
///
/// # Arguments
///
/// * `state` - Current CRC state (typically initialized to 0xFFFF or 0x0000)
/// * `data` - Input data to process
///
/// # Returns
///
/// Updated CRC state after processing the input data.
pub type Crc16Fn = fn(u16, &[u8]) -> u16;

// ─────────────────────────────────────────────────────────────────────────────
// CRC-24 Kernel Function Type
// ─────────────────────────────────────────────────────────────────────────────

/// Function signature for CRC-24 kernels.
///
/// Used by CRC-24 (OpenPGP/Radix-64). The result is a 24-bit value stored
/// in the low 24 bits of a u32.
///
/// # Arguments
///
/// * `state` - Current CRC state (only low 24 bits are used)
/// * `data` - Input data to process
///
/// # Returns
///
/// Updated CRC state with the result in the low 24 bits.
pub type Crc24Fn = fn(u32, &[u8]) -> u32;

// ─────────────────────────────────────────────────────────────────────────────
// CRC-32 Kernel Function Type
// ─────────────────────────────────────────────────────────────────────────────

/// Function signature for CRC-32 kernels.
///
/// Used by:
/// - CRC-32 (IEEE)
/// - CRC-32C (Castagnoli)
///
/// # Hardware Acceleration
///
/// - **x86_64**: SSE4.2 `crc32` (CRC-32C only), PCLMULQDQ, VPCLMULQDQ
/// - **aarch64**: ARMv8 CRC extension, PMULL, PMULL+EOR3
///
/// # Arguments
///
/// * `state` - Current CRC state (typically initialized to 0xFFFFFFFF)
/// * `data` - Input data to process
///
/// # Returns
///
/// Updated CRC state after processing the input data.
pub type Crc32Fn = fn(u32, &[u8]) -> u32;

// ─────────────────────────────────────────────────────────────────────────────
// CRC-64 Kernel Function Type
// ─────────────────────────────────────────────────────────────────────────────

/// Function signature for CRC-64 kernels.
///
/// Used by:
/// - CRC-64-XZ (ECMA-182) - XZ Utils, 7-Zip
/// - CRC-64-NVME - NVMe specification
/// - CRC-64-GO-ISO - Go standard library
///
/// # Hardware Acceleration
///
/// - **x86_64**: PCLMULQDQ, VPCLMULQDQ
/// - **aarch64**: PMULL, PMULL+EOR3
///
/// # Arguments
///
/// * `state` - Current CRC state (typically initialized to 0xFFFFFFFFFFFFFFFF)
/// * `data` - Input data to process
///
/// # Returns
///
/// Updated CRC state after processing the input data.
pub type Crc64Fn = fn(u64, &[u8]) -> u64;
