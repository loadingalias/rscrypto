//! CRC16 variants.
//!
//! CRC16 is not a single algorithm: multiple polynomials and parameter sets
//! exist in the wild. This module provides high-quality, widely used variants.
//!
//! # Variants
//!
//! | Variant | Polynomial | Init | Use Cases |
//! |---------|------------|------|-----------|
//! | CRC16/IBM | 0x8005 | 0x0000 | Modbus, USB, SDLC |
//! | CRC16/CCITT-FALSE | 0x1021 | 0xFFFF | X.25, PPP, Bluetooth |
//!
//! # Example
//!
//! ```
//! use checksum::{Crc16CcittFalse, Crc16Ibm};
//!
//! let ibm = Crc16Ibm::checksum(b"123456789");
//! let ccitt = Crc16CcittFalse::checksum(b"123456789");
//!
//! assert_eq!(ibm, 0xBB3D);
//! assert_eq!(ccitt, 0x29B1);
//! ```

pub mod ccitt_false;
pub mod ibm;

pub use ccitt_false::Crc16CcittFalse;
pub use ibm::Crc16Ibm;

/// Convenience alias for the most common reflected CRC16 variant.
pub type Crc16 = Crc16Ibm;
