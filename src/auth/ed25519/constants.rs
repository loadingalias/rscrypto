//! Internal constants for the Ed25519 implementation plan.

/// Ed25519 secret key length in bytes.
pub(crate) const SECRET_KEY_LENGTH: usize = 32;

/// Ed25519 public key length in bytes.
pub(crate) const PUBLIC_KEY_LENGTH: usize = 32;

/// Ed25519 signature length in bytes.
pub(crate) const SIGNATURE_LENGTH: usize = 64;

/// Internal field element uses a 5x51 radix layout in the planned portable implementation.
pub(crate) const FIELD_LIMBS: usize = 5;

/// Internal scalar uses four 64-bit limbs in the planned portable implementation.
pub(crate) const SCALAR_LIMBS: usize = 4;
