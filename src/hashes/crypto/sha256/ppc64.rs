//! Conservative SHA-256 ppc64 compatibility kernel.
//!
//! # Safety
//!
//! Callers must still verify `power::POWER8_CRYPTO` because this entry point is
//! only reachable through the POWER-specific kernel slot. The current
//! implementation intentionally routes to the portable compressor until the
//! `vshasigmaw` path is reintroduced with target-native verification on
//! `powerpc64le` and big-endian POWER.

#![allow(unsafe_code)]
use super::Sha256;

/// SHA-256 multi-block compression entry point for the POWER kernel slot.
///
/// # Safety
///
/// Caller must ensure POWER8 Crypto features are available.
#[target_feature(enable = "altivec,vsx,power8-vector,power8-crypto")]
pub(crate) unsafe fn compress_blocks_ppc64_crypto(state: &mut [u32; 8], blocks: &[u8]) {
  // Keep the explicit POWER kernel entry point correct even when selected
  // directly by benches or tests.
  Sha256::compress_blocks_portable(state, blocks);
}
