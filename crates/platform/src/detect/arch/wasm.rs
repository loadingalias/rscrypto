// WebAssembly Detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
fn detect_wasm32() -> Detected {
  use crate::caps::wasm;

  let mut caps = Caps::NONE;

  if cfg!(target_feature = "simd128") {
    caps |= wasm::SIMD128;
  }
  if cfg!(target_feature = "relaxed-simd") {
    caps |= wasm::RELAXED_SIMD;
  }

  Detected {
    caps,
    tune: Tune::PORTABLE,
    arch: Arch::Wasm32,
  }
}

#[cfg(target_arch = "wasm64")]
fn detect_wasm64() -> Detected {
  use crate::caps::wasm;

  let mut caps = Caps::NONE;

  if cfg!(target_feature = "simd128") {
    caps |= wasm::SIMD128;
  }
  if cfg!(target_feature = "relaxed-simd") {
    caps |= wasm::RELAXED_SIMD;
  }

  Detected {
    caps,
    tune: Tune::PORTABLE,
    arch: Arch::Wasm64,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

