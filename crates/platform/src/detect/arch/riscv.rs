// RISC-V Detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "riscv64")]
fn detect_riscv64() -> Detected {
  #[cfg(feature = "std")]
  let caps = caps_static() | runtime_riscv();
  #[cfg(not(feature = "std"))]
  let caps = caps_static();

  Detected {
    caps,
    tune: if caps.has(crate::caps::riscv::ZBC) || caps.has(crate::caps::riscv::ZVBC) {
      Tune::DEFAULT
    } else {
      Tune::PORTABLE
    },
    arch: Arch::Riscv64,
  }
}

#[cfg(target_arch = "riscv32")]
fn detect_riscv32() -> Detected {
  #[cfg(feature = "std")]
  let caps = caps_static() | runtime_riscv();
  #[cfg(not(feature = "std"))]
  let caps = caps_static();

  Detected {
    caps,
    tune: if caps.has(crate::caps::riscv::ZBC) || caps.has(crate::caps::riscv::ZVBC) {
      Tune::DEFAULT
    } else {
      Tune::PORTABLE
    },
    arch: Arch::Riscv32,
  }
}

#[cfg(all(
  any(target_arch = "riscv64", target_arch = "riscv32"),
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
fn runtime_riscv() -> Caps {
  use crate::caps::riscv;

  let mut caps = Caps::NONE;

  // IMPORTANT:
  // `is_riscv_feature_detected!` only accepts a subset of RISC-V feature
  // strings today. In current nightlies used by this workspace, vector (`v`)
  // and vector-crypto (`zv*`) strings are rejected at compile time even though
  // they may appear in `rustc --print target-features`.
  //
  // We therefore probe only the runtime-detectable subset here and rely on
  // `caps_static()` for compile-time known features (including `v` and `zv*`).
  //
  // This keeps detection correct and cross-target builds stable without
  // silently dropping supported scalar runtime probes.
  //
  // Also: call `is_riscv_feature_detected!` directly. Routing the feature
  // literal through another macro (e.g. `rt!($f)`) causes this std macro to
  // reject otherwise-valid features at compile time.

  // ─── Runtime-detectable Bit Manipulation ───
  if std::arch::is_riscv_feature_detected!("zbb") {
    caps |= riscv::ZBB;
  }
  if std::arch::is_riscv_feature_detected!("zbs") {
    caps |= riscv::ZBS;
  }
  if std::arch::is_riscv_feature_detected!("zba") {
    caps |= riscv::ZBA;
  }
  if std::arch::is_riscv_feature_detected!("zbc") {
    caps |= riscv::ZBC;
  }

  // ─── Runtime-detectable Scalar Crypto ───
  if std::arch::is_riscv_feature_detected!("zbkb") {
    caps |= riscv::ZBKB;
  }
  if std::arch::is_riscv_feature_detected!("zbkc") {
    caps |= riscv::ZBKC;
  }
  if std::arch::is_riscv_feature_detected!("zbkx") {
    caps |= riscv::ZBKX;
  }
  if std::arch::is_riscv_feature_detected!("zknd") {
    caps |= riscv::ZKND;
  }
  if std::arch::is_riscv_feature_detected!("zkne") {
    caps |= riscv::ZKNE;
  }
  if std::arch::is_riscv_feature_detected!("zknh") {
    caps |= riscv::ZKNH;
  }
  if std::arch::is_riscv_feature_detected!("zksed") {
    caps |= riscv::ZKSED;
  }
  if std::arch::is_riscv_feature_detected!("zksh") {
    caps |= riscv::ZKSH;
  }

  caps
}

#[cfg(all(
  any(target_arch = "riscv64", target_arch = "riscv32"),
  feature = "std",
  not(any(target_os = "linux", target_os = "android"))
))]
fn runtime_riscv() -> Caps {
  // `std::arch::is_riscv_feature_detected!` is only implemented on Linux-like
  // platforms today.
  Caps::NONE
}
