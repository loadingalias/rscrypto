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

  macro_rules! rt {
    ($f:literal => $c:expr) => {
      if std::arch::is_riscv_feature_detected!($f) {
        caps |= $c;
      }
    };
  }

  // ─── Vector Extension ───
  rt!("v" => riscv::V);

  // ─── Bit Manipulation ───
  rt!("zbb" => riscv::ZBB);
  rt!("zbs" => riscv::ZBS);
  rt!("zba" => riscv::ZBA);
  rt!("zbc" => riscv::ZBC);

  // ─── Scalar Crypto ───
  rt!("zbkb" => riscv::ZBKB);
  rt!("zbkc" => riscv::ZBKC);
  rt!("zbkx" => riscv::ZBKX);
  rt!("zknd" => riscv::ZKND);
  rt!("zkne" => riscv::ZKNE);
  rt!("zknh" => riscv::ZKNH);
  rt!("zksed" => riscv::ZKSED);
  rt!("zksh" => riscv::ZKSH);

  // ─── Vector Crypto ───
  rt!("zvbb" => riscv::ZVBB);
  rt!("zvbc" => riscv::ZVBC);
  rt!("zvkb" => riscv::ZVKB);
  rt!("zvkg" => riscv::ZVKG);
  rt!("zvkned" => riscv::ZVKNED);
  rt!("zvknha" => riscv::ZVKNHA);
  rt!("zvknhb" => riscv::ZVKNHB);
  rt!("zvksed" => riscv::ZVKSED);
  rt!("zvksh" => riscv::ZVKSH);

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

