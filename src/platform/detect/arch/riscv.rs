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
    arch: Arch::Riscv32,
  }
}

#[cfg(all(
  any(target_arch = "riscv64", target_arch = "riscv32"),
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
fn runtime_riscv() -> Caps {
  use core::{arch::asm, ptr};
  use crate::platform::caps::riscv;

  #[repr(C)]
  struct RiscvHwprobe {
    key: i64,
    value: u64,
  }

  const __NR_RISCV_HWPROBE: usize = 258;
  const RISCV_HWPROBE_KEY_BASE_BEHAVIOR: i64 = 3;
  const RISCV_HWPROBE_KEY_IMA_EXT_0: i64 = 4;
  const RISCV_HWPROBE_BASE_BEHAVIOR_IMA: u64 = 1 << 0;

  const RISCV_HWPROBE_IMA_V: u64 = 1 << 2;
  const RISCV_HWPROBE_EXT_ZBA: u64 = 1 << 3;
  const RISCV_HWPROBE_EXT_ZBB: u64 = 1 << 4;
  const RISCV_HWPROBE_EXT_ZBS: u64 = 1 << 5;
  const RISCV_HWPROBE_EXT_ZBC: u64 = 1 << 7;
  const RISCV_HWPROBE_EXT_ZBKB: u64 = 1 << 8;
  const RISCV_HWPROBE_EXT_ZBKC: u64 = 1 << 9;
  const RISCV_HWPROBE_EXT_ZBKX: u64 = 1 << 10;
  const RISCV_HWPROBE_EXT_ZKND: u64 = 1 << 11;
  const RISCV_HWPROBE_EXT_ZKNE: u64 = 1 << 12;
  const RISCV_HWPROBE_EXT_ZKNH: u64 = 1 << 13;
  const RISCV_HWPROBE_EXT_ZKSED: u64 = 1 << 14;
  const RISCV_HWPROBE_EXT_ZKSH: u64 = 1 << 15;
  const RISCV_HWPROBE_EXT_ZKT: u64 = 1 << 16;
  const RISCV_HWPROBE_EXT_ZVBB: u64 = 1 << 17;
  const RISCV_HWPROBE_EXT_ZVBC: u64 = 1 << 18;
  const RISCV_HWPROBE_EXT_ZVKB: u64 = 1 << 19;
  const RISCV_HWPROBE_EXT_ZVKG: u64 = 1 << 20;
  const RISCV_HWPROBE_EXT_ZVKNED: u64 = 1 << 21;
  const RISCV_HWPROBE_EXT_ZVKNHA: u64 = 1 << 22;
  const RISCV_HWPROBE_EXT_ZVKNHB: u64 = 1 << 23;
  const RISCV_HWPROBE_EXT_ZVKSED: u64 = 1 << 24;
  const RISCV_HWPROBE_EXT_ZVKSH: u64 = 1 << 25;
  const RISCV_HWPROBE_EXT_ZVKT: u64 = 1 << 26;

  #[inline]
  unsafe fn syscall_riscv_hwprobe(pairs: *mut RiscvHwprobe, pair_count: usize) -> isize {
    let ret: usize;
    // SAFETY: We follow the Linux RISC-V syscall ABI directly:
    // a7 = syscall number, a0..a4 = arguments, return value in a0.
    // The caller guarantees `pairs` is valid for `pair_count` entries.
    unsafe {
      asm!(
        "ecall",
        in("a7") __NR_RISCV_HWPROBE,
        inlateout("a0") pairs as usize => ret,
        in("a1") pair_count,
        in("a2") 0usize,
        in("a3") ptr::null_mut::<usize>() as usize,
        in("a4") 0usize,
        options(nostack, preserves_flags),
      );
    }
    ret as isize
  }

  let mut probes = [
    RiscvHwprobe {
      key: RISCV_HWPROBE_KEY_BASE_BEHAVIOR,
      value: 0,
    },
    RiscvHwprobe {
      key: RISCV_HWPROBE_KEY_IMA_EXT_0,
      value: 0,
    },
  ];

  // Older kernels return ENOSYS/EINVAL here; treat that as "runtime probing
  // unavailable" and fall back to compile-time detection only.
  // SAFETY: `probes` points to two initialized `RiscvHwprobe` entries and remains
  // valid for the duration of the syscall.
  if unsafe { syscall_riscv_hwprobe(probes.as_mut_ptr(), probes.len()) } != 0 {
    return Caps::NONE;
  }

  if probes[0].key == -1 || probes[1].key == -1 {
    return Caps::NONE;
  }

  if probes[0].value & RISCV_HWPROBE_BASE_BEHAVIOR_IMA == 0 {
    return Caps::NONE;
  }

  let mut caps = Caps::NONE;
  let ext0 = probes[1].value;

  if ext0 & RISCV_HWPROBE_IMA_V != 0 {
    caps |= riscv::V;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZBA != 0 {
    caps |= riscv::ZBA;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZBB != 0 {
    caps |= riscv::ZBB;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZBS != 0 {
    caps |= riscv::ZBS;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZBC != 0 {
    caps |= riscv::ZBC;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZBKB != 0 {
    caps |= riscv::ZBKB;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZBKC != 0 {
    caps |= riscv::ZBKC;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZBKX != 0 {
    caps |= riscv::ZBKX;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZKND != 0 {
    caps |= riscv::ZKND;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZKNE != 0 {
    caps |= riscv::ZKNE;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZKNH != 0 {
    caps |= riscv::ZKNH;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZKSED != 0 {
    caps |= riscv::ZKSED;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZKSH != 0 {
    caps |= riscv::ZKSH;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZKT != 0 {
    caps |= riscv::ZKT;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZVBB != 0 {
    caps |= riscv::ZVBB;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZVBC != 0 {
    caps |= riscv::ZVBC;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZVKB != 0 {
    caps |= riscv::ZVKB;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZVKG != 0 {
    caps |= riscv::ZVKG;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZVKNED != 0 {
    caps |= riscv::ZVKNED;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZVKNHA != 0 {
    caps |= riscv::ZVKNHA;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZVKNHB != 0 {
    caps |= riscv::ZVKNHB;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZVKSED != 0 {
    caps |= riscv::ZVKSED;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZVKSH != 0 {
    caps |= riscv::ZVKSH;
  }
  if ext0 & RISCV_HWPROBE_EXT_ZVKT != 0 {
    caps |= riscv::ZVKT;
  }

  // Kernel hwprobe currently reports the leaf crypto extensions, not the
  // synthetic bundles (`zkn`, `zvkn`, etc.), so keep runtime detection leaf-
  // precise and let compile-time detection carry bundle bits when available.
  if caps.has(riscv::ZBKB | riscv::ZBKC | riscv::ZBKX | riscv::ZKND | riscv::ZKNE | riscv::ZKNH) {
    caps |= riscv::ZKN;
  }
  if caps.has(riscv::ZBKB | riscv::ZBKC | riscv::ZBKX | riscv::ZKSED | riscv::ZKSH) {
    caps |= riscv::ZKS;
  }
  if caps.has(riscv::ZVKB | riscv::ZVKNED | riscv::ZVKNHB | riscv::ZVKT) {
    caps |= riscv::ZVKN;
  }
  if caps.has(riscv::ZVKB | riscv::ZVKNED | riscv::ZVKNHB | riscv::ZVBC | riscv::ZVKT) {
    caps |= riscv::ZVKNC;
  }
  if caps.has(riscv::ZVKB | riscv::ZVKNED | riscv::ZVKNHB | riscv::ZVKG | riscv::ZVKT) {
    caps |= riscv::ZVKNG;
  }
  if caps.has(riscv::ZVKB | riscv::ZVKSED | riscv::ZVKSH | riscv::ZVKT) {
    caps |= riscv::ZVKS;
  }
  if caps.has(riscv::ZVKB | riscv::ZVKSED | riscv::ZVKSH | riscv::ZVBC | riscv::ZVKT) {
    caps |= riscv::ZVKSC;
  }
  if caps.has(riscv::ZVKB | riscv::ZVKSED | riscv::ZVKSH | riscv::ZVKG | riscv::ZVKT) {
    caps |= riscv::ZVKSG;
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
