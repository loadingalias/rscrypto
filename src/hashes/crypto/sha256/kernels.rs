use super::Sha256;
use crate::platform::Caps;
#[cfg(target_arch = "aarch64")]
use crate::platform::caps::aarch64;
#[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
use crate::platform::caps::riscv;
#[cfg(target_arch = "s390x")]
use crate::platform::caps::s390x;
#[cfg(target_arch = "wasm32")]
use crate::platform::caps::wasm;
#[cfg(target_arch = "x86_64")]
use crate::platform::caps::x86;

pub(crate) type CompressBlocksFn = fn(&mut [u32; 8], &[u8]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Sha256KernelId {
  Portable = 0,
  #[cfg(target_arch = "x86_64")]
  X86Sha = 1,
  #[cfg(target_arch = "aarch64")]
  Aarch64Sha2 = 2,
  #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
  RiscvZknh = 3,
  #[cfg(target_arch = "wasm32")]
  WasmSimd128 = 4,
  #[cfg(target_arch = "s390x")]
  S390xKimd = 5,
}

impl Sha256KernelId {
  #[cfg(any(test, feature = "diag"))]
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(target_arch = "x86_64")]
      Self::X86Sha => "x86-sha",
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Sha2 => "aarch64-sha2",
      #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
      Self::RiscvZknh => "riscv/zknh",
      #[cfg(target_arch = "wasm32")]
      Self::WasmSimd128 => "wasm/simd128",
      #[cfg(target_arch = "s390x")]
      Self::S390xKimd => "s390x/kimd",
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe wrappers — dispatch validates caps before calling.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn compress_blocks_x86_sha(state: &mut [u32; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `x86::SHA` is available.
  unsafe { super::x86_64::compress_blocks_sha_ni(state, blocks) }
}

#[cfg(target_arch = "aarch64")]
fn compress_blocks_aarch64_sha2(state: &mut [u32; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `aarch64::SHA2` is available.
  unsafe { super::aarch64::compress_blocks_aarch64_sha2(state, blocks) }
}

#[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
fn compress_blocks_riscv_zknh(state: &mut [u32; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `riscv::ZKNH` is available.
  unsafe { super::riscv64::compress_blocks_zknh(state, blocks) }
}

#[cfg(target_arch = "wasm32")]
fn compress_blocks_wasm_simd128(state: &mut [u32; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `wasm::SIMD128` is available.
  unsafe { super::wasm::compress_blocks_wasm_simd(state, blocks) }
}

#[cfg(target_arch = "s390x")]
fn compress_blocks_s390x_kimd(state: &mut [u32; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `s390x::MSA` is available.
  unsafe { super::s390x::compress_blocks_kimd(state, blocks) }
}

#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha256KernelId) -> CompressBlocksFn {
  match id {
    Sha256KernelId::Portable => Sha256::compress_blocks_portable,
    #[cfg(target_arch = "x86_64")]
    Sha256KernelId::X86Sha => compress_blocks_x86_sha,
    #[cfg(target_arch = "aarch64")]
    Sha256KernelId::Aarch64Sha2 => compress_blocks_aarch64_sha2,
    #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
    Sha256KernelId::RiscvZknh => compress_blocks_riscv_zknh,
    #[cfg(target_arch = "wasm32")]
    Sha256KernelId::WasmSimd128 => compress_blocks_wasm_simd128,
    #[cfg(target_arch = "s390x")]
    Sha256KernelId::S390xKimd => compress_blocks_s390x_kimd,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha256KernelId) -> Caps {
  match id {
    Sha256KernelId::Portable => Caps::NONE,
    #[cfg(target_arch = "x86_64")]
    Sha256KernelId::X86Sha => x86::SHA,
    #[cfg(target_arch = "aarch64")]
    Sha256KernelId::Aarch64Sha2 => aarch64::SHA2,
    #[cfg(any(target_arch = "riscv64", target_arch = "riscv32"))]
    Sha256KernelId::RiscvZknh => riscv::ZKNH,
    #[cfg(target_arch = "wasm32")]
    Sha256KernelId::WasmSimd128 => wasm::SIMD128,
    #[cfg(target_arch = "s390x")]
    Sha256KernelId::S390xKimd => s390x::MSA,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compile-time dispatch bypass.
//
// When the target feature is statically known (e.g. `-C target-cpu=native` on
// SHA-NI hardware, or aarch64-apple-darwin where SHA2 is always on), callers
// can skip the OnceCache + function-pointer indirection entirely.
//
// No user-facing feature flags — the compiler sets `target_feature` cfg
// automatically from `-C target-cpu` or platform defaults.
// ─────────────────────────────────────────────────────────────────────────────

/// Whether the best SHA-256 kernel is known at compile time.
pub(crate) const COMPILE_TIME_HW: bool = cfg!(any(
  all(target_arch = "x86_64", target_feature = "sha"),
  all(target_arch = "aarch64", target_feature = "sha2"),
  all(
    any(target_arch = "riscv64", target_arch = "riscv32"),
    target_feature = "zknh"
  ),
  all(target_arch = "wasm32", target_feature = "simd128"),
));

/// Returns the compile-time-best compress function.
///
/// When [`COMPILE_TIME_HW`] is `true`, returns the HW-accelerated kernel.
/// Otherwise returns the portable fallback.  Marked `#[inline(always)]` so
/// LLVM sees a constant function pointer and can devirtualize through it.
#[inline(always)]
pub(crate) fn compile_time_best() -> CompressBlocksFn {
  #[cfg(all(target_arch = "x86_64", target_feature = "sha"))]
  {
    return compress_blocks_x86_sha;
  }
  #[cfg(all(target_arch = "aarch64", target_feature = "sha2"))]
  {
    return compress_blocks_aarch64_sha2;
  }
  #[cfg(all(any(target_arch = "riscv64", target_arch = "riscv32"), target_feature = "zknh"))]
  {
    return compress_blocks_riscv_zknh;
  }
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  {
    return compress_blocks_wasm_simd128;
  }
  #[allow(unreachable_code)]
  Sha256::compress_blocks_portable
}

/// Kernel name for the compile-time-best path (introspection).
#[cfg(any(test, feature = "diag"))]
pub(crate) const COMPILE_TIME_NAME: &str = if cfg!(all(target_arch = "x86_64", target_feature = "sha")) {
  "x86-sha"
} else if cfg!(all(target_arch = "aarch64", target_feature = "sha2")) {
  "aarch64-sha2"
} else if cfg!(all(
  any(target_arch = "riscv64", target_arch = "riscv32"),
  target_feature = "zknh"
)) {
  "riscv/zknh"
} else if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
  "wasm/simd128"
} else {
  "portable"
};
