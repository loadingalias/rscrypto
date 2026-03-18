use super::Sha512;
use crate::platform::Caps;
#[cfg(target_arch = "aarch64")]
use crate::platform::caps::aarch64;
#[cfg(target_arch = "riscv64")]
use crate::platform::caps::riscv;
#[cfg(target_arch = "wasm32")]
use crate::platform::caps::wasm;
#[cfg(target_arch = "x86_64")]
use crate::platform::caps::x86;

pub(crate) type CompressBlocksFn = fn(&mut [u64; 8], &[u8]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Sha512KernelId {
  Portable = 0,
  #[cfg(target_arch = "aarch64")]
  Aarch64Sha512 = 1,
  #[cfg(target_arch = "x86_64")]
  X86Sha512 = 2,
  #[cfg(target_arch = "x86_64")]
  X86Avx512vl = 5,
  #[cfg(target_arch = "x86_64")]
  X86Avx2 = 6,
  #[cfg(target_arch = "riscv64")]
  Riscv64Zknh = 3,
  #[cfg(target_arch = "wasm32")]
  WasmSimd128 = 4,
}

#[cfg(any(test, feature = "std"))]
pub const ALL: &[Sha512KernelId] = &[
  Sha512KernelId::Portable,
  #[cfg(target_arch = "aarch64")]
  Sha512KernelId::Aarch64Sha512,
  #[cfg(target_arch = "x86_64")]
  Sha512KernelId::X86Sha512,
  #[cfg(target_arch = "x86_64")]
  Sha512KernelId::X86Avx512vl,
  #[cfg(target_arch = "x86_64")]
  Sha512KernelId::X86Avx2,
  #[cfg(target_arch = "riscv64")]
  Sha512KernelId::Riscv64Zknh,
  #[cfg(target_arch = "wasm32")]
  Sha512KernelId::WasmSimd128,
];

impl Sha512KernelId {
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Sha512 => "aarch64-sha512",
      #[cfg(target_arch = "x86_64")]
      Self::X86Sha512 => "x86-sha512",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx512vl => "x86-avx512vl",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx2 => "x86-avx2",
      #[cfg(target_arch = "riscv64")]
      Self::Riscv64Zknh => "riscv/zknh",
      #[cfg(target_arch = "wasm32")]
      Self::WasmSimd128 => "wasm/simd128",
    }
  }
}

#[allow(dead_code)]
#[must_use]
pub fn id_from_name(name: &str) -> Option<Sha512KernelId> {
  match name {
    "portable" => Some(Sha512KernelId::Portable),
    #[cfg(target_arch = "aarch64")]
    "aarch64-sha512" => Some(Sha512KernelId::Aarch64Sha512),
    #[cfg(target_arch = "x86_64")]
    "x86-sha512" => Some(Sha512KernelId::X86Sha512),
    #[cfg(target_arch = "x86_64")]
    "x86-avx512vl" => Some(Sha512KernelId::X86Avx512vl),
    #[cfg(target_arch = "x86_64")]
    "x86-avx2" => Some(Sha512KernelId::X86Avx2),
    #[cfg(target_arch = "riscv64")]
    "riscv/zknh" => Some(Sha512KernelId::Riscv64Zknh),
    #[cfg(target_arch = "wasm32")]
    "wasm/simd128" => Some(Sha512KernelId::WasmSimd128),
    _ => None,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Safe wrappers — dispatch validates caps before calling.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
fn compress_blocks_aarch64_sha512(state: &mut [u64; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `aarch64::SHA512` is available.
  unsafe { super::aarch64::compress_blocks_aarch64_sha512(state, blocks) }
}

#[cfg(target_arch = "x86_64")]
fn compress_blocks_x86_sha512(state: &mut [u64; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `x86::SHA512` is available.
  unsafe { super::x86_64::compress_blocks_sha512_ni(state, blocks) }
}

#[cfg(target_arch = "x86_64")]
fn compress_blocks_x86_avx512vl(state: &mut [u64; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified AVX-512VL is available.
  unsafe { super::x86_64_avx512vl::compress_blocks_avx512vl(state, blocks) }
}

#[cfg(target_arch = "x86_64")]
fn compress_blocks_x86_avx2(state: &mut [u64; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `x86::AVX2` is available.
  unsafe { super::x86_64_avx2::compress_blocks_avx2(state, blocks) }
}

#[cfg(target_arch = "riscv64")]
fn compress_blocks_riscv_zknh(state: &mut [u64; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `riscv::ZKNH` is available.
  unsafe { super::riscv64::compress_blocks_zknh(state, blocks) }
}

#[cfg(target_arch = "wasm32")]
fn compress_blocks_wasm_simd128(state: &mut [u64; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `wasm::SIMD128` is available.
  unsafe { super::wasm::compress_blocks_wasm_simd(state, blocks) }
}

#[must_use]
pub(crate) fn compress_blocks_fn(id: Sha512KernelId) -> CompressBlocksFn {
  match id {
    Sha512KernelId::Portable => Sha512::compress_blocks_portable,
    #[cfg(target_arch = "aarch64")]
    Sha512KernelId::Aarch64Sha512 => compress_blocks_aarch64_sha512,
    #[cfg(target_arch = "x86_64")]
    Sha512KernelId::X86Sha512 => compress_blocks_x86_sha512,
    #[cfg(target_arch = "x86_64")]
    Sha512KernelId::X86Avx512vl => compress_blocks_x86_avx512vl,
    #[cfg(target_arch = "x86_64")]
    Sha512KernelId::X86Avx2 => compress_blocks_x86_avx2,
    #[cfg(target_arch = "riscv64")]
    Sha512KernelId::Riscv64Zknh => compress_blocks_riscv_zknh,
    #[cfg(target_arch = "wasm32")]
    Sha512KernelId::WasmSimd128 => compress_blocks_wasm_simd128,
  }
}

#[inline]
#[must_use]
pub const fn required_caps(id: Sha512KernelId) -> Caps {
  match id {
    Sha512KernelId::Portable => Caps::NONE,
    #[cfg(target_arch = "aarch64")]
    Sha512KernelId::Aarch64Sha512 => aarch64::SHA512,
    #[cfg(target_arch = "x86_64")]
    Sha512KernelId::X86Sha512 => x86::SHA512.union(x86::AVX2),
    #[cfg(target_arch = "x86_64")]
    Sha512KernelId::X86Avx512vl => x86::AVX512F.union(x86::AVX512VL),
    #[cfg(target_arch = "x86_64")]
    Sha512KernelId::X86Avx2 => x86::AVX2,
    #[cfg(target_arch = "riscv64")]
    Sha512KernelId::Riscv64Zknh => riscv::ZKNH,
    #[cfg(target_arch = "wasm32")]
    Sha512KernelId::WasmSimd128 => wasm::SIMD128,
  }
}
