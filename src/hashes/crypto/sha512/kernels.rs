use super::Sha512;
use crate::platform::Caps;
#[cfg(target_arch = "aarch64")]
use crate::platform::caps::aarch64;
#[cfg(target_arch = "riscv64")]
use crate::platform::caps::riscv;
#[cfg(target_arch = "s390x")]
use crate::platform::caps::s390x;
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
  #[cfg(target_arch = "riscv64")]
  Riscv64Zknh = 3,
  #[cfg(target_arch = "wasm32")]
  WasmSimd128 = 4,
  #[cfg(target_arch = "s390x")]
  S390xKimd = 7,
  #[cfg(target_arch = "x86_64")]
  X86Avx2Decoupled = 11,
  #[cfg(target_arch = "x86_64")]
  X86Avx512vlDecoupled = 12,
}

impl Sha512KernelId {
  #[cfg(any(test, feature = "diag"))]
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
      #[cfg(target_arch = "riscv64")]
      Self::Riscv64Zknh => "riscv/zknh",
      #[cfg(target_arch = "wasm32")]
      Self::WasmSimd128 => "wasm/simd128",
      #[cfg(target_arch = "s390x")]
      Self::S390xKimd => "s390x/kimd",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx2Decoupled => "x86-avx2-decoupled",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx512vlDecoupled => "x86-avx512vl-decoupled",
    }
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
  // SAFETY: Only called when dispatch has verified AVX-512VL and BMI2 are available.
  unsafe { super::x86_64_avx512vl::compress_blocks_avx512vl(state, blocks) }
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

#[cfg(target_arch = "s390x")]
fn compress_blocks_s390x_kimd(state: &mut [u64; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `s390x::MSA` is available.
  unsafe { super::s390x::compress_blocks_kimd(state, blocks) }
}

#[cfg(target_arch = "x86_64")]
fn compress_blocks_x86_avx2_decoupled(state: &mut [u64; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `x86::AVX2` and `x86::BMI2` are available.
  unsafe { super::x86_64_avx2::compress_blocks_avx2_decoupled(state, blocks) }
}

#[cfg(target_arch = "x86_64")]
fn compress_blocks_x86_avx512vl_decoupled(state: &mut [u64; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified AVX-512VL and BMI2 are available.
  unsafe { super::x86_64_avx512vl::compress_blocks_avx512vl_decoupled(state, blocks) }
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
    #[cfg(target_arch = "riscv64")]
    Sha512KernelId::Riscv64Zknh => compress_blocks_riscv_zknh,
    #[cfg(target_arch = "wasm32")]
    Sha512KernelId::WasmSimd128 => compress_blocks_wasm_simd128,
    #[cfg(target_arch = "s390x")]
    Sha512KernelId::S390xKimd => compress_blocks_s390x_kimd,
    #[cfg(target_arch = "x86_64")]
    Sha512KernelId::X86Avx2Decoupled => compress_blocks_x86_avx2_decoupled,
    #[cfg(target_arch = "x86_64")]
    Sha512KernelId::X86Avx512vlDecoupled => compress_blocks_x86_avx512vl_decoupled,
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
    Sha512KernelId::X86Avx512vl => x86::AVX512F.union(x86::AVX512VL).union(x86::BMI2),
    #[cfg(target_arch = "riscv64")]
    Sha512KernelId::Riscv64Zknh => riscv::ZKNH,
    #[cfg(target_arch = "wasm32")]
    Sha512KernelId::WasmSimd128 => wasm::SIMD128,
    #[cfg(target_arch = "s390x")]
    Sha512KernelId::S390xKimd => s390x::MSA,
    #[cfg(target_arch = "x86_64")]
    Sha512KernelId::X86Avx2Decoupled => x86::AVX2.union(x86::BMI2),
    #[cfg(target_arch = "x86_64")]
    Sha512KernelId::X86Avx512vlDecoupled => x86::AVX512F.union(x86::AVX512VL).union(x86::BMI2),
  }
}

// Keep kernel tests focused on backends that runtime dispatch can actually pick.
#[cfg(test)]
pub const ALL: &[Sha512KernelId] = &[
  Sha512KernelId::Portable,
  #[cfg(target_arch = "aarch64")]
  Sha512KernelId::Aarch64Sha512,
  #[cfg(target_arch = "x86_64")]
  Sha512KernelId::X86Sha512,
  #[cfg(target_arch = "riscv64")]
  Sha512KernelId::Riscv64Zknh,
  #[cfg(target_arch = "wasm32")]
  Sha512KernelId::WasmSimd128,
  #[cfg(target_arch = "s390x")]
  Sha512KernelId::S390xKimd,
  #[cfg(target_arch = "x86_64")]
  Sha512KernelId::X86Avx2Decoupled,
  #[cfg(target_arch = "x86_64")]
  Sha512KernelId::X86Avx512vlDecoupled,
];
