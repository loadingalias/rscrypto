use super::Sha256;
use crate::hashes::crypto::{Compress, WithCompress};
use crate::platform::Caps;
#[cfg(target_arch = "aarch64")]
use crate::platform::caps::aarch64;
#[cfg(all(
  any(target_arch = "riscv64", target_arch = "riscv32"),
  not(feature = "portable-only")
))]
use crate::platform::caps::riscv;
#[cfg(target_arch = "s390x")]
use crate::platform::caps::s390x;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::platform::caps::wasm;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
use crate::platform::caps::x86;

/// The selected compression kernel.
///
/// [`Self::with`] reaches each kernel through a direct call, so no code
/// pointer is stored or called, and a static stack review bounds every path.
#[derive(Clone, Copy)]
pub(crate) enum CompressBlocksFn {
  Kernel(Sha256KernelId),
  /// A PBKDF2 test's instrumented compression function.
  #[cfg(all(test, feature = "pbkdf2"))]
  Test(fn(&mut [u32; 8], &[u8])),
}

impl CompressBlocksFn {
  /// Run `op` with the selected kernel.
  ///
  /// Each kernel gets its own copy of `op`, so each compression in `op` is
  /// one direct call.
  #[inline(always)]
  pub(crate) fn with<O: WithCompress<[u32; 8]>>(self, op: O) -> O::Output {
    match self {
      Self::Kernel(Sha256KernelId::Portable) => op.run(Sha256::compress_blocks_portable),
      #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
      Self::Kernel(Sha256KernelId::X86Sha) => op.run(compress_blocks_x86_sha),
      #[cfg(target_arch = "aarch64")]
      Self::Kernel(Sha256KernelId::Aarch64Sha2) => op.run(compress_blocks_aarch64_sha2),
      #[cfg(all(
        any(target_arch = "riscv64", target_arch = "riscv32"),
        not(feature = "portable-only")
      ))]
      Self::Kernel(Sha256KernelId::RiscvZknh) => op.run(compress_blocks_riscv_zknh),
      #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
      Self::Kernel(Sha256KernelId::WasmSimd128) => op.run(compress_blocks_wasm_simd128),
      #[cfg(target_arch = "s390x")]
      Self::Kernel(Sha256KernelId::S390xKimd) => op.run(compress_blocks_s390x_kimd),
      #[cfg(all(test, feature = "pbkdf2"))]
      Self::Test(compress) => op.run(compress),
    }
  }

  /// Compress the whole blocks of `blocks` into `state`.
  #[inline(always)]
  pub(crate) fn call(self, state: &mut [u32; 8], blocks: &[u8]) {
    self.with(Compress(state, blocks));
  }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub(crate) enum Sha256KernelId {
  Portable = 0,
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  X86Sha = 1,
  #[cfg(target_arch = "aarch64")]
  Aarch64Sha2 = 2,
  #[cfg(all(
    any(target_arch = "riscv64", target_arch = "riscv32"),
    not(feature = "portable-only")
  ))]
  RiscvZknh = 3,
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  WasmSimd128 = 4,
  #[cfg(target_arch = "s390x")]
  S390xKimd = 5,
}

impl Sha256KernelId {
  #[cfg(any(test, feature = "diag"))]
  #[inline]
  #[must_use]
  pub(crate) const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
      Self::X86Sha => "x86-sha",
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Sha2 => "aarch64-sha2",
      #[cfg(all(
        any(target_arch = "riscv64", target_arch = "riscv32"),
        not(feature = "portable-only")
      ))]
      Self::RiscvZknh => "riscv/zknh",
      #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
      Self::WasmSimd128 => "wasm/simd128",
      #[cfg(target_arch = "s390x")]
      Self::S390xKimd => "s390x/kimd",
    }
  }
}

#[cfg(test)]
pub(crate) const ALL: &[Sha256KernelId] = &[
  Sha256KernelId::Portable,
  #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
  Sha256KernelId::X86Sha,
  #[cfg(target_arch = "aarch64")]
  Sha256KernelId::Aarch64Sha2,
  #[cfg(all(
    any(target_arch = "riscv64", target_arch = "riscv32"),
    not(feature = "portable-only")
  ))]
  Sha256KernelId::RiscvZknh,
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  Sha256KernelId::WasmSimd128,
  #[cfg(target_arch = "s390x")]
  Sha256KernelId::S390xKimd,
];

// Safe wrappers — dispatch validates caps before calling. Each wrapper is
// one out-of-line function, so no caller gets its own copy of a kernel.

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline(never)]
fn compress_blocks_x86_sha(state: &mut [u32; 8], blocks: &[u8]) {
  // SAFETY: SHA-NI compression through the dispatch wrapper because:
  // 1. Runtime dispatch only selects this wrapper when `sha` and `sse4.1` are available.
  // 2. Compile-time dispatch only selects this wrapper when `target_feature=sha,sse4.1`.
  // 3. Callers pass complete SHA-256 block slices; the backend asserts that invariant.
  unsafe { super::x86_64::compress_blocks_sha_ni(state, blocks) }
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn compress_blocks_aarch64_sha2(state: &mut [u32; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `aarch64::SHA2` is available.
  unsafe { super::aarch64::compress_blocks_aarch64_sha2(state, blocks) }
}

#[cfg(all(
  any(target_arch = "riscv64", target_arch = "riscv32"),
  not(feature = "portable-only")
))]
#[inline(never)]
fn compress_blocks_riscv_zknh(state: &mut [u32; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `riscv::ZKNH` is available.
  unsafe { super::riscv64::compress_blocks_zknh(state, blocks) }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(never)]
fn compress_blocks_wasm_simd128(state: &mut [u32; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `wasm::SIMD128` is available.
  unsafe { super::wasm::compress_blocks_wasm_simd(state, blocks) }
}

#[cfg(target_arch = "s390x")]
#[inline(never)]
fn compress_blocks_s390x_kimd(state: &mut [u32; 8], blocks: &[u8]) {
  // SAFETY: Only called when dispatch has verified `s390x::MSA` is available.
  unsafe { super::s390x::compress_blocks_kimd(state, blocks) }
}

/// The kernel `id`. Dispatch has already checked `id` against the CPU.
#[must_use]
pub(crate) const fn compress_blocks_fn(id: Sha256KernelId) -> CompressBlocksFn {
  CompressBlocksFn::Kernel(id)
}

#[inline]
#[must_use]
pub(crate) const fn required_caps(id: Sha256KernelId) -> Caps {
  match id {
    Sha256KernelId::Portable => Caps::NONE,
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    Sha256KernelId::X86Sha => x86::SHA.union(x86::SSE41),
    #[cfg(target_arch = "aarch64")]
    Sha256KernelId::Aarch64Sha2 => aarch64::SHA2,
    #[cfg(all(
      any(target_arch = "riscv64", target_arch = "riscv32"),
      not(feature = "portable-only")
    ))]
    Sha256KernelId::RiscvZknh => riscv::ZKNH,
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    Sha256KernelId::WasmSimd128 => wasm::SIMD128,
    #[cfg(target_arch = "s390x")]
    Sha256KernelId::S390xKimd => s390x::MSA,
  }
}

// Compile-time dispatch bypass.
//
// When the target feature is statically known (e.g. `-C target-cpu=native` on
// SHA-NI hardware, or aarch64-apple-darwin where SHA2 is always on), callers
// can skip the OnceCache lookup and the kernel match entirely.
//
// No user-facing feature flags — the compiler sets `target_feature` cfg
// automatically from `-C target-cpu` or platform defaults.

/// Whether the best SHA-256 kernel is known at compile time.
pub(crate) const COMPILE_TIME_HW: bool = cfg!(not(miri))
  && cfg!(any(
    all(
      all(target_arch = "x86_64", target_feature = "sse2"),
      target_feature = "sha",
      target_feature = "sse4.1"
    ),
    all(target_arch = "aarch64", target_os = "macos"),
    all(target_arch = "aarch64", target_feature = "sha2"),
    all(
      any(target_arch = "riscv64", target_arch = "riscv32"),
      not(feature = "portable-only"),
      target_feature = "zknh"
    ),
    all(target_arch = "wasm32", target_feature = "simd128"),
  ));

/// Returns the compile-time-best compress function.
///
/// When [`COMPILE_TIME_HW`] is `true`, returns the HW-accelerated kernel.
/// Otherwise returns the portable fallback.  Marked `#[inline(always)]` so
/// LLVM sees a constant kernel and removes the kernel match.
#[inline(always)]
pub(crate) fn compile_time_best() -> CompressBlocksFn {
  #[cfg(miri)]
  {
    compress_blocks_fn(Sha256KernelId::Portable)
  }
  #[cfg(all(
    not(miri),
    all(target_arch = "x86_64", target_feature = "sse2"),
    target_feature = "sha",
    target_feature = "sse4.1"
  ))]
  {
    compress_blocks_fn(Sha256KernelId::X86Sha)
  }
  #[cfg(all(
    not(miri),
    target_arch = "aarch64",
    any(target_os = "macos", target_feature = "sha2")
  ))]
  {
    compress_blocks_fn(Sha256KernelId::Aarch64Sha2)
  }
  #[cfg(all(
    not(miri),
    any(target_arch = "riscv64", target_arch = "riscv32"),
    not(feature = "portable-only"),
    target_feature = "zknh"
  ))]
  {
    compress_blocks_fn(Sha256KernelId::RiscvZknh)
  }
  #[cfg(all(not(miri), target_arch = "wasm32", target_feature = "simd128"))]
  {
    compress_blocks_fn(Sha256KernelId::WasmSimd128)
  }
  #[cfg(all(
    not(miri),
    not(any(
      all(
        target_arch = "x86_64",
        target_feature = "sse2",
        target_feature = "sha",
        target_feature = "sse4.1"
      ),
      all(target_arch = "aarch64", any(target_os = "macos", target_feature = "sha2")),
      all(
        any(target_arch = "riscv64", target_arch = "riscv32"),
        not(feature = "portable-only"),
        target_feature = "zknh"
      ),
      all(target_arch = "wasm32", target_feature = "simd128")
    ))
  ))]
  {
    compress_blocks_fn(Sha256KernelId::Portable)
  }
}

/// Kernel name for the compile-time-best path (introspection).
#[cfg(feature = "diag")]
pub(crate) const COMPILE_TIME_NAME: &str = if cfg!(miri) {
  "portable"
} else if cfg!(all(
  all(target_arch = "x86_64", target_feature = "sse2"),
  target_feature = "sha",
  target_feature = "sse4.1"
)) {
  "x86-sha"
} else if cfg!(all(
  target_arch = "aarch64",
  any(target_os = "macos", target_feature = "sha2")
)) {
  "aarch64-sha2"
} else if cfg!(all(
  any(
    all(target_arch = "riscv64", not(feature = "portable-only")),
    all(target_arch = "riscv32", not(feature = "portable-only"))
  ),
  target_feature = "zknh"
)) {
  "riscv/zknh"
} else if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
  "wasm/simd128"
} else {
  "portable"
};
