use super::kernels::AsconPermute12KernelId;

/// Single-state hashes, XOFs, and the scalar batch tail use the inline portable permutation.
pub(crate) const fn scalar_kernel_id() -> AsconPermute12KernelId {
  AsconPermute12KernelId::Portable
}

#[cfg(feature = "std")]
#[inline]
#[must_use]
pub(crate) fn batch_kernel_id_for_count(_count: usize) -> AsconPermute12KernelId {
  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
  let caps = crate::platform::caps();

  #[cfg(target_arch = "x86_64")]
  {
    use super::kernels::required_caps;
    if _count >= 8 && caps.has(required_caps(AsconPermute12KernelId::X86Avx512)) {
      return AsconPermute12KernelId::X86Avx512;
    }
    if _count >= 4 && caps.has(required_caps(AsconPermute12KernelId::X86Avx2)) {
      return AsconPermute12KernelId::X86Avx2;
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    use super::kernels::required_caps;
    if _count >= 2 && caps.has(required_caps(AsconPermute12KernelId::Aarch64Neon)) {
      return AsconPermute12KernelId::Aarch64Neon;
    }
  }

  scalar_kernel_id()
}

#[cfg(feature = "std")]
#[inline]
#[must_use]
pub(crate) fn batch_fallback_kernel_id(requested: AsconPermute12KernelId, _count: usize) -> AsconPermute12KernelId {
  match requested {
    AsconPermute12KernelId::Portable => AsconPermute12KernelId::Portable,
    #[cfg(target_arch = "aarch64")]
    AsconPermute12KernelId::Aarch64Neon => {
      if _count >= 2 {
        AsconPermute12KernelId::Aarch64Neon
      } else {
        scalar_kernel_id()
      }
    }
    #[cfg(target_arch = "x86_64")]
    AsconPermute12KernelId::X86Avx2 => {
      if _count >= 4 {
        AsconPermute12KernelId::X86Avx2
      } else {
        scalar_kernel_id()
      }
    }
    #[cfg(target_arch = "x86_64")]
    AsconPermute12KernelId::X86Avx512 => {
      if _count >= 8 {
        AsconPermute12KernelId::X86Avx512
      } else {
        batch_kernel_id_for_count(_count)
      }
    }
  }
}
