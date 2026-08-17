// s390x (IBM Z) Detection

#[cfg(target_arch = "s390x")]
fn detect_s390x() -> Detected {
  // Start with compile-time detected features.
  #[cfg(feature = "std")]
  let caps = caps_static() | runtime_s390x();
  #[cfg(not(feature = "std"))]
  let caps = caps_static();

  Detected {
    caps,
    arch: Arch::S390x,
  }
}

#[cfg(all(target_arch = "s390x", feature = "std"))]
fn runtime_s390x() -> Caps {
  runtime_s390x_linux()
}

#[cfg(all(
  target_arch = "s390x",
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
fn runtime_s390x_linux() -> Caps {
  use std::{fs::File, io::Read};

  use crate::platform::caps::s390x;

  // ELF auxiliary vector type
  const AT_HWCAP: u64 = 16;

  // HWCAP bit positions from Linux s390 ABI.
  const HWCAP_MSA: u64 = 1 << 3;
  const HWCAP_VXRS: u64 = 1 << 11;
  const HWCAP_VXRS_BCD: u64 = 1 << 12;
  const HWCAP_VXRS_EXT: u64 = 1 << 13;
  const HWCAP_VXRS_EXT2: u64 = 1 << 15;
  const HWCAP_SORT: u64 = 1 << 17;
  const HWCAP_DFLT: u64 = 1 << 18;
  const HWCAP_NNPA: u64 = 1 << 20;

  let hwcap = (|| -> Option<u64> {
    let mut file = File::open("/proc/self/auxv").ok()?;
    let mut buf = [0u8; 4096];
    let n = file.read(&mut buf).ok()?;

    for chunk in buf.get(..n)?.as_chunks::<16>().0 {
      let (types, values) = chunk.as_chunks::<8>().0.split_at(1);
      let a_type = u64::from_ne_bytes(types[0]);
      let a_val = u64::from_ne_bytes(values[0]);
      if a_type == AT_HWCAP {
        return Some(a_val);
      }
      if a_type == 0 {
        break;
      }
    }
    None
  })()
  .unwrap_or(0);

  let stfle = stfle_facilities();

  #[inline(always)]
  fn has_facility(words: &[u64; 4], bit: u8) -> bool {
    let word_index = usize::from(bit >> 6);
    let shift = 63u32.strict_sub(u32::from(bit & 63));
    words
      .get(word_index)
      .map(|word| word & 1u64.strict_shl(shift) != 0)
      .unwrap_or(false)
  }

  let mut caps = Caps::NONE;

  // Vector facilities need both hardware and kernel vector support.
  if hwcap & HWCAP_VXRS != 0 {
    caps |= s390x::VECTOR;
    if hwcap & HWCAP_VXRS_EXT != 0 {
      caps |= s390x::VECTOR_ENH1;
    }
    if hwcap & HWCAP_VXRS_EXT2 != 0 {
      caps |= s390x::VECTOR_ENH2;
    }
    if has_facility(&stfle, 198) {
      caps |= s390x::VECTOR_ENH3;
    }
    if hwcap & HWCAP_VXRS_BCD != 0 {
      caps |= s390x::VECTOR_PD;
    }
    if hwcap & HWCAP_NNPA != 0 {
      caps |= s390x::NNP_ASSIST;
    }
  }

  // CPACF / MSA facilities.
  if hwcap & HWCAP_MSA != 0 || has_facility(&stfle, 76) {
    caps |= s390x::MSA;
  }
  if has_facility(&stfle, 77) {
    caps |= s390x::MSA | s390x::MSA4;
  }
  if has_facility(&stfle, 57) {
    caps |= s390x::MSA | s390x::MSA5;
  }
  if has_facility(&stfle, 146) {
    caps |= s390x::MSA | s390x::MSA8;
  }
  if has_facility(&stfle, 155) {
    caps |= s390x::MSA | s390x::MSA9;
  }

  // Misc facilities.
  if has_facility(&stfle, 58) {
    caps |= s390x::MISC_EXT2;
  }
  if has_facility(&stfle, 61) {
    caps |= s390x::MISC_EXT3;
  }
  if hwcap & HWCAP_DFLT != 0 || has_facility(&stfle, 151) {
    caps |= s390x::DEFLATE;
  }
  if hwcap & HWCAP_SORT != 0 || has_facility(&stfle, 150) {
    caps |= s390x::ENHANCED_SORT;
  }

  caps
}

#[cfg(all(
  target_arch = "s390x",
  feature = "std",
  not(any(target_os = "linux", target_os = "android"))
))]
fn runtime_s390x_linux() -> Caps {
  // No stable cross-platform source for these facilities outside Linux/Android.
  Caps::NONE
}

#[cfg(all(
  target_arch = "s390x",
  feature = "std",
  any(target_os = "linux", target_os = "android")
))]
#[inline]
fn stfle_facilities() -> [u64; 4] {
  let mut facilities = [0u64; 4];
  // SAFETY: `stfle` is part of the s390x baseline for Linux user space.
  unsafe {
    core::arch::asm!(
      "stfle 0({ptr})",
      ptr = in(reg) facilities.as_mut_ptr(),
      inout("r0") 3u64 => _,
      options(nostack)
    );
  }
  facilities
}

#[cfg(all(
  target_arch = "s390x",
  feature = "std",
  not(any(target_os = "linux", target_os = "android"))
))]
#[inline]
fn stfle_facilities() -> [u64; 4] {
  [0; 4]
}
