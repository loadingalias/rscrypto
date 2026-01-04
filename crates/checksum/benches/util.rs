use std::sync::Once;

use checksum::{Crc16Ccitt, Crc16Ibm, Crc24OpenPgp, Crc32, Crc32C, Crc64, Crc64Nvme};

#[allow(dead_code)] // Used by `benches/comp.rs` (but not by `benches/kernels.rs`).
pub const BUFFERED_CHUNK_BYTES: usize = 31;

pub const CASES: &[(&str, usize)] = &[
  ("xs", 64),
  ("s", 256),
  ("m", 4usize.strict_mul(1024)),
  ("l", 64usize.strict_mul(1024)),
  ("xl", 1024usize.strict_mul(1024)),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Alignment {
  /// A regular `Vec<u8>` buffer (baseline).
  Vec,
  /// A slice starting at a 16B-aligned address.
  A16,
  /// A slice starting at a 32B-aligned address.
  A32,
  /// A slice starting at a 64B-aligned address.
  A64,
}

impl Alignment {
  pub const ALL: [Self; 4] = [Self::Vec, Self::A16, Self::A32, Self::A64];

  #[inline]
  #[must_use]
  pub const fn label(self) -> &'static str {
    match self {
      Self::Vec => "vec",
      Self::A16 => "a16",
      Self::A32 => "a32",
      Self::A64 => "a64",
    }
  }

  #[inline]
  #[must_use]
  pub const fn bytes(self) -> Option<usize> {
    match self {
      Self::Vec => None,
      Self::A16 => Some(16),
      Self::A32 => Some(32),
      Self::A64 => Some(64),
    }
  }
}

pub struct BenchData {
  alignment: Alignment,
  backing: Vec<u8>,
  offset: usize,
  len: usize,
}

impl BenchData {
  #[inline]
  #[must_use]
  pub fn from_vec(data: Vec<u8>) -> Self {
    let len = data.len();
    Self {
      alignment: Alignment::Vec,
      backing: data,
      offset: 0,
      len,
    }
  }

  #[inline]
  #[must_use]
  pub fn aligned_copy(src: &[u8], alignment: Alignment) -> Self {
    let Some(align_bytes) = alignment.bytes() else {
      return Self::from_vec(src.to_vec());
    };

    debug_assert!(align_bytes.is_power_of_two());

    let len = src.len();
    let backing_len = len.strict_add(align_bytes);
    let mut backing = vec![0u8; backing_len];

    let base = backing.as_ptr() as usize;
    let misalignment = base % align_bytes;
    let offset = if misalignment == 0 {
      0
    } else {
      align_bytes.strict_sub(misalignment)
    };
    let end = offset.strict_add(len);

    backing[offset..end].copy_from_slice(src);

    Self {
      alignment,
      backing,
      offset,
      len,
    }
  }

  #[inline]
  #[must_use]
  pub fn alignment(&self) -> Alignment {
    self.alignment
  }

  #[inline]
  #[must_use]
  pub fn as_slice(&self) -> &[u8] {
    let end = self.offset.strict_add(self.len);
    &self.backing[self.offset..end]
  }
}

#[must_use]
pub fn make_data(len: usize) -> Vec<u8> {
  (0..len)
    .map(|i| (i as u8).wrapping_mul(31).wrapping_add(i.strict_shr(8) as u8))
    .collect()
}

#[must_use]
pub fn make_alignment_variants(src: Vec<u8>) -> Vec<BenchData> {
  let src_ref: &[u8] = &src;
  let mut out = Vec::with_capacity(Alignment::ALL.len());

  for alignment in [Alignment::A16, Alignment::A32, Alignment::A64] {
    out.push(BenchData::aligned_copy(src_ref, alignment));
  }
  out.insert(0, BenchData::from_vec(src));

  out
}

#[inline]
#[must_use]
pub fn bench_param_label(size_label: &str, alignment: Alignment) -> String {
  format!("{size_label}@{}", alignment.label())
}

/// Print platform detection info once at benchmark start.
pub fn print_platform_info() {
  static ONCE: Once = Once::new();
  ONCE.call_once(|| {
    let tune = platform::tune();
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║                   PLATFORM DETECTION INFO                    ║");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║ Platform: {}", platform::describe());
    eprintln!("║ Tune Kind: {:?}", tune.kind());
    eprintln!("║ PCLMUL threshold: {} bytes", tune.pclmul_threshold);
    eprintln!("║ SIMD width: {} bits", tune.effective_simd_width);
    eprintln!("║ Fast wide ops: {}", tune.fast_wide_ops);
    eprintln!("║ Parallel streams: {}", tune.parallel_streams);
    eprintln!("║ Bench alignments: vec, a16, a32, a64");
    eprintln!("╠══════════════════════════════════════════════════════════════╣");
    eprintln!("║ Kernel selection by size:");
    for &(label, size) in CASES {
      let crc64 = Crc64::kernel_name_for_len(size);
      let crc64_nvme = Crc64Nvme::kernel_name_for_len(size);
      let crc32 = Crc32::kernel_name_for_len(size);
      let crc32c = Crc32C::kernel_name_for_len(size);
      let crc16_ccitt = Crc16Ccitt::kernel_name_for_len(size);
      let crc16_ibm = Crc16Ibm::kernel_name_for_len(size);
      let crc24 = Crc24OpenPgp::kernel_name_for_len(size);
      eprintln!(
        "║   {:>3} ({:>7} B): crc64/xz={crc64}  crc64/nvme={crc64_nvme}  crc32={crc32}  crc32c={crc32c}  \
         crc16/ccitt={crc16_ccitt}  crc16/ibm={crc16_ibm}  crc24/openpgp={crc24}",
        label, size
      );
    }
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
  });
}
