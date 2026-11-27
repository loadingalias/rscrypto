#![cfg(all(
  feature = "std",
  feature = "crc32",
  feature = "adler32",
  any(unix, windows),
  not(miri)
))]

use rscrypto::{
  Adler32, Checksum, Crc32C,
  checksum::introspect,
  platform::{Caps, Detected, expert},
};

#[test]
fn checksums_follow_the_process_capabilities() {
  const CRC: &str = "portable/slice16";
  const ADLER: &str = "portable/unroll16";
  const CASES: &[(Caps, &str, &str)] = &[
    (Caps::NONE, CRC, ADLER),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    (rscrypto::platform::caps::x86::SSSE3, CRC, "x86_64/ssse3"),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    (rscrypto::platform::caps::x86::CRC32C_READY, "x86_64/hwcrc", ADLER),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    (
      rscrypto::platform::caps::x86::PCLMUL_READY,
      "x86_64/pclmul",
      "x86_64/ssse3",
    ),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    (
      rscrypto::platform::caps::x86::VPCLMUL_READY,
      "x86_64/vpclmul",
      "x86_64/ssse3",
    ),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    (rscrypto::platform::caps::x86::AVX2, CRC, "x86_64/avx2"),
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    (
      rscrypto::platform::caps::x86::AVX512F
        .union(rscrypto::platform::caps::x86::AVX512BW)
        .union(rscrypto::platform::caps::x86::AVX512VNNI),
      CRC,
      "x86_64/avx512-vnni",
    ),
    #[cfg(target_arch = "aarch64")]
    (rscrypto::platform::caps::aarch64::CRC_READY, "aarch64/hwcrc", ADLER),
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    (rscrypto::platform::caps::aarch64::NEON, CRC, "aarch64/neon"),
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    (
      rscrypto::platform::caps::aarch64::NEON.union(rscrypto::platform::caps::aarch64::DOTPROD),
      CRC,
      "aarch64/neon-dotprod",
    ),
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    (
      rscrypto::platform::caps::aarch64::PMULL_READY.union(rscrypto::platform::caps::aarch64::NEON),
      "aarch64/pmull",
      "aarch64/neon",
    ),
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    (
      rscrypto::platform::caps::aarch64::PMULL_EOR3_READY
        .union(rscrypto::platform::caps::aarch64::NEON)
        .union(rscrypto::platform::caps::aarch64::DOTPROD),
      "aarch64/pmull-eor3",
      "aarch64/neon-dotprod",
    ),
    #[cfg(target_arch = "powerpc64")]
    (rscrypto::platform::caps::power::ALTIVEC, CRC, "power/vmsum"),
    #[cfg(target_arch = "powerpc64")]
    (
      rscrypto::platform::caps::power::VPMSUM_READY,
      "power/vpmsum",
      "power/vmsum",
    ),
    #[cfg(target_arch = "s390x")]
    (rscrypto::platform::caps::s390x::VECTOR, "s390x/vgfm", "s390x/vsum"),
    #[cfg(target_arch = "riscv64")]
    (rscrypto::platform::caps::riscv::V, CRC, "riscv64/v"),
    #[cfg(target_arch = "riscv64")]
    (rscrypto::platform::caps::riscv::ZBC, "riscv64/zbc", ADLER),
    #[cfg(target_arch = "riscv64")]
    (
      rscrypto::platform::caps::riscv::V.union(rscrypto::platform::caps::riscv::ZVBC),
      "riscv64/zvbc",
      "riscv64/v",
    ),
  ];
  const CHILD: &str = "RSCRYPTO_TEST_CHECKSUM_DISPATCH_CASE";
  let host = expert::detect_uncached();
  if let Some(case) = std::env::var_os(CHILD) {
    let (mask, crc, adler) = CASES[case
      .to_str()
      .expect("the parent sets an ASCII index")
      .parse::<usize>()
      .expect("the parent sets a case index")];
    expert::try_set_override(Some(Detected {
      caps: mask,
      arch: host.arch,
    }))
    .expect("supported capabilities before detection");
    assert_eq!(introspect::crc32c(15).selected_kernel, CRC);
    assert_eq!(introspect::crc32c(16).selected_kernel, crc);
    assert_eq!(introspect::kernel_for::<Adler32>(36), adler);
    let data = b"123456789123456789123456789123456789";
    assert_eq!(
      Crc32C::checksum(data),
      crc::Crc::<u32>::new(&crc::CRC_32_ISCSI).checksum(data)
    );
    // Python zlib.adler32 on the same input.
    assert_eq!(Adler32::checksum(data), 0x8916_0775);
    return;
  }
  let executable = std::env::current_exe().expect("checksum test executable");
  for (index, &(mask, crc, adler)) in CASES.iter().enumerate() {
    if host.caps.has(mask) && (!cfg!(feature = "portable-only") || mask.is_empty()) {
      let status = std::process::Command::new(&executable)
        .args(["--exact", "checksums_follow_the_process_capabilities"])
        .env(CHILD, index.to_string())
        .status()
        .expect("start the checksum test in a fresh process");
      assert!(status.success(), "{crc} and {adler}: {status}");
    }
  }
}
