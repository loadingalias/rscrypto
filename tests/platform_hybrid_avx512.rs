//! The hybrid AVX-512 opt-in is accepted only before detection and changes
//! nothing on a CPU that is not hybrid. Each integration test file is its own
//! process, so no other test can start detection first.

#![cfg(all(feature = "std", target_arch = "x86_64", not(miri)))]

use rscrypto::platform::{
  self,
  expert::{self, OverrideError},
};

/// CPUID.(EAX=07H, ECX=0):EDX[15] identifies an Intel hybrid part.
fn is_hybrid_part() -> bool {
  core::arch::x86_64::__cpuid(0).eax >= 7 && core::arch::x86_64::__cpuid_count(7, 0).edx & (1 << 15) != 0
}

#[test]
fn opt_in_is_accepted_only_before_detection() {
  let uncached = expert::detect_uncached();

  if is_hybrid_part() {
    // A hybrid host cannot prove the opt-in's precondition; test only the rejection below.
    assert_eq!(platform::get(), uncached);
  } else {
    // SAFETY: CPUID.(EAX=07H, ECX=0):EDX[15] is clear, so this is not a hybrid
    // part: every logical processor implements the same instructions, including
    // any AVX-512 that CPUID reports.
    unsafe { expert::allow_hybrid_avx512() }.expect("the opt-in must be accepted before detection");
    assert_eq!(
      platform::get(),
      uncached,
      "the opt-in changes nothing on a CPU that is not hybrid"
    );
  }

  // SAFETY: detection has started, so the call is rejected and enables nothing.
  let late = unsafe { expert::allow_hybrid_avx512() };
  assert_eq!(late, Err(OverrideError::AlreadyInitialized));
  assert_eq!(
    platform::get(),
    uncached,
    "a rejected opt-in leaves the cached result unchanged"
  );
}
