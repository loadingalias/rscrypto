#![cfg(all(feature = "std", target_arch = "x86_64", target_os = "linux", not(miri)))]
#![allow(unsafe_code)]

use rscrypto::platform::{self, caps::x86};

const CHILD_MODE: &str = "RSCRYPTO_PLATFORM_AMX_CHILD";
const REQUIRE_AMX: &str = "RSCRYPTO_REQUIRE_AMX";
const CACHE_TRANSITION: &str = "cache-transition";
const REQUEST_BEFORE_CACHE: &str = "request-before-cache";
const ARCH_GET_XCOMP_PERM: usize = 0x1022;
const ARCH_REQ_XCOMP_PERM: usize = 0x1023;
const XFEATURE_XTILEDATA: usize = 18;
const XCOMP_TILE_MASK: u64 = (1 << 17) | (1 << 18);

#[allow(unused_unsafe)]
fn cpu_supports_amx_tile() -> bool {
  // MSRV: CPUID is unsafe on Rust 1.91 but safe on the pinned nightly.
  // SAFETY: CPUID is a non-privileged x86-64 identification instruction.
  let leaf0 = unsafe { core::arch::x86_64::__cpuid(0) };
  if leaf0.eax < 7 {
    return false;
  }
  // SAFETY: CPUID leaf 7, subleaf 0 is valid because leaf 0 reports support
  // for leaf 7; the intrinsic only returns register values.
  let leaf7 = unsafe { core::arch::x86_64::__cpuid_count(7, 0) };
  leaf7.edx & (1 << 24) != 0
}

fn xcomp_permissions() -> Option<u64> {
  const SYS_ARCH_PRCTL: isize = 158;

  let mut permissions = 0u64;
  let mut result = SYS_ARCH_PRCTL;

  // SAFETY: Linux x86-64 syscall ABI invocation because:
  // 1. syscall 158 is arch_prctl on this test's cfg-constrained target;
  // 2. ARCH_GET_XCOMP_PERM writes one u64 through the valid exclusive pointer in RSI and does not
  //    retain it; and
  // 3. syscall clobbers RAX, RCX, and R11, all of which are declared.
  unsafe {
    core::arch::asm!(
      "syscall",
      inlateout("rax") result,
      in("rdi") ARCH_GET_XCOMP_PERM,
      in("rsi") &mut permissions,
      lateout("rcx") _,
      lateout("r11") _,
      options(nostack),
    );
  }

  (result == 0).then_some(permissions)
}

fn request_tile_data_permission() -> bool {
  const SYS_ARCH_PRCTL: isize = 158;

  let mut result = SYS_ARCH_PRCTL;

  // SAFETY: Linux x86-64 syscall ABI invocation because:
  // 1. syscall 158 is arch_prctl on this test's cfg-constrained target;
  // 2. ARCH_REQ_XCOMP_PERM consumes the immediate XFEATURE_XTILEDATA number in RSI and does not
  //    dereference it; and
  // 3. syscall clobbers RAX, RCX, and R11, all of which are declared.
  unsafe {
    core::arch::asm!(
      "syscall",
      inlateout("rax") result,
      in("rdi") ARCH_REQ_XCOMP_PERM,
      in("rsi") XFEATURE_XTILEDATA,
      lateout("rcx") _,
      lateout("r11") _,
      options(nostack),
    );
  }

  result == 0
}

fn require_amx() -> bool {
  std::env::var_os(REQUIRE_AMX).is_some()
}

fn skip_or_fail(reason: &str) {
  if require_amx() {
    panic!("{reason}");
  }
  eprintln!("skipping AMX permission transition: {reason}");
}

fn cache_transition_child() {
  if !cpu_supports_amx_tile() {
    skip_or_fail("CPU does not report AMX-TILE");
    return;
  }

  let Some(before_permissions) = xcomp_permissions() else {
    skip_or_fail("ARCH_GET_XCOMP_PERM is unavailable");
    return;
  };
  if require_amx() {
    assert_eq!(
      before_permissions & XCOMP_TILE_MASK,
      1 << 17,
      "fresh process must begin with tile configuration but not tile-data permission"
    );
  }

  let before = platform::caps();
  assert_eq!(
    before.has(x86::AMX_TILE),
    before_permissions & XCOMP_TILE_MASK == XCOMP_TILE_MASK,
    "cached detection must match the process permission visible at initialization"
  );

  if !request_tile_data_permission() {
    skip_or_fail("ARCH_REQ_XCOMP_PERM rejected XFEATURE_XTILEDATA");
    return;
  }
  let after_permissions = xcomp_permissions().expect("permission query must succeed after a successful request");
  assert_eq!(
    after_permissions & XCOMP_TILE_MASK,
    XCOMP_TILE_MASK,
    "successful tile-data request must authorize both AMX state components"
  );

  assert_eq!(
    platform::caps(),
    before,
    "cached detection must not silently change after initialization"
  );
  assert!(
    platform::expert::detect_uncached().caps.has(x86::AMX_TILE),
    "fresh detection after permission must publish AMX-TILE"
  );
}

fn request_before_cache_child() {
  if !cpu_supports_amx_tile() {
    skip_or_fail("CPU does not report AMX-TILE");
    return;
  }
  if !request_tile_data_permission() {
    skip_or_fail("ARCH_REQ_XCOMP_PERM rejected XFEATURE_XTILEDATA");
    return;
  }

  assert!(
    platform::caps().has(x86::AMX_TILE),
    "permission requested before cached detection must make AMX-TILE available"
  );
}

#[test]
fn linux_x86_64_amx_permission_and_cache_are_process_scoped() {
  match std::env::var(CHILD_MODE).as_deref() {
    Ok(CACHE_TRANSITION) => {
      cache_transition_child();
      return;
    }
    Ok(REQUEST_BEFORE_CACHE) => {
      request_before_cache_child();
      return;
    }
    Ok(other) => panic!("unknown AMX child mode: {other}"),
    Err(std::env::VarError::NotPresent) => {}
    Err(error) => panic!("invalid AMX child mode: {error}"),
  }

  let executable = std::env::current_exe().expect("current test executable");
  for mode in [CACHE_TRANSITION, REQUEST_BEFORE_CACHE] {
    let status = std::process::Command::new(&executable)
      .arg("--exact")
      .arg("linux_x86_64_amx_permission_and_cache_are_process_scoped")
      .arg("--nocapture")
      .env(CHILD_MODE, mode)
      .status()
      .expect("spawn isolated AMX detector process");
    assert!(status.success(), "AMX detector child failed in mode {mode}");
  }
}
