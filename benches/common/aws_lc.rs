//! AWS-LC benchmark availability follows the dev-dependency targets in Cargo.toml.

#[cfg(any(
  all(
    any(unix, windows),
    not(target_arch = "wasm32"),
    not(any(target_arch = "s390x", target_arch = "powerpc64"))
  ),
  all(
    target_arch = "powerpc64",
    target_endian = "little",
    target_os = "linux",
    target_env = "gnu"
  )
))]
macro_rules! aws_lc_bench {
  ($($tokens:tt)*) => { $($tokens)* };
}

#[cfg(not(any(
  all(
    any(unix, windows),
    not(target_arch = "wasm32"),
    not(any(target_arch = "s390x", target_arch = "powerpc64"))
  ),
  all(
    target_arch = "powerpc64",
    target_endian = "little",
    target_os = "linux",
    target_env = "gnu"
  )
)))]
macro_rules! aws_lc_bench {
  ($($tokens:tt)*) => {};
}

pub(crate) use aws_lc_bench;
