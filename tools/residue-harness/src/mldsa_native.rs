//! Native ML-DSA residue measurement on a private, painted stack.
//!
//! The capture build uses a retained patch in an isolated source copy to emit
//! production intermediate values. The measured build has no capture hooks.
//! Signatures bind each capture to its measured call. This is a functional
//! cleanup experiment; it records no timings.

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
mod native_stack;

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
mod native {
  extern crate alloc;

  use core::hint::black_box;
  use std::{fs, path::Path};

  use rscrypto::{MlDsa44, MlDsa65, MlDsa87};

  use crate::native_stack::{hex, input, measure, plant};

  #[cfg(feature = "capture-mldsa")]
  #[unsafe(no_mangle)]
  extern "C" fn rscrypto_residue_observe(attempt: usize, stage: u32, index: usize, data: *const u8, len: usize) {
    // SAFETY: the isolated capture patch calls synchronously with initialized
    // production scratch, borrowing `len` bytes until this function returns.
    let bytes = unsafe { core::slice::from_raw_parts(data, len) };
    println!("capture {attempt} {stage} {index} {}", hex(bytes));
  }

  macro_rules! run_set {
    ($directory:expr, $start:expr, $count:expr, $name:literal, $set:ident) => {{
      let (public, secret) = $set::keypair_from_seed(&input(0, 0)).expect("key generation");
      let prepared = secret.prepare_in(alloc::alloc::Global).expect("preparation");
      for index in $start..$start.strict_add($count) {
        let message = input(1, index);
        let random = input(2, index);
        for hedged in [false, true] {
          for prepared_form in [false, true] {
            let name = format!(
              "{}-{}-{}-{index}",
              $name,
              if prepared_form { "prepared" } else { "compact" },
              if hedged { "hedged" } else { "deterministic" }
            );
            println!("case {name}");
            let mut signature = None;
            let stack = measure(|| {
              let fill = |out: &mut [u8]| {
                out.copy_from_slice(black_box(&random));
                Ok::<(), rscrypto::MlDsaError>(())
              };
              signature = Some(
                match (prepared_form, hedged) {
                  (false, false) => secret.sign_deterministic(black_box(&message), b"residue"),
                  (false, true) => secret.sign_with(black_box(&message), b"residue", fill),
                  (true, false) => prepared.sign_deterministic(black_box(&message), b"residue"),
                  (true, true) => prepared.sign_with(black_box(&message), b"residue", fill),
                }
                .expect("signature"),
              );
            });
            let signature = signature.expect("measured signature");
            public
              .verify_with_context(&message, b"residue", &signature)
              .expect("signature verifies");
            println!("signature {}", hex(signature.as_bytes()));
            if !cfg!(feature = "capture-mldsa") {
              fs::write($directory.join(format!("{name}.stack")), &stack).expect("write stack snapshot");
            }
            println!("end");
          }
        }
      }
    }};
  }

  pub(super) fn run() {
    let arguments: Vec<String> = std::env::args().collect();
    let directory = Path::new(arguments.get(1).expect("output directory"));
    fs::create_dir_all(directory).expect("output directory");
    if let Some(path) = arguments.get(2).and_then(|value| value.strip_prefix("control=")) {
      let needle = fs::read(path).expect("control needle");
      for clear in [false, true] {
        let stack = measure(|| plant(&needle, clear));
        fs::write(
          directory.join(if clear { "cleared.stack" } else { "planted.stack" }),
          stack,
        )
        .expect("control stack snapshot");
      }
      return;
    }
    let count: u32 = arguments.get(2).map_or(1, |value| value.parse().expect("case count"));
    let start: u32 = arguments
      .get(3)
      .map_or(0, |value| value.parse().expect("first message index"));
    assert!((1..=128).contains(&count), "case count 1..=128");
    assert!(start <= 4096, "first message index 0..=4096");
    run_set!(directory, start, count, "mldsa44", MlDsa44);
    run_set!(directory, start, count, "mldsa65", MlDsa65);
    run_set!(directory, start, count, "mldsa87", MlDsa87);
  }
}

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
fn main() {
  native::run();
}

#[cfg(not(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
)))]
fn main() {
  eprintln!("native ML-DSA residue measurement requires macOS AArch64 or Linux AArch64/x86-64");
  std::process::exit(2);
}
