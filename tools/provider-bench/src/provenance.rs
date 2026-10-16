//! Build and runtime identity of the comparison libraries, for retained results.

use core::fmt::Write as _;
use serde_json::{Value, json};

/// Identity of the linked OpenSSL, Rust code-generation flags and capability overrides.
///
/// Records OpenSSL's version, configure flags and platform; the path and
/// SHA-256 of each loaded libssl and libcrypto image (Linux, from
/// `/proc/self/maps`; none means a static link into the executable), or of the
/// libraries under the build's `OPENSSL_DIR` elsewhere; `RUSTFLAGS` and
/// `CARGO_ENCODED_RUSTFLAGS` seen by rustc for this crate and present at run
/// time; and every `OPENSSL_*cap` variable, which OpenSSL and AWS-LC read to
/// override CPU capability detection.
#[must_use]
pub fn provenance() -> Value {
  let overrides = std::env::vars_os()
    .filter_map(|(name, value)| {
      let name = name.into_string().ok()?;
      (name.starts_with("OPENSSL_") && name.ends_with("cap")).then(|| (name, json!(value.to_string_lossy())))
    })
    .collect::<serde_json::Map<_, _>>();
  let runtime = |name: &str| std::env::var_os(name).map(|value| value.to_string_lossy().into_owned());
  json!({
    "openssl": {
      "version": openssl::version::version(),
      "c_flags": openssl::version::c_flags(),
      "platform": openssl::version::platform(),
      "dir": openssl::version::dir(),
      "build_openssl_dir": option_env!("OPENSSL_DIR"),
      "build_openssl_static": option_env!("OPENSSL_STATIC"),
      "images": images(),
    },
    "rustflags": {
      "build": option_env!("RUSTFLAGS"),
      "build_encoded": option_env!("CARGO_ENCODED_RUSTFLAGS"),
      "runtime": runtime("RUSTFLAGS"),
      "runtime_encoded": runtime("CARGO_ENCODED_RUSTFLAGS"),
    },
    "capability_overrides": overrides,
    "target": {"arch": std::env::consts::ARCH, "os": std::env::consts::OS},
  })
}

fn is_openssl_library(path: &str) -> bool {
  let name = path.rsplit('/').next().unwrap_or(path);
  name.starts_with("libssl.") || name.starts_with("libcrypto.")
}

/// The loaded OpenSSL images, from the process's own memory map.
#[cfg(target_os = "linux")]
fn images() -> Value {
  let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
    return json!({"error": "/proc/self/maps is unreadable"});
  };
  let mut paths = maps
    .lines()
    .filter_map(|line| line.split_whitespace().nth(5))
    .filter(|path| is_openssl_library(path))
    .collect::<Vec<_>>();
  paths.sort_unstable();
  paths.dedup();
  if paths.is_empty() {
    let executable = std::env::current_exe().ok();
    return json!({"static": executable.as_deref().map(file)});
  }
  json!({"loaded": paths.into_iter().map(|path| file(std::path::Path::new(path))).collect::<Vec<_>>()})
}

/// The OpenSSL libraries under the build's `OPENSSL_DIR`. This identifies the
/// configured installation; it does not prove which image the loader chose.
#[cfg(not(target_os = "linux"))]
fn images() -> Value {
  let Some(dir) = option_env!("OPENSSL_DIR") else {
    return json!({"error": "OPENSSL_DIR was not set at build time"});
  };
  let Ok(entries) = std::fs::read_dir(std::path::Path::new(dir).join("lib")) else {
    return json!({"error": "OPENSSL_DIR/lib is unreadable"});
  };
  let mut paths = entries
    .filter_map(|entry| entry.ok().map(|entry| entry.path()))
    .filter(|path| path.to_str().is_some_and(is_openssl_library))
    .collect::<Vec<_>>();
  paths.sort_unstable();
  json!({"configured": paths.iter().map(|path| file(path)).collect::<Vec<_>>()})
}

fn file(path: &std::path::Path) -> Value {
  let canonical = std::fs::canonicalize(path).ok();
  let sha256 = canonical
    .as_ref()
    .and_then(|file| std::fs::read(file).ok())
    .map(|bytes| {
      rscrypto_rustls::hash::SHA256
        .hash(&bytes)
        .as_ref()
        .iter()
        .fold(String::new(), |mut hex, byte| {
          write!(hex, "{byte:02x}").expect("string formatting");
          hex
        })
    });
  json!({"path": path, "canonical": canonical, "sha256": sha256})
}
