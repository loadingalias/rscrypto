// ─────────────────────────────────────────────────────────────────────────────
// s390x (IBM Z) Detection
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "s390x")]
fn detect_s390x() -> Detected {
  // Start with compile-time detected features.
  #[cfg(feature = "std")]
  let caps = caps_static() | runtime_s390x();
  #[cfg(not(feature = "std"))]
  let caps = caps_static();

  let tune = select_s390x_tune(caps);

  Detected {
    caps,
    tune,
    arch: Arch::S390x,
  }
}

#[cfg(all(target_arch = "s390x", feature = "std"))]
fn runtime_s390x() -> Caps {
  use crate::caps::s390x;

  let mut caps = Caps::NONE;

  macro_rules! rt {
    ($f:literal => $c:expr) => {
      if std::arch::is_s390x_feature_detected!($f) {
        caps |= $c;
      }
    };
  }

  // ─── Vector Facilities ───
  rt!("vector" => s390x::VECTOR);
  rt!("vector-enhancements-1" => s390x::VECTOR_ENH1);
  rt!("vector-enhancements-2" => s390x::VECTOR_ENH2);
  rt!("vector-enhancements-3" => s390x::VECTOR_ENH3);
  rt!("vector-packed-decimal" => s390x::VECTOR_PD);
  rt!("nnp-assist" => s390x::NNP_ASSIST);

  // ─── Miscellaneous Extensions ───
  rt!("miscellaneous-extensions-2" => s390x::MISC_EXT2);
  rt!("miscellaneous-extensions-3" => s390x::MISC_EXT3);

  // ─── Crypto (CPACF - Message Security Assist) ───
  rt!("message-security-assist-extension3" => s390x::MSA);
  rt!("message-security-assist-extension4" => s390x::MSA | s390x::MSA4);
  rt!("message-security-assist-extension5" => s390x::MSA | s390x::MSA5);
  rt!("message-security-assist-extension8" => s390x::MSA | s390x::MSA8);
  rt!("message-security-assist-extension9" => s390x::MSA | s390x::MSA9);

  // ─── Other Facilities ───
  rt!("deflate-conversion" => s390x::DEFLATE);
  rt!("enhanced-sort" => s390x::ENHANCED_SORT);

  caps
}

#[cfg(target_arch = "s390x")]
fn select_s390x_tune(caps: Caps) -> Tune {
  use crate::caps::s390x;

  if caps.has(s390x::VECTOR_ENH2) {
    // z15+
    Tune::Z15
  } else if caps.has(s390x::VECTOR_ENH1) {
    // z14
    Tune::Z14
  } else if caps.has(s390x::VECTOR) {
    // z13
    Tune::Z13
  } else {
    Tune::PORTABLE
  }
}

