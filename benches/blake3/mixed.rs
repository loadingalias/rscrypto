//! Public batch calls and serial calls over the same ordered input slices.
//!
//! Timed: production calls, their construction/cleanup, and output writes into
//! reusable caller storage. Untimed: deterministic fixtures, output allocation,
//! correctness checks, context hashing, and final caller-output cleanup. Each
//! row preserves its library's internal cleanup policy. Derived outputs remain
//! live between iterations for every implementation.

use core::hint::black_box;

use blake3::hazmat::HasherExt as _;
use criterion::Criterion;
use rscrypto::{Blake3, Blake3DeriveContext, Blake3KeyedHash};

const KEY: [u8; 32] = [0x42; 32];
const CONTEXT: &str = "rscrypto benchmark mixed batch derive-key context";

/// Configure one process before any cached platform or BLAKE3 detection.
/// The override only removes detected capabilities; upstream keeps auto dispatch.
pub(super) fn configure_isa() -> Result<(), String> {
  use rscrypto::platform::{self, Caps, Detected, expert};

  let requested = std::env::var_os("RSCRYPTO_BLAKE3_BENCH_ISA")
    .map(|value| value.into_string().expect("BLAKE3 benchmark ISA is UTF-8"))
    .unwrap_or_else(|| "auto".to_owned());
  let detected = expert::detect_uncached();
  let (required, removed) = match requested.as_str() {
    "auto" => (Caps::NONE, Caps::NONE),
    "portable" => (Caps::NONE, detected.caps),
    #[cfg(target_arch = "x86_64")]
    "sse41" | "avx2" | "avx512" => {
      use rscrypto::platform::caps::x86;
      let wide = [
        x86::AVX512F,
        x86::AVX512VL,
        x86::AVX512BW,
        x86::AVX512DQ,
        x86::AVX512CD,
        x86::AVX512IFMA,
        x86::AVX512VBMI,
        x86::AVX512VBMI2,
        x86::AVX512VNNI,
        x86::AVX512BITALG,
        x86::AVX512VPOPCNTDQ,
        x86::AVX512VP2INTERSECT,
        x86::AVX512FP16,
        x86::AVX512BF16,
        x86::AVX10_1,
        x86::AVX10_2,
      ]
      .into_iter()
      .fold(Caps::NONE, Caps::union);
      match requested.as_str() {
        "sse41" => (
          x86::SSE41.union(x86::SSSE3),
          wide
            .union(x86::AVX)
            .union(x86::AVX2)
            .union(x86::FMA)
            .union(x86::F16C)
            .union(x86::VAES)
            .union(x86::VPCLMULQDQ)
            .union(x86::SHA512),
        ),
        "avx2" => (x86::AVX2, wide),
        _ => (
          x86::AVX512F.union(x86::AVX512VL).union(x86::AVX512DQ).union(x86::AVX2),
          Caps::NONE,
        ),
      }
    }
    #[cfg(target_arch = "aarch64")]
    "neon" => {
      use rscrypto::platform::caps::aarch64;
      let sve = [
        aarch64::SVE,
        aarch64::SVE2,
        aarch64::SVE2_AES,
        aarch64::SVE2_SHA3,
        aarch64::SVE2_SM4,
        aarch64::SVE2_BITPERM,
        aarch64::SVE2_PMULL,
        aarch64::SVE2_I8MM,
        aarch64::SVE2_F32MM,
        aarch64::SVE2_F64MM,
        aarch64::SVE2_BF16,
        aarch64::SVE2_EBF16,
        aarch64::SVE2P1,
        aarch64::SVE_B16B16,
      ]
      .into_iter()
      .fold(Caps::NONE, Caps::union);
      (aarch64::NEON, sve)
    }
    _ => {
      return Err(format!(
        "unsupported BLAKE3 benchmark ISA {requested:?} on {}",
        std::env::consts::ARCH
      ));
    }
  };
  if !detected.caps.has(required) || (cfg!(feature = "portable-only") && !required.is_empty()) {
    return Err(format!(
      "BLAKE3 benchmark ISA {requested:?} is unavailable; detected {}",
      detected.caps
    ));
  }
  if requested != "auto" {
    expert::try_set_override(Some(Detected {
      caps: detected.caps.difference(removed),
      arch: detected.arch,
    }))
    .map_err(|error| format!("set BLAKE3 benchmark ISA before detection: {error}"))?;
  }
  let effective = platform::caps();
  #[cfg(feature = "diag")]
  let selected_bulk = Some(rscrypto::hashes::introspect::kernel_for::<Blake3>(usize::MAX));
  #[cfg(not(feature = "diag"))]
  let selected_bulk: Option<&str> = None;
  #[cfg(feature = "diag")]
  if requested != "auto" {
    let expected = match requested.as_str() {
      "sse41" => "x86_64/sse4.1",
      "avx2" => "x86_64/avx2",
      "avx512" => "x86_64/avx512",
      "neon" => "aarch64/neon",
      _ => "portable",
    };
    if selected_bulk != Some(expected) {
      return Err(format!(
        "requested {requested} but production bulk dispatch selected {selected_bulk:?}"
      ));
    }
  }
  eprintln!(
    "rscrypto-blake3-bench {}",
    serde_json::json!({
      "requested_isa": requested,
      "detected_caps": detected.caps.words(),
      "effective_caps": effective.words(),
      "static_caps": platform::caps_static().words(),
      "selected_bulk": selected_bulk,
      "upstream_dispatch": "auto",
    })
  );
  Ok(())
}

const CASES: [(&str, [usize; 16]); 3] = [
  (
    "short-refill",
    [1, 63, 7, 128, 16, 65, 31, 257, 33, 64, 47, 129, 70, 127, 191, 255],
  ),
  (
    "block-boundaries",
    [0, 1024, 63, 65, 1023, 64, 0, 65, 1024, 63, 64, 1023, 65, 0, 1024, 63],
  ),
  (
    "tree-fallback",
    [
      0, 1025, 64, 2047, 1023, 2048, 1, 2049, 1024, 3073, 63, 4096, 65, 4097, 8193, 16385,
    ],
  ),
];

fn serial_plain<const UPSTREAM: bool>(inputs: &[&[u8]], outputs: &mut [[u8; 32]]) {
  for (input, output) in inputs.iter().zip(outputs) {
    *output = if UPSTREAM {
      *blake3::hash(input).as_bytes()
    } else {
      Blake3::digest(input)
    };
  }
}

fn serial_keyed<const UPSTREAM: bool>(key: &[u8; 32], inputs: &[&[u8]], outputs: &mut [Blake3KeyedHash]) {
  for (input, output) in inputs.iter().zip(outputs) {
    *output = if UPSTREAM {
      Blake3KeyedHash::from_bytes(*blake3::keyed_hash(key, input).as_bytes())
    } else {
      Blake3::keyed_digest(key, input)
    };
  }
}

fn serial_derive(context: &Blake3DeriveContext, inputs: &[&[u8]], outputs: &mut [[u8; 32]]) {
  for (input, output) in inputs.iter().zip(outputs) {
    *output = Blake3::derive_key_with(context, input);
  }
}

fn upstream_derive(context_key: &[u8; 32], inputs: &[&[u8]], outputs: &mut [[u8; 32]]) {
  for (input, output) in inputs.iter().zip(outputs) {
    // Both libraries receive a prehashed context. Construct/update/finalize of
    // the upstream public hazmat hasher remains inside the timed operation.
    let mut hasher = blake3::Hasher::new_from_context_key(context_key);
    hasher.update(input);
    *output = *hasher.finalize().as_bytes();
  }
}

fn plain(c: &mut Criterion, case: &str, inputs: &[&[u8]], bytes: usize) {
  let name = format!("blake3/mixed-batch/plain/{case}/{}", inputs.len());
  if !super::bench_config::selected(&name) {
    return;
  }
  let expected: Vec<_> = inputs.iter().map(|input| *blake3::hash(input).as_bytes()).collect();
  let mut outputs = vec![[0; 32]; inputs.len()];
  Blake3::digest_batch(inputs, &mut outputs);
  assert_eq!(outputs, expected, "plain batch {case}");
  serial_plain::<false>(inputs, &mut outputs);
  assert_eq!(outputs, expected, "plain serial {case}");

  let mut group = c.benchmark_group(name);
  super::common::set_throughput(&mut group, bytes);
  group.bench_function("rscrypto/batch", |b| {
    b.iter(|| {
      Blake3::digest_batch(black_box(inputs), black_box(&mut outputs));
      black_box(&outputs);
    });
  });
  group.bench_function("rscrypto/serial", |b| {
    b.iter(|| {
      serial_plain::<false>(black_box(inputs), black_box(&mut outputs));
      black_box(&outputs);
    });
  });
  group.bench_function("blake3-auto/serial", |b| {
    b.iter(|| {
      serial_plain::<true>(black_box(inputs), black_box(&mut outputs));
      black_box(&outputs);
    });
  });
  group.finish();
}

fn keyed(c: &mut Criterion, case: &str, inputs: &[&[u8]], bytes: usize) {
  let name = format!("blake3/mixed-batch/keyed/{case}/{}", inputs.len());
  if !super::bench_config::selected(&name) {
    return;
  }
  let expected: Vec<_> = inputs
    .iter()
    .map(|input| *blake3::keyed_hash(&KEY, input).as_bytes())
    .collect();
  let mut outputs = vec![Blake3KeyedHash::default(); inputs.len()];
  Blake3::keyed_digest_batch(&KEY, inputs, &mut outputs);
  for (output, expected) in outputs.iter().zip(&expected) {
    assert_eq!(output.as_bytes(), expected, "keyed batch {case}");
  }
  serial_keyed::<false>(&KEY, inputs, &mut outputs);
  for (output, expected) in outputs.iter().zip(&expected) {
    assert_eq!(output.as_bytes(), expected, "keyed serial {case}");
  }

  let mut group = c.benchmark_group(name);
  super::common::set_throughput(&mut group, bytes);
  group.bench_function("rscrypto/batch", |b| {
    b.iter(|| {
      Blake3::keyed_digest_batch(black_box(&KEY), black_box(inputs), black_box(&mut outputs));
      black_box(&outputs);
    });
  });
  group.bench_function("rscrypto/serial", |b| {
    b.iter(|| {
      serial_keyed::<false>(black_box(&KEY), black_box(inputs), black_box(&mut outputs));
      black_box(&outputs);
    });
  });
  group.bench_function("blake3-auto/serial", |b| {
    b.iter(|| {
      serial_keyed::<true>(black_box(&KEY), black_box(inputs), black_box(&mut outputs));
      black_box(&outputs);
    });
  });
  group.finish();
}

fn derive(c: &mut Criterion, case: &str, inputs: &[&[u8]], bytes: usize) {
  let name = format!("blake3/mixed-batch/derive/{case}/{}", inputs.len());
  if !super::bench_config::selected(&name) {
    return;
  }
  let context = Blake3DeriveContext::new(CONTEXT);
  let context_key = blake3::hazmat::hash_derive_key_context(CONTEXT);
  let expected: Vec<_> = inputs.iter().map(|input| blake3::derive_key(CONTEXT, input)).collect();
  let mut outputs = vec![[0; 32]; inputs.len()];
  context.derive_key_batch(inputs, &mut outputs);
  assert_eq!(outputs, expected, "derive batch {case}");
  serial_derive(&context, inputs, &mut outputs);
  assert_eq!(outputs, expected, "derive serial {case}");
  upstream_derive(&context_key, inputs, &mut outputs);
  assert_eq!(outputs, expected, "derive upstream prehashed context {case}");

  let mut group = c.benchmark_group(name);
  super::common::set_throughput(&mut group, bytes);
  group.bench_function("rscrypto/batch", |b| {
    b.iter(|| {
      black_box(&context).derive_key_batch(black_box(inputs), black_box(&mut outputs));
      black_box(&outputs);
    });
  });
  group.bench_function("rscrypto/serial", |b| {
    b.iter(|| {
      serial_derive(black_box(&context), black_box(inputs), black_box(&mut outputs));
      black_box(&outputs);
    });
  });
  group.bench_function("blake3-auto/serial", |b| {
    b.iter(|| {
      upstream_derive(black_box(&context_key), black_box(inputs), black_box(&mut outputs));
      black_box(&outputs);
    });
  });
  group.finish();
  rscrypto::ct::zeroize(outputs.as_flattened_mut());
}

pub(super) fn bench(c: &mut Criterion) {
  if !super::bench_config::selected("blake3/mixed-batch/") {
    return;
  }
  super::print_blake3_diag_once();
  for (case, lengths) in CASES {
    for count in [16usize, 65] {
      let storage: Vec<Vec<u8>> = (0..count)
        .map(|index| {
          let len = lengths[index.strict_rem(lengths.len())];
          (0..len.strict_add(1))
            .map(|offset| u8::try_from(index.strict_mul(17).strict_add(offset) % 251).expect("fixture byte fits"))
            .collect()
        })
        .collect();
      let inputs: Vec<&[u8]> = storage.iter().map(|bytes| &bytes[1..]).collect();
      let bytes = inputs.iter().fold(0usize, |total, input| total.strict_add(input.len()));
      plain(c, case, &inputs, bytes);
      keyed(c, case, &inputs, bytes);
      derive(c, case, &inputs, bytes);
    }
  }
}
