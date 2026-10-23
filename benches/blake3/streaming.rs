//! Equal ordered bytes through one-shot, one-update, and two-update calls.
//!
//! Case IDs: `blake3/serial/ordered[-alignment]/<mode>/<shape>/<implementation>/<input>`.
//! Each (mode, shape) pair is one task. `plain` hashes; `derive-context`
//! derives a key, and every call hashes the context string: rscrypto through
//! `Blake3DeriveContext::new`, upstream inside `derive_key` and
//! `Hasher::new_derive_key`. `oneshot` is one call, `single-update` is
//! construction, one update and finalization, and `two-update` splits the
//! bytes at the case's split point.
//!
//! Timed: construction, context hashing, updates, finalization, the 32-byte
//! output, and destruction. Untimed: fixture allocation and expected outputs.
//! Upstream's `zeroize` feature is off, so cleanup costs differ.
//!
//! Only `blake3_serial` registers these groups. With `parallel`, rscrypto's
//! one-shot path threads from 64 KiB on the AArch64 NEON profile, so the
//! 65,537- to 66,559-byte cases would compare threaded rscrypto with serial
//! `blake3::hash`.

use core::hint::black_box;

use criterion::Criterion;
use rscrypto::{Blake3, Blake3DeriveContext, Digest as _};

use super::bench_config::checked;

const CONTEXT: &str = "rscrypto benchmark derive-key context";

fn ours_oneshot<const DERIVE: bool>(data: &[u8]) -> [u8; 32] {
  if DERIVE {
    Blake3::derive_key_with(&Blake3DeriveContext::new(black_box(CONTEXT)), data)
  } else {
    Blake3::digest(data)
  }
}

fn reference_oneshot<const DERIVE: bool>(data: &[u8]) -> [u8; 32] {
  if DERIVE {
    blake3::derive_key(black_box(CONTEXT), data)
  } else {
    *blake3::hash(data).as_bytes()
  }
}

fn ours_update<const DERIVE: bool, const SPLIT: bool>(data: &[u8], split: usize) -> [u8; 32] {
  let mut h = if DERIVE {
    Blake3::new_derive_key_from(&Blake3DeriveContext::new(black_box(CONTEXT)))
  } else {
    Blake3::new()
  };
  if SPLIT {
    h.update(&data[..split]);
    h.update(&data[split..]);
  } else {
    h.update(data);
  }
  h.finalize()
}

fn reference_update<const DERIVE: bool, const SPLIT: bool>(data: &[u8], split: usize) -> [u8; 32] {
  let mut h = if DERIVE {
    blake3::Hasher::new_derive_key(black_box(CONTEXT))
  } else {
    blake3::Hasher::new()
  };
  if SPLIT {
    h.update(&data[..split]);
    h.update(&data[split..]);
  } else {
    h.update(data);
  }
  *h.finalize().as_bytes()
}

/// One ordered input: its name in case IDs, its bytes, and the call shapes it uses.
struct Case {
  input: String,
  storage: Vec<u8>,
  range: core::ops::Range<usize>,
  split: Option<usize>,
  single: bool,
}

impl Case {
  fn data(&self) -> &[u8] {
    &self.storage[self.range.clone()]
  }
}

#[derive(Clone, Copy, PartialEq)]
enum Shape {
  Oneshot,
  SingleUpdate,
  TwoUpdate,
}

impl Shape {
  const ALL: [Self; 3] = [Self::Oneshot, Self::SingleUpdate, Self::TwoUpdate];

  fn name(self) -> &'static str {
    match self {
      Self::Oneshot => "oneshot",
      Self::SingleUpdate => "single-update",
      Self::TwoUpdate => "two-update",
    }
  }

  fn applies(self, case: &Case) -> bool {
    match self {
      Self::Oneshot => true,
      Self::SingleUpdate => case.single,
      Self::TwoUpdate => case.split.is_some(),
    }
  }
}

/// Register the rscrypto and upstream rows of one (mode, shape) task over every applicable case.
fn group<const DERIVE: bool>(c: &mut Criterion, operation: &str, shape: Shape, cases: &[Case]) {
  let mode = if DERIVE { "derive-context" } else { "plain" };
  let name = format!("{operation}/{mode}/{}", shape.name());
  if !super::bench_config::selected(&name) {
    return;
  }
  let mut g = c.benchmark_group(name);
  for case in cases.iter().filter(|case| shape.applies(case)) {
    let data = case.data();
    let split = case.split.unwrap_or(0);
    let input = &case.input;
    let upstream = reference_oneshot::<DERIVE>(data);
    let ours = ours_oneshot::<DERIVE>(data);
    super::common::set_throughput(&mut g, data.len());
    match shape {
      Shape::Oneshot => {
        g.bench_function(
          format!("rscrypto/{input}"),
          checked(
            || ours_oneshot::<DERIVE>(black_box(data)),
            |out| assert_eq!(out, upstream, "{input} one-shot"),
          ),
        );
        g.bench_function(
          format!("blake3/{input}"),
          checked(
            || reference_oneshot::<DERIVE>(black_box(data)),
            |out| assert_eq!(out, ours),
          ),
        );
      }
      Shape::SingleUpdate => {
        g.bench_function(
          format!("rscrypto/{input}"),
          checked(
            || ours_update::<DERIVE, false>(black_box(data), 0),
            |out| assert_eq!(out, upstream, "{input} one-update"),
          ),
        );
        g.bench_function(
          format!("blake3/{input}"),
          checked(
            || reference_update::<DERIVE, false>(black_box(data), 0),
            |out| assert_eq!(out, ours),
          ),
        );
      }
      Shape::TwoUpdate => {
        g.bench_function(
          format!("rscrypto/{input}"),
          checked(
            || ours_update::<DERIVE, true>(black_box(data), split),
            |out| assert_eq!(out, upstream, "{input} two-update"),
          ),
        );
        g.bench_function(
          format!("blake3/{input}"),
          checked(
            || reference_update::<DERIVE, true>(black_box(data), split),
            |out| assert_eq!(out, ours),
          ),
        );
      }
    }
  }
  g.finish();
}

fn groups(c: &mut Criterion, operation: &str, plain: &[Case], derive: &[Case]) {
  for shape in Shape::ALL {
    group::<false>(c, operation, shape, plain);
    group::<true>(c, operation, shape, derive);
  }
}

/// Bytes `(i + 1) % 251` after one leading byte, so the input starts one byte past the allocation.
fn case(input: String, len: usize, split: Option<usize>, single: bool) -> Case {
  let storage: Vec<u8> = (0..len.strict_add(1))
    .map(|i| u8::try_from(i % 251).expect("fixture remainder fits"))
    .collect();
  Case {
    input,
    storage,
    range: 1..len.strict_add(1),
    split,
    single,
  }
}

/// `case` at an address congruent to `offset` modulo 64, for offsets 0 and 1.
fn aligned(input: &str, len: usize, split: Option<usize>, single: bool) -> impl Iterator<Item = Case> {
  [0usize, 1].into_iter().map(move |offset| {
    let (storage, range) = super::leaf_alignment::fixture(len, offset);
    Case {
      input: format!("{input}/in-{offset}"),
      storage,
      range,
      split,
      single,
    }
  })
}

/// Selected ordered cases at addresses congruent to 0 and 1 modulo 64.
pub(super) fn alignment_bench(c: &mut Criterion) {
  const OPERATION: &str = "blake3/serial/ordered-alignment";
  if !super::bench_config::selected(OPERATION) {
    return;
  }
  let mut plain: Vec<Case> = Vec::new();
  plain.extend(aligned("prefix-70-bulk-4096", 4166, Some(70), false));
  plain.extend(aligned("prefix-24-bulk-3104", 3128, Some(24), false));
  for chunks in [3usize, 5, 6, 7] {
    plain.extend(aligned(
      &format!("chunks-{chunks}-tail-70"),
      chunks.strict_mul(1024).strict_add(70),
      None,
      true,
    ));
  }
  let mut derive: Vec<Case> = Vec::new();
  for bulk in [4052usize, 4096] {
    derive.extend(aligned(
      &format!("bulk-{bulk}-suffix-60"),
      bulk.strict_add(60),
      Some(bulk),
      true,
    ));
  }
  groups(c, OPERATION, &plain, &derive);
}

pub(super) fn bench(c: &mut Criterion) {
  const OPERATION: &str = "blake3/serial/ordered";
  if !super::bench_config::selected(&format!("{OPERATION}/")) {
    return;
  }
  let mut plain = Vec::new();
  for prefix in [1usize, 24, 70, 1023] {
    for bulk in [1024usize, 2048, 4096, 8192, 16384, 32768, 65536] {
      plain.push(case(
        format!("prefix-{prefix}-bulk-{bulk}"),
        prefix.strict_add(bulk),
        Some(prefix),
        false,
      ));
    }
  }
  plain.push(case("prefix-24-bulk-3104".to_owned(), 3128, Some(24), false));
  // 3, 7, 11 and 15 complete chunks before a partial one take the one-pass NEON tail kernel at chunk
  // counter 0, and 3 and 15 the AVX-512 one; 5 and 6 take neither.
  for chunks in [3usize, 5, 6, 7, 11, 15] {
    for tail in [1usize, 70, 1023] {
      plain.push(case(
        format!("chunks-{chunks}-tail-{tail}"),
        chunks.strict_mul(1024).strict_add(tail),
        None,
        true,
      ));
    }
  }
  plain.push(case("chunks-4-tail-70".to_owned(), 4166, None, true));
  for len in [64, 1024] {
    plain.push(case(format!("single-chunk-{len}"), len, None, true));
  }

  // Suffix cases hash envelope || fields, never the prefix cases' reversed bytes.
  let derive: Vec<Case> = [980usize, 1000, 4000, 4052, 4095, 4096, 16340, 16384]
    .into_iter()
    .map(|bulk| case(format!("bulk-{bulk}-suffix-60"), bulk.strict_add(60), Some(bulk), true))
    .collect();
  groups(c, OPERATION, &plain, &derive);
}
