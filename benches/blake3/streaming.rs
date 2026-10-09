//! Equivalent ordered bytes across one-shot, one-update, and two-update calls.

use core::hint::black_box;

use criterion::Criterion;
use rscrypto::{Blake3, Digest as _};

const CONTEXT: &str = "rscrypto benchmark derive-key context";

fn ours_oneshot<const DERIVE: bool>(data: &[u8]) -> [u8; 32] {
  if DERIVE {
    Blake3::derive_key(CONTEXT, data)
  } else {
    Blake3::digest(data)
  }
}

fn reference_oneshot<const DERIVE: bool>(data: &[u8]) -> [u8; 32] {
  if DERIVE {
    blake3::derive_key(CONTEXT, data)
  } else {
    *blake3::hash(data).as_bytes()
  }
}

fn ours_update<const DERIVE: bool, const SPLIT: bool>(data: &[u8], split: usize) -> [u8; 32] {
  let mut h = if DERIVE {
    Blake3::new_derive_key(CONTEXT)
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
    blake3::Hasher::new_derive_key(CONTEXT)
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

fn case<const DERIVE: bool>(c: &mut Criterion, name: &str, len: usize, split: Option<usize>, single: bool) {
  let mode = if DERIVE { "derive" } else { "plain" };
  let group = format!("blake3/ordered/{mode}/{name}");
  if !super::bench_config::selected(&group) {
    return;
  }
  let storage: Vec<u8> = (0..len.strict_add(1))
    .map(|i| u8::try_from(i % 251).expect("fixture remainder fits"))
    .collect();
  let data = &storage[1..];
  measure::<DERIVE>(c, &group, name, data, split, single);
}

fn measure<const DERIVE: bool>(
  c: &mut Criterion,
  group: &str,
  name: &str,
  data: &[u8],
  split: Option<usize>,
  single: bool,
) {
  let expected = reference_oneshot::<DERIVE>(data);
  let mut g = c.benchmark_group(group);
  super::common::set_throughput(&mut g, data.len());

  assert_eq!(ours_oneshot::<DERIVE>(data), expected, "{name} one-shot");
  g.bench_function("rscrypto/oneshot", |b| {
    b.iter(|| black_box(ours_oneshot::<DERIVE>(black_box(data))))
  });
  g.bench_function("blake3/oneshot", |b| {
    b.iter(|| black_box(reference_oneshot::<DERIVE>(black_box(data))))
  });

  if single {
    assert_eq!(ours_update::<DERIVE, false>(data, 0), expected, "{name} one-update");
    assert_eq!(reference_update::<DERIVE, false>(data, 0), expected);
    g.bench_function("rscrypto/single-update", |b| {
      b.iter(|| black_box(ours_update::<DERIVE, false>(black_box(data), 0)))
    });
    g.bench_function("blake3/single-update", |b| {
      b.iter(|| black_box(reference_update::<DERIVE, false>(black_box(data), 0)))
    });
  }

  if let Some(split) = split {
    assert_eq!(ours_update::<DERIVE, true>(data, split), expected, "{name} two-update");
    assert_eq!(reference_update::<DERIVE, true>(data, split), expected);
    g.bench_function("rscrypto/two-update", |b| {
      b.iter(|| black_box(ours_update::<DERIVE, true>(black_box(data), split)))
    });
    g.bench_function("blake3/two-update", |b| {
      b.iter(|| black_box(reference_update::<DERIVE, true>(black_box(data), split)))
    });
  }
  g.finish();
}

fn alignment_case<const DERIVE: bool>(c: &mut Criterion, name: &str, len: usize, split: Option<usize>, single: bool) {
  let mode = if DERIVE { "derive" } else { "plain" };
  for offset in [0usize, 1] {
    let group = format!("blake3/ordered-alignment/{mode}/{name}/in-{offset}");
    if !super::bench_config::selected(&group) {
      continue;
    }
    let (storage, range) = super::leaf_alignment::fixture(len, offset);
    measure::<DERIVE>(c, &group, name, &storage[range], split, single);
  }
}

pub(super) fn alignment_bench(c: &mut Criterion) {
  if !super::bench_config::selected("blake3/ordered-alignment/") {
    return;
  }
  alignment_case::<false>(c, "prefix-70-bulk-4096", 4166, Some(70), false);
  alignment_case::<false>(c, "prefix-24-bulk-3104", 3128, Some(24), false);
  for chunks in [3usize, 5, 6, 7] {
    alignment_case::<false>(
      c,
      &format!("chunks-{chunks}-tail-70"),
      chunks.strict_mul(1024).strict_add(70),
      None,
      true,
    );
  }
  for bulk in [4052usize, 4096] {
    alignment_case::<true>(
      c,
      &format!("bulk-{bulk}-suffix-60"),
      bulk.strict_add(60),
      Some(bulk),
      true,
    );
  }
}

pub(super) fn bench(c: &mut Criterion) {
  if !super::bench_config::selected("blake3/ordered") {
    return;
  }
  for prefix in [1usize, 24, 70, 1023] {
    for bulk in [1024usize, 2048, 4096, 8192, 16384, 32768, 65536] {
      case::<false>(
        c,
        &format!("prefix-{prefix}-bulk-{bulk}"),
        prefix.strict_add(bulk),
        Some(prefix),
        false,
      );
    }
  }
  case::<false>(c, "prefix-24-bulk-3104", 3128, Some(24), false);

  // Suffix rows hash envelope || fields, never the prefix rows' reversed bytes.
  for bulk in [980usize, 1000, 4000, 4052, 4095, 4096, 16340, 16384] {
    case::<true>(
      c,
      &format!("bulk-{bulk}-suffix-60"),
      bulk.strict_add(60),
      Some(bulk),
      true,
    );
  }
  for chunks in [3usize, 5, 6, 7] {
    for tail in [1usize, 70, 1023] {
      case::<false>(
        c,
        &format!("chunks-{chunks}-tail-{tail}"),
        chunks.strict_mul(1024).strict_add(tail),
        None,
        true,
      );
    }
  }
  case::<false>(c, "chunks-4-tail-70", 4166, None, true);
  for len in [64, 1024] {
    case::<false>(c, &format!("single-chunk-{len}"), len, None, true);
  }
}
