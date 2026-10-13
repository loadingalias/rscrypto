//! ML-DSA signing-tail baseline: per-signature latency and attempt counts.
//!
//! `scripts/bench/mldsa_tail.py` runs this program. For every parameter set it
//! signs `KEYS x MESSAGES` deterministic inputs through the compact and the
//! prepared key, deterministically and hedged with fixed randomness, and times
//! each production call. Every target signs the same signatures, so only the
//! latency differs between targets. A build with `--cfg rscrypto_mldsa_attempts`
//! also reports each signature's rejection-sampling attempts; its latency is
//! not production latency.
//!
//! Output: one CSV row per signature between `# csv begin` and `# csv end`,
//! then one summary line per parameter set, key form, and signing mode.

use core::hint::black_box;
use std::time::Instant;

use rscrypto::{
  MlDsa44, MlDsa44PreparedSecretKeyStorage, MlDsa65, MlDsa65PreparedSecretKeyStorage, MlDsa87,
  MlDsa87PreparedSecretKeyStorage,
};

struct Row {
  set: &'static str,
  form: &'static str,
  mode: &'static str,
  key: u32,
  message: u32,
  attempts: Option<usize>,
  nanos: u64,
}

/// Deterministic, non-secret test input: splitmix64 over the tag and indices.
fn input(tag: u64, key: u32, message: u32) -> [u8; 32] {
  let mut state = tag
    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
    .wrapping_add(u64::from(key) << 32 | u64::from(message));
  let mut out = [0u8; 32];
  for chunk in out.as_chunks_mut::<8>().0 {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    *chunk = (z ^ (z >> 31)).to_le_bytes();
  }
  out
}

fn take_attempts() -> Option<usize> {
  #[cfg(rscrypto_mldsa_attempts)]
  {
    Some(rscrypto::auth::diag_mldsa_take_sign_attempts())
  }
  #[cfg(not(rscrypto_mldsa_attempts))]
  {
    None
  }
}

macro_rules! run_set {
  ($rows:expr, $keys:expr, $messages:expr, $name:literal, $set:ident, $storage:ident) => {{
    for key in 0..$keys {
      let (public, secret) = $set::keypair_from_seed(&input(0, key, 0)).expect("key generation");
      let mut storage = Box::new($storage::new());
      let prepared = secret.prepare(&mut storage).expect("preparation");
      if key == 0 {
        for message in 0..16 {
          let message = input(1, u32::MAX, message);
          black_box(secret.sign_deterministic(&message, b"").expect("warm-up"));
          black_box(prepared.sign_deterministic(&message, b"").expect("warm-up"));
        }
      }
      for message in 0..$messages {
        let text = input(1, key, message);
        let random = input(2, key, message);
        for hedged in [false, true] {
          let mut compact_signature = None;
          for prepared_form in [false, true] {
            let fill = |out: &mut [u8]| {
              out.copy_from_slice(&random);
              Ok::<(), rscrypto::MlDsaError>(())
            };
            let _ = take_attempts();
            let start = Instant::now();
            let signature = match (prepared_form, hedged) {
              (false, false) => secret.sign_deterministic(black_box(&text), b""),
              (false, true) => secret.sign_with(black_box(&text), b"", fill),
              (true, false) => prepared.sign_deterministic(black_box(&text), b""),
              (true, true) => prepared.sign_with(black_box(&text), b"", fill),
            }
            .expect("signature");
            let nanos = u64::try_from(start.elapsed().as_nanos()).expect("signature time fits u64");
            let attempts = take_attempts();
            public.verify(&text, &signature).expect("signature verifies");
            // Both key forms sign the same bytes.
            match &compact_signature {
              None => compact_signature = Some(signature),
              Some(expected) => assert_eq!(expected.as_bytes(), signature.as_bytes(), "{} hedged={hedged}", $name),
            }
            $rows.push(Row {
              set: $name,
              form: if prepared_form { "prepared" } else { "compact" },
              mode: if hedged { "hedged" } else { "deterministic" },
              key,
              message,
              attempts,
              nanos,
            });
          }
        }
      }
    }
  }};
}

/// The nearest-rank quantile at `per_mille` / 1000 of a sorted, nonempty slice.
fn quantile(sorted: &[u64], per_mille: usize) -> u64 {
  sorted[sorted.len().strict_sub(1).strict_mul(per_mille).strict_add(500) / 1000]
}

fn main() {
  let mut args = std::env::args().skip(1);
  let keys: u32 = args.next().map_or(16, |value| value.parse().expect("KEYS"));
  let messages: u32 = args.next().map_or(64, |value| value.parse().expect("MESSAGES"));
  assert!((1..=1024).contains(&keys) && (1..=4096).contains(&messages), "KEYS 1..=1024, MESSAGES 1..=4096");

  let mut rows = Vec::new();
  run_set!(rows, keys, messages, "mldsa44", MlDsa44, MlDsa44PreparedSecretKeyStorage);
  run_set!(rows, keys, messages, "mldsa65", MlDsa65, MlDsa65PreparedSecretKeyStorage);
  run_set!(rows, keys, messages, "mldsa87", MlDsa87, MlDsa87PreparedSecretKeyStorage);

  println!("# csv begin");
  println!("set,form,mode,key,message,attempts,nanos");
  for row in &rows {
    let attempts = row.attempts.map_or(String::new(), |attempts| attempts.to_string());
    println!(
      "{},{},{},{},{},{attempts},{}",
      row.set, row.form, row.mode, row.key, row.message, row.nanos
    );
  }
  println!("# csv end");

  for set in ["mldsa44", "mldsa65", "mldsa87"] {
    for mode in ["deterministic", "hedged"] {
      for form in ["compact", "prepared"] {
        let group: Vec<&Row> = rows
          .iter()
          .filter(|row| row.set == set && row.form == form && row.mode == mode)
          .collect();
        let mut nanos: Vec<u64> = group.iter().map(|row| row.nanos).collect();
        nanos.sort_unstable();
        let attempts = group.iter().filter_map(|row| row.attempts).collect::<Vec<_>>();
        let attempt_summary = if attempts.is_empty() {
          String::from("attempts not counted")
        } else {
          let total: usize = attempts.iter().sum();
          format!(
            "attempts mean {:.3} max {}",
            total as f64 / attempts.len() as f64,
            attempts.iter().max().expect("attempts")
          )
        };
        println!(
          "# summary {set} {form} {mode}: n {} latency us p50 {:.1} p95 {:.1} p99 {:.1} max {:.1}; {attempt_summary}",
          nanos.len(),
          quantile(&nanos, 500) as f64 / 1000.0,
          quantile(&nanos, 950) as f64 / 1000.0,
          quantile(&nanos, 990) as f64 / 1000.0,
          *nanos.last().expect("samples") as f64 / 1000.0,
        );
      }
    }
  }
}
