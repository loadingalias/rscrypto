//! Native SLH-DSA-SHA2-128s residue and resource measurement on a private,
//! painted stack.
//!
//! Each scenario runs one production call on a fresh stack. After the call
//! returns, an independent FIPS 205 oracle over the `sha2` crate derives the
//! call's secrets: SK.seed, SK.prf, the hedging bytes, each WOTS+ chain value
//! that no signature reveals, and each unrevealed FORS leaf secret. The oracle
//! must find every revealed value of the measured signature on its chain or
//! leaf. The harness searches the touched stack for each secret in byte order
//! and in SHA-256's big-endian word order, as a whole 16-byte value and as
//! 8-byte halves. It also counts the call's heap allocations. It records no
//! timings.

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
  use core::{
    alloc::{GlobalAlloc, Layout},
    fmt::Write as _,
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
  };
  use std::{alloc::System, collections::HashSet, fs, path::Path};

  use rscrypto::{
    HashSlhDsaSha2_128sWithSha256SecretKey as HashSecretKey, SlhDsaError, SlhDsaSha2_128s as Profile,
    SlhDsaSha2_128sSecretKey as SecretKey,
  };
  use sha2::{Digest, Sha256};

  use crate::native_stack::{PAINT, hex, input, measure, plant};

  /// Counts every allocation, including reallocation, of the process.
  struct Counting;

  static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

  // SAFETY: each method forwards its arguments unchanged to `System`, which
  // upholds the `GlobalAlloc` contract; the counter has no memory effect.
  unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
      ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
      // SAFETY: the caller's layout contract is forwarded unchanged.
      unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
      // SAFETY: `pointer` came from `alloc` with this `layout`.
      unsafe { System.dealloc(pointer, layout) }
    }
  }

  #[global_allocator]
  static GLOBAL: Counting = Counting;

  // FIPS 205 Table 2, SLH-DSA-SHA2-128s.
  const N: usize = 16;
  const HP: u32 = 9;
  const D: u8 = 7;
  const A: u32 = 12;
  const K: usize = 14;
  const LEN: usize = 35;
  const W: u32 = 16;
  const FORS_TREE_NODES: usize = 1 + A as usize;
  const FORS_NODES: usize = K * FORS_TREE_NODES;
  const XMSS_NODES: usize = LEN + HP as usize;

  // FIPS 205 Section 4.2 address types.
  const WOTS_HASH: u8 = 0;
  const WOTS_PRF: u8 = 5;
  const FORS_PRF: u8 = 6;

  /// RFC 9909 DER object identifier of SHA-256, the HashSLH-DSA pre-hash.
  const SHA256_OID: [u8; 11] = [0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01];
  const CONTEXT: &[u8] = b"residue";

  /// SHA-256 after `PK.seed || toByte(0, 64 - n)` (FIPS 205 Section 11.2).
  struct Oracle(Sha256);

  impl Oracle {
    fn new(pk_seed: &[u8; N]) -> Self {
      Self(Sha256::new().chain_update(pk_seed).chain_update([0u8; 64 - N]))
    }

    /// `Trunc_n(SHA-256(PK.seed || toByte(0, 64 - n) || ADRSc || value))`:
    /// PRF when `value` is SK.seed, F otherwise (Section 11.2.1).
    fn tweak(&self, address: &[u8; 22], value: &[u8; N]) -> [u8; N] {
      let digest = self.0.clone().chain_update(address).chain_update(value).finalize();
      digest[..N].try_into().expect("a 16-byte prefix")
    }
  }

  /// ADRSc (Section 11.2): layer, the low 64 bits of the tree address, type,
  /// and the three final words.
  fn address(layer: u8, tree: u64, kind: u8, words: [u32; 3]) -> [u8; 22] {
    let mut out = [0u8; 22];
    out[0] = layer;
    out[1..9].copy_from_slice(&tree.to_be_bytes());
    out[9] = kind;
    for (chunk, word) in out[10..].chunks_mut(4).zip(words) {
      chunk.copy_from_slice(&word.to_be_bytes());
    }
    out
  }

  /// Hands each secret value to the search.
  type Visit<'a> = &'a mut dyn FnMut(&str, &[u8; N], String);

  /// Each secret WOTS+ chain value of tree `tree` in `layer` (Algorithms 5
  /// and 6). For the leaf that signed, the values below each revealed value
  /// are secret. Elsewhere every value below the public chain end is secret.
  fn wots_secrets(
    oracle: &Oracle,
    sk_seed: &[u8; N],
    layer: u8,
    tree: u64,
    signing: Option<(u32, &[[u8; N]])>,
    visit: Visit<'_>,
  ) {
    for leaf in 0..1u32 << HP {
      let revealed = signing.filter(|&(signer, _)| signer == leaf).map(|(_, nodes)| nodes);
      for chain in 0..u32::try_from(LEN).expect("a chain count") {
        let mut value = oracle.tweak(&address(layer, tree, WOTS_PRF, [leaf, chain, 0]), sk_seed);
        let target = revealed.map(|nodes| nodes[chain as usize]);
        for position in 0..W {
          if target == Some(value) {
            break;
          }
          if position == W - 1 {
            assert!(target.is_none(), "revealed WOTS+ value is not on its chain");
            break;
          }
          visit(
            "wots_chain",
            &value,
            format!("layer {layer} leaf {leaf} chain {chain} position {position}"),
          );
          value = oracle.tweak(&address(layer, tree, WOTS_HASH, [leaf, chain, position]), &value);
        }
      }
    }
  }

  /// The secret seeds and derived values of one signature (Algorithm 19).
  fn signing_secrets(public: &[u8; 2 * N], sk_seed: &[u8; N], message: &[&[u8]], signature: &[u8], visit: Visit<'_>) {
    let (pk_seed, pk_root) = public.split_at(N);
    let pk_seed: &[u8; N] = pk_seed.try_into().expect("PK.seed");
    let (nodes, rest) = signature.as_chunks::<N>();
    assert!(
      rest.is_empty()
        && nodes.len()
          == 1usize
            .strict_add(FORS_NODES)
            .strict_add(usize::from(D).strict_mul(XMSS_NODES))
    );
    // H_msg (Section 11.2.1): MGF1-SHA-256(R || PK.seed || SHA-256(R ||
    // PK.seed || PK.root || M), m); m = 30 fits the first MGF1 block.
    let r = &nodes[0];
    let mut inner = Sha256::new()
      .chain_update(r)
      .chain_update(pk_seed)
      .chain_update(pk_root);
    for part in message {
      inner.update(part);
    }
    let digest = Sha256::new()
      .chain_update(r)
      .chain_update(pk_seed)
      .chain_update(inner.finalize())
      .chain_update(0u32.to_be_bytes())
      .finalize();
    let (md, rest) = digest.split_at(21);
    let mut tree =
      u64::from_be_bytes([0, rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6]]) & ((1 << 54) - 1);
    let mut leaf = u32::from(u16::from_be_bytes([rest[7], rest[8]])) & ((1 << HP) - 1);
    let oracle = Oracle::new(pk_seed);
    // FORS (Algorithms 14 and 16): 14 big-endian 12-bit indices.
    for (index, pair) in md.chunks(3).enumerate() {
      let digits = [
        (u32::from(pair[0]) << 4) | u32::from(pair[1] >> 4),
        (u32::from(pair[1] & 0x0f) << 8) | u32::from(pair[2]),
      ];
      for (half, selected) in digits.into_iter().enumerate() {
        let fors_tree = u32::try_from(index.strict_mul(2).strict_add(half)).expect("a FORS tree");
        for leaf_index in 0..1u32 << A {
          let words = [leaf, 0, (fors_tree << A).strict_add(leaf_index)];
          let secret = oracle.tweak(&address(0, tree, FORS_PRF, words), sk_seed);
          if leaf_index == selected {
            assert_eq!(
              secret,
              nodes[1usize.strict_add(
                usize::try_from(fors_tree)
                  .expect("a FORS tree")
                  .strict_mul(FORS_TREE_NODES)
              )],
              "revealed FORS secret"
            );
          } else {
            visit(
              "fors_secret",
              &secret,
              format!("fors tree {fors_tree} leaf {leaf_index}"),
            );
          }
        }
      }
    }
    // The hypertree (Algorithm 12).
    for (layer, xmss) in (0..D).zip(nodes[1 + FORS_NODES..].as_chunks::<XMSS_NODES>().0) {
      wots_secrets(&oracle, sk_seed, layer, tree, Some((leaf, &xmss[..LEN])), visit);
      leaf = u32::try_from(tree & ((1 << HP) - 1)).expect("a leaf index");
      tree >>= HP;
    }
  }

  /// Every 16-byte and 8-byte window of the touched stack, in byte order and
  /// with each aligned 4-byte word reversed (SHA-256's big-endian words).
  struct Search {
    whole: HashSet<[u8; N]>,
    halves: HashSet<[u8; N / 2]>,
    touched: usize,
    views: [Vec<u8>; 2],
  }

  impl Search {
    fn new(stack: &[u8]) -> Self {
      let first = stack.iter().position(|&byte| byte != PAINT).unwrap_or(stack.len());
      // A window may begin in the paint, and the private stack is page aligned.
      let start = first.saturating_sub(N) & !3;
      let raw = &stack[start..];
      let swapped: Vec<u8> = raw.chunks(4).flat_map(|word| word.iter().rev().copied()).collect();
      let mut search = Self {
        whole: HashSet::new(),
        halves: HashSet::new(),
        touched: stack.len().strict_sub(first),
        views: [raw.to_vec(), swapped],
      };
      for view in &search.views {
        search.whole.extend(view.array_windows::<N>().copied());
        search.halves.extend(view.array_windows::<{ N / 2 }>().copied());
      }
      search
    }

    /// Where `fragment` occurs, as `word-order@distance below the stack top`.
    fn locate(&self, fragment: &[u8]) -> String {
      let mut places = Vec::new();
      for (order, view) in ["bytes", "words"].into_iter().zip(&self.views) {
        for (offset, window) in view.windows(fragment.len()).enumerate() {
          if window == fragment {
            places.push(format!("{order}@-{}", view.len().strict_sub(offset)));
          }
        }
      }
      places.join(" ")
    }
  }

  #[derive(Default)]
  struct Class {
    needles: usize,
    whole: usize,
    half: usize,
    examples: Vec<String>,
  }

  /// One measured call and the classes whose secrets must not remain.
  struct Scenario<'a> {
    name: &'a str,
    /// Classes whose residue is recorded but not judged.
    report_only: &'a [&'a str],
    stack: Vec<u8>,
    allocations: usize,
  }

  fn judge(scenario: &Scenario<'_>, sequence: usize, derive: impl FnOnce(Visit<'_>), json: &mut String) -> bool {
    let search = Search::new(&scenario.stack);
    let mut classes: Vec<(String, Class)> = Vec::new();
    derive(&mut |class, needle, detail| {
      let row = match classes.iter().position(|(name, _)| name == class) {
        Some(index) => &mut classes[index].1,
        None => {
          classes.push((class.to_owned(), Class::default()));
          &mut classes.last_mut().expect("pushed").1
        }
      };
      row.needles = row.needles.strict_add(1);
      let (low, high) = needle.split_at(N / 2);
      let found = if search.whole.contains(needle) {
        row.whole = row.whole.strict_add(1);
        format!("whole {}", search.locate(needle))
      } else if search.halves.contains(low) || search.halves.contains(high) {
        row.half = row.half.strict_add(1);
        format!("half low {} high {}", search.locate(low), search.locate(high))
      } else {
        return;
      };
      if row.examples.len() < 8 {
        row.examples.push(format!("{found} {detail}"));
      }
    });
    let passed = scenario.allocations == 0
      && classes
        .iter()
        .all(|(name, row)| scenario.report_only.contains(&name.as_str()) || row.whole.strict_add(row.half) == 0);
    println!(
      "{}: {} B stack, {} allocations: {}",
      scenario.name,
      search.touched,
      scenario.allocations,
      if passed { "pass" } else { "FAIL" }
    );
    write!(
      json,
      "{{\"sequence\":{sequence},\"scenario\":\"{}\",\"passed\":{passed},\"stack_touched_bytes\":{},\"allocations\":{},\"report_only\":{:?},\"classes\":{{",
      scenario.name, search.touched, scenario.allocations, scenario.report_only
    )
    .expect("String formatting");
    for (index, (name, row)) in classes.iter().enumerate() {
      println!(
        "  {name}: {} needles, {} whole, {} half",
        row.needles, row.whole, row.half
      );
      for example in &row.examples {
        println!("    {example}");
      }
      write!(
        json,
        "{}\"{name}\":{{\"needles\":{},\"whole\":{},\"half\":{},\"examples\":{:?}}}",
        if index == 0 { "" } else { "," },
        row.needles,
        row.whole,
        row.half,
        row.examples
      )
      .expect("String formatting");
    }
    json.push_str("}}");
    passed
  }

  /// Run `call` on a fresh private stack, counting its allocations.
  fn run(name: &'static str, report_only: &'static [&'static str], mut call: impl FnMut()) -> Scenario<'static> {
    let mut allocations = 0;
    let stack = measure(|| {
      let before = ALLOCATIONS.load(Ordering::Relaxed);
      call();
      allocations = ALLOCATIONS.load(Ordering::Relaxed).strict_sub(before);
    });
    Scenario {
      name,
      report_only,
      stack,
      allocations,
    }
  }

  #[inline(never)]
  fn plant_words(words: [u32; 4]) {
    let copy = black_box(words);
    black_box(&copy);
  }

  // `input` streams for different tags overlap unless the tags differ by
  // at least four, so each input class takes its own distant tag.
  const SEEDS: u64 = 0x10;
  const PK_SEED: u64 = 0x20;
  const MESSAGE: u64 = 0x30;
  const ADDRND: u64 = 0x40;
  const PREHASH: u64 = 0x50;

  /// SK.seed, SK.prf, and PK.seed for key `index`: synthetic public test data.
  fn seeds(index: u32) -> [u8; 3 * N] {
    let mut out = [0u8; 3 * N];
    out[..32].copy_from_slice(&input(SEEDS, index));
    out[32..].copy_from_slice(&input(PK_SEED, index)[..N]);
    out
  }

  pub(super) fn run_all() -> bool {
    let arguments: Vec<String> = std::env::args().collect();
    let directory = Path::new(arguments.get(1).expect("output directory"));
    let cases: u32 = arguments.get(2).map_or(2, |value| value.parse().expect("case count"));
    assert!((1..=16).contains(&cases), "case count 1..=16");
    fs::create_dir_all(directory).expect("output directory");
    let mut json = String::from("{\"set\":\"SLH-DSA-SHA2-128s\",\"scenarios\":[");
    let mut passed = true;
    let mut sequence = 0usize;
    let mut record = |scenario: &Scenario<'_>, derive: &mut dyn FnMut(Visit<'_>), json: &mut String| {
      if sequence != 0 {
        json.push(',');
      }
      let path = directory.join(format!("{sequence:02}-{}.stack", scenario.name));
      fs::write(path, &scenario.stack).expect("write stack snapshot");
      passed &= judge(scenario, sequence, |visit| derive(visit), json);
      sequence = sequence.strict_add(1);
    };

    for index in 0..cases {
      let seeds = seeds(index);
      let sk_seed: [u8; N] = seeds[..N].try_into().expect("SK.seed");
      let sk_prf: [u8; N] = seeds[N..2 * N].try_into().expect("SK.prf");
      let pk_seed: [u8; N] = seeds[2 * N..].try_into().expect("PK.seed");
      let keys = |visit: Visit<'_>| {
        visit("sk_seed", &sk_seed, String::new());
        visit("sk_prf", &sk_prf, String::new());
      };
      let top_tree = |visit: Visit<'_>| wots_secrets(&Oracle::new(&pk_seed), &sk_seed, D - 1, 0, None, visit);

      let mut generated = None;
      let scenario = run("keygen", &["sk_seed", "sk_prf"], || {
        let fill = |out: &mut [u8]| {
          out.copy_from_slice(black_box(&seeds));
          Ok(())
        };
        generated = Some(Profile::generate_keypair(fill).expect("key generation"));
      });
      record(
        &scenario,
        &mut |visit| {
          keys(visit);
          top_tree(visit);
        },
        &mut json,
      );
      let (public, secret) = generated.expect("generated key");
      let encoded = secret.expose_secret();
      let raw = encoded.as_bytes();
      let public_bytes: [u8; 2 * N] = *public.as_bytes();

      let scenario = run("keygen-fail", &[], || {
        let fill = |out: &mut [u8]| {
          out.copy_from_slice(black_box(&seeds));
          Err(SlhDsaError::RandomGenerationFailed)
        };
        black_box(Profile::generate_keypair(fill).is_err());
      });
      record(&scenario, &mut |visit| keys(visit), &mut json);

      let mut imported = None;
      let scenario = run("import", &["sk_seed", "sk_prf"], || {
        imported = Some(SecretKey::try_from_slice(black_box(raw)).expect("import"));
      });
      record(
        &scenario,
        &mut |visit| {
          keys(visit);
          top_tree(visit);
        },
        &mut json,
      );
      assert_eq!(imported.expect("imported key").public_key(), &public);

      let mut foreign = *raw;
      foreign[4 * N - 1] ^= 1;
      let scenario = run("import-reject", &[], || {
        black_box(SecretKey::try_from_slice(black_box(&foreign)).is_err());
      });
      record(
        &scenario,
        &mut |visit| {
          keys(visit);
          top_tree(visit);
        },
        &mut json,
      );

      let hash_secret = HashSecretKey::try_from_slice(raw).expect("pre-hash key import");
      let message = input(MESSAGE, index);
      let addrnd: [u8; N] = input(ADDRND, index)[..N].try_into().expect("addrnd");
      let prehash = input(PREHASH, index);
      for mode in ["sign", "sign-hedged", "sign-hedged-fail", "sign-prehash"] {
        let mut signature = vec![0u8; Profile::SIGNATURE_LENGTH];
        let output: &mut [u8; Profile::SIGNATURE_LENGTH] = signature.as_mut_slice().try_into().expect("length");
        let mut result = None;
        let scenario = run(mode, &[], || {
          let fill = |out: &mut [u8]| {
            out.copy_from_slice(black_box(&addrnd));
            if mode == "sign-hedged-fail" {
              Err(SlhDsaError::RandomGenerationFailed)
            } else {
              Ok(())
            }
          };
          result = Some(match mode {
            "sign" => secret.sign_deterministic(black_box(&message), CONTEXT, output),
            "sign-prehash" => hash_secret.sign_prehash_deterministic(black_box(&prehash), CONTEXT, output),
            _ => secret.sign_with(black_box(&message), CONTEXT, fill, output),
          });
        });
        let failed = mode == "sign-hedged-fail";
        assert_eq!(result.expect("signing ran").is_err(), failed);
        let context_length = [u8::try_from(CONTEXT.len()).expect("short context")];
        let pure: [&[u8]; 4] = [&[0], &context_length, CONTEXT, &message];
        let hashed: [&[u8]; 5] = [&[1], &context_length, CONTEXT, &SHA256_OID, &prehash];
        if mode == "sign-prehash" {
          hash_secret
            .public_key()
            .verify_prehash(&prehash, CONTEXT, &signature)
            .expect("pre-hash signature verifies");
        } else if !failed {
          public
            .verify_with_context(&message, CONTEXT, &signature)
            .expect("signature verifies");
        }
        record(
          &scenario,
          &mut |visit| {
            keys(visit);
            if mode.starts_with("sign-hedged") {
              visit("addrnd", &addrnd, String::new());
            }
            if !failed {
              let parts: &[&[u8]] = if mode == "sign-prehash" { &hashed } else { &pure };
              signing_secrets(&public_bytes, &sk_seed, parts, &signature, visit);
            }
          },
          &mut json,
        );
      }
    }

    // Controls: the search must find a secret planted in a returned frame in
    // either word order, and must not find it once cleared.
    let pk_seed: [u8; N] = seeds(0)[2 * N..].try_into().expect("PK.seed");
    let sk_seed: [u8; N] = seeds(0)[..N].try_into().expect("SK.seed");
    let control = Oracle::new(&pk_seed).tweak(&address(D - 1, 0, WOTS_PRF, [0, 0, 0]), &sk_seed);
    let words = core::array::from_fn(|index| u32::from_be_bytes(control.as_chunks::<4>().0[index]));
    for (name, expect_found, scenario) in [
      (
        "control-planted",
        true,
        run("control-planted", &[], || plant(&control, false)),
      ),
      (
        "control-planted-words",
        true,
        run("control-planted-words", &[], || plant_words(words)),
      ),
      (
        "control-cleared",
        false,
        run("control-cleared", &[], || plant(&control, true)),
      ),
    ] {
      let search = Search::new(&scenario.stack);
      let found = search.whole.contains(&control);
      let ok = found == expect_found;
      println!("{name}: {}", if ok { "pass" } else { "FAIL" });
      write!(json, ",{{\"scenario\":\"{name}\",\"passed\":{ok},\"found\":{found}}}").expect("String formatting");
      passed &= ok;
    }
    write!(json, "],\"passed\":{passed},\"control_needle\":\"{}\"}}", hex(&control)).expect("String formatting");
    fs::write(directory.join("report.json"), json + "\n").expect("write report");
    passed
  }
}

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
fn main() {
  std::process::exit(if native::run_all() { 0 } else { 1 });
}

#[cfg(not(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
)))]
fn main() {
  eprintln!("native SLH-DSA residue measurement requires macOS AArch64 or Linux AArch64/x86-64");
  std::process::exit(2);
}
