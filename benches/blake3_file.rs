//! Complete file hashing, with verified Linux page-cache preconditions.

#[path = "common/criterion.rs"]
mod bench_config;

use core::{hint::black_box, time::Duration};
use std::{
  fs::{File, OpenOptions},
  io::{Read, Seek, SeekFrom, Write},
  path::{Path, PathBuf},
  process::Command,
  sync::OnceLock,
  time::Instant,
};

use criterion::{BenchmarkId, Criterion, SamplingMode, Throughput};
use rscrypto::{Blake3, Digest, hashes::expert::blake3_tree::Blake3Tree};

const MIB: u64 = 1024 * 1024;
const SIZES: [u64; 5] = [MIB, 16 * MIB, 256 * MIB, 1024 * MIB, 10 * 1024 * MIB];
const B3SUM_EXECUTABLE: &str = "RSCRYPTO_BLAKE3_B3SUM";
const B3SUM_RECEIPT: &str = "RSCRYPTO_BLAKE3_B3SUM_RECEIPT";
const B3SUM_OPERATION: &str = "upstream-b3sum-4";
const UPSTREAM_VERSION: &str = "1.8.7";
// The selected blake3 package's .cargo_vcs_info.json pins this upstream commit.
// Its b3sum package has the same version and enables blake3's mmap/rayon features.
const UPSTREAM_COMMIT: &str = "f3149ec5bb5449af877ba20377a11008ff499fa2";
static B3SUM: OnceLock<B3sum> = OnceLock::new();

struct B3sum {
  executable: PathBuf,
  sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
  use core::fmt::Write as _;

  use sha2::Digest as _;
  let mut encoded = String::with_capacity(64);
  for byte in sha2::Sha256::digest(bytes) {
    write!(encoded, "{byte:02x}").expect("format SHA-256 identity");
  }
  encoded
}

impl B3sum {
  /// The receipt is a build attestation, not independent proof of source-to-binary
  /// correspondence. A retained build log/source inventory must substantiate it.
  fn load() -> Self {
    let executable = PathBuf::from(std::env::var_os(B3SUM_EXECUTABLE).expect("RSCRYPTO_BLAKE3_B3SUM is required"))
      .canonicalize()
      .expect("resolve explicit b3sum executable");
    let receipt_path = PathBuf::from(
      std::env::var_os(B3SUM_RECEIPT).expect("RSCRYPTO_BLAKE3_B3SUM_RECEIPT is required for the b3sum row"),
    )
    .canonicalize()
    .expect("resolve b3sum build receipt");
    let bytes = std::fs::read(&receipt_path).expect("read b3sum build receipt");
    let receipt: serde_json::Value = serde_json::from_slice(&bytes).expect("b3sum receipt JSON");
    assert_eq!(receipt["schema"].as_u64(), Some(1), "b3sum receipt schema");
    assert_eq!(receipt["package"].as_str(), Some("b3sum"), "b3sum receipt package");
    assert_eq!(receipt["version"].as_str(), Some(UPSTREAM_VERSION), "b3sum version pin");
    assert_eq!(
      receipt["blake3_version"].as_str(),
      Some(UPSTREAM_VERSION),
      "b3sum BLAKE3 dependency pin"
    );
    assert_eq!(
      receipt["repository"].as_str(),
      Some("https://github.com/BLAKE3-team/BLAKE3"),
      "b3sum upstream source"
    );
    assert_eq!(
      receipt["git_commit"].as_str(),
      Some(UPSTREAM_COMMIT),
      "b3sum source pin"
    );
    assert_eq!(
      receipt["source_clean"].as_bool(),
      Some(true),
      "b3sum source must be clean"
    );
    let binary_sha256 = receipt["binary_sha256"]
      .as_str()
      .expect("b3sum binary SHA-256")
      .to_owned();
    let cli = Self {
      executable,
      sha256: binary_sha256,
    };
    cli.verify_unchanged();
    let version = Command::new(&cli.executable)
      .arg("--version")
      .output()
      .expect("run b3sum --version");
    assert!(
      version.status.success() && version.stderr.is_empty(),
      "b3sum version query failed"
    );
    assert_eq!(version.stdout, b"b3sum 1.8.7\n", "b3sum executable version pin");
    cli.verify_unchanged();
    eprintln!(
      "rscrypto-b3sum {}",
      serde_json::json!({
        "executable": cli.executable,
        "binary_sha256": cli.sha256,
        "receipt": receipt_path,
        "receipt_sha256": sha256(&bytes),
        "package": "b3sum", "version": UPSTREAM_VERSION,
        "blake3_version": UPSTREAM_VERSION, "git_commit": UPSTREAM_COMMIT,
        "source_evidence": "build attestation; retain build log and source inventory",
        "arguments": ["--num-threads", "4", "--raw", "--", "FIXTURE"],
      })
    );
    cli
  }

  fn verify_unchanged(&self) {
    assert_eq!(
      sha256(&std::fs::read(&self.executable).expect("read explicit b3sum executable")),
      self.sha256,
      "b3sum executable differs from its pinned build receipt"
    );
  }

  fn digest(&self, fixture: &Fixture) -> [u8; 32] {
    // Process setup, four-worker pool startup, hashing, stdout capture and exit
    // all belong to the timed CLI operation. Version/provenance checks do not.
    let output = Command::new(&self.executable)
      .args(["--num-threads", "4", "--raw", "--"])
      .arg(&fixture.path)
      .output()
      .expect("execute b3sum file hashing");
    assert!(
      output.status.success() && output.stderr.is_empty(),
      "b3sum failed: {}",
      String::from_utf8_lossy(&output.stderr)
    );
    output.stdout.try_into().expect("b3sum --raw returns exactly 32 bytes")
  }
}

struct Fixture {
  path: PathBuf,
  len: u64,
  digest: [u8; 32],
}

impl Fixture {
  fn load(directory: &Path, len: u64) -> Self {
    let manifest: serde_json::Value =
      serde_json::from_slice(&std::fs::read(directory.join(format!("{len}.json"))).expect("fixture manifest"))
        .expect("fixture JSON");
    let path = directory.join(format!("{len}.bin"));
    assert_eq!(manifest["length"].as_u64(), Some(len));
    assert_eq!(path.metadata().expect("fixture metadata").len(), len);
    let encoded = manifest["blake3"].as_str().expect("fixture digest");
    assert_eq!(encoded.len(), 64);
    let digest = core::array::from_fn(|i| {
      let start = i.strict_mul(2);
      u8::from_str_radix(encoded.get(start..start.strict_add(2)).expect("ASCII hex pair"), 16).expect("hex digest")
    });
    Self { path, len, digest }
  }

  fn condition(&self, mode: &str, logfile: &Path, case: &str) {
    let status = Command::new("python3")
      .arg("-c")
      .arg(include_str!("../scripts/bench/file_cache.py"))
      .arg(mode)
      .arg(&self.path)
      .arg(logfile)
      .arg(case)
      .status()
      .expect("run Linux file-cache precondition");
    assert!(status.success(), "file-cache precondition failed");
  }
}

fn reader(fixture: &Fixture) -> [u8; 32] {
  let mut file = File::open(&fixture.path).expect("open fixture");
  let mut hasher = Blake3::new();
  assert_eq!(hasher.update_reader(&mut file).expect("read fixture"), fixture.len);
  hasher.finalize()
}

fn subtrees(fixture: &Fixture) -> [u8; 32] {
  let tree = Blake3Tree::new();
  let left_len = Blake3Tree::left_subtree_len(fixture.len).expect("multi-chunk file");
  let right_len = fixture.len.strict_sub(left_len);
  let left_split = Blake3Tree::left_subtree_len(left_len).expect("multi-chunk left child");
  let right_split = Blake3Tree::left_subtree_len(right_len).expect("multi-chunk right child");
  let ranges = [
    (0, left_split),
    (left_split, left_len.strict_sub(left_split)),
    (left_len, right_split),
    (left_len.strict_add(right_split), right_len.strict_sub(right_split)),
  ];
  // Caller scheduling only: every leaf and parent invokes the public production
  // subtree API. Independent descriptors avoid a shared seek position.
  let cvs = std::thread::scope(|scope| {
    let workers = ranges.map(|(offset, len)| {
      let tree = &tree;
      scope.spawn(move || {
        let mut file = File::open(&fixture.path).expect("open subtree fixture");
        file.seek(SeekFrom::Start(offset)).expect("seek subtree range");
        let mut subtree = tree.subtree(offset).expect("canonical subtree offset");
        assert_eq!(subtree.update_reader(&mut file.take(len)).expect("read subtree"), len);
        subtree.finalize().expect("nonempty subtree")
      })
    });
    workers.map(|worker| worker.join().expect("subtree worker"))
  });
  let left = tree.merge(&cvs[0], &cvs[1]).expect("left parent");
  let right = tree.merge(&cvs[2], &cvs[3]).expect("right parent");
  tree.merge_root(&left, &right).expect("root parent")
}

fn mmap_rayon(fixture: &Fixture) -> [u8; 32] {
  let mut hasher = blake3::Hasher::new();
  hasher.update_mmap_rayon(&fixture.path).expect("upstream file read");
  *hasher.finalize().as_bytes()
}

fn b3sum(fixture: &Fixture) -> [u8; 32] {
  B3SUM.get_or_init(B3sum::load).digest(fixture)
}

type FileOperation = fn(&Fixture) -> [u8; 32];
const OPERATIONS: [(&str, FileOperation); 4] = [
  ("rscrypto-reader", reader),
  ("rscrypto-subtrees-4", subtrees),
  ("upstream-mmap-rayon-4", mmap_rayon),
  (B3SUM_OPERATION, b3sum),
];

fn files(c: &mut Criterion) {
  let listing = std::env::args().any(|arg| arg == "--list");
  let directory = std::env::var_os("RSCRYPTO_BLAKE3_FILE_DIR").map(PathBuf::from);
  let logfile = std::env::var_os("RSCRYPTO_BLAKE3_FILE_CACHE_LOG").map(PathBuf::from);
  if !listing {
    // The long-lived pool used by upstream and the Linux AArch64 reader is
    // untimed. Caller-scheduled subtree worker creation and destruction stay timed.
    rayon::ThreadPoolBuilder::new()
      .num_threads(4)
      .build_global()
      .expect("initialize four-worker pool");
  }
  for cache in ["cold", "warm"] {
    let prefix = format!("blake3/file/{cache}/");
    if !bench_config::selected(&prefix) {
      continue;
    }
    let mut group = c.benchmark_group(prefix.trim_end_matches('/'));
    group.sampling_mode(SamplingMode::Flat);
    for len in SIZES {
      group.throughput(Throughput::Bytes(len));
      for (name, operation) in OPERATIONS {
        let case = format!("{prefix}{name}/{len}B");
        if !bench_config::selected(&case) {
          continue;
        }
        let fixture = OnceLock::new();
        group.bench_function(BenchmarkId::new(name, format!("{len}B")), |b| {
          // Criterion invokes this only for selected measurements, not listing
          // or filtered rows. In particular, ordinary rows do not require b3sum.
          let fixture = fixture.get_or_init(|| {
            if name == B3SUM_OPERATION {
              B3SUM.get_or_init(B3sum::load);
            }
            let fixture = Fixture::load(directory.as_deref().expect("RSCRYPTO_BLAKE3_FILE_DIR is required"), len);
            // Validate once before the timed loop, including repeated samples.
            assert_eq!(operation(&fixture), fixture.digest, "file output: {case}");
            fixture
          });
          let logfile = logfile.as_deref().expect("RSCRYPTO_BLAKE3_FILE_CACHE_LOG is required");
          // Timed: open/seek, construction, reader allocations and clears,
          // hashing, caller threads, public merges, finalization and destruction.
          // The b3sum row also includes command setup, process/worker startup,
          // output capture and process exit; its build/version check is untimed.
          // Untimed: fixture generation, validation and cache conditioning.
          // Cold means page-cache cold; no EBS/controller-cache claim is made.
          b.iter_custom(|iters| {
            let mut elapsed = Duration::ZERO;
            for _ in 0..iters {
              fixture.condition(cache, logfile, &case);
              let start = Instant::now();
              let digest = black_box(operation(black_box(fixture)));
              elapsed = elapsed.checked_add(start.elapsed()).expect("elapsed duration fits");
              assert_eq!(digest, fixture.digest, "measured file output: {case}");
            }
            elapsed
          });
        });
      }
    }
    group.finish();
  }
  if let Some(cli) = B3SUM.get() {
    cli.verify_unchanged();
  }
}

fn prepare(directory: &Path, len: u64) {
  assert!(SIZES.contains(&len), "size must be in the fixed file catalog");
  std::fs::create_dir_all(directory).expect("create fixture directory");
  let path = directory.join(format!("{len}.bin"));
  let mut file = OpenOptions::new()
    .write(true)
    .create_new(true)
    .open(&path)
    .expect("create new fixture");
  let block: Vec<u8> = (0..MIB).map(|i| u8::try_from(i % 251).expect("pattern byte")).collect();
  let mut hasher = blake3::Hasher::new();
  for _ in 0..len / MIB {
    file.write_all(&block).expect("write complete fixture block");
    hasher.update(&block);
  }
  file.sync_all().expect("sync fixture");
  let manifest = serde_json::json!({
    "length": len, "blake3": hasher.finalize().to_hex().as_str(),
    "pattern": "repeat a 1 MiB block whose byte i is i modulo 251",
  });
  let mut output = OpenOptions::new()
    .write(true)
    .create_new(true)
    .open(directory.join(format!("{len}.json")))
    .expect("create new fixture identity");
  writeln!(
    output,
    "{}",
    serde_json::to_string_pretty(&manifest).expect("fixture JSON")
  )
  .expect("write identity");
}

fn main() {
  let args: Vec<_> = std::env::args_os().collect();
  if args.get(1).is_some_and(|arg| arg == "--rscrypto-verify-b3sum") {
    assert_eq!(args.len(), 2, "b3sum verification takes no arguments");
    B3SUM.get_or_init(B3sum::load);
    return;
  }
  if args.get(1).is_some_and(|arg| arg == "--rscrypto-prepare-file") {
    assert_eq!(args.len(), 4, "expected DIRECTORY BYTES");
    prepare(
      Path::new(&args[2]),
      args[3].to_str().expect("length text").parse().expect("length integer"),
    );
    return;
  }
  if args.get(1).is_some_and(|arg| arg == "--rscrypto-verify-file") {
    assert_eq!(args.len(), 4, "expected DIRECTORY BYTES");
    let len = args[3].to_str().expect("length text").parse().expect("length integer");
    let fixture = Fixture::load(Path::new(&args[2]), len);
    for (name, operation) in OPERATIONS {
      if name == B3SUM_OPERATION
        && std::env::var_os(B3SUM_EXECUTABLE).is_none()
        && std::env::var_os(B3SUM_RECEIPT).is_none()
      {
        continue;
      }
      assert_eq!(operation(&fixture), fixture.digest, "file output: {name}");
      println!("verified {name}: {len} bytes");
    }
    if let Some(cli) = B3SUM.get() {
      cli.verify_unchanged();
    }
    return;
  }
  bench_config::run(&[files]);
}
