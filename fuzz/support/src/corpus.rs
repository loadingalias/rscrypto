use std::{fs, path::Path, path::PathBuf};

/// Replay the explicit committed seed set against the runner used by libFuzzer.
///
/// `fuzz/committed-seeds.txt` owns the repository-relative seed paths. Set
/// `RSCRYPTO_FUZZ_CORPUS=local` to replay every file in the corpus directory,
/// including untracked discoveries. The default is `committed`; unknown modes,
/// missing seeds, and empty replay sets fail instead of reducing coverage.
pub fn replay_corpus_dir<F, P>(target: &str, corpus_dir: P, run: F) -> usize
where
  F: Fn(&[u8]),
  P: AsRef<Path>,
{
  let mode = std::env::var("RSCRYPTO_FUZZ_CORPUS")
    .or_else(|error| match error {
      std::env::VarError::NotPresent => Ok("committed".to_owned()),
      error => Err(error),
    })
    .expect("RSCRYPTO_FUZZ_CORPUS must be committed or local");
  assert!(
    mode == "committed" || mode == "local",
    "RSCRYPTO_FUZZ_CORPUS must be committed or local"
  );
  let root = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../..")
    .canonicalize()
    .expect("corpus replay repository root must exist");
  let corpus_dir = corpus_dir
    .as_ref()
    .canonicalize()
    .expect("corpus replay directory must exist");
  let relative = corpus_dir
    .strip_prefix(&root)
    .expect("corpus replay directory must be in the repository");
  let seeds = include_str!("../../committed-seeds.txt")
    .lines()
    .map(Path::new)
    .filter(|path| path.parent() == Some(relative))
    .map(|path| root.join(path))
    .collect();
  replay(target, &corpus_dir, seeds, mode == "local", run)
}

fn replay(target: &str, corpus_dir: &Path, seeds: Vec<PathBuf>, local: bool, run: impl Fn(&[u8])) -> usize {
  let mut files = if local {
    fs::read_dir(corpus_dir)
      .expect("corpus replay directory must be readable")
      .map(|entry| entry.expect("corpus replay directory entries must be readable"))
      .filter_map(|entry| {
        let kind = entry
          .file_type()
          .expect("corpus replay entry metadata must be readable");
        (kind.is_file() || kind.is_symlink()).then(|| entry.path())
      })
      .collect::<Vec<_>>()
  } else {
    seeds
  };
  files.sort();
  assert!(
    !files.is_empty(),
    "corpus replay: target `{target}` has an empty replay set at {}",
    corpus_dir.display()
  );
  for path in &files {
    let data = fs::read(path)
      .map_err(|error| format!("{}: {error}", path.display()))
      .expect("corpus replay input must be readable");
    run(&data);
  }
  eprintln!(
    "corpus replay: {target}: {} inputs ({})",
    files.len(),
    if local { "local" } else { "committed" }
  );
  files.len()
}

#[cfg(test)]
mod tests {
  use super::replay;
  use core::{
    cell::RefCell,
    sync::atomic::{AtomicUsize, Ordering},
  };
  use std::{fs, path::PathBuf};

  struct Fixture(PathBuf);

  impl Fixture {
    fn new() -> Self {
      static NEXT: AtomicUsize = AtomicUsize::new(0);
      let path = std::env::temp_dir().join(format!(
        "rscrypto-corpus-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
      ));
      fs::create_dir(&path).expect("create corpus fixture directory");
      Self(path)
    }
  }

  impl Drop for Fixture {
    fn drop(&mut self) {
      fs::remove_dir_all(&self.0).expect("remove corpus fixture directory");
    }
  }

  #[test]
  fn discoveries_only_change_local_replay() {
    let fixture = Fixture::new();
    let seed = fixture.0.join("seed");
    fs::write(&seed, b"committed").expect("write committed seed");
    let seen = RefCell::new(Vec::new());
    let run = |data: &[u8]| seen.borrow_mut().push(data.to_vec());
    assert_eq!(replay("fixture", &fixture.0, vec![seed.clone()], false, run), 1);
    assert_eq!(*seen.borrow(), [b"committed".to_vec()]);
    fs::write(fixture.0.join("discovery"), b"local").expect("write local discovery");
    seen.borrow_mut().clear();
    assert_eq!(replay("fixture", &fixture.0, vec![seed.clone()], false, run), 1);
    assert_eq!(*seen.borrow(), [b"committed".to_vec()]);
    seen.borrow_mut().clear();
    assert_eq!(replay("fixture", &fixture.0, vec![seed], true, run), 2);
    assert_eq!(*seen.borrow(), [b"local".to_vec(), b"committed".to_vec()]);
    assert_eq!(
      fs::read(fixture.0.join("discovery")).expect("read preserved local discovery"),
      b"local"
    );
  }

  #[test]
  #[should_panic(expected = "corpus replay input")]
  fn missing_seed_fails_even_with_local_discoveries() {
    let fixture = Fixture::new();
    fs::write(fixture.0.join("discovery"), b"local").expect("write local discovery");
    replay("fixture", &fixture.0, vec![fixture.0.join("missing")], false, |_| {});
  }

  #[test]
  #[should_panic(expected = "empty replay set")]
  fn empty_inventory_fails_even_with_local_discoveries() {
    let fixture = Fixture::new();
    fs::write(fixture.0.join("discovery"), b"local").expect("write local discovery");
    replay("fixture", &fixture.0, vec![], false, |_| {});
  }
}
