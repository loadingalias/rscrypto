use core::sync::atomic::{AtomicU64, Ordering};
use std::{
  fs::{self, File, OpenOptions},
  io::Write,
  path::{Path, PathBuf},
};

pub(crate) struct TemporaryFile {
  path: PathBuf,
}

impl TemporaryFile {
  pub(crate) fn new(input: &[u8]) -> Self {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut created = None;
    for _ in 0..64 {
      let id = NEXT
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| value.checked_add(1))
        .expect("file fixture identifier fits u64");
      let path = std::env::temp_dir().join(format!("rscrypto-blake3-file-{}-{id}", std::process::id()));
      let result = OpenOptions::new().write(true).create_new(true).open(&path);
      if result
        .as_ref()
        .is_err_and(|error| error.kind() == std::io::ErrorKind::AlreadyExists)
      {
        continue;
      }
      let mut file = result.expect("create a private file fixture");
      file.write_all(input).expect("write file fixture");
      drop(file);
      created = Some(Self { path });
      break;
    }
    created.expect("reserve an unused file-fixture path")
  }

  pub(crate) fn path(&self) -> &Path {
    &self.path
  }

  pub(crate) fn open(&self) -> File {
    File::open(self.path()).expect("open file fixture")
  }
}

impl Drop for TemporaryFile {
  fn drop(&mut self) {
    let _cleanup_result = fs::remove_file(&self.path);
  }
}
