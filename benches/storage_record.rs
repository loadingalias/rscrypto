//! Via T1411. Check the ciphertext root, then optionally time each complete crypto operation.
//! A single extent has one nonce and a detached tag. The overlap case requires two existing units.

use core::hint::black_box;
use std::{
  sync::mpsc::{Receiver, SyncSender, sync_channel},
  time::Instant,
};

use rscrypto::{Aegis256, Aegis256Key, Aegis256Tag, Blake3, Digest, aead::Nonce256, aead::expert::AeadWithNonce};

const KEY: [u8; 32] = [0x31; 32];
const SAMPLES: usize = 2_001;

fn nonce(size: usize, sample: usize, unit: u8) -> Nonce256 {
  let mut bytes = [0; 32];
  bytes[..8].copy_from_slice(&u64::try_from(size).expect("record size").to_le_bytes());
  bytes[8..16].copy_from_slice(&u64::try_from(sample).expect("sample number").to_le_bytes());
  bytes[16] = unit;
  Nonce256::from_bytes(bytes)
}

fn seal(cipher: &Aegis256, nonce: &Nonce256, plaintext: &[u8], output: &mut [u8]) -> Aegis256Tag {
  output.copy_from_slice(plaintext);
  cipher.encrypt_in_place(nonce, b"", output).expect("bounded record")
}

fn reference(plaintext: &[u8], nonce: &Nonce256) -> (Vec<u8>, [u8; 16]) {
  let mut ciphertext = plaintext.to_vec();
  let tag = aegis::aegis256::Aegis256::<16>::new(&KEY, nonce.as_bytes()).encrypt_in_place(&mut ciphertext, b"");
  (ciphertext, tag)
}

// One worker owns the sealed prefix while the caller seals the next unit.
// The returned hasher has consumed exactly the prefix at offset zero.
fn hash_prefix(input: Receiver<Vec<u8>>, output: SyncSender<(Blake3, Vec<u8>)>) {
  for prefix in input {
    let mut hash = Blake3::new();
    hash.update(&prefix);
    output.send((hash, prefix)).expect("active caller");
  }
}

fn samples<T>(measure: bool, name: &str, size: usize, mut op: impl FnMut(usize) -> T) {
  if measure {
    for sample in 1..=SAMPLES {
      black_box(op(sample));
    }
    let mut ns = Vec::with_capacity(SAMPLES);
    for sample in SAMPLES + 1..=2 * SAMPLES {
      let start = Instant::now();
      black_box(op(sample));
      ns.push(start.elapsed().as_nanos());
    }
    ns.sort_unstable();
    println!("{name},{size},{SAMPLES},{},{}", ns[SAMPLES / 2], ns[SAMPLES * 99 / 100]);
  }
}

fn record(size: usize, measure: bool) {
  let cipher = Aegis256::new(&Aegis256Key::from_bytes(KEY));
  let plaintext: Vec<_> = (0..size).map(|i| u8::try_from(i % 251).expect("byte")).collect();
  let mut ciphertext = vec![0; size];
  let single_nonce = nonce(size, 0, 0);
  let (expected, expected_tag) = reference(&plaintext, &single_nonce);
  let tag = seal(&cipher, &single_nonce, &plaintext, &mut ciphertext);
  assert_eq!(ciphertext, expected);
  assert_eq!(tag.as_ref(), expected_tag);
  assert_eq!(Blake3::digest(&ciphertext), *blake3::hash(&expected).as_bytes());
  samples(measure, "one-seal-and-root", size, |sample| {
    let tag = seal(&cipher, &nonce(size, sample, 0), &plaintext, &mut ciphertext);
    (tag, Blake3::digest(&ciphertext))
  });
  samples(measure, "digest", size, |_| Blake3::digest(black_box(&expected)));
  samples(measure, "seal-in-place", size, |sample| {
    cipher
      .encrypt_in_place(&nonce(size, sample, 3), b"", black_box(&mut ciphertext))
      .expect("bounded record")
  });

  let (first, last) = plaintext.split_at(size / 2);
  let (mut expected, first_tag) = reference(first, &nonce(size, 0, 1));
  let (suffix, last_tag) = reference(last, &nonce(size, 0, 2));
  expected.extend_from_slice(&suffix);
  let expected_root = *blake3::hash(&expected).as_bytes();

  let (mut prefix, mut suffix) = (vec![0; first.len()], vec![0; last.len()]);
  std::thread::scope(|scope| {
    let (send_prefix, input) = sync_channel(1);
    let (send_hash, hash_ready) = sync_channel(1);
    scope.spawn(move || hash_prefix(input, send_hash));
    for overlap in [false, true] {
      let mut two_units = |sample| {
        let tag1 = seal(&cipher, &nonce(size, sample, 1), first, &mut prefix);
        if overlap {
          send_prefix.send(core::mem::take(&mut prefix)).expect("active worker");
        }
        let tag2 = seal(&cipher, &nonce(size, sample, 2), last, &mut suffix);
        let mut hasher = if overlap {
          let (hasher, buffer) = hash_ready.recv().expect("hashed prefix");
          prefix = buffer;
          hasher
        } else {
          let mut hasher = Blake3::new();
          hasher.update(&prefix);
          hasher
        };
        hasher.update(&suffix);
        (tag1, tag2, hasher.finalize())
      };
      let (tag1, tag2, root) = two_units(0);
      assert_eq!(tag1.as_ref(), first_tag);
      assert_eq!(tag2.as_ref(), last_tag);
      assert_eq!(root, expected_root);
      samples(
        measure,
        if overlap {
          "two-seals-overlap"
        } else {
          "two-seals-serial"
        },
        size,
        two_units,
      );
    }
  });
}

// Each extent retains its original nonce, tag, ciphertext, and independent root.
fn record_pair(size: usize, measure: bool) {
  let cipher = Aegis256::new(&Aegis256Key::from_bytes(KEY));
  let plaintext: Vec<_> = (0..size).map(|i| u8::try_from(i % 251).expect("byte")).collect();
  let mut first = vec![0; size];
  let mut second = vec![0; size];
  let expected = [4, 5].map(|unit| {
    let (bytes, tag) = reference(&plaintext, &nonce(size, 0, unit));
    (tag, *blake3::hash(&bytes).as_bytes())
  });
  std::thread::scope(|scope| {
    let (jobs, inputs) = sync_channel::<Vec<u8>>(1);
    let (done, results) = sync_channel(1);
    scope.spawn(move || {
      for bytes in inputs {
        done.send((Blake3::digest(&bytes), bytes)).expect("active caller");
      }
    });
    for overlap in [false, true] {
      let mut pair = |sample| {
        let tag1 = seal(&cipher, &nonce(size, sample, 4), &plaintext, &mut first);
        let first_root = if overlap {
          jobs.send(core::mem::take(&mut first)).expect("active worker");
          None
        } else {
          Some(Blake3::digest(&first))
        };
        let tag2 = seal(&cipher, &nonce(size, sample, 5), &plaintext, &mut second);
        let second_root = Blake3::digest(&second);
        let first_root = first_root.unwrap_or_else(|| {
          let (root, bytes) = results.recv().expect("worker root");
          first = bytes;
          root
        });
        [(tag1, first_root), (tag2, second_root)]
      };
      for ((tag, root), (expected_tag, expected_root)) in pair(0).iter().zip(expected) {
        assert_eq!(tag.as_ref(), expected_tag);
        assert_eq!(*root, expected_root);
      }
      samples(
        measure,
        if overlap {
          "two-records-overlap"
        } else {
          "two-records-serial"
        },
        size,
        pair,
      );
    }
  });
}

fn main() {
  let measure = std::env::args().any(|arg| arg == "--measure");
  if measure {
    println!("operation,payload_bytes,samples,p50_ns,p99_ns");
  }
  for size in [1024, 4096, 65_536] {
    record(size, measure);
    record_pair(size, measure);
  }
  if !measure {
    println!("Ciphertext roots and detached tags passed for 1, 4, and 64 KiB.");
  }
}
