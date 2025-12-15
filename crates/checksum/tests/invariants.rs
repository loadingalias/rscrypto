use checksum::{Crc16CcittFalse, Crc16Ibm, Crc24, Crc32, Crc32c, Crc64, Crc64Nvme};

fn gen_bytes(len: usize, seed: u64) -> Vec<u8> {
  let mut out = vec![0u8; len];
  let mut x = seed;
  for b in &mut out {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *b = (x as u8).wrapping_add((x >> 8) as u8);
  }
  out
}

fn crc32_reflected_bitwise(poly_reflected: u32, data: &[u8]) -> u32 {
  let mut crc = 0xffff_ffffu32;
  for &b in data {
    crc ^= b as u32;
    for _ in 0..8 {
      let mask = 0u32.wrapping_sub(crc & 1);
      crc = (crc >> 1) ^ (poly_reflected & mask);
    }
  }
  crc ^ 0xffff_ffff
}

fn crc_reflected_bitwise_u64(poly_reflected: u64, width: u8, init: u64, xor_out: u64, data: &[u8]) -> u64 {
  let mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
  let mut crc = init & mask;
  for &b in data {
    crc ^= u64::from(b);
    for _ in 0..8 {
      let mask = 0u64.wrapping_sub(crc & 1);
      crc = (crc >> 1) ^ (poly_reflected & mask);
    }
  }
  (crc ^ xor_out) & mask
}

fn crc_normal_bitwise_u64(poly: u64, width: u8, init: u64, xor_out: u64, data: &[u8]) -> u64 {
  let mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
  let top = 1u64 << (width as u32 - 1);
  let shift = width as u32 - 8;

  let mut crc = init & mask;
  for &b in data {
    crc ^= u64::from(b) << shift;
    for _ in 0..8 {
      if (crc & top) != 0 {
        crc = ((crc << 1) ^ poly) & mask;
      } else {
        crc = (crc << 1) & mask;
      }
    }
  }
  (crc ^ xor_out) & mask
}

#[test]
fn crc32_invariants() {
  let lengths = [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 1024, 2048];
  let seeds = [0u64, 1, 0x0123_4567_89ab_cdef, 0xd1b5_4a32_d192_ed03];

  for &len in &lengths {
    for &seed in &seeds {
      let data = gen_bytes(len, seed ^ len as u64);

      let oneshot = Crc32::checksum(&data);
      let reference = crc32_reflected_bitwise(0xedb8_8320, &data);
      assert_eq!(oneshot, reference, "crc32 reference mismatch at len={}", len);

      for &split in &[0usize, 1, len / 2, len.saturating_sub(1), len] {
        if split > len {
          continue;
        }
        let (a, b) = data.split_at(split);

        let mut h = Crc32::new();
        h.update(a);
        h.update(b);
        assert_eq!(
          h.finalize(),
          oneshot,
          "crc32 incremental mismatch at len={} split={}",
          len,
          split
        );

        let crc_a = Crc32::checksum(a);
        let mut r = Crc32::resume(crc_a);
        r.update(b);
        assert_eq!(
          r.finalize(),
          oneshot,
          "crc32 resume mismatch at len={} split={}",
          len,
          split
        );

        let crc_b = Crc32::checksum(b);
        let combined = Crc32::combine(crc_a, crc_b, b.len());
        assert_eq!(
          combined, oneshot,
          "crc32 combine mismatch at len={} split={}",
          len, split
        );
      }
    }
  }
}

#[test]
fn crc32c_invariants() {
  let lengths = [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 1024, 2048];
  let seeds = [0u64, 1, 0x0123_4567_89ab_cdef, 0x5d58_39a7_3d87_1ceb];

  for &len in &lengths {
    for &seed in &seeds {
      let data = gen_bytes(len, seed ^ len as u64);

      let oneshot = Crc32c::checksum(&data);
      let reference = crc32_reflected_bitwise(0x82f6_3b78, &data);
      assert_eq!(oneshot, reference, "crc32c reference mismatch at len={}", len);

      for &split in &[0usize, 1, len / 2, len.saturating_sub(1), len] {
        if split > len {
          continue;
        }
        let (a, b) = data.split_at(split);

        let mut h = Crc32c::new();
        h.update(a);
        h.update(b);
        assert_eq!(
          h.finalize(),
          oneshot,
          "crc32c incremental mismatch at len={} split={}",
          len,
          split
        );

        let crc_a = Crc32c::checksum(a);
        let mut r = Crc32c::resume(crc_a);
        r.update(b);
        assert_eq!(
          r.finalize(),
          oneshot,
          "crc32c resume mismatch at len={} split={}",
          len,
          split
        );

        let crc_b = Crc32c::checksum(b);
        let combined = Crc32c::combine(crc_a, crc_b, b.len());
        assert_eq!(
          combined, oneshot,
          "crc32c combine mismatch at len={} split={}",
          len, split
        );
      }
    }
  }
}

#[test]
fn crc16_ibm_invariants() {
  let lengths = [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 1024, 2048];
  let seeds = [0u64, 1, 0x0123_4567_89ab_cdef, 0xd1b5_4a32_d192_ed03];

  for &len in &lengths {
    for &seed in &seeds {
      let data = gen_bytes(len, seed ^ len as u64);

      let oneshot = Crc16Ibm::checksum(&data);
      let reference = crc_reflected_bitwise_u64(0xA001, 16, 0x0000, 0x0000, &data) as u16;
      assert_eq!(oneshot, reference, "crc16/ibm reference mismatch at len={}", len);

      for &split in &[0usize, 1, len / 2, len.saturating_sub(1), len] {
        if split > len {
          continue;
        }
        let (a, b) = data.split_at(split);

        let mut h = Crc16Ibm::new();
        h.update(a);
        h.update(b);
        assert_eq!(
          h.finalize(),
          oneshot,
          "crc16/ibm incremental mismatch at len={} split={}",
          len,
          split
        );

        let crc_a = Crc16Ibm::checksum(a);
        let mut r = Crc16Ibm::resume(crc_a);
        r.update(b);
        assert_eq!(
          r.finalize(),
          oneshot,
          "crc16/ibm resume mismatch at len={} split={}",
          len,
          split
        );

        let crc_b = Crc16Ibm::checksum(b);
        let combined = Crc16Ibm::combine(crc_a, crc_b, b.len());
        assert_eq!(
          combined, oneshot,
          "crc16/ibm combine mismatch at len={} split={}",
          len, split
        );
      }
    }
  }
}

#[test]
fn crc16_ccitt_false_invariants() {
  let lengths = [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 1024, 2048];
  let seeds = [0u64, 1, 0x0123_4567_89ab_cdef, 0x5d58_39a7_3d87_1ceb];

  for &len in &lengths {
    for &seed in &seeds {
      let data = gen_bytes(len, seed ^ len as u64);

      let oneshot = Crc16CcittFalse::checksum(&data);
      let reference = crc_normal_bitwise_u64(0x1021, 16, 0xFFFF, 0x0000, &data) as u16;
      assert_eq!(
        oneshot, reference,
        "crc16/ccitt-false reference mismatch at len={}",
        len
      );

      for &split in &[0usize, 1, len / 2, len.saturating_sub(1), len] {
        if split > len {
          continue;
        }
        let (a, b) = data.split_at(split);

        let mut h = Crc16CcittFalse::new();
        h.update(a);
        h.update(b);
        assert_eq!(
          h.finalize(),
          oneshot,
          "crc16/ccitt-false incremental mismatch at len={} split={}",
          len,
          split
        );

        let crc_a = Crc16CcittFalse::checksum(a);
        let mut r = Crc16CcittFalse::resume(crc_a);
        r.update(b);
        assert_eq!(
          r.finalize(),
          oneshot,
          "crc16/ccitt-false resume mismatch at len={} split={}",
          len,
          split
        );

        let crc_b = Crc16CcittFalse::checksum(b);
        let combined = Crc16CcittFalse::combine(crc_a, crc_b, b.len());
        assert_eq!(
          combined, oneshot,
          "crc16/ccitt-false combine mismatch at len={} split={}",
          len, split
        );
      }
    }
  }
}

#[test]
fn crc24_openpgp_invariants() {
  let lengths = [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 1024, 2048];
  let seeds = [0u64, 1, 0x0123_4567_89ab_cdef, 0x5d58_39a7_3d87_1ceb];

  for &len in &lengths {
    for &seed in &seeds {
      let data = gen_bytes(len, seed ^ len as u64);

      let oneshot = Crc24::checksum(&data);
      let reference = crc_normal_bitwise_u64(0x86_4C_FB, 24, 0xB7_04_CE, 0x00_00_00, &data) as u32;
      assert_eq!(oneshot, reference, "crc24/openpgp reference mismatch at len={}", len);

      for &split in &[0usize, 1, len / 2, len.saturating_sub(1), len] {
        if split > len {
          continue;
        }
        let (a, b) = data.split_at(split);

        let mut h = Crc24::new();
        h.update(a);
        h.update(b);
        assert_eq!(
          h.finalize(),
          oneshot,
          "crc24/openpgp incremental mismatch at len={} split={}",
          len,
          split
        );

        let crc_a = Crc24::checksum(a);
        let mut r = Crc24::resume(crc_a);
        r.update(b);
        assert_eq!(
          r.finalize(),
          oneshot,
          "crc24/openpgp resume mismatch at len={} split={}",
          len,
          split
        );

        let crc_b = Crc24::checksum(b);
        let combined = Crc24::combine(crc_a, crc_b, b.len());
        assert_eq!(
          combined, oneshot,
          "crc24/openpgp combine mismatch at len={} split={}",
          len, split
        );
      }
    }
  }
}

#[test]
fn crc64_invariants() {
  let lengths = [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 1024, 2048];
  let seeds = [0u64, 1, 0x0123_4567_89ab_cdef, 0xd1b5_4a32_d192_ed03];

  for &len in &lengths {
    for &seed in &seeds {
      let data = gen_bytes(len, seed ^ len as u64);

      let oneshot = Crc64::checksum(&data);
      let reference = crc_reflected_bitwise_u64(
        0xC96C_5795_D787_0F42,
        64,
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
        &data,
      );
      assert_eq!(oneshot, reference, "crc64/xz reference mismatch at len={}", len);

      for &split in &[0usize, 1, len / 2, len.saturating_sub(1), len] {
        if split > len {
          continue;
        }
        let (a, b) = data.split_at(split);

        let mut h = Crc64::new();
        h.update(a);
        h.update(b);
        assert_eq!(
          h.finalize(),
          oneshot,
          "crc64/xz incremental mismatch at len={} split={}",
          len,
          split
        );

        let crc_a = Crc64::checksum(a);
        let mut r = Crc64::resume(crc_a);
        r.update(b);
        assert_eq!(
          r.finalize(),
          oneshot,
          "crc64/xz resume mismatch at len={} split={}",
          len,
          split
        );

        let crc_b = Crc64::checksum(b);
        let combined = Crc64::combine(crc_a, crc_b, b.len());
        assert_eq!(
          combined, oneshot,
          "crc64/xz combine mismatch at len={} split={}",
          len, split
        );
      }
    }
  }
}

#[test]
fn crc64_nvme_invariants() {
  let lengths = [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 255, 256, 1024, 2048];
  let seeds = [0u64, 1, 0x0123_4567_89ab_cdef, 0x5d58_39a7_3d87_1ceb];

  for &len in &lengths {
    for &seed in &seeds {
      let data = gen_bytes(len, seed ^ len as u64);

      let oneshot = Crc64Nvme::checksum(&data);
      let reference = crc_reflected_bitwise_u64(
        0x9A6C_9329_AC4B_C9B5,
        64,
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
        &data,
      );
      assert_eq!(oneshot, reference, "crc64/nvme reference mismatch at len={}", len);

      for &split in &[0usize, 1, len / 2, len.saturating_sub(1), len] {
        if split > len {
          continue;
        }
        let (a, b) = data.split_at(split);

        let mut h = Crc64Nvme::new();
        h.update(a);
        h.update(b);
        assert_eq!(
          h.finalize(),
          oneshot,
          "crc64/nvme incremental mismatch at len={} split={}",
          len,
          split
        );

        let crc_a = Crc64Nvme::checksum(a);
        let mut r = Crc64Nvme::resume(crc_a);
        r.update(b);
        assert_eq!(
          r.finalize(),
          oneshot,
          "crc64/nvme resume mismatch at len={} split={}",
          len,
          split
        );

        let crc_b = Crc64Nvme::checksum(b);
        let combined = Crc64Nvme::combine(crc_a, crc_b, b.len());
        assert_eq!(
          combined, oneshot,
          "crc64/nvme combine mismatch at len={} split={}",
          len, split
        );
      }
    }
  }
}
