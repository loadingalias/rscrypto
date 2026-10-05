//! Diagnostic controls for short HMAC timings. This binary is outside the release gate.

use dudect_bencher::{BenchRng, Class, CtRunner, ctbench_main_with_seeds, rand::RngExt};
use rscrypto::{HmacSha256, HmacSha256Tag};

fn hmac_sha256_control(runner: &mut CtRunner, rng: &mut BenchRng) {
  let mode = std::env::var("RSCRYPTO_CT_HMAC_CONTROL").expect("set RSCRYPTO_CT_HMAC_CONTROL explicitly");
  let invalid = match mode.as_str() {
    "valid-invalid" => [false, true],
    "invalid-valid" => [true, false],
    "valid-valid" => [false, false],
    "invalid-invalid" => [true, true],
    _ => {
      eprintln!("unknown HMAC control: {mode}");
      std::process::exit(2);
    }
  };
  let count: usize = std::env::var("RSCRYPTO_CT_DUDECT_SAMPLES")
    .expect("set RSCRYPTO_CT_DUDECT_SAMPLES explicitly")
    .parse()
    .expect("sample count must be an integer");
  assert!(count > 0, "sample count must be positive");
  eprintln!("diagnostic HMAC control: {mode}; {count} samples");

  // Match the release case's RNG consumption, key distribution, message, and
  // timed public call. Only the untimed expected-tag preparation varies.
  const MESSAGE: &[u8] = b"rscrypto constant-time dudect timing lane input";
  let mut inputs = Vec::with_capacity(count);
  for _ in 0..count {
    let class = if rng.random::<bool>() {
      Class::Left
    } else {
      Class::Right
    };
    let mut key = [0u8; 32];
    rng.fill(&mut key);
    let mut expected = HmacSha256::mac(&key, MESSAGE).to_bytes();
    let class_index = usize::from(matches!(class, Class::Right));
    if invalid[class_index] {
      expected[0] ^= 1;
    }
    inputs.push((class, key, HmacSha256Tag::from_bytes(expected)));
  }
  for (class, key, expected) in inputs {
    runner.run_one(class, || HmacSha256::verify_tag(&key, MESSAGE, &expected).is_ok());
  }
}

ctbench_main_with_seeds!((hmac_sha256_control, Some(0x686d61635f736861)));
