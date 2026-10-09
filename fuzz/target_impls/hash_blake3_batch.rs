use rscrypto::{Blake3, Blake3DeriveContext, Blake3KeyedHash};

pub(super) fn run(data: &[u8]) {
  let Some(&control) = data.first() else { return };
  let count = usize::from(control) % 66;
  let mut key = [0u8; 32];
  for (index, byte) in key.iter_mut().enumerate() {
    *byte = data[index.strict_rem(data.len())];
  }
  let inputs: Vec<&[u8]> = (0..count)
    .map(|index| {
      let first = usize::from(data[index.strict_mul(3).strict_rem(data.len())]);
      let second = usize::from(data[index.strict_mul(3).strict_add(1).strict_rem(data.len())]);
      let third = usize::from(data[index.strict_mul(3).strict_add(2).strict_rem(data.len())]);
      let start = first
        .strict_mul(256)
        .strict_add(second)
        .strict_rem(data.len().strict_add(1));
      let length = match third % 8 {
        0 => 0,
        1 => 63,
        2 => 64,
        3 => 65,
        4 => 1023,
        5 => 1024,
        6 => 1025,
        _ => second.strict_mul(256).strict_add(third),
      }
      .min(data.len().strict_sub(start));
      &data[start..start.strict_add(length)]
    })
    .collect();
  let mut plain = vec![[0u8; 32]; count];
  let mut keyed = vec![Blake3KeyedHash::default(); count];
  let mut derived = vec![[0u8; 32]; count];
  const CONTEXT: &str = "rscrypto fuzz mixed batch context";
  Blake3::digest_batch(&inputs, &mut plain);
  Blake3::keyed_digest_batch(&key, &inputs, &mut keyed);
  Blake3DeriveContext::new(CONTEXT).derive_key_batch(&inputs, &mut derived);
  for (index, input) in inputs.iter().enumerate() {
    assert_eq!(&plain[index], blake3::hash(input).as_bytes());
    assert_eq!(keyed[index].as_bytes(), blake3::keyed_hash(&key, input).as_bytes());
    assert_eq!(derived[index], blake3::derive_key(CONTEXT, input));
  }
}
