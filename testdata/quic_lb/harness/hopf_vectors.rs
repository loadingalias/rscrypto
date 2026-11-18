//! Emit QUIC-LB vectors with hopf-quic 0.7.0 for the length pairs that quiche refuses.
//!
//! Output lines match testdata/quic_lb/quiche-vectors.txt: "rotation key server_id nonce
//! connection_id".
use hopf_quic::QuicLbConfig;

const DRAFT_KEY: [u8; 16] = [
  0x8f, 0x95, 0xf0, 0x92, 0x45, 0x76, 0x5f, 0x80, 0x25, 0x69, 0x34, 0xe5, 0x0c, 0x66, 0x20, 0x7f,
];

/// splitmix64 with a fixed seed, so the file is reproducible from this source alone.
struct Stream(u64);

impl Stream {
  fn byte(&mut self) -> u8 {
    self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 56) as u8
  }
}

fn hex(bytes: &[u8]) -> String {
  bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
  let mut stream = Stream(0x686f_7066_2d71_7569);
  let mut keys = [DRAFT_KEY; 3];
  for key in &mut keys[1..] {
    key.iter_mut().for_each(|byte| *byte = stream.byte());
  }
  let mut line = 0u8;
  for (server_id_len, nonce_len) in [(1usize, 17usize), (1, 18), (2, 17)] {
    for key in keys {
      let rotation = line % 7;
      line += 1;
      let server_id: Vec<u8> = (0..server_id_len).map(|_| stream.byte()).collect();
      let nonce: Vec<u8> = (0..nonce_len).map(|_| stream.byte()).collect();
      let config = QuicLbConfig::new(rotation, &server_id, nonce_len)
        .and_then(|config| config.with_key(key))
        .expect("draft lengths")
        .with_length_self_description(true);
      let id = config.encode(&nonce);
      assert_eq!(config.decode_server_id(id.as_slice()).as_deref(), Some(server_id.as_slice()));
      println!("{rotation} {} {} {} {}", hex(&key), hex(&server_id), hex(&nonce), hex(id.as_slice()));
    }
  }
}
