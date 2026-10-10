//! FIPS 205 Section 4.2 hash-function addresses (ADRS).
//!
//! The fields are kept typed and encoded on use: the SHAKE sets hash the
//! 32-byte form (Figure 2) and the SHA-2 sets the 22-byte compressed form
//! (Figure 18). A layer address and type fit one byte and a tree address
//! eight bytes for every supported set, so the encodings are exact.

/// `type` values (Section 4.2).
pub(crate) const WOTS_HASH: u8 = 0;
pub(crate) const WOTS_PK: u8 = 1;
pub(crate) const TREE: u8 = 2;
pub(crate) const FORS_TREE: u8 = 3;
pub(crate) const FORS_ROOTS: u8 = 4;
pub(crate) const WOTS_PRF: u8 = 5;
pub(crate) const FORS_PRF: u8 = 6;

/// One address. The three final words are the key pair address, the chain
/// address or tree height, and the hash address or tree index.
#[derive(Clone, Copy)]
pub(crate) struct Address {
  layer: u8,
  tree: u64,
  kind: u8,
  words: [u32; 3],
}

impl Address {
  /// `toByte(0, 32)`.
  pub(crate) const fn new() -> Self {
    Self {
      layer: 0,
      tree: 0,
      kind: 0,
      words: [0; 3],
    }
  }

  pub(crate) fn set_layer(&mut self, layer: u8) {
    self.layer = layer;
  }

  pub(crate) fn set_tree(&mut self, tree: u64) {
    self.tree = tree;
  }

  /// Set the type and clear the final 12 bytes.
  pub(crate) fn set_type_and_clear(&mut self, kind: u8) {
    self.kind = kind;
    self.words = [0; 3];
  }

  pub(crate) fn set_key_pair(&mut self, key_pair: u32) {
    self.words[0] = key_pair;
  }

  pub(crate) const fn key_pair(&self) -> u32 {
    self.words[0]
  }

  pub(crate) fn set_chain(&mut self, chain: u32) {
    self.words[1] = chain;
  }

  pub(crate) fn set_tree_height(&mut self, height: u32) {
    self.words[1] = height;
  }

  pub(crate) fn set_hash(&mut self, hash: u32) {
    self.words[2] = hash;
  }

  pub(crate) fn set_tree_index(&mut self, index: u32) {
    self.words[2] = index;
  }

  /// A copy of this address with type `kind` that keeps its key pair address,
  /// as FIPS 205 derives key-generation and public-key addresses.
  pub(crate) fn with_type_keeping_key_pair(&self, kind: u8) -> Self {
    let mut address = *self;
    address.set_type_and_clear(kind);
    address.set_key_pair(self.key_pair());
    address
  }

  /// The 32-byte ADRS. The tree address occupies 12 bytes; its high four
  /// are zero.
  pub(crate) fn full(&self) -> [u8; 32] {
    let mut out = [0; 32];
    out[3] = self.layer;
    out[8..16].copy_from_slice(&self.tree.to_be_bytes());
    out[19] = self.kind;
    for (chunk, word) in out[20..].as_chunks_mut::<4>().0.iter_mut().zip(self.words) {
      *chunk = word.to_be_bytes();
    }
    out
  }

  /// The 22-byte ADRSc: `ADRS[3] || ADRS[8:16] || ADRS[19] || ADRS[20:32]`.
  pub(crate) fn compressed(&self) -> [u8; 22] {
    let mut out = [0; 22];
    out[0] = self.layer;
    out[1..9].copy_from_slice(&self.tree.to_be_bytes());
    out[9] = self.kind;
    for (chunk, word) in out[10..].as_chunks_mut::<4>().0.iter_mut().zip(self.words) {
      *chunk = word.to_be_bytes();
    }
    out
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Every member function of FIPS 205 Table 1 writes its documented bytes,
  /// and the compressed form selects the Table 3 bytes from the full form.
  #[test]
  fn encodings_follow_tables_1_and_3() {
    let mut address = Address::new();
    address.set_layer(0x05);
    address.set_tree(0x0102_0304_0506_0708);
    address.set_type_and_clear(FORS_TREE);
    address.set_key_pair(0x1112_1314);
    address.set_tree_height(0x2122_2324);
    address.set_tree_index(0x3132_3334);
    let full = address.full();
    assert_eq!(
      full,
      [
        0, 0, 0, 5, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 3, 0x11, 0x12, 0x13, 0x14, 0x21, 0x22, 0x23, 0x24,
        0x31, 0x32, 0x33, 0x34
      ]
    );
    let mut expected = [0; 22];
    expected[0] = full[3];
    expected[1..9].copy_from_slice(&full[8..16]);
    expected[9] = full[19];
    expected[10..].copy_from_slice(&full[20..]);
    assert_eq!(address.compressed(), expected);

    address.set_type_and_clear(WOTS_PK);
    assert_eq!(address.full()[16..], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(address.full()[..16], full[..16]);
  }
}
