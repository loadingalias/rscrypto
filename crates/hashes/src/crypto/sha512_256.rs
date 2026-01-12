#![allow(clippy::indexing_slicing)] // Fixed-size arrays + compression schedule

use traits::Digest;

use crate::util::rotr64;

const BLOCK_LEN: usize = 128;

// SHA-512/256 initial hash value (FIPS 180-4).
const H0: [u64; 8] = [
  0x2231_2194_fc2b_f72c,
  0x9f55_5fa3_c84c_64c2,
  0x2393_b86b_6f53_b151,
  0x9638_7719_5940_eabd,
  0x9628_3ee2_a88e_ffe3,
  0xbe5e_1e25_5386_3992,
  0x2b01_99fc_2c85_b8aa,
  0x0eb7_2ddc_81c5_2ca2,
];

// SHA-512 K constants (shared).
const K: [u64; 80] = [
  0x428a_2f98_d728_ae22,
  0x7137_4491_23ef_65cd,
  0xb5c0_fbcf_ec4d_3b2f,
  0xe9b5_dba5_8189_dbbc,
  0x3956_c25b_f348_b538,
  0x59f1_11f1_b605_d019,
  0x923f_82a4_af19_4f9b,
  0xab1c_5ed5_da6d_8118,
  0xd807_aa98_a303_0242,
  0x1283_5b01_4570_6fbe,
  0x2431_85be_4ee4_b28c,
  0x550c_7dc3_d5ff_b4e2,
  0x72be_5d74_f27b_896f,
  0x80de_b1fe_3b16_96b1,
  0x9bdc_06a7_25c7_1235,
  0xc19b_f174_cf69_2694,
  0xe49b_69c1_9ef1_4ad2,
  0xefbe_4786_384f_25e3,
  0x0fc1_9dc6_8b8c_d5b5,
  0x240c_a1cc_77ac_9c65,
  0x2de9_2c6f_592b_0275,
  0x4a74_84aa_6ea6_e483,
  0x5cb0_a9dc_bd41_fbd4,
  0x76f9_88da_8311_53b5,
  0x983e_5152_ee66_dfab,
  0xa831_c66d_2db4_3210,
  0xb003_27c8_98fb_213f,
  0xbf59_7fc7_beef_0ee4,
  0xc6e0_0bf3_3da8_8fc2,
  0xd5a7_9147_930a_a725,
  0x06ca_6351_e003_826f,
  0x1429_2967_0a0e_6e70,
  0x27b7_0a85_46d2_2ffc,
  0x2e1b_2138_5c26_c926,
  0x4d2c_6dfc_5ac4_2aed,
  0x5338_0d13_9d95_b3df,
  0x650a_7354_8baf_63de,
  0x766a_0abb_3c77_b2a8,
  0x81c2_c92e_47ed_aee6,
  0x9272_2c85_1482_353b,
  0xa2bf_e8a1_4cf1_0364,
  0xa81a_664b_bc42_3001,
  0xc24b_8b70_d0f8_9791,
  0xc76c_51a3_0654_be30,
  0xd192_e819_d6ef_5218,
  0xd699_0624_5565_a910,
  0xf40e_3585_5771_202a,
  0x106a_a070_32bb_d1b8,
  0x19a4_c116_b8d2_d0c8,
  0x1e37_6c08_5141_ab53,
  0x2748_774c_df8e_eb99,
  0x34b0_bcb5_e19b_48a8,
  0x391c_0cb3_c5c9_5a63,
  0x4ed8_aa4a_e341_8acb,
  0x5b9c_ca4f_7763_e373,
  0x682e_6ff3_d6b2_b8a3,
  0x748f_82ee_5def_b2fc,
  0x78a5_636f_4317_2f60,
  0x84c8_7814_a1f0_ab72,
  0x8cc7_0208_1a64_39ec,
  0x90be_fffa_2363_1e28,
  0xa450_6ceb_de82_bde9,
  0xbef9_a3f7_b2c6_7915,
  0xc671_78f2_e372_532b,
  0xca27_3ece_ea26_619c,
  0xd186_b8c7_21c0_c207,
  0xeada_7dd6_cde0_eb1e,
  0xf57d_4f7f_ee6e_d178,
  0x06f0_67aa_7217_6fba,
  0x0a63_7dc5_a2c8_98a6,
  0x113f_9804_bef9_0dae,
  0x1b71_0b35_131c_471b,
  0x28db_77f5_2304_7d84,
  0x32ca_ab7b_40c7_2493,
  0x3c9e_be0a_15c9_bebc,
  0x431d_67c4_9c10_0d4c,
  0x4cc5_d4be_cb3e_42b6,
  0x597f_299c_fc65_7e2a,
  0x5fcb_6fab_3ad6_faec,
  0x6c44_198c_4a47_5817,
];

#[inline(always)]
fn ch(x: u64, y: u64, z: u64) -> u64 {
  (x & y) ^ (!x & z)
}

#[inline(always)]
fn maj(x: u64, y: u64, z: u64) -> u64 {
  (x & y) ^ (x & z) ^ (y & z)
}

#[inline(always)]
fn big_sigma0(x: u64) -> u64 {
  rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39)
}

#[inline(always)]
fn big_sigma1(x: u64) -> u64 {
  rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41)
}

#[inline(always)]
fn small_sigma0(x: u64) -> u64 {
  rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7)
}

#[inline(always)]
fn small_sigma1(x: u64) -> u64 {
  rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6)
}

#[derive(Clone)]
pub struct Sha512_256 {
  state: [u64; 8],
  block: [u8; BLOCK_LEN],
  block_len: usize,
  bytes_hashed: u128,
}

impl Default for Sha512_256 {
  #[inline]
  fn default() -> Self {
    Self {
      state: H0,
      block: [0u8; BLOCK_LEN],
      block_len: 0,
      bytes_hashed: 0,
    }
  }
}

impl Sha512_256 {
  #[inline(always)]
  fn compress_block(state: &mut [u64; 8], block: &[u8; BLOCK_LEN]) {
    let mut w = [0u64; 16];
    let (chunks, _) = block.as_chunks::<8>();
    for (i, c) in chunks.iter().enumerate() {
      w[i] = u64::from_be_bytes(*c);
    }
    let [
      mut w0,
      mut w1,
      mut w2,
      mut w3,
      mut w4,
      mut w5,
      mut w6,
      mut w7,
      mut w8,
      mut w9,
      mut w10,
      mut w11,
      mut w12,
      mut w13,
      mut w14,
      mut w15,
    ] = w;

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    macro_rules! round {
      ($k:expr, $wi:expr) => {{
        let t1 = h
          .wrapping_add(big_sigma1(e))
          .wrapping_add(ch(e, f, g))
          .wrapping_add($k)
          .wrapping_add($wi);
        let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
      }};
    }

    macro_rules! sched {
      ($w_im2:expr, $w_im7:expr, $w_im15:expr, $w_im16:expr) => {{
        small_sigma1($w_im2)
          .wrapping_add($w_im7)
          .wrapping_add(small_sigma0($w_im15))
          .wrapping_add($w_im16)
      }};
    }

    round!(K[0], w0);
    round!(K[1], w1);
    round!(K[2], w2);
    round!(K[3], w3);
    round!(K[4], w4);
    round!(K[5], w5);
    round!(K[6], w6);
    round!(K[7], w7);
    round!(K[8], w8);
    round!(K[9], w9);
    round!(K[10], w10);
    round!(K[11], w11);
    round!(K[12], w12);
    round!(K[13], w13);
    round!(K[14], w14);
    round!(K[15], w15);

    w0 = sched!(w14, w9, w1, w0);
    round!(K[16], w0);
    w1 = sched!(w15, w10, w2, w1);
    round!(K[17], w1);
    w2 = sched!(w0, w11, w3, w2);
    round!(K[18], w2);
    w3 = sched!(w1, w12, w4, w3);
    round!(K[19], w3);
    w4 = sched!(w2, w13, w5, w4);
    round!(K[20], w4);
    w5 = sched!(w3, w14, w6, w5);
    round!(K[21], w5);
    w6 = sched!(w4, w15, w7, w6);
    round!(K[22], w6);
    w7 = sched!(w5, w0, w8, w7);
    round!(K[23], w7);
    w8 = sched!(w6, w1, w9, w8);
    round!(K[24], w8);
    w9 = sched!(w7, w2, w10, w9);
    round!(K[25], w9);
    w10 = sched!(w8, w3, w11, w10);
    round!(K[26], w10);
    w11 = sched!(w9, w4, w12, w11);
    round!(K[27], w11);
    w12 = sched!(w10, w5, w13, w12);
    round!(K[28], w12);
    w13 = sched!(w11, w6, w14, w13);
    round!(K[29], w13);
    w14 = sched!(w12, w7, w15, w14);
    round!(K[30], w14);
    w15 = sched!(w13, w8, w0, w15);
    round!(K[31], w15);
    w0 = sched!(w14, w9, w1, w0);
    round!(K[32], w0);
    w1 = sched!(w15, w10, w2, w1);
    round!(K[33], w1);
    w2 = sched!(w0, w11, w3, w2);
    round!(K[34], w2);
    w3 = sched!(w1, w12, w4, w3);
    round!(K[35], w3);
    w4 = sched!(w2, w13, w5, w4);
    round!(K[36], w4);
    w5 = sched!(w3, w14, w6, w5);
    round!(K[37], w5);
    w6 = sched!(w4, w15, w7, w6);
    round!(K[38], w6);
    w7 = sched!(w5, w0, w8, w7);
    round!(K[39], w7);
    w8 = sched!(w6, w1, w9, w8);
    round!(K[40], w8);
    w9 = sched!(w7, w2, w10, w9);
    round!(K[41], w9);
    w10 = sched!(w8, w3, w11, w10);
    round!(K[42], w10);
    w11 = sched!(w9, w4, w12, w11);
    round!(K[43], w11);
    w12 = sched!(w10, w5, w13, w12);
    round!(K[44], w12);
    w13 = sched!(w11, w6, w14, w13);
    round!(K[45], w13);
    w14 = sched!(w12, w7, w15, w14);
    round!(K[46], w14);
    w15 = sched!(w13, w8, w0, w15);
    round!(K[47], w15);
    w0 = sched!(w14, w9, w1, w0);
    round!(K[48], w0);
    w1 = sched!(w15, w10, w2, w1);
    round!(K[49], w1);
    w2 = sched!(w0, w11, w3, w2);
    round!(K[50], w2);
    w3 = sched!(w1, w12, w4, w3);
    round!(K[51], w3);
    w4 = sched!(w2, w13, w5, w4);
    round!(K[52], w4);
    w5 = sched!(w3, w14, w6, w5);
    round!(K[53], w5);
    w6 = sched!(w4, w15, w7, w6);
    round!(K[54], w6);
    w7 = sched!(w5, w0, w8, w7);
    round!(K[55], w7);
    w8 = sched!(w6, w1, w9, w8);
    round!(K[56], w8);
    w9 = sched!(w7, w2, w10, w9);
    round!(K[57], w9);
    w10 = sched!(w8, w3, w11, w10);
    round!(K[58], w10);
    w11 = sched!(w9, w4, w12, w11);
    round!(K[59], w11);
    w12 = sched!(w10, w5, w13, w12);
    round!(K[60], w12);
    w13 = sched!(w11, w6, w14, w13);
    round!(K[61], w13);
    w14 = sched!(w12, w7, w15, w14);
    round!(K[62], w14);
    w15 = sched!(w13, w8, w0, w15);
    round!(K[63], w15);
    w0 = sched!(w14, w9, w1, w0);
    round!(K[64], w0);
    w1 = sched!(w15, w10, w2, w1);
    round!(K[65], w1);
    w2 = sched!(w0, w11, w3, w2);
    round!(K[66], w2);
    w3 = sched!(w1, w12, w4, w3);
    round!(K[67], w3);
    w4 = sched!(w2, w13, w5, w4);
    round!(K[68], w4);
    w5 = sched!(w3, w14, w6, w5);
    round!(K[69], w5);
    w6 = sched!(w4, w15, w7, w6);
    round!(K[70], w6);
    w7 = sched!(w5, w0, w8, w7);
    round!(K[71], w7);
    w8 = sched!(w6, w1, w9, w8);
    round!(K[72], w8);
    w9 = sched!(w7, w2, w10, w9);
    round!(K[73], w9);
    w10 = sched!(w8, w3, w11, w10);
    round!(K[74], w10);
    w11 = sched!(w9, w4, w12, w11);
    round!(K[75], w11);
    w12 = sched!(w10, w5, w13, w12);
    round!(K[76], w12);
    w13 = sched!(w11, w6, w14, w13);
    round!(K[77], w13);
    w14 = sched!(w12, w7, w15, w14);
    round!(K[78], w14);
    w15 = sched!(w13, w8, w0, w15);
    round!(K[79], w15);

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
  }

  #[inline]
  fn update_block(state: &mut [u64; 8], bytes_hashed: &mut u128, block: &[u8; BLOCK_LEN]) {
    Self::compress_block(state, block);
    *bytes_hashed = (*bytes_hashed).wrapping_add(BLOCK_LEN as u128);
  }

  #[inline]
  fn finalize_inner(&self) -> [u8; 32] {
    let mut state = self.state;
    let mut block = self.block;
    let mut block_len = self.block_len;
    let total_len = self.bytes_hashed.wrapping_add(block_len as u128);

    block[block_len] = 0x80;
    block_len += 1;

    if block_len > 112 {
      block[block_len..].fill(0);
      Self::compress_block(&mut state, &block);
      block = [0u8; BLOCK_LEN];
      block_len = 0;
    }

    block[block_len..112].fill(0);

    let bit_len = total_len << 3;
    block[112..128].copy_from_slice(&bit_len.to_be_bytes());
    Self::compress_block(&mut state, &block);

    let mut out = [0u8; 32];
    for (i, word) in state.iter().copied().enumerate().take(4) {
      let offset = i * 8;
      out[offset..offset + 8].copy_from_slice(&word.to_be_bytes());
    }
    out
  }
}

impl Digest for Sha512_256 {
  const OUTPUT_SIZE: usize = 32;
  type Output = [u8; 32];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  fn update(&mut self, mut data: &[u8]) {
    if data.is_empty() {
      return;
    }

    if self.block_len != 0 {
      let take = core::cmp::min(BLOCK_LEN - self.block_len, data.len());
      self.block[self.block_len..self.block_len + take].copy_from_slice(&data[..take]);
      self.block_len += take;
      data = &data[take..];

      if self.block_len == BLOCK_LEN {
        Self::update_block(&mut self.state, &mut self.bytes_hashed, &self.block);
        self.block_len = 0;
      }
    }

    let (blocks, rest) = data.as_chunks::<BLOCK_LEN>();
    if !blocks.is_empty() {
      for block in blocks {
        Self::compress_block(&mut self.state, block);
      }
      self.bytes_hashed = self.bytes_hashed.wrapping_add((blocks.len() * BLOCK_LEN) as u128);
    }
    data = rest;

    if !data.is_empty() {
      self.block[..data.len()].copy_from_slice(data);
      self.block_len = data.len();
    }
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    self.finalize_inner()
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}
