macro_rules! impl_riscv_aegis {
  ($features:literal) => {
    #[inline]
    fn keystream(s: &State) -> Block {
      let s2_and_s3 = and_block(&s[2], &s[3]);
      let mut z = xor_block(&s[1], &s[4]);
      z = xor_block(&z, &s[5]);
      xor_block(&z, &s2_and_s3)
    }

    #[target_feature(enable = $features)]
    /// # Safety
    ///
    /// The caller must establish the AES features of this backend.
    pub(super) unsafe fn encrypt_fused(
      key: &[u8; KEY_SIZE],
      nonce: &[u8; NONCE_SIZE],
      aad: &[u8],
      buffer: &mut [u8],
    ) -> [u8; TAG_SIZE] {
      // SAFETY: the public dispatcher only calls this entry point after runtime
      // feature detection confirms the AES features of this backend.
      unsafe {
        let (kh0, kh1) = split_halves(key);
        let (nh0, nh1) = split_halves(nonce);
        let k0_xor_n0 = xor_block(kh0, nh0);
        let k1_xor_n1 = xor_block(kh1, nh1);
        let mut s: State = [k0_xor_n0, k1_xor_n1, C1, C0, xor_block(kh0, &C0), xor_block(kh1, &C1)];

        for _ in 0..4 {
          update(&mut s, kh0);
          update(&mut s, kh1);
          update(&mut s, &k0_xor_n0);
          update(&mut s, &k1_xor_n1);
        }

        let mut offset = 0usize;
        while offset.strict_add(BLOCK_SIZE) <= aad.len() {
          let mut tmp = [0u8; BLOCK_SIZE];
          tmp.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
          update(&mut s, &tmp);
          offset = offset.strict_add(BLOCK_SIZE);
        }
        if offset < aad.len() {
          let mut pad = [0u8; BLOCK_SIZE];
          pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
          update(&mut s, &pad);
        }

        let msg_len = buffer.len();
        let len = buffer.len();
        offset = 0;
        while offset.strict_add(BLOCK_SIZE) <= len {
          let z = keystream(&s);
          let mut xi = [0u8; BLOCK_SIZE];
          xi.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
          update(&mut s, &xi);
          buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xor_block(&xi, &z));
          offset = offset.strict_add(BLOCK_SIZE);
        }
        if offset < len {
          let z = keystream(&s);
          let tail_len = len.strict_sub(offset);
          let mut pad = [0u8; BLOCK_SIZE];
          pad[..tail_len].copy_from_slice(&buffer[offset..]);
          update(&mut s, &pad);
          let ct = xor_block(&pad, &z);
          buffer[offset..].copy_from_slice(&ct[..tail_len]);
        }

        let ad_bits = (aad.len() as u64).strict_mul(8);
        let msg_bits = (msg_len as u64).strict_mul(8);
        let mut len_bytes = [0u8; BLOCK_SIZE];
        len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
        len_bytes[8..].copy_from_slice(&msg_bits.to_le_bytes());
        let t = xor_block(&s[3], &len_bytes);
        for _ in 0..7 {
          update(&mut s, &t);
        }

        xor_block(
          &xor_block(&xor_block(&s[0], &s[1]), &xor_block(&s[2], &s[3])),
          &xor_block(&s[4], &s[5]),
        )
      }
    }

    #[target_feature(enable = $features)]
    /// # Safety
    ///
    /// The caller must establish the AES features of this backend.
    pub(super) unsafe fn decrypt_fused(
      key: &[u8; KEY_SIZE],
      nonce: &[u8; NONCE_SIZE],
      aad: &[u8],
      buffer: &mut [u8],
    ) -> [u8; TAG_SIZE] {
      // SAFETY: the public dispatcher only calls this entry point after runtime
      // feature detection confirms the AES features of this backend.
      unsafe {
        let (kh0, kh1) = split_halves(key);
        let (nh0, nh1) = split_halves(nonce);
        let k0_xor_n0 = xor_block(kh0, nh0);
        let k1_xor_n1 = xor_block(kh1, nh1);
        let mut s: State = [k0_xor_n0, k1_xor_n1, C1, C0, xor_block(kh0, &C0), xor_block(kh1, &C1)];

        for _ in 0..4 {
          update(&mut s, kh0);
          update(&mut s, kh1);
          update(&mut s, &k0_xor_n0);
          update(&mut s, &k1_xor_n1);
        }

        let mut offset = 0usize;
        while offset.strict_add(BLOCK_SIZE) <= aad.len() {
          let mut tmp = [0u8; BLOCK_SIZE];
          tmp.copy_from_slice(&aad[offset..offset.strict_add(BLOCK_SIZE)]);
          update(&mut s, &tmp);
          offset = offset.strict_add(BLOCK_SIZE);
        }
        if offset < aad.len() {
          let mut pad = [0u8; BLOCK_SIZE];
          pad[..aad.len().strict_sub(offset)].copy_from_slice(&aad[offset..]);
          update(&mut s, &pad);
        }

        let ct_len = buffer.len();
        let len = buffer.len();
        offset = 0;
        while offset.strict_add(BLOCK_SIZE) <= len {
          let z = keystream(&s);
          let mut ci = [0u8; BLOCK_SIZE];
          ci.copy_from_slice(&buffer[offset..offset.strict_add(BLOCK_SIZE)]);
          let xi = xor_block(&ci, &z);
          update(&mut s, &xi);
          buffer[offset..offset.strict_add(BLOCK_SIZE)].copy_from_slice(&xi);
          offset = offset.strict_add(BLOCK_SIZE);
        }
        if offset < len {
          let z = keystream(&s);
          let tail_len = len.strict_sub(offset);
          let mut pad = [0u8; BLOCK_SIZE];
          pad[..tail_len].copy_from_slice(&buffer[offset..]);
          let mut pt_pad = [0u8; BLOCK_SIZE];
          for i in 0..tail_len {
            pt_pad[i] = pad[i] ^ z[i];
          }
          update(&mut s, &pt_pad);
          buffer[offset..].copy_from_slice(&pt_pad[..tail_len]);
        }

        let ad_bits = (aad.len() as u64).strict_mul(8);
        let ct_bits = (ct_len as u64).strict_mul(8);
        let mut len_bytes = [0u8; BLOCK_SIZE];
        len_bytes[..8].copy_from_slice(&ad_bits.to_le_bytes());
        len_bytes[8..].copy_from_slice(&ct_bits.to_le_bytes());
        let t = xor_block(&s[3], &len_bytes);
        for _ in 0..7 {
          update(&mut s, &t);
        }

        xor_block(
          &xor_block(&xor_block(&s[0], &s[1]), &xor_block(&s[2], &s[3])),
          &xor_block(&s[4], &s[5]),
        )
      }
    }
  };
}
