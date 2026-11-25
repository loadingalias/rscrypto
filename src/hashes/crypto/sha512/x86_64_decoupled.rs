macro_rules! sha512_decoupled_blocks {
  ($state:expr, $blocks:expr, prefetch = $prefetch:literal) => {
    let state = $state;
    let blocks = $blocks;
    debug_assert_eq!(blocks.len() % BLOCK_LEN, 0);
    let num_blocks = blocks.len() / BLOCK_LEN;
    if num_blocks == 0 {
      return;
    }

    // SAFETY: The caller establishes the features of this kernel. Its selected intrinsics use this
    // feature set. Every block access is within the input slice.
    unsafe {
      let mut ptr = blocks.as_ptr();
      let mut remaining = num_blocks;

      let bswap = _mm_loadu_si128(BSWAP64_128.as_ptr().cast());

      // Round-value buffer: holds one group of 16 W[t]+K[t] values for block 1.
      // All 16 entries are written (load or schedule) before rounds consume them,
      // so the initial value is irrelevant. Hoisted to avoid stack reallocation.
      let mut rv = [0u64; 16];

      // Block-2 round-value buffer: holds all 80 W[t]+K[t] for block 2.
      let mut t2_buf = [0u64; 80];

      // ── Dual-block loop ──────────────────────────────────────────────────
      while remaining >= 2 {
        // Prefetch next block pair into L1D to hide L2 latency at large sizes.
        if $prefetch && remaining >= 4 {
          _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(2)).cast());
          _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(2).strict_add(64)).cast());
          _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(3)).cast());
          _mm_prefetch::<_MM_HINT_T0>(ptr.add(BLOCK_LEN.strict_mul(3).strict_add(64)).cast());
        }

        let blk1 = ptr;
        let blk2 = ptr.add(BLOCK_LEN);

        let mut w: [__m256i; 8] = [
          load_two_blocks(blk1, blk2, 0, bswap),
          load_two_blocks(blk1, blk2, 16, bswap),
          load_two_blocks(blk1, blk2, 32, bswap),
          load_two_blocks(blk1, blk2, 48, bswap),
          load_two_blocks(blk1, blk2, 64, bswap),
          load_two_blocks(blk1, blk2, 80, bswap),
          load_two_blocks(blk1, blk2, 96, bswap),
          load_two_blocks(blk1, blk2, 112, bswap),
        ];

        // Extract initial W[0-15]+K[0-15] for both blocks.
        // K is added as a vector op (1 load + 1 broadcast + 1 add vs 4 scalar adds).
        for (i, &wv) in w.iter().enumerate() {
          let r = i.strict_mul(2);
          let k_pair: __m128i = _mm_loadu_si128(K.as_ptr().add(r).cast());
          let wk = _mm256_add_epi64(wv, _mm256_set_m128i(k_pair, k_pair));
          let (lo0, lo1) = extract_lo(wk);
          let (hi0, hi1) = extract_hi(wk);
          rv[r] = lo0;
          rv[r.strict_add(1)] = lo1;
          t2_buf[r] = hi0;
          t2_buf[r.strict_add(1)] = hi1;
        }

        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];

        macro_rules! round {
          ($wk:expr) => {{
            let t1 = h
              .wrapping_add(big_sigma1(e))
              .wrapping_add(ch(e, f, g))
              .wrapping_add($wk);
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

        // Decoupled loop: 4 outer × 8 inner = 64 rounds + 64 schedule words.
        // Rounds consume rv[] from the PREVIOUS group (load or prior outer).
        // Schedule computes the NEXT group via rotation (no cross-lane permute).
        for outer in 0..4usize {
          for j in 0..8usize {
            // Consume 2 rounds from current rv[] (previous group).
            let r = j.strict_mul(2);
            round!(rv[r]);
            round!(rv[r.strict_add(1)]);

            // Compute next schedule pair via rotation (independent of rounds).
            let kr = 8usize.strict_add(outer.strict_mul(8)).strict_add(j).strict_mul(2);
            let k_pair: __m128i = _mm_loadu_si128(K.as_ptr().add(kr).cast());
            let wk = schedule_rotate_256(&mut w, _mm256_set_m128i(k_pair, k_pair));
            let (lo0, lo1) = extract_lo(wk);
            let (hi0, hi1) = extract_hi(wk);

            // Store for next group (overwrite consumed rv[] slots).
            rv[r] = lo0;
            rv[r.strict_add(1)] = lo1;
            t2_buf[kr] = hi0;
            t2_buf[kr.strict_add(1)] = hi1;
          }
        }

        // Final 16 rounds (64-79) from last rv[] — pure scalar.
        for &wk in &rv {
          round!(wk);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);

        // Block 2: pure scalar from pre-computed buffer.
        a = state[0];
        b = state[1];
        c = state[2];
        d = state[3];
        e = state[4];
        f = state[5];
        g = state[6];
        h = state[7];

        for &wk in &t2_buf {
          round!(wk);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);

        ptr = ptr.add(BLOCK_LEN.strict_mul(2));
        remaining = remaining.strict_sub(2);
      }

      // ── Single-block (odd trailing) ──────────────────────────────────────
      // Uses rotation-based 128-bit schedule (same approach as 256-bit) and
      // vector K addition for consistency with the dual-block path.
      if remaining == 1 {
        let mut wv: [__m128i; 8] = [
          _mm_shuffle_epi8(_mm_loadu_si128(ptr.cast()), bswap),
          _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(16).cast()), bswap),
          _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(32).cast()), bswap),
          _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(48).cast()), bswap),
          _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(64).cast()), bswap),
          _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(80).cast()), bswap),
          _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(96).cast()), bswap),
          _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(112).cast()), bswap),
        ];

        // Extract initial W[0-15]+K[0-15] with vector K addition.
        for (i, &v) in wv.iter().enumerate() {
          let r = i.strict_mul(2);
          let k_pair: __m128i = _mm_loadu_si128(K.as_ptr().add(r).cast());
          let wk = _mm_add_epi64(v, k_pair);
          let (w0, w1) = extract_128(wk);
          rv[r] = w0;
          rv[r.strict_add(1)] = w1;
        }

        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];

        macro_rules! round1 {
          ($wk:expr) => {{
            let t1 = h
              .wrapping_add(big_sigma1(e))
              .wrapping_add(ch(e, f, g))
              .wrapping_add($wk);
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

        // Decoupled loop: 4 outer × 8 inner = 64 rounds + 64 schedule words.
        // Rotation-based schedule with vector K addition.
        for outer in 0..4usize {
          for j in 0..8usize {
            let r = j.strict_mul(2);
            round1!(rv[r]);
            round1!(rv[r.strict_add(1)]);

            let kr = 8usize.strict_add(outer.strict_mul(8)).strict_add(j).strict_mul(2);
            let k_pair: __m128i = _mm_loadu_si128(K.as_ptr().add(kr).cast());
            let wk = schedule_rotate_128(&mut wv, k_pair);
            let (w0, w1) = extract_128(wk);
            rv[r] = w0;
            rv[r.strict_add(1)] = w1;
          }
        }

        // Final 16 rounds (64-79).
        for &wk in &rv {
          round1!(wk);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
      }
    } // unsafe
  };
}
