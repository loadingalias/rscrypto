    .arch armv8.4-a+crypto+sha3
    .text

    .macro LOAD_COUNTER_BASE
        ldr d25, [x1]
        ldr w9, [x1, #8]
        fmov s31, w9
        mov v25.s[2], v31.s[0]
        movi v31.2d, #0
        mov v25.s[3], v31.s[0]
        movi v29.4s, #0
        movi v30.4s, #4
        mov w9, #1
        fmov s31, w9
        mov v29.s[1], v31.s[0]
        mov w9, #2
        fmov s31, w9
        mov v29.s[2], v31.s[0]
        mov w9, #3
        fmov s31, w9
        mov v29.s[3], v31.s[0]
        mov w9, #5
        fmov s31, w9
        mov v30.s[1], v31.s[0]
        mov w9, #6
        fmov s31, w9
        mov v30.s[2], v31.s[0]
        mov w9, #7
        fmov s31, w9
        mov v30.s[3], v31.s[0]
    .endm

    .macro BUILD_COUNTER8_SCALAR
        orr v0.16b, v25.16b, v25.16b
        orr v1.16b, v25.16b, v25.16b
        orr v2.16b, v25.16b, v25.16b
        orr v3.16b, v25.16b, v25.16b
        orr v4.16b, v25.16b, v25.16b
        orr v5.16b, v25.16b, v25.16b
        orr v6.16b, v25.16b, v25.16b
        orr v7.16b, v25.16b, v25.16b

        rev w9, w6
        fmov s31, w9
        mov v0.s[3], v31.s[0]
        add w9, w6, #1
        rev w9, w9
        fmov s31, w9
        mov v1.s[3], v31.s[0]
        add w9, w6, #2
        rev w9, w9
        fmov s31, w9
        mov v2.s[3], v31.s[0]
        add w9, w6, #3
        rev w9, w9
        fmov s31, w9
        mov v3.s[3], v31.s[0]
        add w9, w6, #4
        rev w9, w9
        fmov s31, w9
        mov v4.s[3], v31.s[0]
        add w9, w6, #5
        rev w9, w9
        fmov s31, w9
        mov v5.s[3], v31.s[0]
        add w9, w6, #6
        rev w9, w9
        fmov s31, w9
        mov v6.s[3], v31.s[0]
        add w9, w6, #7
        rev w9, w9
        fmov s31, w9
        mov v7.s[3], v31.s[0]
    .endm

    .macro BUILD_COUNTER8_VECTOR
        orr v0.16b, v25.16b, v25.16b
        orr v1.16b, v25.16b, v25.16b
        orr v2.16b, v25.16b, v25.16b
        orr v3.16b, v25.16b, v25.16b
        orr v4.16b, v25.16b, v25.16b
        orr v5.16b, v25.16b, v25.16b
        orr v6.16b, v25.16b, v25.16b
        orr v7.16b, v25.16b, v25.16b

        dup v31.4s, w6
        add v24.4s, v31.4s, v29.4s
        add v31.4s, v31.4s, v30.4s
        rev32 v24.16b, v24.16b
        rev32 v31.16b, v31.16b
        mov v0.s[3], v24.s[0]
        mov v1.s[3], v24.s[1]
        mov v2.s[3], v24.s[2]
        mov v3.s[3], v24.s[3]
        mov v4.s[3], v31.s[0]
        mov v5.s[3], v31.s[1]
        mov v6.s[3], v31.s[2]
        mov v7.s[3], v31.s[3]
    .endm

    .macro BUILD_COUNTER8 rounds
        BUILD_COUNTER8_VECTOR
    .endm

    .macro PREP_NEXT_COUNTER8
        add w11, w6, #8
        dup v31.4s, w11
        add v24.4s, v31.4s, v29.4s
        add v31.4s, v31.4s, v30.4s
        rev32 v24.16b, v24.16b
        rev32 v31.16b, v31.16b
    .endm

    .macro AES_ROUND8 off
        ldr q24, [x0, #\off]
        aese v0.16b, v24.16b
        aesmc v0.16b, v0.16b
        aese v1.16b, v24.16b
        aesmc v1.16b, v1.16b
        aese v2.16b, v24.16b
        aesmc v2.16b, v2.16b
        aese v3.16b, v24.16b
        aesmc v3.16b, v3.16b
        aese v4.16b, v24.16b
        aesmc v4.16b, v4.16b
        aese v5.16b, v24.16b
        aesmc v5.16b, v5.16b
        aese v6.16b, v24.16b
        aesmc v6.16b, v6.16b
        aese v7.16b, v24.16b
        aesmc v7.16b, v7.16b
    .endm

    .macro AES_FINAL8 aese_off, xor_off
        ldr q24, [x0, #\aese_off]
        ldr q23, [x0, #\xor_off]
        aese v0.16b, v24.16b
        eor v0.16b, v0.16b, v23.16b
        aese v1.16b, v24.16b
        eor v1.16b, v1.16b, v23.16b
        aese v2.16b, v24.16b
        eor v2.16b, v2.16b, v23.16b
        aese v3.16b, v24.16b
        eor v3.16b, v3.16b, v23.16b
        aese v4.16b, v24.16b
        eor v4.16b, v4.16b, v23.16b
        aese v5.16b, v24.16b
        eor v5.16b, v5.16b, v23.16b
        aese v6.16b, v24.16b
        eor v6.16b, v6.16b, v23.16b
        aese v7.16b, v24.16b
        eor v7.16b, v7.16b, v23.16b
    .endm

    .macro AES_FINAL8_RAW aese_off, xor_off
        ldr q24, [x0, #\aese_off]
        ldr q23, [x0, #\xor_off]
        aese v0.16b, v24.16b
        aese v1.16b, v24.16b
        aese v2.16b, v24.16b
        aese v3.16b, v24.16b
        aese v4.16b, v24.16b
        aese v5.16b, v24.16b
        aese v6.16b, v24.16b
        aese v7.16b, v24.16b
    .endm

    .macro AES_ENCRYPT8 rounds
        AES_ROUND8 0
        AES_ROUND8 16
        AES_ROUND8 32
        AES_ROUND8 48
        AES_ROUND8 64
        AES_ROUND8 80
        AES_ROUND8 96
        AES_ROUND8 112
        AES_ROUND8 128
        .if \rounds == 14
            AES_ROUND8 144
            AES_ROUND8 160
            AES_ROUND8 176
            AES_ROUND8 192
            AES_FINAL8 208, 224
        .else
            AES_FINAL8 144, 160
        .endif
    .endm

    .macro AES_ENCRYPT8_RAW rounds
        AES_ROUND8 0
        AES_ROUND8 16
        AES_ROUND8 32
        AES_ROUND8 48
        AES_ROUND8 64
        AES_ROUND8 80
        AES_ROUND8 96
        AES_ROUND8 112
        AES_ROUND8 128
        .if \rounds == 14
            AES_ROUND8 144
            AES_ROUND8 160
            AES_ROUND8 176
            AES_ROUND8 192
            AES_FINAL8_RAW 208, 224
        .else
            AES_FINAL8_RAW 144, 160
        .endif
    .endm

    .macro OPEN_GHASH_ONLY ll_b, hh_b, mm_b, data_off, h_off, with_acc
        ldr q24, [x2, #\data_off]
        GHASH_FOLD_INTO \ll_b, \hh_b, \mm_b, v24.16b, \h_off, \with_acc
    .endm

    .macro LOAD_OPEN_CT8
        ldp q8, q9, [x2]
        ldp q10, q11, [x2, #32]
        ldp q12, q13, [x2, #64]
        ldp q14, q15, [x2, #96]
    .endm

    .macro AES_ENCRYPT8_GHASH_OPEN rounds
        movi v16.2d, #0
        movi v17.2d, #0
        movi v18.2d, #0
        movi v26.2d, #0
        movi v27.2d, #0
        movi v28.2d, #0
        AES_ROUND8 0
        OPEN_GHASH_ONLY v16.16b, v17.16b, v18.16b, 0, 0, 1
        AES_ROUND8 16
        OPEN_GHASH_ONLY v26.16b, v27.16b, v28.16b, 16, 16, 0
        AES_ROUND8 32
        OPEN_GHASH_ONLY v16.16b, v17.16b, v18.16b, 32, 32, 0
        AES_ROUND8 48
        OPEN_GHASH_ONLY v26.16b, v27.16b, v28.16b, 48, 48, 0
        AES_ROUND8 64
        OPEN_GHASH_ONLY v16.16b, v17.16b, v18.16b, 64, 64, 0
        AES_ROUND8 80
        OPEN_GHASH_ONLY v26.16b, v27.16b, v28.16b, 80, 80, 0
        AES_ROUND8 96
        OPEN_GHASH_ONLY v16.16b, v17.16b, v18.16b, 96, 96, 0
        AES_ROUND8 112
        OPEN_GHASH_ONLY v26.16b, v27.16b, v28.16b, 112, 112, 0
        AES_ROUND8 128
        .if \rounds == 14
            AES_ROUND8 144
            AES_ROUND8 160
            GHASH_COMBINE_REDUCE
            AES_ROUND8 176
            AES_ROUND8 192
            AES_FINAL8 208, 224
        .else
            GHASH_COMBINE_REDUCE
            AES_FINAL8 144, 160
        .endif
    .endm

    .macro AES_ENCRYPT8_GHASH_OPEN_REG rounds
        movi v16.2d, #0
        movi v17.2d, #0
        movi v18.2d, #0
        movi v26.2d, #0
        movi v27.2d, #0
        movi v28.2d, #0
        AES_ROUND8 0
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, v8.16b, 0, 1
        AES_ROUND8 16
        GHASH_FOLD_INTO v26.16b, v27.16b, v28.16b, v9.16b, 16, 0
        AES_ROUND8 32
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, v10.16b, 32, 0
        AES_ROUND8 48
        GHASH_FOLD_INTO v26.16b, v27.16b, v28.16b, v11.16b, 48, 0
        AES_ROUND8 64
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, v12.16b, 64, 0
        AES_ROUND8 80
        GHASH_FOLD_INTO v26.16b, v27.16b, v28.16b, v13.16b, 80, 0
        AES_ROUND8 96
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, v14.16b, 96, 0
        AES_ROUND8 112
        GHASH_FOLD_INTO v26.16b, v27.16b, v28.16b, v15.16b, 112, 0
        AES_ROUND8 128
        .if \rounds == 14
            AES_ROUND8 144
            AES_ROUND8 160
            GHASH_COMBINE_REDUCE
            AES_ROUND8 176
            AES_ROUND8 192
            AES_FINAL8 208, 224
        .else
            GHASH_COMBINE_REDUCE
            AES_FINAL8 144, 160
        .endif
    .endm

    .macro AES_ENCRYPT8_GHASH_PREV rounds
        movi v16.2d, #0
        movi v17.2d, #0
        movi v18.2d, #0
        movi v26.2d, #0
        movi v27.2d, #0
        movi v28.2d, #0
        AES_ROUND8 0
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, v8.16b, 0, 1
        AES_ROUND8 16
        GHASH_FOLD_INTO v26.16b, v27.16b, v28.16b, v9.16b, 16, 0
        AES_ROUND8 32
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, v10.16b, 32, 0
        AES_ROUND8 48
        GHASH_FOLD_INTO v26.16b, v27.16b, v28.16b, v11.16b, 48, 0
        AES_ROUND8 64
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, v12.16b, 64, 0
        AES_ROUND8 80
        GHASH_FOLD_INTO v26.16b, v27.16b, v28.16b, v13.16b, 80, 0
        AES_ROUND8 96
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, v14.16b, 96, 0
        AES_ROUND8 112
        GHASH_FOLD_INTO v26.16b, v27.16b, v28.16b, v15.16b, 112, 0
        AES_ROUND8 128
        .if \rounds == 14
            AES_ROUND8 144
            AES_ROUND8 160
            GHASH_COMBINE_REDUCE
            AES_ROUND8 176
            AES_ROUND8 192
            AES_FINAL8 208, 224
        .else
            GHASH_COMBINE_REDUCE
            AES_FINAL8 144, 160
        .endif
    .endm

    .macro GHASH_ZERO_PRODUCTS
        movi v16.2d, #0
        movi v17.2d, #0
        movi v18.2d, #0
        movi v26.2d, #0
        movi v27.2d, #0
        movi v28.2d, #0
    .endm

    .macro AES_ENCRYPT8_GHASH_OPEN_REG_ACCUM_EOR3 rounds, h_base, h_pair_base, with_acc, reduce, use_gpr_reduce
        GHASH_LOAD_PREMID_PAIR \h_pair_base + 0
        AES_ROUND8 0
        GHASH_FOLD2_INTO_PREMID_LOADED v16, v17, v18, v8, v9, \with_acc
        GHASH_LOAD_PREMID_PAIR \h_pair_base + 48
        AES_ROUND8 16
        GHASH_FOLD2_INTO_PREMID_LOADED v26, v27, v28, v10, v11, 0
        GHASH_LOAD_PREMID_PAIR \h_pair_base + 96
        AES_ROUND8 32
        GHASH_FOLD2_INTO_PREMID_LOADED v16, v17, v18, v12, v13, 0
        GHASH_LOAD_PREMID_PAIR \h_pair_base + 144
        AES_ROUND8 48
        GHASH_FOLD2_INTO_PREMID_LOADED v26, v27, v28, v14, v15, 0
        AES_ROUND8 64
        AES_ROUND8 80
        AES_ROUND8 96
        AES_ROUND8 112
        .if \reduce
            GHASH_COMBINE_REDUCE_EOR3 \use_gpr_reduce
        .endif
        AES_ROUND8 128
        .if \rounds == 14
            AES_ROUND8 144
            AES_ROUND8 160
            AES_ROUND8 176
            AES_ROUND8 192
            AES_FINAL8_RAW 208, 224
        .else
            AES_FINAL8_RAW 144, 160
        .endif
    .endm

    .macro AES_ENCRYPT8_GHASH_PREV_ACCUM_EOR3 rounds, h_base, h_pair_base, with_acc, reduce, use_gpr_reduce
        GHASH_LOAD_PREMID_PAIR \h_pair_base + 0
        AES_ROUND8 0
        GHASH_FOLD2_INTO_PREMID_LOADED v16, v17, v18, v8, v9, \with_acc
        GHASH_LOAD_PREMID_PAIR \h_pair_base + 48
        AES_ROUND8 16
        GHASH_FOLD2_INTO_PREMID_LOADED v26, v27, v28, v10, v11, 0
        GHASH_LOAD_PREMID_PAIR \h_pair_base + 96
        AES_ROUND8 32
        GHASH_FOLD2_INTO_PREMID_LOADED v16, v17, v18, v12, v13, 0
        GHASH_LOAD_PREMID_PAIR \h_pair_base + 144
        AES_ROUND8 48
        GHASH_FOLD2_INTO_PREMID_LOADED v26, v27, v28, v14, v15, 0
        AES_ROUND8 64
        AES_ROUND8 80
        AES_ROUND8 96
        AES_ROUND8 112
        .if \reduce
            GHASH_COMBINE_REDUCE_EOR3 \use_gpr_reduce
        .endif
        AES_ROUND8 128
        .if \rounds == 14
            AES_ROUND8 144
            AES_ROUND8 160
            AES_ROUND8 176
            AES_ROUND8 192
            AES_FINAL8_RAW 208, 224
        .else
            AES_FINAL8_RAW 144, 160
        .endif
    .endm

    .macro AES_ENCRYPT8_GHASH_REG_INIT_EOR3 rounds, h_pair_base, with_acc
        GHASH_LOAD_PREMID_PAIR \h_pair_base + 0
        AES_ROUND8 0
        GHASH_FOLD2_SET_PREMID_LOADED v16, v17, v18, v8, v9, \with_acc
        GHASH_LOAD_PREMID_PAIR \h_pair_base + 48
        AES_ROUND8 16
        GHASH_FOLD2_SET_PREMID_LOADED v26, v27, v28, v10, v11, 0
        GHASH_LOAD_PREMID_PAIR \h_pair_base + 96
        AES_ROUND8 32
        GHASH_FOLD2_INTO_PREMID_LOADED v16, v17, v18, v12, v13, 0
        GHASH_LOAD_PREMID_PAIR \h_pair_base + 144
        AES_ROUND8 48
        GHASH_FOLD2_INTO_PREMID_LOADED v26, v27, v28, v14, v15, 0
        AES_ROUND8 64
        AES_ROUND8 80
        AES_ROUND8 96
        AES_ROUND8 112
        AES_ROUND8 128
        .if \rounds == 14
            AES_ROUND8 144
            AES_ROUND8 160
            AES_ROUND8 176
            AES_ROUND8 192
            AES_FINAL8_RAW 208, 224
        .else
            AES_FINAL8_RAW 144, 160
        .endif
    .endm

    .macro GHASH_FOLD_INTO ll_b, hh_b, mm_b, block_b, h_off, with_acc
        rev64 v21.16b, \block_b
        ext v21.16b, v21.16b, v21.16b, #8
        .if \with_acc
            eor v21.16b, v21.16b, v19.16b
        .endif
        ldr q20, [x4, #(\h_off)]
        pmull v22.1q, v21.1d, v20.1d
        eor \ll_b, \ll_b, v22.16b
        pmull2 v22.1q, v21.2d, v20.2d
        eor \hh_b, \hh_b, v22.16b
        ext v22.16b, v21.16b, v21.16b, #8
        eor v22.16b, v22.16b, v21.16b
        ext v23.16b, v20.16b, v20.16b, #8
        eor v23.16b, v23.16b, v20.16b
        pmull v22.1q, v22.1d, v23.1d
        eor \mm_b, \mm_b, v22.16b
    .endm

    .macro GHASH_FOLD_INTO_PREMID ll_b, hh_b, mm_b, block_b, h_off, with_acc
        rev64 v21.16b, \block_b
        ext v21.16b, v21.16b, v21.16b, #8
        .if \with_acc
            eor v21.16b, v21.16b, v19.16b
        .endif
        ldr q20, [x4, #(\h_off)]
        pmull v22.1q, v21.1d, v20.1d
        eor \ll_b, \ll_b, v22.16b
        pmull2 v22.1q, v21.2d, v20.2d
        eor \hh_b, \hh_b, v22.16b
        ext v22.16b, v21.16b, v21.16b, #8
        eor v22.16b, v22.16b, v21.16b
        ldr d23, [x13, #(\h_off)]
        pmull v22.1q, v22.1d, v23.1d
        eor \mm_b, \mm_b, v22.16b
    .endm

    .macro GHASH_LOAD_PREMID h_off
        ldr q20, [x4, #(\h_off)]
        ldr d23, [x13, #(\h_off)]
    .endm

    .macro GHASH_LOAD_PREMID_PAIR pair_off
        ldp q31, q23, [x12, #(\pair_off)]
        ldr q20, [x12, #(\pair_off + 32)]
    .endm

    .macro GHASH_FOLD_INTO_PREMID_LOADED ll_b, hh_b, mm_b, block_b, with_acc
        rev64 v21.16b, \block_b
        ext v21.16b, v21.16b, v21.16b, #8
        .if \with_acc
            eor v21.16b, v21.16b, v19.16b
        .endif
        pmull v22.1q, v21.1d, v20.1d
        eor \ll_b, \ll_b, v22.16b
        pmull2 v22.1q, v21.2d, v20.2d
        eor \hh_b, \hh_b, v22.16b
        ext v22.16b, v21.16b, v21.16b, #8
        eor v22.16b, v22.16b, v21.16b
        pmull v22.1q, v22.1d, v23.1d
        eor \mm_b, \mm_b, v22.16b
    .endm

    .macro GHASH_FOLD2_INTO_PREMID_LOADED ll, hh, mm, block_a, block_b, with_acc
        rev64 v21.16b, \block_a\().16b
        rev64 v22.16b, \block_b\().16b
        .if \with_acc
            ext v24.16b, v19.16b, v19.16b, #8
            eor v21.16b, v21.16b, v24.16b
        .endif
        trn2 v24.2d, v21.2d, v22.2d
        trn1 v22.2d, v21.2d, v22.2d

        pmull v21.1q, v24.1d, v31.1d
        pmull2 v31.1q, v24.2d, v31.2d
        eor3 \ll\().16b, \ll\().16b, v21.16b, v31.16b

        pmull v21.1q, v22.1d, v23.1d
        pmull2 v23.1q, v22.2d, v23.2d
        eor3 \hh\().16b, \hh\().16b, v21.16b, v23.16b

        eor v24.16b, v24.16b, v22.16b
        pmull v21.1q, v24.1d, v20.1d
        pmull2 v20.1q, v24.2d, v20.2d
        eor3 \mm\().16b, \mm\().16b, v21.16b, v20.16b
    .endm

    .macro GHASH_FOLD2_SET_PREMID_LOADED ll, hh, mm, block_a, block_b, with_acc
        rev64 v21.16b, \block_a\().16b
        rev64 v22.16b, \block_b\().16b
        .if \with_acc
            ext v24.16b, v19.16b, v19.16b, #8
            eor v21.16b, v21.16b, v24.16b
        .endif
        trn2 v24.2d, v21.2d, v22.2d
        trn1 v22.2d, v21.2d, v22.2d

        pmull v21.1q, v24.1d, v31.1d
        pmull2 v31.1q, v24.2d, v31.2d
        eor \ll\().16b, v21.16b, v31.16b

        pmull v21.1q, v22.1d, v23.1d
        pmull2 v23.1q, v22.2d, v23.2d
        eor \hh\().16b, v21.16b, v23.16b

        eor v24.16b, v24.16b, v22.16b
        pmull v21.1q, v24.1d, v20.1d
        pmull2 v20.1q, v24.2d, v20.2d
        eor \mm\().16b, v21.16b, v20.16b
    .endm

    .macro GHASH_FOLD block_b, block_1d, block_2d, h_off, with_acc
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, \block_b, \h_off, \with_acc
    .endm

    .macro GHASH_COMBINE_REDUCE
        eor v16.16b, v16.16b, v26.16b
        eor v17.16b, v17.16b, v27.16b
        eor v18.16b, v18.16b, v28.16b
        GHASH_REDUCE
    .endm

    .macro GHASH_REDUCE
        eor v18.16b, v18.16b, v16.16b
        eor v18.16b, v18.16b, v17.16b
        movi v20.2d, #0
        ext v21.16b, v20.16b, v18.16b, #8
        eor v16.16b, v16.16b, v21.16b
        ext v21.16b, v18.16b, v20.16b, #8
        eor v17.16b, v17.16b, v21.16b

        movi v20.16b, #0xe1
        shl v20.2d, v20.2d, #57
        pmull v21.1q, v16.1d, v20.1d
        ext v22.16b, v16.16b, v16.16b, #8
        eor v22.16b, v22.16b, v21.16b
        pmull v21.1q, v22.1d, v20.1d
        ext v22.16b, v22.16b, v22.16b, #8
        eor v19.16b, v17.16b, v22.16b
        eor v19.16b, v19.16b, v21.16b
    .endm

    .macro GHASH_COMBINE_REDUCE_EOR3 use_gpr_reduce
        eor v16.16b, v16.16b, v26.16b
        eor v17.16b, v17.16b, v27.16b
        eor v18.16b, v18.16b, v28.16b
        GHASH_REDUCE_EOR3 \use_gpr_reduce
    .endm

    .macro GHASH_REDUCE_EOR3 use_gpr_reduce
        eor3 v18.16b, v18.16b, v16.16b, v17.16b
        movi v20.2d, #0
        ext v21.16b, v20.16b, v18.16b, #8
        eor v16.16b, v16.16b, v21.16b
        ext v21.16b, v18.16b, v20.16b, #8
        eor v17.16b, v17.16b, v21.16b

        .if \use_gpr_reduce
            fmov d20, x14
        .else
            movi v20.16b, #0xe1
            shl v20.2d, v20.2d, #57
        .endif
        pmull v21.1q, v16.1d, v20.1d
        ext v22.16b, v16.16b, v16.16b, #8
        eor v22.16b, v22.16b, v21.16b
        pmull v21.1q, v22.1d, v20.1d
        ext v22.16b, v22.16b, v22.16b, #8
        eor3 v19.16b, v17.16b, v22.16b, v21.16b
    .endm

    .macro SEAL_XOR_STORE
        ldr q24, [x2]
        eor v0.16b, v0.16b, v24.16b
        str q0, [x2]
        ldr q24, [x2, #16]
        eor v1.16b, v1.16b, v24.16b
        str q1, [x2, #16]
        ldr q24, [x2, #32]
        eor v2.16b, v2.16b, v24.16b
        str q2, [x2, #32]
        ldr q24, [x2, #48]
        eor v3.16b, v3.16b, v24.16b
        str q3, [x2, #48]
        ldr q24, [x2, #64]
        eor v4.16b, v4.16b, v24.16b
        str q4, [x2, #64]
        ldr q24, [x2, #80]
        eor v5.16b, v5.16b, v24.16b
        str q5, [x2, #80]
        ldr q24, [x2, #96]
        eor v6.16b, v6.16b, v24.16b
        str q6, [x2, #96]
        ldr q24, [x2, #112]
        eor v7.16b, v7.16b, v24.16b
        str q7, [x2, #112]
    .endm

    .macro SEAL_XOR_STORE_TO_PREV
        ldp q8, q9, [x2]
        eor v8.16b, v8.16b, v0.16b
        eor v9.16b, v9.16b, v1.16b
        stp q8, q9, [x2]
        ldp q10, q11, [x2, #32]
        eor v10.16b, v10.16b, v2.16b
        eor v11.16b, v11.16b, v3.16b
        stp q10, q11, [x2, #32]
        ldp q12, q13, [x2, #64]
        eor v12.16b, v12.16b, v4.16b
        eor v13.16b, v13.16b, v5.16b
        stp q12, q13, [x2, #64]
        ldp q14, q15, [x2, #96]
        eor v14.16b, v14.16b, v6.16b
        eor v15.16b, v15.16b, v7.16b
        stp q14, q15, [x2, #96]
    .endm

    .macro SEAL_EOR3_STORE_TO_PREV
        ldp q8, q9, [x2]
        eor3 v8.16b, v8.16b, v0.16b, v23.16b
        eor3 v9.16b, v9.16b, v1.16b, v23.16b
        stp q8, q9, [x2]
        ldp q10, q11, [x2, #32]
        eor3 v10.16b, v10.16b, v2.16b, v23.16b
        eor3 v11.16b, v11.16b, v3.16b, v23.16b
        stp q10, q11, [x2, #32]
        ldp q12, q13, [x2, #64]
        eor3 v12.16b, v12.16b, v4.16b, v23.16b
        eor3 v13.16b, v13.16b, v5.16b, v23.16b
        stp q12, q13, [x2, #64]
        ldp q14, q15, [x2, #96]
        eor3 v14.16b, v14.16b, v6.16b, v23.16b
        eor3 v15.16b, v15.16b, v7.16b, v23.16b
        stp q14, q15, [x2, #96]
    .endm

    .macro SEAL_XOR_STORE_TO_PREV_BUILD_NEXT
        PREP_NEXT_COUNTER8
        ldp q8, q9, [x2]
        eor v8.16b, v8.16b, v0.16b
        eor v9.16b, v9.16b, v1.16b
        stp q8, q9, [x2]
        orr v0.16b, v25.16b, v25.16b
        orr v1.16b, v25.16b, v25.16b
        mov v0.s[3], v24.s[0]
        mov v1.s[3], v24.s[1]
        ldp q10, q11, [x2, #32]
        eor v10.16b, v10.16b, v2.16b
        eor v11.16b, v11.16b, v3.16b
        stp q10, q11, [x2, #32]
        orr v2.16b, v25.16b, v25.16b
        orr v3.16b, v25.16b, v25.16b
        mov v2.s[3], v24.s[2]
        mov v3.s[3], v24.s[3]
        ldp q12, q13, [x2, #64]
        eor v12.16b, v12.16b, v4.16b
        eor v13.16b, v13.16b, v5.16b
        stp q12, q13, [x2, #64]
        orr v4.16b, v25.16b, v25.16b
        orr v5.16b, v25.16b, v25.16b
        mov v4.s[3], v31.s[0]
        mov v5.s[3], v31.s[1]
        ldp q14, q15, [x2, #96]
        eor v14.16b, v14.16b, v6.16b
        eor v15.16b, v15.16b, v7.16b
        stp q14, q15, [x2, #96]
        orr v6.16b, v25.16b, v25.16b
        orr v7.16b, v25.16b, v25.16b
        mov v6.s[3], v31.s[2]
        mov v7.s[3], v31.s[3]
    .endm

    .macro SEAL_EOR3_STORE_TO_PREV_BUILD_NEXT
        PREP_NEXT_COUNTER8
        ldp q8, q9, [x2]
        eor3 v8.16b, v8.16b, v0.16b, v23.16b
        eor3 v9.16b, v9.16b, v1.16b, v23.16b
        stp q8, q9, [x2]
        orr v0.16b, v25.16b, v25.16b
        orr v1.16b, v25.16b, v25.16b
        mov v0.s[3], v24.s[0]
        mov v1.s[3], v24.s[1]
        ldp q10, q11, [x2, #32]
        eor3 v10.16b, v10.16b, v2.16b, v23.16b
        eor3 v11.16b, v11.16b, v3.16b, v23.16b
        stp q10, q11, [x2, #32]
        orr v2.16b, v25.16b, v25.16b
        orr v3.16b, v25.16b, v25.16b
        mov v2.s[3], v24.s[2]
        mov v3.s[3], v24.s[3]
        ldp q12, q13, [x2, #64]
        eor3 v12.16b, v12.16b, v4.16b, v23.16b
        eor3 v13.16b, v13.16b, v5.16b, v23.16b
        stp q12, q13, [x2, #64]
        orr v4.16b, v25.16b, v25.16b
        orr v5.16b, v25.16b, v25.16b
        mov v4.s[3], v31.s[0]
        mov v5.s[3], v31.s[1]
        ldp q14, q15, [x2, #96]
        eor3 v14.16b, v14.16b, v6.16b, v23.16b
        eor3 v15.16b, v15.16b, v7.16b, v23.16b
        stp q14, q15, [x2, #96]
        orr v6.16b, v25.16b, v25.16b
        orr v7.16b, v25.16b, v25.16b
        mov v6.s[3], v31.s[2]
        mov v7.s[3], v31.s[3]
    .endm

    .macro GHASH_PREV_8
        movi v16.2d, #0
        movi v17.2d, #0
        movi v18.2d, #0
        movi v26.2d, #0
        movi v27.2d, #0
        movi v28.2d, #0
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, v8.16b, 0, 1
        GHASH_FOLD_INTO v26.16b, v27.16b, v28.16b, v9.16b, 16, 0
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, v10.16b, 32, 0
        GHASH_FOLD_INTO v26.16b, v27.16b, v28.16b, v11.16b, 48, 0
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, v12.16b, 64, 0
        GHASH_FOLD_INTO v26.16b, v27.16b, v28.16b, v13.16b, 80, 0
        GHASH_FOLD_INTO v16.16b, v17.16b, v18.16b, v14.16b, 96, 0
        GHASH_FOLD_INTO v26.16b, v27.16b, v28.16b, v15.16b, 112, 0
        GHASH_COMBINE_REDUCE
    .endm

    .macro GHASH_PREV_8_ACCUM_EOR3 h_base, with_acc, reduce, use_gpr_reduce
        GHASH_FOLD_INTO_PREMID v16.16b, v17.16b, v18.16b, v8.16b, \h_base + 0, \with_acc
        GHASH_FOLD_INTO_PREMID v26.16b, v27.16b, v28.16b, v9.16b, \h_base + 16, 0
        GHASH_FOLD_INTO_PREMID v16.16b, v17.16b, v18.16b, v10.16b, \h_base + 32, 0
        GHASH_FOLD_INTO_PREMID v26.16b, v27.16b, v28.16b, v11.16b, \h_base + 48, 0
        GHASH_FOLD_INTO_PREMID v16.16b, v17.16b, v18.16b, v12.16b, \h_base + 64, 0
        GHASH_FOLD_INTO_PREMID v26.16b, v27.16b, v28.16b, v13.16b, \h_base + 80, 0
        GHASH_FOLD_INTO_PREMID v16.16b, v17.16b, v18.16b, v14.16b, \h_base + 96, 0
        GHASH_FOLD_INTO_PREMID v26.16b, v27.16b, v28.16b, v15.16b, \h_base + 112, 0
        .if \reduce
            GHASH_COMBINE_REDUCE_EOR3 \use_gpr_reduce
        .endif
    .endm

    .macro COPY_CURRENT_TO_PREV
        orr v8.16b, v0.16b, v0.16b
        orr v9.16b, v1.16b, v1.16b
        orr v10.16b, v2.16b, v2.16b
        orr v11.16b, v3.16b, v3.16b
        orr v12.16b, v4.16b, v4.16b
        orr v13.16b, v5.16b, v5.16b
        orr v14.16b, v6.16b, v6.16b
        orr v15.16b, v7.16b, v7.16b
    .endm

    .macro OPEN_STORE_REG state_b, state_q, ct_b, data_off
        eor \state_b, \state_b, \ct_b
        str \state_q, [x2, #\data_off]
    .endm

    .macro OPEN_STORE_REG8
        eor v0.16b, v0.16b, v8.16b
        eor v1.16b, v1.16b, v9.16b
        stp q0, q1, [x2]
        eor v2.16b, v2.16b, v10.16b
        eor v3.16b, v3.16b, v11.16b
        stp q2, q3, [x2, #32]
        eor v4.16b, v4.16b, v12.16b
        eor v5.16b, v5.16b, v13.16b
        stp q4, q5, [x2, #64]
        eor v6.16b, v6.16b, v14.16b
        eor v7.16b, v7.16b, v15.16b
        stp q6, q7, [x2, #96]
    .endm

    .macro OPEN_STORE_REG8_EOR3
        eor3 v0.16b, v0.16b, v8.16b, v23.16b
        eor3 v1.16b, v1.16b, v9.16b, v23.16b
        stp q0, q1, [x2]
        eor3 v2.16b, v2.16b, v10.16b, v23.16b
        eor3 v3.16b, v3.16b, v11.16b, v23.16b
        stp q2, q3, [x2, #32]
        eor3 v4.16b, v4.16b, v12.16b, v23.16b
        eor3 v5.16b, v5.16b, v13.16b, v23.16b
        stp q4, q5, [x2, #64]
        eor3 v6.16b, v6.16b, v14.16b, v23.16b
        eor3 v7.16b, v7.16b, v15.16b, v23.16b
        stp q6, q7, [x2, #96]
    .endm

    .macro OPEN_STORE_REG8_BUILD_NEXT
        PREP_NEXT_COUNTER8
        eor v0.16b, v0.16b, v8.16b
        eor v1.16b, v1.16b, v9.16b
        stp q0, q1, [x2]
        orr v0.16b, v25.16b, v25.16b
        orr v1.16b, v25.16b, v25.16b
        mov v0.s[3], v24.s[0]
        mov v1.s[3], v24.s[1]
        eor v2.16b, v2.16b, v10.16b
        eor v3.16b, v3.16b, v11.16b
        stp q2, q3, [x2, #32]
        orr v2.16b, v25.16b, v25.16b
        orr v3.16b, v25.16b, v25.16b
        mov v2.s[3], v24.s[2]
        mov v3.s[3], v24.s[3]
        eor v4.16b, v4.16b, v12.16b
        eor v5.16b, v5.16b, v13.16b
        stp q4, q5, [x2, #64]
        orr v4.16b, v25.16b, v25.16b
        orr v5.16b, v25.16b, v25.16b
        mov v4.s[3], v31.s[0]
        mov v5.s[3], v31.s[1]
        eor v6.16b, v6.16b, v14.16b
        eor v7.16b, v7.16b, v15.16b
        stp q6, q7, [x2, #96]
        orr v6.16b, v25.16b, v25.16b
        orr v7.16b, v25.16b, v25.16b
        mov v6.s[3], v31.s[2]
        mov v7.s[3], v31.s[3]
    .endm

    .macro OPEN_STORE_REG8_EOR3_BUILD_NEXT
        PREP_NEXT_COUNTER8
        eor3 v0.16b, v0.16b, v8.16b, v23.16b
        eor3 v1.16b, v1.16b, v9.16b, v23.16b
        stp q0, q1, [x2]
        orr v0.16b, v25.16b, v25.16b
        orr v1.16b, v25.16b, v25.16b
        mov v0.s[3], v24.s[0]
        mov v1.s[3], v24.s[1]
        eor3 v2.16b, v2.16b, v10.16b, v23.16b
        eor3 v3.16b, v3.16b, v11.16b, v23.16b
        stp q2, q3, [x2, #32]
        orr v2.16b, v25.16b, v25.16b
        orr v3.16b, v25.16b, v25.16b
        mov v2.s[3], v24.s[2]
        mov v3.s[3], v24.s[3]
        eor3 v4.16b, v4.16b, v12.16b, v23.16b
        eor3 v5.16b, v5.16b, v13.16b, v23.16b
        stp q4, q5, [x2, #64]
        orr v4.16b, v25.16b, v25.16b
        orr v5.16b, v25.16b, v25.16b
        mov v4.s[3], v31.s[0]
        mov v5.s[3], v31.s[1]
        eor3 v6.16b, v6.16b, v14.16b, v23.16b
        eor3 v7.16b, v7.16b, v15.16b, v23.16b
        stp q6, q7, [x2, #96]
        orr v6.16b, v25.16b, v25.16b
        orr v7.16b, v25.16b, v25.16b
        mov v6.s[3], v31.s[2]
        mov v7.s[3], v31.s[3]
    .endm

    .macro DEFINE_SEAL name, rounds
        .globl _\name
        .p2align 4
_\name:
        sub sp, sp, #64
        stp d8, d9, [sp]
        stp d10, d11, [sp, #16]
        stp d12, d13, [sp, #32]
        stp d14, d15, [sp, #48]
        ldr q19, [x5]
        ldr w6, [x5, #16]
        mov x7, #0
        LOAD_COUNTER_BASE
        cmp x3, #128
        b.lo 3f

        BUILD_COUNTER8 \rounds
        AES_ENCRYPT8 \rounds
        SEAL_XOR_STORE_TO_PREV
        add w6, w6, #8
        add x2, x2, #128
        add x7, x7, #128
        subs x3, x3, #128
        cmp x3, #128
        b.lo 2f
1:
        BUILD_COUNTER8 \rounds
        AES_ENCRYPT8_GHASH_PREV \rounds
        SEAL_XOR_STORE_TO_PREV
        add w6, w6, #8
        add x2, x2, #128
        add x7, x7, #128
        subs x3, x3, #128
        cmp x3, #128
        b.hs 1b
2:
        GHASH_PREV_8
3:
        str q19, [x5]
        str w6, [x5, #16]
        str x7, [x5, #24]
        ldp d14, d15, [sp, #48]
        ldp d12, d13, [sp, #32]
        ldp d10, d11, [sp, #16]
        ldp d8, d9, [sp]
        add sp, sp, #64
        ret
    .endm

    .macro DEFINE_OPEN name, rounds
        .globl _\name
        .p2align 4
_\name:
        sub sp, sp, #64
        stp d8, d9, [sp]
        stp d10, d11, [sp, #16]
        stp d12, d13, [sp, #32]
        stp d14, d15, [sp, #48]
        ldr q19, [x5]
        ldr w6, [x5, #16]
        mov x7, #0
        LOAD_COUNTER_BASE
        cmp x3, #128
        b.lo 2f
1:
        LOAD_OPEN_CT8
        BUILD_COUNTER8 \rounds
        AES_ENCRYPT8_GHASH_OPEN_REG \rounds
        OPEN_STORE_REG8
        add w6, w6, #8
        add x2, x2, #128
        add x7, x7, #128
        subs x3, x3, #128
        cmp x3, #128
        b.hs 1b
2:
        str q19, [x5]
        str w6, [x5, #16]
        str x7, [x5, #24]
        ldp d14, d15, [sp, #48]
        ldp d12, d13, [sp, #32]
        ldp d10, d11, [sp, #16]
        ldp d8, d9, [sp]
        add sp, sp, #64
        ret
    .endm

    .macro DEFINE_SEAL16_EOR3 name, rounds
        .globl _\name
        .p2align 4
_\name:
        sub sp, sp, #64
        stp d8, d9, [sp]
        stp d10, d11, [sp, #16]
        stp d12, d13, [sp, #32]
        stp d14, d15, [sp, #48]
        mov x10, x7
        mov x12, x6
        mov x13, x5
        .if \rounds == 14
            mov x14, #0xc200000000000000
        .endif
        ldr q19, [x10]
        ldr w6, [x10, #16]
        mov x7, #0
        LOAD_COUNTER_BASE
        cmp x3, #128
        b.lo 5f

        BUILD_COUNTER8 \rounds
        AES_ENCRYPT8_RAW \rounds
        SEAL_EOR3_STORE_TO_PREV_BUILD_NEXT
        add w6, w6, #8
        add x2, x2, #128
        add x7, x7, #128
        subs x3, x3, #128
        cmp x3, #256
        b.lo 2f
1:
        AES_ENCRYPT8_GHASH_REG_INIT_EOR3 \rounds, 0, 1
        SEAL_EOR3_STORE_TO_PREV_BUILD_NEXT
        add w6, w6, #8
        add x2, x2, #128
        add x7, x7, #128
        subs x3, x3, #128

        .if \rounds == 14
            AES_ENCRYPT8_GHASH_PREV_ACCUM_EOR3 \rounds, 128, 192, 0, 1, 1
        .else
            AES_ENCRYPT8_GHASH_PREV_ACCUM_EOR3 \rounds, 128, 192, 0, 1, 0
        .endif
        SEAL_EOR3_STORE_TO_PREV_BUILD_NEXT
        add w6, w6, #8
        add x2, x2, #128
        add x7, x7, #128
        subs x3, x3, #128
        cmp x3, #256
        b.hs 1b
2:
        cmp x3, #128
        b.lo 4f
        GHASH_ZERO_PRODUCTS
        AES_ENCRYPT8_GHASH_PREV_ACCUM_EOR3 \rounds, 0, 0, 1, 0, 0
        SEAL_EOR3_STORE_TO_PREV
        add w6, w6, #8
        add x2, x2, #128
        add x7, x7, #128
        subs x3, x3, #128
        .if \rounds == 14
            GHASH_PREV_8_ACCUM_EOR3 128, 0, 1, 1
        .else
            GHASH_PREV_8_ACCUM_EOR3 128, 0, 1, 0
        .endif
        b 5f
4:
        GHASH_ZERO_PRODUCTS
        .if \rounds == 14
            GHASH_PREV_8_ACCUM_EOR3 128, 1, 1, 1
        .else
            GHASH_PREV_8_ACCUM_EOR3 128, 1, 1, 0
        .endif
5:
        str q19, [x10]
        str w6, [x10, #16]
        str x7, [x10, #24]
        ldp d14, d15, [sp, #48]
        ldp d12, d13, [sp, #32]
        ldp d10, d11, [sp, #16]
        ldp d8, d9, [sp]
        add sp, sp, #64
        ret
    .endm

    .macro DEFINE_OPEN16_EOR3 name, rounds
        .globl _\name
        .p2align 4
_\name:
        sub sp, sp, #64
        stp d8, d9, [sp]
        stp d10, d11, [sp, #16]
        stp d12, d13, [sp, #32]
        stp d14, d15, [sp, #48]
        mov x10, x7
        mov x12, x6
        mov x13, x5
        .if \rounds == 10
            mov x14, #0xc200000000000000
        .endif
        ldr q19, [x10]
        ldr w6, [x10, #16]
        mov x7, #0
        LOAD_COUNTER_BASE
        cmp x3, #256
        b.lo 2f
        BUILD_COUNTER8 \rounds
1:
        LOAD_OPEN_CT8
        AES_ENCRYPT8_GHASH_REG_INIT_EOR3 \rounds, 0, 1
        OPEN_STORE_REG8_EOR3_BUILD_NEXT
        add w6, w6, #8
        add x2, x2, #128
        add x7, x7, #128
        subs x3, x3, #128

        LOAD_OPEN_CT8
        .if \rounds == 10
            AES_ENCRYPT8_GHASH_OPEN_REG_ACCUM_EOR3 \rounds, 128, 192, 0, 1, 1
        .else
            AES_ENCRYPT8_GHASH_OPEN_REG_ACCUM_EOR3 \rounds, 128, 192, 0, 1, 0
        .endif
        OPEN_STORE_REG8_EOR3_BUILD_NEXT
        add w6, w6, #8
        add x2, x2, #128
        add x7, x7, #128
        subs x3, x3, #128
        cmp x3, #256
        b.hs 1b
2:
        cmp x3, #128
        b.lo 3f
        GHASH_ZERO_PRODUCTS
        LOAD_OPEN_CT8
        BUILD_COUNTER8 \rounds
        .if \rounds == 10
            AES_ENCRYPT8_GHASH_OPEN_REG_ACCUM_EOR3 \rounds, 128, 192, 1, 1, 1
        .else
            AES_ENCRYPT8_GHASH_OPEN_REG_ACCUM_EOR3 \rounds, 128, 192, 1, 1, 0
        .endif
        OPEN_STORE_REG8_EOR3
        add w6, w6, #8
        add x2, x2, #128
        add x7, x7, #128
3:
        str q19, [x10]
        str w6, [x10, #16]
        str x7, [x10, #24]
        ldp d14, d15, [sp, #48]
        ldp d12, d13, [sp, #32]
        ldp d10, d11, [sp, #16]
        ldp d8, d9, [sp]
        add sp, sp, #64
        ret
    .endm

    DEFINE_SEAL rscrypto_aes128_gcm_seal_8x_aarch64, 10
    DEFINE_OPEN rscrypto_aes128_gcm_open_8x_aarch64, 10
    DEFINE_SEAL rscrypto_aes256_gcm_seal_8x_aarch64, 14
    DEFINE_OPEN rscrypto_aes256_gcm_open_8x_aarch64, 14
    DEFINE_SEAL16_EOR3 rscrypto_aes128_gcm_seal_16x_eor3_aarch64, 10
    DEFINE_OPEN16_EOR3 rscrypto_aes128_gcm_open_16x_eor3_aarch64, 10
    DEFINE_SEAL16_EOR3 rscrypto_aes256_gcm_seal_16x_eor3_aarch64, 14
    DEFINE_OPEN16_EOR3 rscrypto_aes256_gcm_open_16x_eor3_aarch64, 14
