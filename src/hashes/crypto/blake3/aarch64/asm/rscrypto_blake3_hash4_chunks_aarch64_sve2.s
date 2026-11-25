// Copyright (c) 2026 rscrypto contributors
// SPDX-License-Identifier: Apache-2.0
// Four contiguous BLAKE3 chunks. A lane holds one chunk through all sixteen blocks.
// x0 = four input chunks, x1 = eight key words, x2 = counter, w3 = flags, x4 = four output CVs.
// Input and output are disjoint. Each SVE2 instruction acts independently on 32-bit words.
// Only the low four words of each vector contribute to an output, at every supported vector length.
.text
.p2align 4
.globl rscrypto_blake3_hash4_chunks_aarch64_sve2
.hidden rscrypto_blake3_hash4_chunks_aarch64_sve2
.type rscrypto_blake3_hash4_chunks_aarch64_sve2, %function
rscrypto_blake3_hash4_chunks_aarch64_sve2:
.cfi_startproc
    sub sp, sp, #96
.cfi_def_cfa_offset 96
    stp d8, d9, [sp]
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
.cfi_offset 72, -96
.cfi_offset 73, -88
.cfi_offset 74, -80
.cfi_offset 75, -72
.cfi_offset 76, -64
.cfi_offset 77, -56
.cfi_offset 78, -48
.cfi_offset 79, -40
// Form the high and low counter words once, including a carry between lanes.
    add x10, x2, #1
    add x11, x2, #2
    add x12, x2, #3
    mov v12.s[0], w2
    mov v12.s[1], w10
    mov v12.s[2], w11
    mov v12.s[3], w12
    lsr x2, x2, #32
    mov v13.s[0], w2
    lsr x10, x10, #32
    mov v13.s[1], w10
    lsr x11, x11, #32
    mov v13.s[2], w11
    lsr x12, x12, #32
    mov v13.s[3], w12
    stp q12, q13, [sp, #64]
    add x5, x0, #1024
    add x6, x0, #2048
    add x7, x0, #3072
    adr x12, 8f
    ldp w13, w14, [x12]
    ldp w15, w16, [x12, #8]
    mov w8, #16
    orr w9, w3, #1
    orr w10, w3, #2
    ld1r {v0.4s}, [x1], #4
    ld1r {v1.4s}, [x1], #4
    ld1r {v2.4s}, [x1], #4
    ld1r {v3.4s}, [x1], #4
    ld1r {v4.4s}, [x1], #4
    ld1r {v5.4s}, [x1], #4
    ld1r {v6.4s}, [x1], #4
    ld1r {v7.4s}, [x1], #4
.arch_extension sve2
// Transpose four adjacent words from each of the four chunks.
.macro b3load a,b,c,d
        ldr q8, [x0], #16
        ldr q9, [x5], #16
        ldr q10, [x6], #16
        ldr q11, [x7], #16
        trn1 v12.4s, v8.4s, v9.4s
        trn2 v13.4s, v8.4s, v9.4s
        trn1 v14.4s, v10.4s, v11.4s
        trn2 v15.4s, v10.4s, v11.4s
        zip1 v\a\().2d, v12.2d, v14.2d
        zip1 v\b\().2d, v13.2d, v15.2d
        zip2 v\c\().2d, v12.2d, v14.2d
        zip2 v\d\().2d, v13.2d, v15.2d
.endm
.macro b3g a,b,c,d,m,n
        add v\a\().4s, v\a\().4s, v\m\().4s
        add v\a\().4s, v\a\().4s, v\b\().4s
        xar z\d\().s, z\d\().s, z\a\().s, #16
        add v\c\().4s, v\c\().4s, v\d\().4s
        xar z\b\().s, z\b\().s, z\c\().s, #12
        add v\a\().4s, v\a\().4s, v\n\().4s
        add v\a\().4s, v\a\().4s, v\b\().4s
        xar z\d\().s, z\d\().s, z\a\().s, #8
        add v\c\().4s, v\c\().4s, v\d\().4s
        xar z\b\().s, z\b\().s, z\c\().s, #7
.endm

// Each round has four column G operations followed by four diagonal G operations.
.macro b3round m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15
        b3g 0,4,8,12,\m0,\m1
        b3g 1,5,9,13,\m2,\m3
        b3g 2,6,10,14,\m4,\m5
        b3g 3,7,11,15,\m6,\m7
        b3g 0,5,10,15,\m8,\m9
        b3g 1,6,11,12,\m10,\m11
        b3g 2,7,8,13,\m12,\m13
        b3g 3,4,9,14,\m14,\m15
.endm

1:
    b3load 16,17,18,19
    b3load 20,21,22,23
    b3load 24,25,26,27
    b3load 28,29,30,31
    dup v8.4s, w13
    dup v9.4s, w14
    dup v10.4s, w15
    dup v11.4s, w16
    ldp q12, q13, [sp, #64]
    movi v14.4s, #64
    cmp w8, #1
    csel w11, w10, w9, eq
    dup v15.4s, w11
    mov w9, w3

// Message word i lives in register 16 + i for all seven rounds.
    b3round 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    b3round 18,22,19,26,23,16,20,29,17,27,28,21,25,30,31,24
    b3round 19,20,26,28,29,18,23,30,22,21,25,16,27,31,24,17
    b3round 26,23,28,25,30,19,29,31,20,16,27,18,21,24,17,22
    b3round 28,29,25,27,31,26,30,24,23,18,21,19,16,17,22,20
    b3round 25,30,27,21,24,28,31,17,29,19,16,26,18,22,20,23
    b3round 27,31,21,16,17,25,24,22,30,26,18,28,19,20,23,29
    eor v0.16b, v0.16b, v8.16b
    eor v1.16b, v1.16b, v9.16b
    eor v2.16b, v2.16b, v10.16b
    eor v3.16b, v3.16b, v11.16b
    eor v4.16b, v4.16b, v12.16b
    eor v5.16b, v5.16b, v13.16b
    eor v6.16b, v6.16b, v14.16b
    eor v7.16b, v7.16b, v15.16b
    subs w8, w8, #1
    b.ne 1b
    st4 {v0.s, v1.s, v2.s, v3.s}[0], [x4], #16
    st4 {v4.s, v5.s, v6.s, v7.s}[0], [x4], #16
    st4 {v0.s, v1.s, v2.s, v3.s}[1], [x4], #16
    st4 {v4.s, v5.s, v6.s, v7.s}[1], [x4], #16
    st4 {v0.s, v1.s, v2.s, v3.s}[2], [x4], #16
    st4 {v4.s, v5.s, v6.s, v7.s}[2], [x4], #16
    st4 {v0.s, v1.s, v2.s, v3.s}[3], [x4], #16
    st4 {v4.s, v5.s, v6.s, v7.s}[3], [x4], #16
    ldp d8, d9, [sp]
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    add sp, sp, #96
.cfi_def_cfa_offset 0
.cfi_restore 72
.cfi_restore 73
.cfi_restore 74
.cfi_restore 75
.cfi_restore 76
.cfi_restore 77
.cfi_restore 78
.cfi_restore 79
    ret
.cfi_endproc
.size rscrypto_blake3_hash4_chunks_aarch64_sve2, . - rscrypto_blake3_hash4_chunks_aarch64_sve2
.arch_extension nosve2
.p2align 4
8:
.word 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a
.purgem b3round
.purgem b3g
.purgem b3load
