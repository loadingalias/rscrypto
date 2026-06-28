// Copyright (c) 2026 rscrypto contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// rscrypto-owned aarch64 ChaCha20 XOR bulk kernel.
//
// ABI:
//   x0: uint8_t *buffer       in-place plaintext/ciphertext buffer
//   x1: size_t chunks         number of 512-byte chunks
//   x2: const uint8_t key[32]
//   w3: uint32_t counter      initial ChaCha20 block counter
//   x4: const uint8_t nonce[12]
//
// The loop processes eight ChaCha20 blocks per iteration with 30 live vector
// state registers and two stack-spilled nonce lanes. All branches and memory
// addresses depend only on public length/counter inputs.

.text
.balign 16

.macro DUP_CONST dst, hi, lo
        mov     w5, #\lo
        movk    w5, #\hi, lsl #16
        dup     \dst\().4s, w5
.endm

.macro ROTL_T dst, tmp, right, left
        ushr    \tmp\().4s, \dst\().4s, #\right
        sli     \tmp\().4s, \dst\().4s, #\left
        mov     \dst\().16b, \tmp\().16b
.endm

.macro QR_T a, b, c, d, tmp
        add     \a\().4s, \a\().4s, \b\().4s
        eor     \d\().16b, \d\().16b, \a\().16b
        rev32   \d\().8h, \d\().8h
        add     \c\().4s, \c\().4s, \d\().4s
        eor     \b\().16b, \b\().16b, \c\().16b
        ROTL_T  \b, \tmp, 20, 12
        add     \a\().4s, \a\().4s, \b\().4s
        eor     \d\().16b, \d\().16b, \a\().16b
        ROTL_T  \d, \tmp, 24, 8
        add     \c\().4s, \c\().4s, \d\().4s
        eor     \b\().16b, \b\().16b, \c\().16b
        ROTL_T  \b, \tmp, 25, 7
.endm

.macro LOAD_STATE8
        DUP_CONST v0, 0x6170, 0x7865
        DUP_CONST v1, 0x3320, 0x646e
        DUP_CONST v2, 0x7962, 0x2d32
        DUP_CONST v3, 0x6b20, 0x6574
        mov     v16.16b, v0.16b
        mov     v17.16b, v1.16b
        mov     v18.16b, v2.16b
        mov     v19.16b, v3.16b

        ldr     w5, [x2, #0]
        dup     v4.4s, w5
        mov     v20.16b, v4.16b
        ldr     w5, [x2, #4]
        dup     v5.4s, w5
        mov     v21.16b, v5.16b
        ldr     w5, [x2, #8]
        dup     v6.4s, w5
        mov     v22.16b, v6.16b
        ldr     w5, [x2, #12]
        dup     v7.4s, w5
        mov     v23.16b, v7.16b
        ldr     w5, [x2, #16]
        dup     v8.4s, w5
        mov     v24.16b, v8.16b
        ldr     w5, [x2, #20]
        dup     v9.4s, w5
        mov     v25.16b, v9.16b
        ldr     w5, [x2, #24]
        dup     v10.4s, w5
        mov     v26.16b, v10.16b
        ldr     w5, [x2, #28]
        dup     v11.4s, w5
        mov     v27.16b, v11.16b

        mov     w6, w3
        mov     v12.s[0], w6
        add     w6, w6, #1
        mov     v12.s[1], w6
        add     w6, w6, #1
        mov     v12.s[2], w6
        add     w6, w6, #1
        mov     v12.s[3], w6
        add     w6, w3, #4
        mov     v28.s[0], w6
        add     w6, w6, #1
        mov     v28.s[1], w6
        add     w6, w6, #1
        mov     v28.s[2], w6
        add     w6, w6, #1
        mov     v28.s[3], w6

        ldr     w5, [x4, #0]
        dup     v13.4s, w5
        mov     v29.16b, v13.16b
        ldr     w5, [x4, #4]
        dup     v14.4s, w5
        str     q14, [sp, #64]
        ldr     w5, [x4, #8]
        dup     v15.4s, w5
        str     q15, [sp, #80]
.endm

.macro ADD_INITIAL8
        DUP_CONST v31, 0x6170, 0x7865
        add     v0.4s, v0.4s, v31.4s
        add     v16.4s, v16.4s, v31.4s
        DUP_CONST v31, 0x3320, 0x646e
        add     v1.4s, v1.4s, v31.4s
        add     v17.4s, v17.4s, v31.4s
        DUP_CONST v31, 0x7962, 0x2d32
        add     v2.4s, v2.4s, v31.4s
        add     v18.4s, v18.4s, v31.4s
        DUP_CONST v31, 0x6b20, 0x6574
        add     v3.4s, v3.4s, v31.4s
        add     v19.4s, v19.4s, v31.4s

        ldr     w5, [x2, #0]
        dup     v31.4s, w5
        add     v4.4s, v4.4s, v31.4s
        add     v20.4s, v20.4s, v31.4s
        ldr     w5, [x2, #4]
        dup     v31.4s, w5
        add     v5.4s, v5.4s, v31.4s
        add     v21.4s, v21.4s, v31.4s
        ldr     w5, [x2, #8]
        dup     v31.4s, w5
        add     v6.4s, v6.4s, v31.4s
        add     v22.4s, v22.4s, v31.4s
        ldr     w5, [x2, #12]
        dup     v31.4s, w5
        add     v7.4s, v7.4s, v31.4s
        add     v23.4s, v23.4s, v31.4s
        ldr     w5, [x2, #16]
        dup     v31.4s, w5
        add     v8.4s, v8.4s, v31.4s
        add     v24.4s, v24.4s, v31.4s
        ldr     w5, [x2, #20]
        dup     v31.4s, w5
        add     v9.4s, v9.4s, v31.4s
        add     v25.4s, v25.4s, v31.4s
        ldr     w5, [x2, #24]
        dup     v31.4s, w5
        add     v10.4s, v10.4s, v31.4s
        add     v26.4s, v26.4s, v31.4s
        ldr     w5, [x2, #28]
        dup     v31.4s, w5
        add     v11.4s, v11.4s, v31.4s
        add     v27.4s, v27.4s, v31.4s

        mov     w6, w3
        mov     v31.s[0], w6
        add     w6, w6, #1
        mov     v31.s[1], w6
        add     w6, w6, #1
        mov     v31.s[2], w6
        add     w6, w6, #1
        mov     v31.s[3], w6
        add     v12.4s, v12.4s, v31.4s
        add     w6, w3, #4
        mov     v31.s[0], w6
        add     w6, w6, #1
        mov     v31.s[1], w6
        add     w6, w6, #1
        mov     v31.s[2], w6
        add     w6, w6, #1
        mov     v31.s[3], w6
        add     v28.4s, v28.4s, v31.4s

        ldr     w5, [x4, #0]
        dup     v31.4s, w5
        add     v13.4s, v13.4s, v31.4s
        add     v29.4s, v29.4s, v31.4s
        ldr     w5, [x4, #4]
        dup     v31.4s, w5
        add     v14.4s, v14.4s, v31.4s
        ldr     q30, [sp, #64]
        add     v30.4s, v30.4s, v31.4s
        str     q30, [sp, #64]
        ldr     w5, [x4, #8]
        dup     v31.4s, w5
        add     v15.4s, v15.4s, v31.4s
        ldr     q30, [sp, #80]
        add     v30.4s, v30.4s, v31.4s
        str     q30, [sp, #80]
.endm

.macro XOR_STORE_GROUP_D off0, off1, off2, off3, a, b, c, d
        zip1    v30.4s, \a\().4s, \b\().4s
        zip2    v31.4s, \a\().4s, \b\().4s
        zip1    \a\().4s, \c\().4s, \d\().4s
        zip2    \b\().4s, \c\().4s, \d\().4s
        zip1    \c\().2d, v30.2d, \a\().2d
        zip2    \d\().2d, v30.2d, \a\().2d
        zip1    \a\().2d, v31.2d, \b\().2d
        zip2    \b\().2d, v31.2d, \b\().2d

        add     x5, x0, #\off0
        ld1     {{ v30.16b }}, [x5]
        eor     \c\().16b, \c\().16b, v30.16b
        st1     {{ \c\().16b }}, [x5]
        add     x5, x0, #\off1
        ld1     {{ v30.16b }}, [x5]
        eor     \d\().16b, \d\().16b, v30.16b
        st1     {{ \d\().16b }}, [x5]
        add     x5, x0, #\off2
        ld1     {{ v30.16b }}, [x5]
        eor     \a\().16b, \a\().16b, v30.16b
        st1     {{ \a\().16b }}, [x5]
        add     x5, x0, #\off3
        ld1     {{ v30.16b }}, [x5]
        eor     \b\().16b, \b\().16b, v30.16b
        st1     {{ \b\().16b }}, [x5]
.endm

.macro PROCESS8
        LOAD_STATE8

        mov     w7, #10
4:
        QR_T    v0, v4, v8, v12, v31
        QR_T    v16, v20, v24, v28, v31
        QR_T    v1, v5, v9, v13, v31
        QR_T    v17, v21, v25, v29, v31
        QR_T    v2, v6, v10, v14, v31
        ldr     q30, [sp, #64]
        QR_T    v18, v22, v26, v30, v31
        str     q30, [sp, #64]
        QR_T    v3, v7, v11, v15, v31
        ldr     q30, [sp, #80]
        QR_T    v19, v23, v27, v30, v31
        str     q30, [sp, #80]

        QR_T    v0, v5, v10, v15, v31
        ldr     q30, [sp, #80]
        QR_T    v16, v21, v26, v30, v31
        str     q30, [sp, #80]
        QR_T    v1, v6, v11, v12, v31
        QR_T    v17, v22, v27, v28, v31
        QR_T    v2, v7, v8, v13, v31
        QR_T    v18, v23, v24, v29, v31
        QR_T    v3, v4, v9, v14, v31
        ldr     q30, [sp, #64]
        QR_T    v19, v20, v25, v30, v31
        str     q30, [sp, #64]
        subs    w7, w7, #1
        b.ne    4b

        ADD_INITIAL8

        XOR_STORE_GROUP_D 0, 64, 128, 192, v0, v1, v2, v3
        XOR_STORE_GROUP_D 16, 80, 144, 208, v4, v5, v6, v7
        XOR_STORE_GROUP_D 32, 96, 160, 224, v8, v9, v10, v11
        XOR_STORE_GROUP_D 48, 112, 176, 240, v12, v13, v14, v15

        XOR_STORE_GROUP_D 256, 320, 384, 448, v16, v17, v18, v19
        XOR_STORE_GROUP_D 272, 336, 400, 464, v20, v21, v22, v23
        XOR_STORE_GROUP_D 288, 352, 416, 480, v24, v25, v26, v27
        ldr     q0, [sp, #64]
        ldr     q1, [sp, #80]
        XOR_STORE_GROUP_D 304, 368, 432, 496, v28, v29, v0, v1
.endm

.globl rscrypto_chacha20_xor_8block_aarch64_linux
.type rscrypto_chacha20_xor_8block_aarch64_linux,%function
rscrypto_chacha20_xor_8block_aarch64_linux:
        sub     sp, sp, #96
        stp     d8, d9, [sp, #0]
        stp     d10, d11, [sp, #16]
        stp     d12, d13, [sp, #32]
        stp     d14, d15, [sp, #48]

        cbz     x1, 3f
2:
        PROCESS8
        add     x0, x0, #512
        add     w3, w3, #8
        subs    x1, x1, #1
        b.ne    2b

3:
        ldp     d8, d9, [sp, #0]
        ldp     d10, d11, [sp, #16]
        ldp     d12, d13, [sp, #32]
        ldp     d14, d15, [sp, #48]
        add     sp, sp, #96
        ret
.size rscrypto_chacha20_xor_8block_aarch64_linux, .-rscrypto_chacha20_xor_8block_aarch64_linux
