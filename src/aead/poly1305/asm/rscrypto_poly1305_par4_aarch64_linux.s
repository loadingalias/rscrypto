// Copyright (c) 2026 rscrypto contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// rscrypto-owned aarch64 Poly1305 par4 accumulator.
//
// ABI:
//   x0: uint32_t h[5]              Poly1305 accumulator limbs, updated in place
//   x1: const uint32_t powers[20]  r, r^2, r^3, r^4 as four five-limb arrays
//   x2: const uint8_t *input       complete 64-byte groups
//   x3: size_t groups              number of 64-byte groups
//
// The loop mirrors the Rust NEON par4 formulation:
//   h' = h * r^4 + b0 * r^4 + b1 * r^3 + b2 * r^2 + b3 * r
// followed by the standard 26-bit Poly1305 reduction. Branches and addresses
// depend only on public group count.

.text
.align 4

.macro LOAD_POWER_VEC dst, off4, off3, off2, off1
        ldr     w15, [x1, #\off4]
        mov     \dst\().s[0], w15
        ldr     w15, [x1, #\off3]
        mov     \dst\().s[1], w15
        ldr     w15, [x1, #\off2]
        mov     \dst\().s[2], w15
        ldr     w15, [x1, #\off1]
        mov     \dst\().s[3], w15
.endm

.macro LOAD_POWER_VEC5 dst, off4, off3, off2, off1
        ldr     w15, [x1, #\off4]
        add     w15, w15, w15, lsl #2
        mov     \dst\().s[0], w15
        ldr     w15, [x1, #\off3]
        add     w15, w15, w15, lsl #2
        mov     \dst\().s[1], w15
        ldr     w15, [x1, #\off2]
        add     w15, w15, w15, lsl #2
        mov     \dst\().s[2], w15
        ldr     w15, [x1, #\off1]
        add     w15, w15, w15, lsl #2
        mov     \dst\().s[3], w15
.endm

.macro LOAD_LIMB_MASKED dst, off0, off1, off2, off3
        ldur    w15, [x2, #\off0]
        and     w15, w15, w4
        mov     \dst\().s[0], w15
        ldur    w15, [x2, #\off1]
        and     w15, w15, w4
        mov     \dst\().s[1], w15
        ldur    w15, [x2, #\off2]
        and     w15, w15, w4
        mov     \dst\().s[2], w15
        ldur    w15, [x2, #\off3]
        and     w15, w15, w4
        mov     \dst\().s[3], w15
.endm

.macro LOAD_LIMB_SHIFTED dst, shift, off0, off1, off2, off3
        ldur    w15, [x2, #\off0]
        lsr     w15, w15, #\shift
        and     w15, w15, w4
        mov     \dst\().s[0], w15
        ldur    w15, [x2, #\off1]
        lsr     w15, w15, #\shift
        and     w15, w15, w4
        mov     \dst\().s[1], w15
        ldur    w15, [x2, #\off2]
        lsr     w15, w15, #\shift
        and     w15, w15, w4
        mov     \dst\().s[2], w15
        ldur    w15, [x2, #\off3]
        lsr     w15, w15, #\shift
        and     w15, w15, w4
        mov     \dst\().s[3], w15
.endm

.macro LOAD_LIMB_HIBIT dst, off0, off1, off2, off3
        ldur    w15, [x2, #\off0]
        lsr     w15, w15, #8
        orr     w15, w15, w5
        mov     \dst\().s[0], w15
        ldur    w15, [x2, #\off1]
        lsr     w15, w15, #8
        orr     w15, w15, w5
        mov     \dst\().s[1], w15
        ldur    w15, [x2, #\off2]
        lsr     w15, w15, #8
        orr     w15, w15, w5
        mov     \dst\().s[2], w15
        ldur    w15, [x2, #\off3]
        lsr     w15, w15, #8
        orr     w15, w15, w5
        mov     \dst\().s[3], w15
.endm

.macro DOT5_SUM dst, x0v, x1v, x2v, x3v, x4v, y0v, y1v, y2v, y3v, y4v
        umull   v16.2d, \x0v\().2s, \y0v\().2s
        umull2  v17.2d, \x0v\().4s, \y0v\().4s
        umlal   v16.2d, \x1v\().2s, \y1v\().2s
        umlal2  v17.2d, \x1v\().4s, \y1v\().4s
        umlal   v16.2d, \x2v\().2s, \y2v\().2s
        umlal2  v17.2d, \x2v\().4s, \y2v\().4s
        umlal   v16.2d, \x3v\().2s, \y3v\().2s
        umlal2  v17.2d, \x3v\().4s, \y3v\().4s
        umlal   v16.2d, \x4v\().2s, \y4v\().2s
        umlal2  v17.2d, \x4v\().4s, \y4v\().4s
        add     v16.2d, v16.2d, v17.2d
        addp    v16.2d, v16.2d, v16.2d
        fmov    \dst, d16
.endm

.globl rscrypto_poly1305_accumulate4_aarch64_linux
.type rscrypto_poly1305_accumulate4_aarch64_linux,%function
rscrypto_poly1305_accumulate4_aarch64_linux:
        cbz     x3, .Lpoly1305_par4_done

        stp     x19, x20, [sp, #-80]!
        stp     x21, x22, [sp, #16]
        stp     x23, x24, [sp, #32]
        stp     x25, x26, [sp, #48]
        stp     x27, x28, [sp, #64]

        mov     x4, #0xffff
        movk    x4, #0x03ff, lsl #16
        mov     w5, #1
        lsl     w5, w5, #24

        ldr     w24, [x1, #60]
        ldr     w25, [x1, #64]
        ldr     w26, [x1, #68]
        ldr     w27, [x1, #72]
        ldr     w28, [x1, #76]
        add     w11, w25, w25, lsl #2
        add     w12, w26, w26, lsl #2
        add     w13, w27, w27, lsl #2
        add     w14, w28, w28, lsl #2

        LOAD_POWER_VEC  v5,  60, 40, 20, 0
        LOAD_POWER_VEC  v6,  64, 44, 24, 4
        LOAD_POWER_VEC  v7,  68, 48, 28, 8
        LOAD_POWER_VEC  v8,  72, 52, 32, 12
        LOAD_POWER_VEC  v9,  76, 56, 36, 16
        LOAD_POWER_VEC5 v10, 64, 44, 24, 4
        LOAD_POWER_VEC5 v11, 68, 48, 28, 8
        LOAD_POWER_VEC5 v12, 72, 52, 32, 12
        LOAD_POWER_VEC5 v13, 76, 56, 36, 16

.Lpoly1305_par4_loop:
        ldp     w19, w20, [x0]
        ldp     w21, w22, [x0, #8]
        ldr     w23, [x0, #16]

        umull   x6,  w19, w24
        umaddl  x6,  w20, w14, x6
        umaddl  x6,  w21, w13, x6
        umaddl  x6,  w22, w12, x6
        umaddl  x6,  w23, w11, x6

        umull   x7,  w19, w25
        umaddl  x7,  w20, w24, x7
        umaddl  x7,  w21, w14, x7
        umaddl  x7,  w22, w13, x7
        umaddl  x7,  w23, w12, x7

        umull   x8,  w19, w26
        umaddl  x8,  w20, w25, x8
        umaddl  x8,  w21, w24, x8
        umaddl  x8,  w22, w14, x8
        umaddl  x8,  w23, w13, x8

        umull   x9,  w19, w27
        umaddl  x9,  w20, w26, x9
        umaddl  x9,  w21, w25, x9
        umaddl  x9,  w22, w24, x9
        umaddl  x9,  w23, w14, x9

        umull   x10, w19, w28
        umaddl  x10, w20, w27, x10
        umaddl  x10, w21, w26, x10
        umaddl  x10, w22, w25, x10
        umaddl  x10, w23, w24, x10

        LOAD_LIMB_MASKED  v0, 0, 16, 32, 48
        LOAD_LIMB_SHIFTED v1, 2, 3, 19, 35, 51
        LOAD_LIMB_SHIFTED v2, 4, 6, 22, 38, 54
        LOAD_LIMB_SHIFTED v3, 6, 9, 25, 41, 57
        LOAD_LIMB_HIBIT   v4, 12, 28, 44, 60

        DOT5_SUM x15, v0, v1, v2, v3, v4, v5, v13, v12, v11, v10
        add     x6, x6, x15
        DOT5_SUM x15, v0, v1, v2, v3, v4, v6, v5, v13, v12, v11
        add     x7, x7, x15
        DOT5_SUM x15, v0, v1, v2, v3, v4, v7, v6, v5, v13, v12
        add     x8, x8, x15
        DOT5_SUM x15, v0, v1, v2, v3, v4, v8, v7, v6, v5, v13
        add     x9, x9, x15
        DOT5_SUM x15, v0, v1, v2, v3, v4, v9, v8, v7, v6, v5
        add     x10, x10, x15

        lsr     x15, x6, #26
        and     x19, x6, x4
        add     x7, x7, x15

        lsr     x15, x7, #26
        and     x20, x7, x4
        add     x8, x8, x15

        lsr     x15, x8, #26
        and     x21, x8, x4
        add     x9, x9, x15

        lsr     x15, x9, #26
        and     x22, x9, x4
        add     x10, x10, x15

        lsr     x15, x10, #26
        and     x23, x10, x4
        add     x19, x19, x15
        add     x19, x19, x15, lsl #2

        lsr     x15, x19, #26
        and     x19, x19, x4
        add     x20, x20, x15

        stp     w19, w20, [x0]
        stp     w21, w22, [x0, #8]
        str     w23, [x0, #16]

        add     x2, x2, #64
        subs    x3, x3, #1
        b.ne    .Lpoly1305_par4_loop

        ldp     x27, x28, [sp, #64]
        ldp     x25, x26, [sp, #48]
        ldp     x23, x24, [sp, #32]
        ldp     x21, x22, [sp, #16]
        ldp     x19, x20, [sp], #80

.Lpoly1305_par4_done:
        ret
.size rscrypto_poly1305_accumulate4_aarch64_linux, .-rscrypto_poly1305_accumulate4_aarch64_linux
