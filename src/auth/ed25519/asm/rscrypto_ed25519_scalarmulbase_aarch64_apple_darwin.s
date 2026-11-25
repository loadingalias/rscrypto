// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - edwards25519_scalarmulbase_alt.S
//
// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.



        .globl _rscrypto_edwards25519_scalarmulbase_alt
        .private_extern _rscrypto_edwards25519_scalarmulbase_alt
        .text
        .balign 4
_rscrypto_edwards25519_scalarmulbase_alt:
        .cfi_startproc
        stp x19, x20, [sp, #-16]! %% .cfi_adjust_cfa_offset 16 %% .cfi_rel_offset x19, 0 %% .cfi_rel_offset x20, 8
        stp x21, x22, [sp, #-16]! %% .cfi_adjust_cfa_offset 16 %% .cfi_rel_offset x21, 0 %% .cfi_rel_offset x22, 8
        stp x23, x24, [sp, #-16]! %% .cfi_adjust_cfa_offset 16 %% .cfi_rel_offset x23, 0 %% .cfi_rel_offset x24, 8
        sub sp, sp, #(14*32 +0) %% .cfi_adjust_cfa_offset 14*32
        mov x23, x0
        ldp x10, x11, [x1]
        ldp x12, x13, [x1, #16]
        lsr x9, x13, #60
        movz x0, #0xd3ed %% movk x0, #0x5cf5, lsl #16 %% movk x0, #0x631a, lsl #32 %% movk x0, #0x5812, lsl #48;
        movz x1, #0x9cd6 %% movk x1, #0xa2f7, lsl #16 %% movk x1, #0xf9de, lsl #32 %% movk x1, #0x14de, lsl #48;
        mul x2, x9, x0
        mul x3, x9, x1
        umulh x4, x9, x0
        umulh x5, x9, x1
        adds x3, x3, x4
        adc x4, x5, xzr
        lsl x5, x9, #60
        subs x10, x10, x2
        sbcs x11, x11, x3
        sbcs x12, x12, x4
        sbcs x13, x13, x5
        csetm x9, cc
        adds xzr, x9, x9
        eor x10, x10, x9
        adcs x10, x10, xzr
        eor x11, x11, x9
        adcs x11, x11, xzr
        eor x12, x12, x9
        adcs x12, x12, xzr
        eor x13, x13, x9
        adc x13, x13, xzr
        and x9, x9, #0x8000000000000000
        orr x13, x13, x9
        stp x10, x11, [sp, #(0*32)]
        tst x13, #0x0800000000000000
        bic x13, x13, #0x0800000000000000
        stp x12, x13, [sp, #(0*32)+16]
        adrp x19, _rscrypto_edwards25519_scalarmulbase_alt_constant@PAGE
        add x19, x19, _rscrypto_edwards25519_scalarmulbase_alt_constant@PAGEOFF
        ldp x0, x1, [x19]
        ldp x2, x3, [x19, #96]
        csel x0, x0, x2, eq
        csel x1, x1, x3, eq
        stp x0, x1, [sp, #(4*32)]
        ldp x0, x1, [x19, #1*16]
        ldp x2, x3, [x19, #(96+1*16)]
        csel x0, x0, x2, eq
        csel x1, x1, x3, eq
        stp x0, x1, [sp, #(4*32)+1*16]
        ldp x0, x1, [x19, #2*16]
        ldp x2, x3, [x19, #(96+2*16)]
        csel x0, x0, x2, eq
        csel x1, x1, x3, eq
        stp x0, x1, [sp, #(4*32)+2*16]
        ldp x0, x1, [x19, #3*16]
        ldp x2, x3, [x19, #(96+3*16)]
        csel x0, x0, x2, eq
        csel x1, x1, x3, eq
        stp x0, x1, [sp, #(4*32)+3*16]
        mov x0, #1
        stp x0, xzr, [sp, #(4*32)+4*16]
        stp xzr, xzr, [sp, #(4*32)+5*16]
        ldp x0, x1, [x19, #4*16]
        ldp x2, x3, [x19, #(96+4*16)]
        csel x0, x0, x2, eq
        csel x1, x1, x3, eq
        stp x0, x1, [sp, #(4*32)+6*16]
        ldp x0, x1, [x19, #5*16]
        ldp x2, x3, [x19, #(96+5*16)]
        csel x0, x0, x2, eq
        csel x1, x1, x3, eq
        stp x0, x1, [sp, #(4*32)+7*16]
        mov x20, 0
        add x19, x19, #192
        mov x21, xzr
Ledwards25519_scalarmulbase_alt_scalarloop:
        lsr x0, x20, #6
        ldr x2, [sp, x0, lsl #3]
        lsr x2, x2, x20
        and x2, x2, #15
        add x22, x2, x21
        cmp x22, 9
        cset x21, cs
        mov x0, 16
        sub x0, x0, x22
        cmp x21, xzr
        csel x22, x0, x22, ne
        mov x0, #1
        mov x1, xzr
        mov x2, xzr
        mov x3, xzr
        mov x4, #1
        mov x5, xzr
        mov x6, xzr
        mov x7, xzr
        mov x8, xzr
        mov x9, xzr
        mov x10, xzr
        mov x11, xzr
        cmp x22, #1
        ldp x12, x13, [x19]
        csel x0, x0, x12, ne
        csel x1, x1, x13, ne
        ldp x12, x13, [x19, #16]
        csel x2, x2, x12, ne
        csel x3, x3, x13, ne
        ldp x12, x13, [x19, #32]
        csel x4, x4, x12, ne
        csel x5, x5, x13, ne
        ldp x12, x13, [x19, #48]
        csel x6, x6, x12, ne
        csel x7, x7, x13, ne
        ldp x12, x13, [x19, #64]
        csel x8, x8, x12, ne
        csel x9, x9, x13, ne
        ldp x12, x13, [x19, #80]
        csel x10, x10, x12, ne
        csel x11, x11, x13, ne
        add x19, x19, #96
        cmp x22, #2
        ldp x12, x13, [x19]
        csel x0, x0, x12, ne
        csel x1, x1, x13, ne
        ldp x12, x13, [x19, #16]
        csel x2, x2, x12, ne
        csel x3, x3, x13, ne
        ldp x12, x13, [x19, #32]
        csel x4, x4, x12, ne
        csel x5, x5, x13, ne
        ldp x12, x13, [x19, #48]
        csel x6, x6, x12, ne
        csel x7, x7, x13, ne
        ldp x12, x13, [x19, #64]
        csel x8, x8, x12, ne
        csel x9, x9, x13, ne
        ldp x12, x13, [x19, #80]
        csel x10, x10, x12, ne
        csel x11, x11, x13, ne
        add x19, x19, #96
        cmp x22, #3
        ldp x12, x13, [x19]
        csel x0, x0, x12, ne
        csel x1, x1, x13, ne
        ldp x12, x13, [x19, #16]
        csel x2, x2, x12, ne
        csel x3, x3, x13, ne
        ldp x12, x13, [x19, #32]
        csel x4, x4, x12, ne
        csel x5, x5, x13, ne
        ldp x12, x13, [x19, #48]
        csel x6, x6, x12, ne
        csel x7, x7, x13, ne
        ldp x12, x13, [x19, #64]
        csel x8, x8, x12, ne
        csel x9, x9, x13, ne
        ldp x12, x13, [x19, #80]
        csel x10, x10, x12, ne
        csel x11, x11, x13, ne
        add x19, x19, #96
        cmp x22, #4
        ldp x12, x13, [x19]
        csel x0, x0, x12, ne
        csel x1, x1, x13, ne
        ldp x12, x13, [x19, #16]
        csel x2, x2, x12, ne
        csel x3, x3, x13, ne
        ldp x12, x13, [x19, #32]
        csel x4, x4, x12, ne
        csel x5, x5, x13, ne
        ldp x12, x13, [x19, #48]
        csel x6, x6, x12, ne
        csel x7, x7, x13, ne
        ldp x12, x13, [x19, #64]
        csel x8, x8, x12, ne
        csel x9, x9, x13, ne
        ldp x12, x13, [x19, #80]
        csel x10, x10, x12, ne
        csel x11, x11, x13, ne
        add x19, x19, #96
        cmp x22, #5
        ldp x12, x13, [x19]
        csel x0, x0, x12, ne
        csel x1, x1, x13, ne
        ldp x12, x13, [x19, #16]
        csel x2, x2, x12, ne
        csel x3, x3, x13, ne
        ldp x12, x13, [x19, #32]
        csel x4, x4, x12, ne
        csel x5, x5, x13, ne
        ldp x12, x13, [x19, #48]
        csel x6, x6, x12, ne
        csel x7, x7, x13, ne
        ldp x12, x13, [x19, #64]
        csel x8, x8, x12, ne
        csel x9, x9, x13, ne
        ldp x12, x13, [x19, #80]
        csel x10, x10, x12, ne
        csel x11, x11, x13, ne
        add x19, x19, #96
        cmp x22, #6
        ldp x12, x13, [x19]
        csel x0, x0, x12, ne
        csel x1, x1, x13, ne
        ldp x12, x13, [x19, #16]
        csel x2, x2, x12, ne
        csel x3, x3, x13, ne
        ldp x12, x13, [x19, #32]
        csel x4, x4, x12, ne
        csel x5, x5, x13, ne
        ldp x12, x13, [x19, #48]
        csel x6, x6, x12, ne
        csel x7, x7, x13, ne
        ldp x12, x13, [x19, #64]
        csel x8, x8, x12, ne
        csel x9, x9, x13, ne
        ldp x12, x13, [x19, #80]
        csel x10, x10, x12, ne
        csel x11, x11, x13, ne
        add x19, x19, #96
        cmp x22, #7
        ldp x12, x13, [x19]
        csel x0, x0, x12, ne
        csel x1, x1, x13, ne
        ldp x12, x13, [x19, #16]
        csel x2, x2, x12, ne
        csel x3, x3, x13, ne
        ldp x12, x13, [x19, #32]
        csel x4, x4, x12, ne
        csel x5, x5, x13, ne
        ldp x12, x13, [x19, #48]
        csel x6, x6, x12, ne
        csel x7, x7, x13, ne
        ldp x12, x13, [x19, #64]
        csel x8, x8, x12, ne
        csel x9, x9, x13, ne
        ldp x12, x13, [x19, #80]
        csel x10, x10, x12, ne
        csel x11, x11, x13, ne
        add x19, x19, #96
        cmp x22, #8
        ldp x12, x13, [x19]
        csel x0, x0, x12, ne
        csel x1, x1, x13, ne
        ldp x12, x13, [x19, #16]
        csel x2, x2, x12, ne
        csel x3, x3, x13, ne
        ldp x12, x13, [x19, #32]
        csel x4, x4, x12, ne
        csel x5, x5, x13, ne
        ldp x12, x13, [x19, #48]
        csel x6, x6, x12, ne
        csel x7, x7, x13, ne
        ldp x12, x13, [x19, #64]
        csel x8, x8, x12, ne
        csel x9, x9, x13, ne
        ldp x12, x13, [x19, #80]
        csel x10, x10, x12, ne
        csel x11, x11, x13, ne
        add x19, x19, #96
        cmp x21, #0
        csel x12, x0, x4, eq
        csel x13, x1, x5, eq
        csel x14, x2, x6, eq
        csel x15, x3, x7, eq
        stp x12, x13, [sp, #(1*32)]
        stp x14, x15, [sp, #(1*32)+16]
        csel x12, x0, x4, ne
        csel x13, x1, x5, ne
        csel x14, x2, x6, ne
        csel x15, x3, x7, ne
        stp x12, x13, [sp, #(1*32)+32]
        stp x14, x15, [sp, #(1*32)+48]
        mov x0, #-19
        subs x0, x0, x8
        mov x2, #-1
        sbcs x1, x2, x9
        sbcs x2, x2, x10
        mov x3, #0x7FFFFFFFFFFFFFFF
        sbc x3, x3, x11
        cmp x22, xzr
        ccmp x21, xzr, #4, ne
        csel x0, x0, x8, ne
        csel x1, x1, x9, ne
        stp x0, x1, [sp, #(1*32)+64]
        csel x2, x2, x10, ne
        csel x3, x3, x11, ne
        stp x2, x3, [sp, #(1*32)+80]
        ldp x3, x4, [sp, #(6*32)] %% adds x3, x3, x3 %% adcs x4, x4, x4 %% ldp x5, x6, [sp, #(6*32)+16] %% adcs x5, x5, x5 %% adcs x6, x6, x6 %% mov x9, #38 %% csel x9, x9, xzr, cs %% adds x3, x3, x9 %% adcs x4, x4, xzr %% adcs x5, x5, xzr %% adc x6, x6, xzr %% stp x3, x4, [sp, #(8*32)] %% stp x5, x6, [sp, #(8*32)+16]
        ldp x5, x6, [sp, #(5*32)] %% ldp x4, x3, [sp, #(4*32)] %% subs x5, x5, x4 %% sbcs x6, x6, x3 %% ldp x7, x8, [sp, #(5*32)+16] %% ldp x4, x3, [sp, #(4*32)+16] %% sbcs x7, x7, x4 %% sbcs x8, x8, x3 %% mov x4, #38 %% csel x3, x4, xzr, lo %% subs x5, x5, x3 %% sbcs x6, x6, xzr %% sbcs x7, x7, xzr %% sbc x8, x8, xzr %% stp x5, x6, [sp, #(9*32)] %% stp x7, x8, [sp, #(9*32)+16]
        ldp x3, x4, [sp, #(5*32)] %% ldp x7, x8, [sp, #(4*32)] %% adds x3, x3, x7 %% adcs x4, x4, x8 %% ldp x5, x6, [sp, #(5*32)+16] %% ldp x7, x8, [sp, #(4*32)+16] %% adcs x5, x5, x7 %% adcs x6, x6, x8 %% mov x9, #38 %% csel x9, x9, xzr, cs %% adds x3, x3, x9 %% adcs x4, x4, xzr %% adcs x5, x5, xzr %% adc x6, x6, xzr %% stp x3, x4, [sp, #(10*32)] %% stp x5, x6, [sp, #(10*32)+16]
        ldp x3, x4, [sp, #(7*32)] %% ldp x7, x8, [sp, #(3*32)] %% mul x12, x3, x7 %% umulh x13, x3, x7 %% mul x11, x3, x8 %% umulh x14, x3, x8 %% adds x13, x13, x11 %% ldp x9, x10, [sp, #(3*32)+16] %% mul x11, x3, x9 %% umulh x15, x3, x9 %% adcs x14, x14, x11 %% mul x11, x3, x10 %% umulh x16, x3, x10 %% adcs x15, x15, x11 %% adc x16, x16, xzr %% ldp x5, x6, [sp, #(7*32)+16] %% mul x11, x4, x7 %% adds x13, x13, x11 %% mul x11, x4, x8 %% adcs x14, x14, x11 %% mul x11, x4, x9 %% adcs x15, x15, x11 %% mul x11, x4, x10 %% adcs x16, x16, x11 %% umulh x3, x4, x10 %% adc x3, x3, xzr %% umulh x11, x4, x7 %% adds x14, x14, x11 %% umulh x11, x4, x8 %% adcs x15, x15, x11 %% umulh x11, x4, x9 %% adcs x16, x16, x11 %% adc x3, x3, xzr %% mul x11, x5, x7 %% adds x14, x14, x11 %% mul x11, x5, x8 %% adcs x15, x15, x11 %% mul x11, x5, x9 %% adcs x16, x16, x11 %% mul x11, x5, x10 %% adcs x3, x3, x11 %% umulh x4, x5, x10 %% adc x4, x4, xzr %% umulh x11, x5, x7 %% adds x15, x15, x11 %% umulh x11, x5, x8 %% adcs x16, x16, x11 %% umulh x11, x5, x9 %% adcs x3, x3, x11 %% adc x4, x4, xzr %% mul x11, x6, x7 %% adds x15, x15, x11 %% mul x11, x6, x8 %% adcs x16, x16, x11 %% mul x11, x6, x9 %% adcs x3, x3, x11 %% mul x11, x6, x10 %% adcs x4, x4, x11 %% umulh x5, x6, x10 %% adc x5, x5, xzr %% umulh x11, x6, x7 %% adds x16, x16, x11 %% umulh x11, x6, x8 %% adcs x3, x3, x11 %% umulh x11, x6, x9 %% adcs x4, x4, x11 %% adc x5, x5, xzr %% mov x7, #0x26 %% mul x11, x7, x16 %% umulh x9, x7, x16 %% adds x12, x12, x11 %% mul x11, x7, x3 %% umulh x3, x7, x3 %% adcs x13, x13, x11 %% mul x11, x7, x4 %% umulh x4, x7, x4 %% adcs x14, x14, x11 %% mul x11, x7, x5 %% umulh x5, x7, x5 %% adcs x15, x15, x11 %% cset x16, cs %% adds x15, x15, x4 %% adc x16, x16, x5 %% cmn x15, x15 %% bic x15, x15, #0x8000000000000000 %% adc x8, x16, x16 %% mov x7, #0x13 %% mul x11, x7, x8 %% adds x12, x12, x11 %% adcs x13, x13, x9 %% adcs x14, x14, x3 %% adc x15, x15, xzr %% stp x12, x13, [sp, #(11*32)] %% stp x14, x15, [sp, #(11*32)+16]
        ldp x3, x4, [sp, #(9*32)] %% ldp x7, x8, [sp, #(1*32)] %% mul x12, x3, x7 %% umulh x13, x3, x7 %% mul x11, x3, x8 %% umulh x14, x3, x8 %% adds x13, x13, x11 %% ldp x9, x10, [sp, #(1*32)+16] %% mul x11, x3, x9 %% umulh x15, x3, x9 %% adcs x14, x14, x11 %% mul x11, x3, x10 %% umulh x16, x3, x10 %% adcs x15, x15, x11 %% adc x16, x16, xzr %% ldp x5, x6, [sp, #(9*32)+16] %% mul x11, x4, x7 %% adds x13, x13, x11 %% mul x11, x4, x8 %% adcs x14, x14, x11 %% mul x11, x4, x9 %% adcs x15, x15, x11 %% mul x11, x4, x10 %% adcs x16, x16, x11 %% umulh x3, x4, x10 %% adc x3, x3, xzr %% umulh x11, x4, x7 %% adds x14, x14, x11 %% umulh x11, x4, x8 %% adcs x15, x15, x11 %% umulh x11, x4, x9 %% adcs x16, x16, x11 %% adc x3, x3, xzr %% mul x11, x5, x7 %% adds x14, x14, x11 %% mul x11, x5, x8 %% adcs x15, x15, x11 %% mul x11, x5, x9 %% adcs x16, x16, x11 %% mul x11, x5, x10 %% adcs x3, x3, x11 %% umulh x4, x5, x10 %% adc x4, x4, xzr %% umulh x11, x5, x7 %% adds x15, x15, x11 %% umulh x11, x5, x8 %% adcs x16, x16, x11 %% umulh x11, x5, x9 %% adcs x3, x3, x11 %% adc x4, x4, xzr %% mul x11, x6, x7 %% adds x15, x15, x11 %% mul x11, x6, x8 %% adcs x16, x16, x11 %% mul x11, x6, x9 %% adcs x3, x3, x11 %% mul x11, x6, x10 %% adcs x4, x4, x11 %% umulh x5, x6, x10 %% adc x5, x5, xzr %% umulh x11, x6, x7 %% adds x16, x16, x11 %% umulh x11, x6, x8 %% adcs x3, x3, x11 %% umulh x11, x6, x9 %% adcs x4, x4, x11 %% adc x5, x5, xzr %% mov x7, #0x26 %% mul x11, x7, x16 %% umulh x9, x7, x16 %% adds x12, x12, x11 %% mul x11, x7, x3 %% umulh x3, x7, x3 %% adcs x13, x13, x11 %% mul x11, x7, x4 %% umulh x4, x7, x4 %% adcs x14, x14, x11 %% mul x11, x7, x5 %% umulh x5, x7, x5 %% adcs x15, x15, x11 %% cset x16, cs %% adds x15, x15, x4 %% adc x16, x16, x5 %% cmn x15, x15 %% bic x15, x15, #0x8000000000000000 %% adc x8, x16, x16 %% mov x7, #0x13 %% mul x11, x7, x8 %% adds x12, x12, x11 %% adcs x13, x13, x9 %% adcs x14, x14, x3 %% adc x15, x15, xzr %% stp x12, x13, [sp, #(9*32)] %% stp x14, x15, [sp, #(9*32)+16]
        ldp x3, x4, [sp, #(10*32)] %% ldp x7, x8, [sp, #(2*32)] %% mul x12, x3, x7 %% umulh x13, x3, x7 %% mul x11, x3, x8 %% umulh x14, x3, x8 %% adds x13, x13, x11 %% ldp x9, x10, [sp, #(2*32)+16] %% mul x11, x3, x9 %% umulh x15, x3, x9 %% adcs x14, x14, x11 %% mul x11, x3, x10 %% umulh x16, x3, x10 %% adcs x15, x15, x11 %% adc x16, x16, xzr %% ldp x5, x6, [sp, #(10*32)+16] %% mul x11, x4, x7 %% adds x13, x13, x11 %% mul x11, x4, x8 %% adcs x14, x14, x11 %% mul x11, x4, x9 %% adcs x15, x15, x11 %% mul x11, x4, x10 %% adcs x16, x16, x11 %% umulh x3, x4, x10 %% adc x3, x3, xzr %% umulh x11, x4, x7 %% adds x14, x14, x11 %% umulh x11, x4, x8 %% adcs x15, x15, x11 %% umulh x11, x4, x9 %% adcs x16, x16, x11 %% adc x3, x3, xzr %% mul x11, x5, x7 %% adds x14, x14, x11 %% mul x11, x5, x8 %% adcs x15, x15, x11 %% mul x11, x5, x9 %% adcs x16, x16, x11 %% mul x11, x5, x10 %% adcs x3, x3, x11 %% umulh x4, x5, x10 %% adc x4, x4, xzr %% umulh x11, x5, x7 %% adds x15, x15, x11 %% umulh x11, x5, x8 %% adcs x16, x16, x11 %% umulh x11, x5, x9 %% adcs x3, x3, x11 %% adc x4, x4, xzr %% mul x11, x6, x7 %% adds x15, x15, x11 %% mul x11, x6, x8 %% adcs x16, x16, x11 %% mul x11, x6, x9 %% adcs x3, x3, x11 %% mul x11, x6, x10 %% adcs x4, x4, x11 %% umulh x5, x6, x10 %% adc x5, x5, xzr %% umulh x11, x6, x7 %% adds x16, x16, x11 %% umulh x11, x6, x8 %% adcs x3, x3, x11 %% umulh x11, x6, x9 %% adcs x4, x4, x11 %% adc x5, x5, xzr %% mov x7, #0x26 %% mul x11, x7, x16 %% umulh x9, x7, x16 %% adds x12, x12, x11 %% mul x11, x7, x3 %% umulh x3, x7, x3 %% adcs x13, x13, x11 %% mul x11, x7, x4 %% umulh x4, x7, x4 %% adcs x14, x14, x11 %% mul x11, x7, x5 %% umulh x5, x7, x5 %% adcs x15, x15, x11 %% cset x16, cs %% adds x15, x15, x4 %% adc x16, x16, x5 %% cmn x15, x15 %% bic x15, x15, #0x8000000000000000 %% adc x8, x16, x16 %% mov x7, #0x13 %% mul x11, x7, x8 %% adds x12, x12, x11 %% adcs x13, x13, x9 %% adcs x14, x14, x3 %% adc x15, x15, xzr %% stp x12, x13, [sp, #(10*32)] %% stp x14, x15, [sp, #(10*32)+16]
        ldp x5, x6, [sp, #(8*32)] %% ldp x4, x3, [sp, #(11*32)] %% subs x5, x5, x4 %% sbcs x6, x6, x3 %% ldp x7, x8, [sp, #(8*32)+16] %% ldp x4, x3, [sp, #(11*32)+16] %% sbcs x7, x7, x4 %% sbcs x8, x8, x3 %% mov x4, #38 %% csel x3, x4, xzr, lo %% subs x5, x5, x3 %% sbcs x6, x6, xzr %% sbcs x7, x7, xzr %% sbc x8, x8, xzr %% stp x5, x6, [sp, #(12*32)] %% stp x7, x8, [sp, #(12*32)+16]
        ldp x3, x4, [sp, #(8*32)] %% ldp x7, x8, [sp, #(11*32)] %% adds x3, x3, x7 %% adcs x4, x4, x8 %% ldp x5, x6, [sp, #(8*32)+16] %% ldp x7, x8, [sp, #(11*32)+16] %% adcs x5, x5, x7 %% adcs x6, x6, x8 %% mov x9, #38 %% csel x9, x9, xzr, cs %% adds x3, x3, x9 %% adcs x4, x4, xzr %% adcs x5, x5, xzr %% adc x6, x6, xzr %% stp x3, x4, [sp, #(8*32)] %% stp x5, x6, [sp, #(8*32)+16]
        ldp x5, x6, [sp, #(10*32)] %% ldp x4, x3, [sp, #(9*32)] %% subs x5, x5, x4 %% sbcs x6, x6, x3 %% ldp x7, x8, [sp, #(10*32)+16] %% ldp x4, x3, [sp, #(9*32)+16] %% sbcs x7, x7, x4 %% sbcs x8, x8, x3 %% mov x4, #38 %% csel x3, x4, xzr, lo %% subs x5, x5, x3 %% sbcs x6, x6, xzr %% sbcs x7, x7, xzr %% sbc x8, x8, xzr %% stp x5, x6, [sp, #(13*32)] %% stp x7, x8, [sp, #(13*32)+16]
        ldp x3, x4, [sp, #(10*32)] %% ldp x7, x8, [sp, #(9*32)] %% adds x3, x3, x7 %% adcs x4, x4, x8 %% ldp x5, x6, [sp, #(10*32)+16] %% ldp x7, x8, [sp, #(9*32)+16] %% adcs x5, x5, x7 %% adcs x6, x6, x8 %% mov x9, #38 %% csel x9, x9, xzr, cs %% adds x3, x3, x9 %% adcs x4, x4, xzr %% adcs x5, x5, xzr %% adc x6, x6, xzr %% stp x3, x4, [sp, #(9*32)] %% stp x5, x6, [sp, #(9*32)+16]
        ldp x3, x4, [sp, #(12*32)] %% ldp x7, x8, [sp, #(8*32)] %% mul x12, x3, x7 %% umulh x13, x3, x7 %% mul x11, x3, x8 %% umulh x14, x3, x8 %% adds x13, x13, x11 %% ldp x9, x10, [sp, #(8*32)+16] %% mul x11, x3, x9 %% umulh x15, x3, x9 %% adcs x14, x14, x11 %% mul x11, x3, x10 %% umulh x16, x3, x10 %% adcs x15, x15, x11 %% adc x16, x16, xzr %% ldp x5, x6, [sp, #(12*32)+16] %% mul x11, x4, x7 %% adds x13, x13, x11 %% mul x11, x4, x8 %% adcs x14, x14, x11 %% mul x11, x4, x9 %% adcs x15, x15, x11 %% mul x11, x4, x10 %% adcs x16, x16, x11 %% umulh x3, x4, x10 %% adc x3, x3, xzr %% umulh x11, x4, x7 %% adds x14, x14, x11 %% umulh x11, x4, x8 %% adcs x15, x15, x11 %% umulh x11, x4, x9 %% adcs x16, x16, x11 %% adc x3, x3, xzr %% mul x11, x5, x7 %% adds x14, x14, x11 %% mul x11, x5, x8 %% adcs x15, x15, x11 %% mul x11, x5, x9 %% adcs x16, x16, x11 %% mul x11, x5, x10 %% adcs x3, x3, x11 %% umulh x4, x5, x10 %% adc x4, x4, xzr %% umulh x11, x5, x7 %% adds x15, x15, x11 %% umulh x11, x5, x8 %% adcs x16, x16, x11 %% umulh x11, x5, x9 %% adcs x3, x3, x11 %% adc x4, x4, xzr %% mul x11, x6, x7 %% adds x15, x15, x11 %% mul x11, x6, x8 %% adcs x16, x16, x11 %% mul x11, x6, x9 %% adcs x3, x3, x11 %% mul x11, x6, x10 %% adcs x4, x4, x11 %% umulh x5, x6, x10 %% adc x5, x5, xzr %% umulh x11, x6, x7 %% adds x16, x16, x11 %% umulh x11, x6, x8 %% adcs x3, x3, x11 %% umulh x11, x6, x9 %% adcs x4, x4, x11 %% adc x5, x5, xzr %% mov x7, #0x26 %% mul x11, x7, x16 %% umulh x9, x7, x16 %% adds x12, x12, x11 %% mul x11, x7, x3 %% umulh x3, x7, x3 %% adcs x13, x13, x11 %% mul x11, x7, x4 %% umulh x4, x7, x4 %% adcs x14, x14, x11 %% mul x11, x7, x5 %% umulh x5, x7, x5 %% adcs x15, x15, x11 %% cset x16, cs %% adds x15, x15, x4 %% adc x16, x16, x5 %% cmn x15, x15 %% bic x15, x15, #0x8000000000000000 %% adc x8, x16, x16 %% mov x7, #0x13 %% mul x11, x7, x8 %% adds x12, x12, x11 %% adcs x13, x13, x9 %% adcs x14, x14, x3 %% adc x15, x15, xzr %% stp x12, x13, [sp, #(6*32)] %% stp x14, x15, [sp, #(6*32)+16]
        ldp x3, x4, [sp, #(13*32)] %% ldp x7, x8, [sp, #(12*32)] %% mul x12, x3, x7 %% umulh x13, x3, x7 %% mul x11, x3, x8 %% umulh x14, x3, x8 %% adds x13, x13, x11 %% ldp x9, x10, [sp, #(12*32)+16] %% mul x11, x3, x9 %% umulh x15, x3, x9 %% adcs x14, x14, x11 %% mul x11, x3, x10 %% umulh x16, x3, x10 %% adcs x15, x15, x11 %% adc x16, x16, xzr %% ldp x5, x6, [sp, #(13*32)+16] %% mul x11, x4, x7 %% adds x13, x13, x11 %% mul x11, x4, x8 %% adcs x14, x14, x11 %% mul x11, x4, x9 %% adcs x15, x15, x11 %% mul x11, x4, x10 %% adcs x16, x16, x11 %% umulh x3, x4, x10 %% adc x3, x3, xzr %% umulh x11, x4, x7 %% adds x14, x14, x11 %% umulh x11, x4, x8 %% adcs x15, x15, x11 %% umulh x11, x4, x9 %% adcs x16, x16, x11 %% adc x3, x3, xzr %% mul x11, x5, x7 %% adds x14, x14, x11 %% mul x11, x5, x8 %% adcs x15, x15, x11 %% mul x11, x5, x9 %% adcs x16, x16, x11 %% mul x11, x5, x10 %% adcs x3, x3, x11 %% umulh x4, x5, x10 %% adc x4, x4, xzr %% umulh x11, x5, x7 %% adds x15, x15, x11 %% umulh x11, x5, x8 %% adcs x16, x16, x11 %% umulh x11, x5, x9 %% adcs x3, x3, x11 %% adc x4, x4, xzr %% mul x11, x6, x7 %% adds x15, x15, x11 %% mul x11, x6, x8 %% adcs x16, x16, x11 %% mul x11, x6, x9 %% adcs x3, x3, x11 %% mul x11, x6, x10 %% adcs x4, x4, x11 %% umulh x5, x6, x10 %% adc x5, x5, xzr %% umulh x11, x6, x7 %% adds x16, x16, x11 %% umulh x11, x6, x8 %% adcs x3, x3, x11 %% umulh x11, x6, x9 %% adcs x4, x4, x11 %% adc x5, x5, xzr %% mov x7, #0x26 %% mul x11, x7, x16 %% umulh x9, x7, x16 %% adds x12, x12, x11 %% mul x11, x7, x3 %% umulh x3, x7, x3 %% adcs x13, x13, x11 %% mul x11, x7, x4 %% umulh x4, x7, x4 %% adcs x14, x14, x11 %% mul x11, x7, x5 %% umulh x5, x7, x5 %% adcs x15, x15, x11 %% cset x16, cs %% adds x15, x15, x4 %% adc x16, x16, x5 %% cmn x15, x15 %% bic x15, x15, #0x8000000000000000 %% adc x8, x16, x16 %% mov x7, #0x13 %% mul x11, x7, x8 %% adds x12, x12, x11 %% adcs x13, x13, x9 %% adcs x14, x14, x3 %% adc x15, x15, xzr %% stp x12, x13, [sp, #(4*32)] %% stp x14, x15, [sp, #(4*32)+16]
        ldp x3, x4, [sp, #(8*32)] %% ldp x7, x8, [sp, #(9*32)] %% mul x12, x3, x7 %% umulh x13, x3, x7 %% mul x11, x3, x8 %% umulh x14, x3, x8 %% adds x13, x13, x11 %% ldp x9, x10, [sp, #(9*32)+16] %% mul x11, x3, x9 %% umulh x15, x3, x9 %% adcs x14, x14, x11 %% mul x11, x3, x10 %% umulh x16, x3, x10 %% adcs x15, x15, x11 %% adc x16, x16, xzr %% ldp x5, x6, [sp, #(8*32)+16] %% mul x11, x4, x7 %% adds x13, x13, x11 %% mul x11, x4, x8 %% adcs x14, x14, x11 %% mul x11, x4, x9 %% adcs x15, x15, x11 %% mul x11, x4, x10 %% adcs x16, x16, x11 %% umulh x3, x4, x10 %% adc x3, x3, xzr %% umulh x11, x4, x7 %% adds x14, x14, x11 %% umulh x11, x4, x8 %% adcs x15, x15, x11 %% umulh x11, x4, x9 %% adcs x16, x16, x11 %% adc x3, x3, xzr %% mul x11, x5, x7 %% adds x14, x14, x11 %% mul x11, x5, x8 %% adcs x15, x15, x11 %% mul x11, x5, x9 %% adcs x16, x16, x11 %% mul x11, x5, x10 %% adcs x3, x3, x11 %% umulh x4, x5, x10 %% adc x4, x4, xzr %% umulh x11, x5, x7 %% adds x15, x15, x11 %% umulh x11, x5, x8 %% adcs x16, x16, x11 %% umulh x11, x5, x9 %% adcs x3, x3, x11 %% adc x4, x4, xzr %% mul x11, x6, x7 %% adds x15, x15, x11 %% mul x11, x6, x8 %% adcs x16, x16, x11 %% mul x11, x6, x9 %% adcs x3, x3, x11 %% mul x11, x6, x10 %% adcs x4, x4, x11 %% umulh x5, x6, x10 %% adc x5, x5, xzr %% umulh x11, x6, x7 %% adds x16, x16, x11 %% umulh x11, x6, x8 %% adcs x3, x3, x11 %% umulh x11, x6, x9 %% adcs x4, x4, x11 %% adc x5, x5, xzr %% mov x7, #0x26 %% mul x11, x7, x16 %% umulh x9, x7, x16 %% adds x12, x12, x11 %% mul x11, x7, x3 %% umulh x3, x7, x3 %% adcs x13, x13, x11 %% mul x11, x7, x4 %% umulh x4, x7, x4 %% adcs x14, x14, x11 %% mul x11, x7, x5 %% umulh x5, x7, x5 %% adcs x15, x15, x11 %% cset x16, cs %% adds x15, x15, x4 %% adc x16, x16, x5 %% cmn x15, x15 %% bic x15, x15, #0x8000000000000000 %% adc x8, x16, x16 %% mov x7, #0x13 %% mul x11, x7, x8 %% adds x12, x12, x11 %% adcs x13, x13, x9 %% adcs x14, x14, x3 %% adc x15, x15, xzr %% stp x12, x13, [sp, #(5*32)] %% stp x14, x15, [sp, #(5*32)+16]
        ldp x3, x4, [sp, #(13*32)] %% ldp x7, x8, [sp, #(9*32)] %% mul x12, x3, x7 %% umulh x13, x3, x7 %% mul x11, x3, x8 %% umulh x14, x3, x8 %% adds x13, x13, x11 %% ldp x9, x10, [sp, #(9*32)+16] %% mul x11, x3, x9 %% umulh x15, x3, x9 %% adcs x14, x14, x11 %% mul x11, x3, x10 %% umulh x16, x3, x10 %% adcs x15, x15, x11 %% adc x16, x16, xzr %% ldp x5, x6, [sp, #(13*32)+16] %% mul x11, x4, x7 %% adds x13, x13, x11 %% mul x11, x4, x8 %% adcs x14, x14, x11 %% mul x11, x4, x9 %% adcs x15, x15, x11 %% mul x11, x4, x10 %% adcs x16, x16, x11 %% umulh x3, x4, x10 %% adc x3, x3, xzr %% umulh x11, x4, x7 %% adds x14, x14, x11 %% umulh x11, x4, x8 %% adcs x15, x15, x11 %% umulh x11, x4, x9 %% adcs x16, x16, x11 %% adc x3, x3, xzr %% mul x11, x5, x7 %% adds x14, x14, x11 %% mul x11, x5, x8 %% adcs x15, x15, x11 %% mul x11, x5, x9 %% adcs x16, x16, x11 %% mul x11, x5, x10 %% adcs x3, x3, x11 %% umulh x4, x5, x10 %% adc x4, x4, xzr %% umulh x11, x5, x7 %% adds x15, x15, x11 %% umulh x11, x5, x8 %% adcs x16, x16, x11 %% umulh x11, x5, x9 %% adcs x3, x3, x11 %% adc x4, x4, xzr %% mul x11, x6, x7 %% adds x15, x15, x11 %% mul x11, x6, x8 %% adcs x16, x16, x11 %% mul x11, x6, x9 %% adcs x3, x3, x11 %% mul x11, x6, x10 %% adcs x4, x4, x11 %% umulh x5, x6, x10 %% adc x5, x5, xzr %% umulh x11, x6, x7 %% adds x16, x16, x11 %% umulh x11, x6, x8 %% adcs x3, x3, x11 %% umulh x11, x6, x9 %% adcs x4, x4, x11 %% adc x5, x5, xzr %% mov x7, #0x26 %% mul x11, x7, x16 %% umulh x9, x7, x16 %% adds x12, x12, x11 %% mul x11, x7, x3 %% umulh x3, x7, x3 %% adcs x13, x13, x11 %% mul x11, x7, x4 %% umulh x4, x7, x4 %% adcs x14, x14, x11 %% mul x11, x7, x5 %% umulh x5, x7, x5 %% adcs x15, x15, x11 %% cset x16, cs %% adds x15, x15, x4 %% adc x16, x16, x5 %% cmn x15, x15 %% bic x15, x15, #0x8000000000000000 %% adc x8, x16, x16 %% mov x7, #0x13 %% mul x11, x7, x8 %% adds x12, x12, x11 %% adcs x13, x13, x9 %% adcs x14, x14, x3 %% adc x15, x15, xzr %% stp x12, x13, [sp, #(7*32)] %% stp x14, x15, [sp, #(7*32)+16]
        add x20, x20, 4
        cmp x20, 252
        bcc Ledwards25519_scalarmulbase_alt_scalarloop
        ldp x0, x1, [sp, #(4*32)]
        ldp x2, x3, [sp, #(4*32)+16]
        mov x4, #0xffffffffffffffda
        subs x4, x4, x0
        mov x7, #0xffffffffffffffff
        sbcs x5, x7, x1
        sbcs x6, x7, x2
        sbc x7, x7, x3
        ldr x10, [sp, #(0*32)+24]
        tst x10, #0x8000000000000000
        csel x0, x4, x0, ne
        csel x1, x5, x1, ne
        csel x2, x6, x2, ne
        csel x3, x7, x3, ne
        stp x0, x1, [sp, #(4*32)]
        stp x2, x3, [sp, #(4*32)+16]
        add x0, sp, #(7*32)
        add x1, sp, #(6*32)
        mov x20, x0
        mov x10, #0xffffffffffffffed
        mov x11, #0xffffffffffffffff
        stp x10, x11, [sp]
        mov x12, #0x7fffffffffffffff
        stp x11, x12, [sp, #16]
        ldp x2, x3, [x1]
        ldp x4, x5, [x1, #16]
        mov x7, #0x13
        lsr x6, x5, #63
        madd x6, x7, x6, x7
        adds x2, x2, x6
        adcs x3, x3, xzr
        adcs x4, x4, xzr
        orr x5, x5, #0x8000000000000000
        adcs x5, x5, xzr
        csel x6, x7, xzr, cc
        subs x2, x2, x6
        sbcs x3, x3, xzr
        sbcs x4, x4, xzr
        sbc x5, x5, xzr
        and x5, x5, #0x7fffffffffffffff
        stp x2, x3, [sp, #32]
        stp x4, x5, [sp, #48]
        stp xzr, xzr, [sp, #64]
        stp xzr, xzr, [sp, #80]
        mov x10, #0x2099
        movk x10, #0x7502, lsl #16
        movk x10, #0x9e23, lsl #32
        movk x10, #0xa0f9, lsl #48
        mov x11, #0x2595
        movk x11, #0x1d13, lsl #16
        movk x11, #0x8f3f, lsl #32
        movk x11, #0xa8c6, lsl #48
        mov x12, #0x5242
        movk x12, #0x5ac, lsl #16
        movk x12, #0x8938, lsl #32
        movk x12, #0x6c6c, lsl #48
        mov x13, #0x615
        movk x13, #0x4177, lsl #16
        movk x13, #0x8b2, lsl #32
        movk x13, #0x2765, lsl #48
        stp x10, x11, [sp, #96]
        stp x12, x13, [sp, #112]
        mov x21, #0xa
        mov x22, #0x1
        b Ledwards25519_scalarmulbase_alt_invmidloop
Ledwards25519_scalarmulbase_alt_invloop:
        cmp x10, xzr
        csetm x14, mi
        cneg x10, x10, mi
        cmp x11, xzr
        csetm x15, mi
        cneg x11, x11, mi
        cmp x12, xzr
        csetm x16, mi
        cneg x12, x12, mi
        cmp x13, xzr
        csetm x17, mi
        cneg x13, x13, mi
        and x0, x10, x14
        and x1, x11, x15
        add x9, x0, x1
        and x0, x12, x16
        and x1, x13, x17
        add x19, x0, x1
        ldr x7, [sp]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x4, x9, x0
        adc x2, xzr, x1
        ldr x8, [sp, #32]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x4, x4, x0
        adc x2, x2, x1
        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x5, x19, x0
        adc x3, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x5, x5, x0
        adc x3, x3, x1
        ldr x7, [sp, #8]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x2, x2, x0
        adc x6, xzr, x1
        ldr x8, [sp, #40]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x2, x2, x0
        adc x6, x6, x1
        extr x4, x2, x4, #59
        str x4, [sp]
        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x3, x3, x0
        adc x4, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x3, x3, x0
        adc x4, x4, x1
        extr x5, x3, x5, #59
        str x5, [sp, #32]
        ldr x7, [sp, #16]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x6, x6, x0
        adc x5, xzr, x1
        ldr x8, [sp, #48]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x6, x6, x0
        adc x5, x5, x1
        extr x2, x6, x2, #59
        str x2, [sp, #8]
        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x4, x4, x0
        adc x2, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x4, x4, x0
        adc x2, x2, x1
        extr x3, x4, x3, #59
        str x3, [sp, #40]
        ldr x7, [sp, #24]
        eor x1, x7, x14
        asr x3, x1, #63
        and x3, x3, x10
        neg x3, x3
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x5, x5, x0
        adc x3, x3, x1
        ldr x8, [sp, #56]
        eor x1, x8, x15
        asr x0, x1, #63
        and x0, x0, x11
        sub x3, x3, x0
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x5, x5, x0
        adc x3, x3, x1
        extr x6, x5, x6, #59
        str x6, [sp, #16]
        extr x5, x3, x5, #59
        str x5, [sp, #24]
        eor x1, x7, x16
        asr x5, x1, #63
        and x5, x5, x12
        neg x5, x5
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x2, x2, x0
        adc x5, x5, x1
        eor x1, x8, x17
        asr x0, x1, #63
        and x0, x0, x13
        sub x5, x5, x0
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x2, x2, x0
        adc x5, x5, x1
        extr x4, x2, x4, #59
        str x4, [sp, #48]
        extr x2, x5, x2, #59
        str x2, [sp, #56]
        ldr x7, [sp, #64]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x4, x9, x0
        adc x2, xzr, x1
        ldr x8, [sp, #96]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x4, x4, x0
        str x4, [sp, #64]
        adc x2, x2, x1
        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x5, x19, x0
        adc x3, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x5, x5, x0
        str x5, [sp, #96]
        adc x3, x3, x1
        ldr x7, [sp, #72]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x2, x2, x0
        adc x6, xzr, x1
        ldr x8, [sp, #104]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x2, x2, x0
        str x2, [sp, #72]
        adc x6, x6, x1
        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x3, x3, x0
        adc x4, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x3, x3, x0
        str x3, [sp, #104]
        adc x4, x4, x1
        ldr x7, [sp, #80]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x6, x6, x0
        adc x5, xzr, x1
        ldr x8, [sp, #112]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x6, x6, x0
        str x6, [sp, #80]
        adc x5, x5, x1
        eor x1, x7, x16
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x4, x4, x0
        adc x2, xzr, x1
        eor x1, x8, x17
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x4, x4, x0
        str x4, [sp, #112]
        adc x2, x2, x1
        ldr x7, [sp, #88]
        eor x1, x7, x14
        and x3, x14, x10
        neg x3, x3
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x5, x5, x0
        adc x3, x3, x1
        ldr x8, [sp, #120]
        eor x1, x8, x15
        and x0, x15, x11
        sub x3, x3, x0
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x5, x5, x0
        adc x3, x3, x1
        extr x6, x3, x5, #63
        ldp x0, x1, [sp, #64]
        add x6, x6, x3, asr #63
        mov x3, #0x13
        mul x4, x6, x3
        add x5, x5, x6, lsl #63
        smulh x3, x6, x3
        ldr x6, [sp, #80]
        adds x0, x0, x4
        adcs x1, x1, x3
        asr x3, x3, #63
        adcs x6, x6, x3
        adc x5, x5, x3
        stp x0, x1, [sp, #64]
        stp x6, x5, [sp, #80]
        eor x1, x7, x16
        and x5, x16, x12
        neg x5, x5
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x2, x2, x0
        adc x5, x5, x1
        eor x1, x8, x17
        and x0, x17, x13
        sub x5, x5, x0
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x2, x2, x0
        adc x5, x5, x1
        extr x6, x5, x2, #63
        ldp x0, x1, [sp, #96]
        add x6, x6, x5, asr #63
        mov x5, #0x13
        mul x4, x6, x5
        add x2, x2, x6, lsl #63
        smulh x5, x6, x5
        ldr x3, [sp, #112]
        adds x0, x0, x4
        adcs x1, x1, x5
        asr x5, x5, #63
        adcs x3, x3, x5
        adc x2, x2, x5
        stp x0, x1, [sp, #96]
        stp x3, x2, [sp, #112]
Ledwards25519_scalarmulbase_alt_invmidloop:
        mov x1, x22
        ldr x2, [sp]
        ldr x3, [sp, #32]
        and x4, x2, #0xfffff
        orr x4, x4, #0xfffffe0000000000
        and x5, x3, #0xfffff
        orr x5, x5, #0xc000000000000000
        tst x5, #0x1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        asr x5, x5, #1
        add x8, x4, #0x100, lsl #12
        sbfx x8, x8, #21, #21
        mov x11, #0x100000
        add x11, x11, x11, lsl #21
        add x9, x4, x11
        asr x9, x9, #42
        add x10, x5, #0x100, lsl #12
        sbfx x10, x10, #21, #21
        add x11, x5, x11
        asr x11, x11, #42
        mul x6, x8, x2
        mul x7, x9, x3
        mul x2, x10, x2
        mul x3, x11, x3
        add x4, x6, x7
        add x5, x2, x3
        asr x2, x4, #20
        asr x3, x5, #20
        and x4, x2, #0xfffff
        orr x4, x4, #0xfffffe0000000000
        and x5, x3, #0xfffff
        orr x5, x5, #0xc000000000000000
        tst x5, #0x1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        asr x5, x5, #1
        add x12, x4, #0x100, lsl #12
        sbfx x12, x12, #21, #21
        mov x15, #0x100000
        add x15, x15, x15, lsl #21
        add x13, x4, x15
        asr x13, x13, #42
        add x14, x5, #0x100, lsl #12
        sbfx x14, x14, #21, #21
        add x15, x5, x15
        asr x15, x15, #42
        mul x6, x12, x2
        mul x7, x13, x3
        mul x2, x14, x2
        mul x3, x15, x3
        add x4, x6, x7
        add x5, x2, x3
        asr x2, x4, #20
        asr x3, x5, #20
        and x4, x2, #0xfffff
        orr x4, x4, #0xfffffe0000000000
        and x5, x3, #0xfffff
        orr x5, x5, #0xc000000000000000
        tst x5, #0x1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        mul x2, x12, x8
        mul x3, x12, x9
        mul x6, x14, x8
        mul x7, x14, x9
        madd x8, x13, x10, x2
        madd x9, x13, x11, x3
        madd x16, x15, x10, x6
        madd x17, x15, x11, x7
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        tst x5, #0x2
        asr x5, x5, #1
        csel x6, x4, xzr, ne
        ccmp x1, xzr, #0x8, ne
        cneg x1, x1, ge
        cneg x6, x6, ge
        csel x4, x5, x4, ge
        add x5, x5, x6
        add x1, x1, #0x2
        asr x5, x5, #1
        add x12, x4, #0x100, lsl #12
        sbfx x12, x12, #22, #21
        mov x15, #0x100000
        add x15, x15, x15, lsl #21
        add x13, x4, x15
        asr x13, x13, #43
        add x14, x5, #0x100, lsl #12
        sbfx x14, x14, #22, #21
        add x15, x5, x15
        asr x15, x15, #43
        mneg x2, x12, x8
        mneg x3, x12, x9
        mneg x4, x14, x8
        mneg x5, x14, x9
        msub x10, x13, x16, x2
        msub x11, x13, x17, x3
        msub x12, x15, x16, x4
        msub x13, x15, x17, x5
        mov x22, x1
        subs x21, x21, #0x1
        b.ne Ledwards25519_scalarmulbase_alt_invloop
        ldr x0, [sp]
        ldr x1, [sp, #32]
        mul x0, x0, x10
        madd x1, x1, x11, x0
        asr x0, x1, #63
        cmp x10, xzr
        csetm x14, mi
        cneg x10, x10, mi
        eor x14, x14, x0
        cmp x11, xzr
        csetm x15, mi
        cneg x11, x11, mi
        eor x15, x15, x0
        cmp x12, xzr
        csetm x16, mi
        cneg x12, x12, mi
        eor x16, x16, x0
        cmp x13, xzr
        csetm x17, mi
        cneg x13, x13, mi
        eor x17, x17, x0
        and x0, x10, x14
        and x1, x11, x15
        add x9, x0, x1
        ldr x7, [sp, #64]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x4, x9, x0
        adc x2, xzr, x1
        ldr x8, [sp, #96]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x4, x4, x0
        str x4, [sp, #64]
        adc x2, x2, x1
        ldr x7, [sp, #72]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x2, x2, x0
        adc x6, xzr, x1
        ldr x8, [sp, #104]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x2, x2, x0
        str x2, [sp, #72]
        adc x6, x6, x1
        ldr x7, [sp, #80]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x6, x6, x0
        adc x5, xzr, x1
        ldr x8, [sp, #112]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x6, x6, x0
        str x6, [sp, #80]
        adc x5, x5, x1
        ldr x7, [sp, #88]
        eor x1, x7, x14
        and x3, x14, x10
        neg x3, x3
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x5, x5, x0
        adc x3, x3, x1
        ldr x8, [sp, #120]
        eor x1, x8, x15
        and x0, x15, x11
        sub x3, x3, x0
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x5, x5, x0
        adc x3, x3, x1
        extr x6, x3, x5, #63
        ldp x0, x1, [sp, #64]
        tst x3, x3
        cinc x6, x6, pl
        mov x3, #0x13
        mul x4, x6, x3
        add x5, x5, x6, lsl #63
        smulh x6, x6, x3
        ldr x2, [sp, #80]
        adds x0, x0, x4
        adcs x1, x1, x6
        asr x6, x6, #63
        adcs x2, x2, x6
        adcs x5, x5, x6
        csel x3, x3, xzr, mi
        subs x0, x0, x3
        sbcs x1, x1, xzr
        sbcs x2, x2, xzr
        sbc x5, x5, xzr
        and x5, x5, #0x7fffffffffffffff
        mov x4, x20
        stp x0, x1, [x4]
        stp x2, x5, [x4, #16]
        ldp x3, x4, [sp, #(4*32)] %% ldp x7, x8, [sp, #(7*32)] %% mul x12, x3, x7 %% umulh x13, x3, x7 %% mul x11, x3, x8 %% umulh x14, x3, x8 %% adds x13, x13, x11 %% ldp x9, x10, [sp, #(7*32)+16] %% mul x11, x3, x9 %% umulh x15, x3, x9 %% adcs x14, x14, x11 %% mul x11, x3, x10 %% umulh x16, x3, x10 %% adcs x15, x15, x11 %% adc x16, x16, xzr %% ldp x5, x6, [sp, #(4*32)+16] %% mul x11, x4, x7 %% adds x13, x13, x11 %% mul x11, x4, x8 %% adcs x14, x14, x11 %% mul x11, x4, x9 %% adcs x15, x15, x11 %% mul x11, x4, x10 %% adcs x16, x16, x11 %% umulh x3, x4, x10 %% adc x3, x3, xzr %% umulh x11, x4, x7 %% adds x14, x14, x11 %% umulh x11, x4, x8 %% adcs x15, x15, x11 %% umulh x11, x4, x9 %% adcs x16, x16, x11 %% adc x3, x3, xzr %% mul x11, x5, x7 %% adds x14, x14, x11 %% mul x11, x5, x8 %% adcs x15, x15, x11 %% mul x11, x5, x9 %% adcs x16, x16, x11 %% mul x11, x5, x10 %% adcs x3, x3, x11 %% umulh x4, x5, x10 %% adc x4, x4, xzr %% umulh x11, x5, x7 %% adds x15, x15, x11 %% umulh x11, x5, x8 %% adcs x16, x16, x11 %% umulh x11, x5, x9 %% adcs x3, x3, x11 %% adc x4, x4, xzr %% mul x11, x6, x7 %% adds x15, x15, x11 %% mul x11, x6, x8 %% adcs x16, x16, x11 %% mul x11, x6, x9 %% adcs x3, x3, x11 %% mul x11, x6, x10 %% adcs x4, x4, x11 %% umulh x5, x6, x10 %% adc x5, x5, xzr %% umulh x11, x6, x7 %% adds x16, x16, x11 %% umulh x11, x6, x8 %% adcs x3, x3, x11 %% umulh x11, x6, x9 %% adcs x4, x4, x11 %% adc x5, x5, xzr %% mov x7, #0x26 %% mul x11, x7, x16 %% umulh x9, x7, x16 %% adds x12, x12, x11 %% mul x11, x7, x3 %% umulh x3, x7, x3 %% adcs x13, x13, x11 %% mul x11, x7, x4 %% umulh x4, x7, x4 %% adcs x14, x14, x11 %% mul x11, x7, x5 %% umulh x5, x7, x5 %% adcs x15, x15, x11 %% cset x16, cs %% adds x15, x15, x4 %% adc x16, x16, x5 %% cmn x15, x15 %% orr x15, x15, #0x8000000000000000 %% adc x8, x16, x16 %% mov x7, #0x13 %% madd x11, x7, x8, x7 %% adds x12, x12, x11 %% adcs x13, x13, x9 %% adcs x14, x14, x3 %% adcs x15, x15, xzr %% csel x7, x7, xzr, cc %% subs x12, x12, x7 %% sbcs x13, x13, xzr %% sbcs x14, x14, xzr %% sbc x15, x15, xzr %% and x15, x15, #0x7fffffffffffffff %% stp x12, x13, [x23, #(0*32)] %% stp x14, x15, [x23, #(0*32)+16]
        ldp x3, x4, [sp, #(5*32)] %% ldp x7, x8, [sp, #(7*32)] %% mul x12, x3, x7 %% umulh x13, x3, x7 %% mul x11, x3, x8 %% umulh x14, x3, x8 %% adds x13, x13, x11 %% ldp x9, x10, [sp, #(7*32)+16] %% mul x11, x3, x9 %% umulh x15, x3, x9 %% adcs x14, x14, x11 %% mul x11, x3, x10 %% umulh x16, x3, x10 %% adcs x15, x15, x11 %% adc x16, x16, xzr %% ldp x5, x6, [sp, #(5*32)+16] %% mul x11, x4, x7 %% adds x13, x13, x11 %% mul x11, x4, x8 %% adcs x14, x14, x11 %% mul x11, x4, x9 %% adcs x15, x15, x11 %% mul x11, x4, x10 %% adcs x16, x16, x11 %% umulh x3, x4, x10 %% adc x3, x3, xzr %% umulh x11, x4, x7 %% adds x14, x14, x11 %% umulh x11, x4, x8 %% adcs x15, x15, x11 %% umulh x11, x4, x9 %% adcs x16, x16, x11 %% adc x3, x3, xzr %% mul x11, x5, x7 %% adds x14, x14, x11 %% mul x11, x5, x8 %% adcs x15, x15, x11 %% mul x11, x5, x9 %% adcs x16, x16, x11 %% mul x11, x5, x10 %% adcs x3, x3, x11 %% umulh x4, x5, x10 %% adc x4, x4, xzr %% umulh x11, x5, x7 %% adds x15, x15, x11 %% umulh x11, x5, x8 %% adcs x16, x16, x11 %% umulh x11, x5, x9 %% adcs x3, x3, x11 %% adc x4, x4, xzr %% mul x11, x6, x7 %% adds x15, x15, x11 %% mul x11, x6, x8 %% adcs x16, x16, x11 %% mul x11, x6, x9 %% adcs x3, x3, x11 %% mul x11, x6, x10 %% adcs x4, x4, x11 %% umulh x5, x6, x10 %% adc x5, x5, xzr %% umulh x11, x6, x7 %% adds x16, x16, x11 %% umulh x11, x6, x8 %% adcs x3, x3, x11 %% umulh x11, x6, x9 %% adcs x4, x4, x11 %% adc x5, x5, xzr %% mov x7, #0x26 %% mul x11, x7, x16 %% umulh x9, x7, x16 %% adds x12, x12, x11 %% mul x11, x7, x3 %% umulh x3, x7, x3 %% adcs x13, x13, x11 %% mul x11, x7, x4 %% umulh x4, x7, x4 %% adcs x14, x14, x11 %% mul x11, x7, x5 %% umulh x5, x7, x5 %% adcs x15, x15, x11 %% cset x16, cs %% adds x15, x15, x4 %% adc x16, x16, x5 %% cmn x15, x15 %% orr x15, x15, #0x8000000000000000 %% adc x8, x16, x16 %% mov x7, #0x13 %% madd x11, x7, x8, x7 %% adds x12, x12, x11 %% adcs x13, x13, x9 %% adcs x14, x14, x3 %% adcs x15, x15, xzr %% csel x7, x7, xzr, cc %% subs x12, x12, x7 %% sbcs x13, x13, xzr %% sbcs x14, x14, xzr %% sbc x15, x15, xzr %% and x15, x15, #0x7fffffffffffffff %% stp x12, x13, [x23, #(1*32)] %% stp x14, x15, [x23, #(1*32)+16]
        add sp, sp, #(14*32 +0) %% .cfi_adjust_cfa_offset -14*32
        ldp x23, x24, [sp], #16 %% .cfi_adjust_cfa_offset -16 %% .cfi_restore x23 %% .cfi_restore x24
        ldp x21, x22, [sp], #16 %% .cfi_adjust_cfa_offset -16 %% .cfi_restore x21 %% .cfi_restore x22
        ldp x19, x20, [sp], #16 %% .cfi_adjust_cfa_offset -16 %% .cfi_restore x19 %% .cfi_restore x20
        ret %% .cfi_endproc
.const_data
_rscrypto_edwards25519_scalarmulbase_alt_constant:
        .quad 0x0000000000000000
        .quad 0x0000000000000000
        .quad 0x0000000000000000
        .quad 0x0000000000000000
        .quad 0x0000000000000001
        .quad 0x0000000000000000
        .quad 0x0000000000000000
        .quad 0x0000000000000000
        .quad 0x0000000000000000
        .quad 0x0000000000000000
        .quad 0x0000000000000000
        .quad 0x0000000000000000
        .quad 0x525f946d7c7220e7
        .quad 0x4636b0b2f1e35444
        .quad 0x796e9d70e892ae0f
        .quad 0x03dec05fa937adb1
        .quad 0x6d1c271cc6375515
        .quad 0x462588c4a4ca4f14
        .quad 0x691129fee55afc39
        .quad 0x15949f784d8472f5
        .quad 0xbd89e510afad0049
        .quad 0x4d1f08c073b9860e
        .quad 0x07716e8b2d00af9d
        .quad 0x70d685f68f859714
        rscrypto_ed25519_base_table
