// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - p256/p256_scalarmul_alt.S
// - deterministic secret-frame and volatile-register cleanup
//
// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.

        .globl rscrypto_p256_scalarmul_alt

        .hidden rscrypto_p256_scalarmul_alt


        .text
        .balign 4
rscrypto_p256_scalarmul_alt:
        .cfi_startproc

        stp x19, x20, [sp, #-16]!
        .cfi_adjust_cfa_offset 16
        .cfi_rel_offset x19, 0
        .cfi_rel_offset x20, 8
        stp x21, x30, [sp, #-16]!
        .cfi_adjust_cfa_offset 16
        .cfi_rel_offset x21, 0
        .cfi_rel_offset x30, 8
        sub sp, sp, #(31*32 +0)
        .cfi_adjust_cfa_offset 31*32





        mov x19, x0
        mov x21, x2



        movz x12, #0x2551
        movk x12, #0xfc63, lsl #16
        movk x12, #0xcac2, lsl #32
        movk x12, #0xf3b9, lsl #48
        movz x13, #0x9e84
        movk x13, #0xa717, lsl #16
        movk x13, #0xfaad, lsl #32
        movk x13, #0xbce6, lsl #48
        mov x14, #0xffffffffffffffff
        mov x15, #0xffffffff00000000



        ldp x2, x3, [x1]
        ldp x4, x5, [x1, #16]

        subs x6, x2, x12
        sbcs x7, x3, x13
        sbcs x8, x4, x14
        sbcs x9, x5, x15

        csel x2, x2, x6, cc
        csel x3, x3, x7, cc
        csel x4, x4, x8, cc
        csel x5, x5, x9, cc





        subs x6, x12, x2
        sbcs x7, x13, x3
        sbcs x8, x14, x4
        sbc x9, x15, x5

        tst x5, #0x8000000000000000
        csel x2, x2, x6, eq
        csel x3, x3, x7, eq
        csel x4, x4, x8, eq
        csel x5, x5, x9, eq
        cset x20, ne




        mov x6, 0x8888888888888888
        adds x2, x2, x6
        adcs x3, x3, x6
        bic x7, x6, #0xF000000000000000
        adcs x4, x4, x6
        adc x5, x5, x7

        stp x2, x3, [sp, #(0*32)]
        stp x4, x5, [sp, #(0*32)+16]




        add x0, sp, #(7*32)
        mov x1, x21
        bl Lp256_scalarmul_alt_local_tomont_p256

        add x1, x21, #32
        add x0, sp, #(7*32)+32
        bl Lp256_scalarmul_alt_local_tomont_p256

        mov x0, #0x0000000000000001
        mov x1, #0xffffffff00000000
        stp x0, x1, [sp, #(7*32)+64]
        mov x2, #0xffffffffffffffff
        mov x3, #0x00000000fffffffe
        stp x2, x3, [sp, #(7*32)+80]



        ldp x4, x5, [sp, #(7*32)+32]
        ldp x6, x7, [sp, #(7*32)+48]

        mov x0, 0xffffffffffffffff
        subs x0, x0, x4
        mov x1, 0x00000000ffffffff
        sbcs x1, x1, x5
        mov x3, 0xffffffff00000001
        sbcs x2, xzr, x6
        sbc x3, x3, x7

        cmp x20, xzr
        csel x4, x0, x4, ne
        csel x5, x1, x5, ne
        csel x6, x2, x6, ne
        csel x7, x3, x7, ne

        stp x4, x5, [sp, #(7*32)+32]
        stp x6, x7, [sp, #(7*32)+48]



        add x0, sp, #(7*32)+96*1
        add x1, sp, #(7*32)
        bl Lp256_scalarmul_alt_local_p256_montjdouble

        add x0, sp, #(7*32)+96*2
        add x1, sp, #(7*32)+96*1
        add x2, sp, #(7*32)
        bl Lp256_scalarmul_alt_local_p256_montjmixadd

        add x0, sp, #(7*32)+96*3
        add x1, sp, #(7*32)+96*1
        bl Lp256_scalarmul_alt_local_p256_montjdouble

        add x0, sp, #(7*32)+96*4
        add x1, sp, #(7*32)+96*3
        add x2, sp, #(7*32)
        bl Lp256_scalarmul_alt_local_p256_montjmixadd

        add x0, sp, #(7*32)+96*5
        add x1, sp, #(7*32)+96*2
        bl Lp256_scalarmul_alt_local_p256_montjdouble

        add x0, sp, #(7*32)+96*6
        add x1, sp, #(7*32)+96*5
        add x2, sp, #(7*32)
        bl Lp256_scalarmul_alt_local_p256_montjmixadd

        add x0, sp, #(7*32)+96*7
        add x1, sp, #(7*32)+96*3
        bl Lp256_scalarmul_alt_local_p256_montjdouble



        ldr x14, [sp, #(0*32)+24]
        lsr x14, x14, #60

        mov x0, xzr
        mov x1, xzr
        mov x2, xzr
        mov x3, xzr
        mov x4, xzr
        mov x5, xzr
        mov x6, xzr
        mov x7, xzr
        mov x8, xzr
        mov x9, xzr
        mov x10, xzr
        mov x11, xzr
        add x15, sp, #(7*32)

        .set i, 1
.rep 8
        cmp x14, #i
        ldp x12, x13, [x15]
        csel x0, x12, x0, eq
        csel x1, x13, x1, eq
        ldp x12, x13, [x15, #16]
        csel x2, x12, x2, eq
        csel x3, x13, x3, eq
        ldp x12, x13, [x15, #32]
        csel x4, x12, x4, eq
        csel x5, x13, x5, eq
        ldp x12, x13, [x15, #48]
        csel x6, x12, x6, eq
        csel x7, x13, x7, eq
        ldp x12, x13, [x15, #64]
        csel x8, x12, x8, eq
        csel x9, x13, x9, eq
        ldp x12, x13, [x15, #80]
        csel x10, x12, x10, eq
        csel x11, x13, x11, eq
        add x15, x15, #96
        .set i, (i+1)
.endr
        stp x0, x1, [sp, #(1*32)]
        stp x2, x3, [sp, #(1*32)+16]
        stp x4, x5, [sp, #(1*32)+32]
        stp x6, x7, [sp, #(1*32)+48]
        stp x8, x9, [sp, #(1*32)+64]
        stp x10, x11, [sp, #(1*32)+80]

        mov x20, #252



Lp256_scalarmul_alt_loop:
        sub x20, x20, #4

        add x0, sp, #(1*32)
        add x1, sp, #(1*32)
        bl Lp256_scalarmul_alt_local_p256_montjdouble

        add x0, sp, #(1*32)
        add x1, sp, #(1*32)
        bl Lp256_scalarmul_alt_local_p256_montjdouble

        add x0, sp, #(1*32)
        add x1, sp, #(1*32)
        bl Lp256_scalarmul_alt_local_p256_montjdouble

        add x0, sp, #(1*32)
        add x1, sp, #(1*32)
        bl Lp256_scalarmul_alt_local_p256_montjdouble

        lsr x2, x20, #6
        ldr x14, [sp, x2, lsl #3]
        lsr x14, x14, x20
        and x14, x14, #15

        subs x14, x14, #8
        cset x16, lo
        cneg x14, x14, lo



        mov x0, xzr
        mov x1, xzr
        mov x2, xzr
        mov x3, xzr
        mov x4, xzr
        mov x5, xzr
        mov x6, xzr
        mov x7, xzr
        mov x8, xzr
        mov x9, xzr
        mov x10, xzr
        mov x11, xzr
        add x15, sp, #(7*32)
        .set i, 1
.rep 8
        cmp x14, #i
        ldp x12, x13, [x15]
        csel x0, x12, x0, eq
        csel x1, x13, x1, eq
        ldp x12, x13, [x15, #16]
        csel x2, x12, x2, eq
        csel x3, x13, x3, eq
        ldp x12, x13, [x15, #32]
        csel x4, x12, x4, eq
        csel x5, x13, x5, eq
        ldp x12, x13, [x15, #48]
        csel x6, x12, x6, eq
        csel x7, x13, x7, eq
        ldp x12, x13, [x15, #64]
        csel x8, x12, x8, eq
        csel x9, x13, x9, eq
        ldp x12, x13, [x15, #80]
        csel x10, x12, x10, eq
        csel x11, x13, x11, eq
        add x15, x15, #96
        .set i, (i+1)
.endr



        stp x0, x1, [sp, #(4*32)]
        stp x2, x3, [sp, #(4*32)+16]

        mov x0, 0xffffffffffffffff
        subs x0, x0, x4
        mov x1, 0x00000000ffffffff
        sbcs x1, x1, x5
        mov x3, 0xffffffff00000001
        sbcs x2, xzr, x6
        sbc x3, x3, x7

        cmp x16, xzr
        csel x4, x0, x4, ne
        csel x5, x1, x5, ne
        csel x6, x2, x6, ne
        csel x7, x3, x7, ne

        stp x4, x5, [sp, #(4*32)+32]
        stp x6, x7, [sp, #(4*32)+48]
        stp x8, x9, [sp, #(4*32)+64]
        stp x10, x11, [sp, #(4*32)+80]

        add x0, sp, #(1*32)
        add x1, sp, #(1*32)
        add x2, sp, #(4*32)
        bl Lp256_scalarmul_alt_local_p256_montjadd

        cbnz x20, Lp256_scalarmul_alt_loop





        add x0, sp, #(7*32)
        add x1, sp, #(1*32)+64
        bl Lp256_scalarmul_alt_local_montsqr_p256

        add x0, sp, #(8*32)
        add x2, sp, #(7*32)
        add x1, sp, #(1*32)+64
        bl Lp256_scalarmul_alt_local_montmul_p256

        add x0, sp, #(7*32)
        add x1, sp, #(8*32)
        bl Lp256_scalarmul_alt_local_demont_p256

        add x0, sp, #(8*32)
        add x1, sp, #(7*32)
        bl Lp256_scalarmul_alt_local_inv_p256

        add x0, sp, #(7*32)
        add x2, sp, #(8*32)
        add x1, sp, #(1*32)+64
        bl Lp256_scalarmul_alt_local_montmul_p256



        add x1, sp, #(1*32)
        add x2, sp, #(7*32)
        mov x0, x19
        bl Lp256_scalarmul_alt_local_montmul_p256

        add x0, x19, #32
        add x1, sp, #(1*32)+32
        add x2, sp, #(8*32)
        bl Lp256_scalarmul_alt_local_montmul_p256



        mov x16, sp
        mov x17, #62
.Lrscrypto_p256_scalarmul_alt_clear_frame:
        stp xzr, xzr, [x16], #16
        subs x17, x17, #1
        b.ne .Lrscrypto_p256_scalarmul_alt_clear_frame
        mov x0, xzr
        mov x1, xzr
        mov x2, xzr
        mov x3, xzr
        mov x4, xzr
        mov x5, xzr
        mov x6, xzr
        mov x7, xzr
        mov x8, xzr
        mov x9, xzr
        mov x10, xzr
        mov x11, xzr
        mov x12, xzr
        mov x13, xzr
        mov x14, xzr
        mov x15, xzr
        mov x16, xzr
        mov x17, xzr
        add sp, sp, #(31*32 +0)
        .cfi_adjust_cfa_offset -31*32
        ldp x21, x30, [sp]
        stp xzr, xzr, [sp]
        add sp, sp, #16
        .cfi_adjust_cfa_offset -16
        .cfi_restore x21
        .cfi_restore x30
        ldp x19, x20, [sp]
        stp xzr, xzr, [sp]
        add sp, sp, #16
        .cfi_adjust_cfa_offset -16
        .cfi_restore x19
        .cfi_restore x20
        ret
        .cfi_endproc







Lp256_scalarmul_alt_local_demont_p256:
        .cfi_startproc
        ldp x2, x3, [x1]
        ldp x4, x5, [x1, #16]
        lsl x7, x2, #32
        subs x8, x2, x7
        lsr x6, x2, #32
        sbc x2, x2, x6
        adds x3, x3, x7
        adcs x4, x4, x6
        adcs x5, x5, x8
        adc x2, x2, xzr
        lsl x7, x3, #32
        subs x8, x3, x7
        lsr x6, x3, #32
        sbc x3, x3, x6
        adds x4, x4, x7
        adcs x5, x5, x6
        adcs x2, x2, x8
        adc x3, x3, xzr
        lsl x7, x4, #32
        subs x8, x4, x7
        lsr x6, x4, #32
        sbc x4, x4, x6
        adds x5, x5, x7
        adcs x2, x2, x6
        adcs x3, x3, x8
        adc x4, x4, xzr
        lsl x7, x5, #32
        subs x8, x5, x7
        lsr x6, x5, #32
        sbc x5, x5, x6
        adds x2, x2, x7
        adcs x3, x3, x6
        adcs x4, x4, x8
        adc x5, x5, xzr
        stp x2, x3, [x0]
        stp x4, x5, [x0, #16]
        ret
        .cfi_endproc





Lp256_scalarmul_alt_local_inv_p256:
        .cfi_startproc
        stp x19, x20, [sp, #-16]!
        .cfi_adjust_cfa_offset 16
        .cfi_rel_offset x19, 0
        .cfi_rel_offset x20, 8
        stp x21, x22, [sp, #-16]!
        .cfi_adjust_cfa_offset 16
        .cfi_rel_offset x21, 0
        .cfi_rel_offset x22, 8
        stp x23, x24, [sp, #-16]!
        .cfi_adjust_cfa_offset 16
        .cfi_rel_offset x23, 0
        .cfi_rel_offset x24, 8
        sub sp, sp, #(160 +0)
        .cfi_adjust_cfa_offset 160
        mov x20, x0
        mov x10, #0xffffffffffffffff
        mov x11, #0xffffffff
        mov x13, #0xffffffff00000001
        stp x10, x11, [sp]
        stp xzr, x13, [sp, #16]
        str xzr, [sp, #32]
        ldp x2, x3, [x1]
        subs x10, x2, x10
        sbcs x11, x3, x11
        ldp x4, x5, [x1, #16]
        sbcs x12, x4, xzr
        sbcs x13, x5, x13
        csel x2, x2, x10, cc
        csel x3, x3, x11, cc
        csel x4, x4, x12, cc
        csel x5, x5, x13, cc
        stp x2, x3, [sp, #48]
        stp x4, x5, [sp, #64]
        str xzr, [sp, #80]
        stp xzr, xzr, [sp, #96]
        stp xzr, xzr, [sp, #112]
        mov x10, #0x4000000000000
        stp x10, xzr, [sp, #128]
        stp xzr, xzr, [sp, #144]
        mov x21, #0xa
        mov x22, #0x1
        b Lp256_scalarmul_alt_inv_midloop
Lp256_scalarmul_alt_inv_loop:
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
        ldr x8, [sp, #48]
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
        ldr x8, [sp, #56]
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
        str x5, [sp, #48]
        ldr x7, [sp, #16]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x6, x6, x0
        adc x5, xzr, x1
        ldr x8, [sp, #64]
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
        str x3, [sp, #56]
        ldr x7, [sp, #24]
        eor x1, x7, x14
        ldr x23, [sp, #32]
        eor x3, x23, x14
        and x3, x3, x10
        neg x3, x3
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x5, x5, x0
        adc x3, x3, x1
        ldr x8, [sp, #72]
        eor x1, x8, x15
        ldr x24, [sp, #80]
        eor x0, x24, x15
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
        asr x3, x3, #59
        str x3, [sp, #32]
        eor x1, x7, x16
        eor x5, x23, x16
        and x5, x5, x12
        neg x5, x5
        mul x0, x1, x12
        umulh x1, x1, x12
        adds x2, x2, x0
        adc x5, x5, x1
        eor x1, x8, x17
        eor x0, x24, x17
        and x0, x0, x13
        sub x5, x5, x0
        mul x0, x1, x13
        umulh x1, x1, x13
        adds x2, x2, x0
        adc x5, x5, x1
        extr x4, x2, x4, #59
        str x4, [sp, #64]
        extr x2, x5, x2, #59
        str x2, [sp, #72]
        asr x5, x5, #59
        str x5, [sp, #80]
        ldr x7, [sp, #96]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x4, x9, x0
        adc x2, xzr, x1
        ldr x8, [sp, #128]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x4, x4, x0
        str x4, [sp, #96]
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
        str x5, [sp, #128]
        adc x3, x3, x1
        ldr x7, [sp, #104]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x2, x2, x0
        adc x6, xzr, x1
        ldr x8, [sp, #136]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x2, x2, x0
        str x2, [sp, #104]
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
        str x3, [sp, #136]
        adc x4, x4, x1
        ldr x7, [sp, #112]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x6, x6, x0
        adc x5, xzr, x1
        ldr x8, [sp, #144]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x6, x6, x0
        str x6, [sp, #112]
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
        str x4, [sp, #144]
        adc x2, x2, x1
        ldr x7, [sp, #120]
        eor x1, x7, x14
        and x3, x14, x10
        neg x3, x3
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x5, x5, x0
        adc x3, x3, x1
        ldr x8, [sp, #152]
        eor x1, x8, x15
        and x0, x15, x11
        sub x3, x3, x0
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x5, x5, x0
        adc x3, x3, x1
        ldp x0, x1, [sp, #96]
        ldr x6, [sp, #112]
        mov x14, #0xe000000000000000
        adds x0, x0, x14
        sbcs x1, x1, xzr
        mov x11, #0x1fffffff
        adcs x6, x6, x11
        mov x10, #0x2000000000000000
        adcs x5, x5, x10
        mov x14, #0x1fffffffe0000000
        adc x3, x3, x14
        lsl x11, x0, #32
        subs x14, x0, x11
        lsr x10, x0, #32
        sbc x0, x0, x10
        adds x1, x1, x11
        adcs x6, x6, x10
        adcs x5, x5, x14
        adcs x3, x3, x0
        mov x14, #0xffffffffffffffff
        mov x11, #0xffffffff
        mov x10, #0xffffffff00000001
        csel x14, x14, xzr, cs
        csel x11, x11, xzr, cs
        csel x10, x10, xzr, cs
        subs x1, x1, x14
        sbcs x6, x6, x11
        sbcs x5, x5, xzr
        sbc x3, x3, x10
        stp x1, x6, [sp, #96]
        stp x5, x3, [sp, #112]
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
        ldp x0, x1, [sp, #128]
        ldr x3, [sp, #144]
        mov x14, #0xe000000000000000
        adds x0, x0, x14
        sbcs x1, x1, xzr
        mov x11, #0x1fffffff
        adcs x3, x3, x11
        mov x10, #0x2000000000000000
        adcs x2, x2, x10
        mov x14, #0x1fffffffe0000000
        adc x5, x5, x14
        lsl x11, x0, #32
        subs x14, x0, x11
        lsr x10, x0, #32
        sbc x0, x0, x10
        adds x1, x1, x11
        adcs x3, x3, x10
        adcs x2, x2, x14
        adcs x5, x5, x0
        mov x14, #0xffffffffffffffff
        mov x11, #0xffffffff
        mov x10, #0xffffffff00000001
        csel x14, x14, xzr, cs
        csel x11, x11, xzr, cs
        csel x10, x10, xzr, cs
        subs x1, x1, x14
        sbcs x3, x3, x11
        sbcs x2, x2, xzr
        sbc x5, x5, x10
        stp x1, x3, [sp, #128]
        stp x2, x5, [sp, #144]
Lp256_scalarmul_alt_inv_midloop:
        mov x1, x22
        ldr x2, [sp]
        ldr x3, [sp, #48]
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
        b.ne Lp256_scalarmul_alt_inv_loop
        ldr x0, [sp]
        ldr x1, [sp, #48]
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
        ldr x7, [sp, #96]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x4, x9, x0
        adc x2, xzr, x1
        ldr x8, [sp, #128]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x4, x4, x0
        str x4, [sp, #96]
        adc x2, x2, x1
        ldr x7, [sp, #104]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x2, x2, x0
        adc x6, xzr, x1
        ldr x8, [sp, #136]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x2, x2, x0
        str x2, [sp, #104]
        adc x6, x6, x1
        ldr x7, [sp, #112]
        eor x1, x7, x14
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x6, x6, x0
        adc x5, xzr, x1
        ldr x8, [sp, #144]
        eor x1, x8, x15
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x6, x6, x0
        str x6, [sp, #112]
        adc x5, x5, x1
        ldr x7, [sp, #120]
        eor x1, x7, x14
        and x3, x14, x10
        neg x3, x3
        mul x0, x1, x10
        umulh x1, x1, x10
        adds x5, x5, x0
        adc x3, x3, x1
        ldr x8, [sp, #152]
        eor x1, x8, x15
        and x0, x15, x11
        sub x3, x3, x0
        mul x0, x1, x11
        umulh x1, x1, x11
        adds x5, x5, x0
        adc x3, x3, x1
        ldp x0, x1, [sp, #96]
        ldr x2, [sp, #112]
        mov x14, #0xe000000000000000
        adds x0, x0, x14
        sbcs x1, x1, xzr
        mov x11, #0x1fffffff
        adcs x2, x2, x11
        mov x10, #0x2000000000000000
        adcs x5, x5, x10
        mov x14, #0x1fffffffe0000000
        adc x3, x3, x14
        lsl x11, x0, #32
        subs x14, x0, x11
        lsr x10, x0, #32
        sbc x0, x0, x10
        adds x1, x1, x11
        adcs x2, x2, x10
        adcs x5, x5, x14
        adcs x3, x3, x0
        mov x14, #0xffffffffffffffff
        mov x11, #0xffffffff
        mov x10, #0xffffffff00000001
        csel x14, x14, xzr, cs
        csel x11, x11, xzr, cs
        csel x10, x10, xzr, cs
        subs x1, x1, x14
        sbcs x2, x2, x11
        sbcs x5, x5, xzr
        sbc x3, x3, x10
        mov x10, #0xffffffffffffffff
        subs x10, x1, x10
        mov x11, #0xffffffff
        sbcs x11, x2, x11
        mov x13, #0xffffffff00000001
        sbcs x12, x5, xzr
        sbcs x13, x3, x13
        csel x10, x1, x10, cc
        csel x11, x2, x11, cc
        csel x12, x5, x12, cc
        csel x13, x3, x13, cc
        stp x10, x11, [x20]
        stp x12, x13, [x20, #16]
        mov x16, sp
        mov x17, #10
.Lrscrypto_p256_scalarmul_alt_clear_inverse_frame:
        stp xzr, xzr, [x16], #16
        subs x17, x17, #1
        b.ne .Lrscrypto_p256_scalarmul_alt_clear_inverse_frame
        add sp, sp, #(160 +0)
        .cfi_adjust_cfa_offset -160
        ldp x23, x24, [sp]
        stp xzr, xzr, [sp]
        add sp, sp, #16
        .cfi_adjust_cfa_offset -16
        .cfi_restore x23
        .cfi_restore x24
        ldp x21, x22, [sp]
        stp xzr, xzr, [sp]
        add sp, sp, #16
        .cfi_adjust_cfa_offset -16
        .cfi_restore x21
        .cfi_restore x22
        ldp x19, x20, [sp]
        stp xzr, xzr, [sp]
        add sp, sp, #16
        .cfi_adjust_cfa_offset -16
        .cfi_restore x19
        .cfi_restore x20
        ret
        .cfi_endproc





Lp256_scalarmul_alt_local_montmul_p256:
        .cfi_startproc
        ldp x3, x4, [x1]
        ldp x7, x8, [x2]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [x2, #16]
        mul x11, x3, x9
        umulh x15, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x16, x3, x10
        adcs x15, x15, x11
        adc x16, x16, xzr
        ldp x5, x6, [x1, #16]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x15, x15, x11
        mul x11, x4, x10
        adcs x16, x16, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x15, x15, x11
        umulh x11, x4, x9
        adcs x16, x16, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x15, x15, x11
        mul x11, x5, x9
        adcs x16, x16, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x15, x15, x11
        umulh x11, x5, x8
        adcs x16, x16, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x15, x15, x11
        mul x11, x6, x8
        adcs x16, x16, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x15, x15, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x16, x16, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x15, x15, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x15, x15, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x15, lsl #32
        lsr x11, x15, #32
        adcs x13, x13, x11
        mul x11, x15, x10
        umulh x15, x15, x10
        adcs x14, x14, x11
        adc x15, x15, xzr
        adds x12, x12, x16
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x15, x15, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x16, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x15, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x16, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x15, x15, x5, cc
        stp x12, x13, [x0]
        stp x14, x15, [x0, #16]
        ret
        .cfi_endproc





Lp256_scalarmul_alt_local_montsqr_p256:
        .cfi_startproc
        ldp x2, x3, [x1]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [x1, #16]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        mov x5, #0xffffffff00000001
        adds x9, x9, x8, lsl #32
        lsr x2, x8, #32
        adcs x10, x10, x2
        mul x2, x8, x5
        umulh x8, x8, x5
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x2, x9, #32
        adcs x11, x11, x2
        mul x2, x9, x5
        umulh x9, x9, x5
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x2, x10, #32
        adcs x8, x8, x2
        mul x2, x10, x5
        umulh x10, x10, x5
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x2, x11, #32
        adcs x9, x9, x2
        mul x2, x11, x5
        umulh x11, x11, x5
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        cset x2, cs
        mov x3, #0xffffffff
        adds x12, x8, #0x1
        sbcs x13, x9, x3
        sbcs x14, x10, xzr
        sbcs x7, x11, x5
        sbcs xzr, x2, xzr
        csel x8, x8, x12, cc
        csel x9, x9, x13, cc
        csel x10, x10, x14, cc
        csel x11, x11, x7, cc
        stp x8, x9, [x0]
        stp x10, x11, [x0, #16]
        ret
        .cfi_endproc





Lp256_scalarmul_alt_local_tomont_p256:
        .cfi_startproc
        ldp x2, x3, [x1]
        ldp x4, x5, [x1, #16]
        mov x1, #0xffffffffffffffff
        mov x7, #0xffffffff
        mov x9, #0xffffffff00000001
        subs x1, x2, x1
        sbcs x7, x3, x7
        sbcs x8, x4, xzr
        sbcs x9, x5, x9
        csel x2, x2, x1, cc
        csel x3, x3, x7, cc
        csel x4, x4, x8, cc
        csel x5, x5, x9, cc
        cmp xzr, xzr
        extr x9, x5, x4, #32
        adcs xzr, x4, x9
        lsr x9, x5, #32
        adcs x9, x5, x9
        csetm x6, cs
        orr x9, x9, x6
        lsl x7, x9, #32
        lsr x8, x9, #32
        adds x4, x4, x7
        adc x5, x5, x8
        negs x6, x9
        sbcs x7, x7, xzr
        sbc x8, x8, xzr
        negs x6, x6
        sbcs x2, x2, x7
        sbcs x3, x3, x8
        sbcs x4, x4, x9
        sbcs x5, x5, x9
        adds x6, x6, x5
        mov x7, #0xffffffff
        and x7, x7, x5
        adcs x2, x2, x7
        adcs x3, x3, xzr
        mov x7, #0xffffffff00000001
        and x7, x7, x5
        adc x4, x4, x7
        cmp xzr, xzr
        extr x9, x4, x3, #32
        adcs xzr, x3, x9
        lsr x9, x4, #32
        adcs x9, x4, x9
        csetm x5, cs
        orr x9, x9, x5
        lsl x7, x9, #32
        lsr x8, x9, #32
        adds x3, x3, x7
        adc x4, x4, x8
        negs x5, x9
        sbcs x7, x7, xzr
        sbc x8, x8, xzr
        negs x5, x5
        sbcs x6, x6, x7
        sbcs x2, x2, x8
        sbcs x3, x3, x9
        sbcs x4, x4, x9
        adds x5, x5, x4
        mov x7, #0xffffffff
        and x7, x7, x4
        adcs x6, x6, x7
        adcs x2, x2, xzr
        mov x7, #0xffffffff00000001
        and x7, x7, x4
        adc x3, x3, x7
        cmp xzr, xzr
        extr x9, x3, x2, #32
        adcs xzr, x2, x9
        lsr x9, x3, #32
        adcs x9, x3, x9
        csetm x4, cs
        orr x9, x9, x4
        lsl x7, x9, #32
        lsr x8, x9, #32
        adds x2, x2, x7
        adc x3, x3, x8
        negs x4, x9
        sbcs x7, x7, xzr
        sbc x8, x8, xzr
        negs x4, x4
        sbcs x5, x5, x7
        sbcs x6, x6, x8
        sbcs x2, x2, x9
        sbcs x3, x3, x9
        adds x4, x4, x3
        mov x7, #0xffffffff
        and x7, x7, x3
        adcs x5, x5, x7
        adcs x6, x6, xzr
        mov x7, #0xffffffff00000001
        and x7, x7, x3
        adc x2, x2, x7
        cmp xzr, xzr
        extr x9, x2, x6, #32
        adcs xzr, x6, x9
        lsr x9, x2, #32
        adcs x9, x2, x9
        csetm x3, cs
        orr x9, x9, x3
        lsl x7, x9, #32
        lsr x8, x9, #32
        adds x6, x6, x7
        adc x2, x2, x8
        negs x3, x9
        sbcs x7, x7, xzr
        sbc x8, x8, xzr
        negs x3, x3
        sbcs x4, x4, x7
        sbcs x5, x5, x8
        sbcs x6, x6, x9
        sbcs x2, x2, x9
        adds x3, x3, x2
        mov x7, #0xffffffff
        and x7, x7, x2
        adcs x4, x4, x7
        adcs x5, x5, xzr
        mov x7, #0xffffffff00000001
        and x7, x7, x2
        adc x6, x6, x7
        stp x3, x4, [x0]
        stp x5, x6, [x0, #16]
        ret
        .cfi_endproc





Lp256_scalarmul_alt_local_p256_montjadd:
        .cfi_startproc
        sub sp, sp, #(224 +0)
        .cfi_adjust_cfa_offset 224
        mov x15, x0
        mov x16, x1
        mov x17, x2
        ldp x2, x3, [x16, #64]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [x16, #80]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        adds x9, x9, x8, lsl #32
        lsr x3, x8, #32
        adcs x10, x10, x3
        mov x3, #0xffffffff00000001
        mul x2, x8, x3
        umulh x8, x8, x3
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x3, x9, #32
        adcs x11, x11, x3
        mov x3, #0xffffffff00000001
        mul x2, x9, x3
        umulh x9, x9, x3
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x3, x10, #32
        adcs x8, x8, x3
        mov x3, #0xffffffff00000001
        mul x2, x10, x3
        umulh x10, x10, x3
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x3, x11, #32
        adcs x9, x9, x3
        mov x3, #0xffffffff00000001
        mul x2, x11, x3
        umulh x11, x11, x3
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        mov x2, #0xffffffffffffffff
        csel x2, xzr, x2, cc
        mov x3, #0xffffffff
        csel x3, xzr, x3, cc
        mov x5, #0xffffffff00000001
        csel x5, xzr, x5, cc
        subs x8, x8, x2
        sbcs x9, x9, x3
        sbcs x10, x10, xzr
        sbc x11, x11, x5
        stp x8, x9, [sp]
        stp x10, x11, [sp, #16]
        ldp x2, x3, [x17, #64]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [x17, #80]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        adds x9, x9, x8, lsl #32
        lsr x3, x8, #32
        adcs x10, x10, x3
        mov x3, #0xffffffff00000001
        mul x2, x8, x3
        umulh x8, x8, x3
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x3, x9, #32
        adcs x11, x11, x3
        mov x3, #0xffffffff00000001
        mul x2, x9, x3
        umulh x9, x9, x3
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x3, x10, #32
        adcs x8, x8, x3
        mov x3, #0xffffffff00000001
        mul x2, x10, x3
        umulh x10, x10, x3
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x3, x11, #32
        adcs x9, x9, x3
        mov x3, #0xffffffff00000001
        mul x2, x11, x3
        umulh x11, x11, x3
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        mov x2, #0xffffffffffffffff
        csel x2, xzr, x2, cc
        mov x3, #0xffffffff
        csel x3, xzr, x3, cc
        mov x5, #0xffffffff00000001
        csel x5, xzr, x5, cc
        subs x8, x8, x2
        sbcs x9, x9, x3
        sbcs x10, x10, xzr
        sbc x11, x11, x5
        stp x8, x9, [sp, #160]
        stp x10, x11, [sp, #176]
        ldp x3, x4, [x17, #64]
        ldp x7, x8, [x16, #32]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [x16, #48]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [x17, #80]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #192]
        stp x14, x0, [sp, #208]
        ldp x3, x4, [x16, #64]
        ldp x7, x8, [x17, #32]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [x17, #48]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [x16, #80]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #32]
        stp x14, x0, [sp, #48]
        ldp x3, x4, [sp]
        ldp x7, x8, [x17]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [x17, #16]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #16]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #64]
        stp x14, x0, [sp, #80]
        ldp x3, x4, [sp, #160]
        ldp x7, x8, [x16]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [x16, #16]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #176]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #128]
        stp x14, x0, [sp, #144]
        ldp x3, x4, [sp]
        ldp x7, x8, [sp, #32]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [sp, #48]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #16]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #32]
        stp x14, x0, [sp, #48]
        ldp x3, x4, [sp, #160]
        ldp x7, x8, [sp, #192]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [sp, #208]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #176]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #192]
        stp x14, x0, [sp, #208]
        ldp x5, x6, [sp, #64]
        ldp x4, x3, [sp, #128]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #80]
        ldp x4, x3, [sp, #144]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp, #160]
        stp x7, x8, [sp, #176]
        ldp x5, x6, [sp, #32]
        ldp x4, x3, [sp, #192]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #48]
        ldp x4, x3, [sp, #208]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp, #32]
        stp x7, x8, [sp, #48]
        ldp x2, x3, [sp, #160]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [sp, #176]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        adds x9, x9, x8, lsl #32
        lsr x3, x8, #32
        adcs x10, x10, x3
        mov x3, #0xffffffff00000001
        mul x2, x8, x3
        umulh x8, x8, x3
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x3, x9, #32
        adcs x11, x11, x3
        mov x3, #0xffffffff00000001
        mul x2, x9, x3
        umulh x9, x9, x3
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x3, x10, #32
        adcs x8, x8, x3
        mov x3, #0xffffffff00000001
        mul x2, x10, x3
        umulh x10, x10, x3
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x3, x11, #32
        adcs x9, x9, x3
        mov x3, #0xffffffff00000001
        mul x2, x11, x3
        umulh x11, x11, x3
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        mov x2, #0xffffffffffffffff
        csel x2, xzr, x2, cc
        mov x3, #0xffffffff
        csel x3, xzr, x3, cc
        mov x5, #0xffffffff00000001
        csel x5, xzr, x5, cc
        subs x8, x8, x2
        sbcs x9, x9, x3
        sbcs x10, x10, xzr
        sbc x11, x11, x5
        stp x8, x9, [sp, #96]
        stp x10, x11, [sp, #112]
        ldp x2, x3, [sp, #32]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [sp, #48]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        adds x9, x9, x8, lsl #32
        lsr x3, x8, #32
        adcs x10, x10, x3
        mov x3, #0xffffffff00000001
        mul x2, x8, x3
        umulh x8, x8, x3
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x3, x9, #32
        adcs x11, x11, x3
        mov x3, #0xffffffff00000001
        mul x2, x9, x3
        umulh x9, x9, x3
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x3, x10, #32
        adcs x8, x8, x3
        mov x3, #0xffffffff00000001
        mul x2, x10, x3
        umulh x10, x10, x3
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x3, x11, #32
        adcs x9, x9, x3
        mov x3, #0xffffffff00000001
        mul x2, x11, x3
        umulh x11, x11, x3
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        cset x2, cs
        mov x3, #0xffffffff
        mov x5, #0xffffffff00000001
        adds x12, x8, #0x1
        sbcs x13, x9, x3
        sbcs x14, x10, xzr
        sbcs x7, x11, x5
        sbcs xzr, x2, xzr
        csel x8, x8, x12, cc
        csel x9, x9, x13, cc
        csel x10, x10, x14, cc
        csel x11, x11, x7, cc
        stp x8, x9, [sp]
        stp x10, x11, [sp, #16]
        ldp x3, x4, [sp, #96]
        ldp x7, x8, [sp, #128]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [sp, #144]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #112]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #128]
        stp x14, x0, [sp, #144]
        ldp x3, x4, [sp, #96]
        ldp x7, x8, [sp, #64]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [sp, #80]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #112]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #64]
        stp x14, x0, [sp, #80]
        ldp x5, x6, [sp]
        ldp x4, x3, [sp, #128]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #16]
        ldp x4, x3, [sp, #144]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp]
        stp x7, x8, [sp, #16]
        ldp x5, x6, [sp, #64]
        ldp x4, x3, [sp, #128]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #80]
        ldp x4, x3, [sp, #144]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp, #96]
        stp x7, x8, [sp, #112]
        ldp x3, x4, [sp, #160]
        ldp x7, x8, [x16, #64]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [x16, #80]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #176]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #160]
        stp x14, x0, [sp, #176]
        ldp x5, x6, [sp]
        ldp x4, x3, [sp, #64]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #16]
        ldp x4, x3, [sp, #80]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp]
        stp x7, x8, [sp, #16]
        ldp x5, x6, [sp, #128]
        ldp x4, x3, [sp]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #144]
        ldp x4, x3, [sp, #16]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp, #128]
        stp x7, x8, [sp, #144]
        ldp x3, x4, [sp, #96]
        ldp x7, x8, [sp, #192]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [sp, #208]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #112]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #96]
        stp x14, x0, [sp, #112]
        ldp x3, x4, [sp, #160]
        ldp x7, x8, [x17, #64]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [x17, #80]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #176]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #160]
        stp x14, x0, [sp, #176]
        ldp x3, x4, [sp, #32]
        ldp x7, x8, [sp, #128]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [sp, #144]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #48]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #128]
        stp x14, x0, [sp, #144]
        ldp x5, x6, [sp, #128]
        ldp x4, x3, [sp, #96]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #144]
        ldp x4, x3, [sp, #112]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp, #128]
        stp x7, x8, [sp, #144]
        ldp x0, x1, [x16, #64]
        ldp x2, x3, [x16, #80]
        orr x12, x0, x1
        orr x13, x2, x3
        orr x12, x12, x13
        cmp x12, xzr
        cset x12, ne
        ldp x4, x5, [x17, #64]
        ldp x6, x7, [x17, #80]
        orr x13, x4, x5
        orr x14, x6, x7
        orr x13, x13, x14
        cmp x13, xzr
        cset x13, ne
        cmp x13, x12
        ldp x8, x9, [sp, #160]
        csel x8, x0, x8, cc
        csel x9, x1, x9, cc
        csel x8, x4, x8, hi
        csel x9, x5, x9, hi
        ldp x10, x11, [sp, #176]
        csel x10, x2, x10, cc
        csel x11, x3, x11, cc
        csel x10, x6, x10, hi
        csel x11, x7, x11, hi
        ldp x12, x13, [x16]
        ldp x0, x1, [sp]
        csel x0, x12, x0, cc
        csel x1, x13, x1, cc
        ldp x12, x13, [x17]
        csel x0, x12, x0, hi
        csel x1, x13, x1, hi
        ldp x12, x13, [x16, #16]
        ldp x2, x3, [sp, #16]
        csel x2, x12, x2, cc
        csel x3, x13, x3, cc
        ldp x12, x13, [x17, #16]
        csel x2, x12, x2, hi
        csel x3, x13, x3, hi
        ldp x12, x13, [x16, #32]
        ldp x4, x5, [sp, #128]
        csel x4, x12, x4, cc
        csel x5, x13, x5, cc
        ldp x12, x13, [x17, #32]
        csel x4, x12, x4, hi
        csel x5, x13, x5, hi
        ldp x12, x13, [x16, #48]
        ldp x6, x7, [sp, #144]
        csel x6, x12, x6, cc
        csel x7, x13, x7, cc
        ldp x12, x13, [x17, #48]
        csel x6, x12, x6, hi
        csel x7, x13, x7, hi
        stp x0, x1, [x15]
        stp x2, x3, [x15, #16]
        stp x4, x5, [x15, #32]
        stp x6, x7, [x15, #48]
        stp x8, x9, [x15, #64]
        stp x10, x11, [x15, #80]
        mov x16, sp
        mov x17, #14
.Lrscrypto_p256_scalarmul_alt_clear_add_frame:
        stp xzr, xzr, [x16], #16
        subs x17, x17, #1
        b.ne .Lrscrypto_p256_scalarmul_alt_clear_add_frame
        add sp, sp, #(224 +0)
        .cfi_adjust_cfa_offset -224
        ret
        .cfi_endproc





Lp256_scalarmul_alt_local_p256_montjdouble:
        .cfi_startproc
        sub sp, sp, #(192 +0)
        .cfi_adjust_cfa_offset 192
        mov x15, x0
        mov x16, x1
        ldp x2, x3, [x16, #64]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [x16, #80]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        mov x5, #0xffffffff00000001
        adds x9, x9, x8, lsl #32
        lsr x3, x8, #32
        adcs x10, x10, x3
        mul x2, x8, x5
        umulh x8, x8, x5
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x3, x9, #32
        adcs x11, x11, x3
        mul x2, x9, x5
        umulh x9, x9, x5
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x3, x10, #32
        adcs x8, x8, x3
        mul x2, x10, x5
        umulh x10, x10, x5
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x3, x11, #32
        adcs x9, x9, x3
        mul x2, x11, x5
        umulh x11, x11, x5
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        cset x2, cs
        mov x3, #0xffffffff
        adds x12, x8, #0x1
        sbcs x13, x9, x3
        sbcs x14, x10, xzr
        sbcs x7, x11, x5
        sbcs xzr, x2, xzr
        csel x8, x8, x12, cc
        csel x9, x9, x13, cc
        csel x10, x10, x14, cc
        csel x11, x11, x7, cc
        stp x8, x9, [sp]
        stp x10, x11, [sp, #16]
        ldp x2, x3, [x16, #32]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [x16, #48]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        mov x5, #0xffffffff00000001
        adds x9, x9, x8, lsl #32
        lsr x3, x8, #32
        adcs x10, x10, x3
        mul x2, x8, x5
        umulh x8, x8, x5
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x3, x9, #32
        adcs x11, x11, x3
        mul x2, x9, x5
        umulh x9, x9, x5
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x3, x10, #32
        adcs x8, x8, x3
        mul x2, x10, x5
        umulh x10, x10, x5
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x3, x11, #32
        adcs x9, x9, x3
        mul x2, x11, x5
        umulh x11, x11, x5
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        cset x2, cs
        mov x3, #0xffffffff
        adds x12, x8, #0x1
        sbcs x13, x9, x3
        sbcs x14, x10, xzr
        sbcs x7, x11, x5
        sbcs xzr, x2, xzr
        csel x8, x8, x12, cc
        csel x9, x9, x13, cc
        csel x10, x10, x14, cc
        csel x11, x11, x7, cc
        stp x8, x9, [sp, #32]
        stp x10, x11, [sp, #48]
        ldp x5, x6, [x16]
        ldp x4, x3, [sp]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [x16, #16]
        ldp x4, x3, [sp, #16]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        and x4, x3, #0xffffffff
        adcs x6, x6, x4
        adcs x7, x7, xzr
        and x4, x3, #0xffffffff00000001
        adc x8, x8, x4
        stp x5, x6, [sp, #96]
        stp x7, x8, [sp, #112]
        ldp x5, x6, [x16]
        ldp x4, x3, [sp]
        adds x5, x5, x4
        adcs x6, x6, x3
        ldp x7, x8, [x16, #16]
        ldp x4, x3, [sp, #16]
        adcs x7, x7, x4
        adcs x8, x8, x3
        csetm x3, cs
        subs x5, x5, x3
        and x1, x3, #0xffffffff
        sbcs x6, x6, x1
        sbcs x7, x7, xzr
        and x2, x3, #0xffffffff00000001
        sbc x8, x8, x2
        stp x5, x6, [sp, #64]
        stp x7, x8, [sp, #80]
        ldp x3, x4, [sp, #64]
        ldp x7, x8, [sp, #96]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [sp, #112]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #80]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #96]
        stp x14, x0, [sp, #112]
        ldp x5, x6, [x16, #32]
        ldp x4, x3, [x16, #64]
        adds x5, x5, x4
        adcs x6, x6, x3
        ldp x7, x8, [x16, #48]
        ldp x4, x3, [x16, #80]
        adcs x7, x7, x4
        adcs x8, x8, x3
        adc x3, xzr, xzr
        cmn x5, #0x1
        mov x4, #0xffffffff
        sbcs xzr, x6, x4
        sbcs xzr, x7, xzr
        mov x4, #0xffffffff00000001
        sbcs xzr, x8, x4
        adcs x3, x3, xzr
        csetm x3, ne
        subs x5, x5, x3
        and x4, x3, #0xffffffff
        sbcs x6, x6, x4
        sbcs x7, x7, xzr
        and x4, x3, #0xffffffff00000001
        sbc x8, x8, x4
        stp x5, x6, [sp, #64]
        stp x7, x8, [sp, #80]
        ldp x3, x4, [x16]
        ldp x7, x8, [sp, #32]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [sp, #48]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [x16, #16]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #128]
        stp x14, x0, [sp, #144]
        ldp x2, x3, [sp, #96]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [sp, #112]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        mov x5, #0xffffffff00000001
        adds x9, x9, x8, lsl #32
        lsr x3, x8, #32
        adcs x10, x10, x3
        mul x2, x8, x5
        umulh x8, x8, x5
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x3, x9, #32
        adcs x11, x11, x3
        mul x2, x9, x5
        umulh x9, x9, x5
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x3, x10, #32
        adcs x8, x8, x3
        mul x2, x10, x5
        umulh x10, x10, x5
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x3, x11, #32
        adcs x9, x9, x3
        mul x2, x11, x5
        umulh x11, x11, x5
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        cset x2, cs
        mov x3, #0xffffffff
        adds x12, x8, #0x1
        sbcs x13, x9, x3
        sbcs x14, x10, xzr
        sbcs x7, x11, x5
        sbcs xzr, x2, xzr
        csel x8, x8, x12, cc
        csel x9, x9, x13, cc
        csel x10, x10, x14, cc
        csel x11, x11, x7, cc
        stp x8, x9, [sp, #160]
        stp x10, x11, [sp, #176]
        ldp x2, x3, [sp, #64]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [sp, #80]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        mov x5, #0xffffffff00000001
        adds x9, x9, x8, lsl #32
        lsr x3, x8, #32
        adcs x10, x10, x3
        mul x2, x8, x5
        umulh x8, x8, x5
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x3, x9, #32
        adcs x11, x11, x3
        mul x2, x9, x5
        umulh x9, x9, x5
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x3, x10, #32
        adcs x8, x8, x3
        mul x2, x10, x5
        umulh x10, x10, x5
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x3, x11, #32
        adcs x9, x9, x3
        mul x2, x11, x5
        umulh x11, x11, x5
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        cset x2, cs
        mov x3, #0xffffffff
        adds x12, x8, #0x1
        sbcs x13, x9, x3
        sbcs x14, x10, xzr
        sbcs x7, x11, x5
        sbcs xzr, x2, xzr
        csel x8, x8, x12, cc
        csel x9, x9, x13, cc
        csel x10, x10, x14, cc
        csel x11, x11, x7, cc
        stp x8, x9, [sp, #64]
        stp x10, x11, [sp, #80]
        mov x1, #0x9
        mov x2, #0xffffffffffffffff
        ldp x9, x10, [sp, #160]
        subs x9, x2, x9
        mov x2, #0xffffffff
        sbcs x10, x2, x10
        ldp x11, x12, [sp, #176]
        ngcs x11, x11
        mov x2, #0xffffffff00000001
        sbc x12, x2, x12
        mul x3, x1, x9
        mul x4, x1, x10
        mul x5, x1, x11
        mul x6, x1, x12
        umulh x9, x1, x9
        umulh x10, x1, x10
        umulh x11, x1, x11
        umulh x7, x1, x12
        adds x4, x4, x9
        adcs x5, x5, x10
        adcs x6, x6, x11
        adc x7, x7, xzr
        mov x1, #0xc
        ldp x9, x10, [sp, #128]
        mul x8, x9, x1
        umulh x9, x9, x1
        adds x3, x3, x8
        mul x8, x10, x1
        umulh x10, x10, x1
        adcs x4, x4, x8
        ldp x11, x12, [sp, #144]
        mul x8, x11, x1
        umulh x11, x11, x1
        adcs x5, x5, x8
        mul x8, x12, x1
        umulh x12, x12, x1
        adcs x6, x6, x8
        adc x7, x7, xzr
        adds x4, x4, x9
        adcs x5, x5, x10
        adcs x6, x6, x11
        adc x7, x7, x12
        add x8, x7, #0x1
        lsl x10, x8, #32
        adds x6, x6, x10
        adc x7, x7, xzr
        neg x9, x8
        sub x10, x10, #0x1
        subs x3, x3, x9
        sbcs x4, x4, x10
        sbcs x5, x5, xzr
        sbcs x6, x6, x8
        sbc x8, x7, x8
        adds x3, x3, x8
        and x9, x8, #0xffffffff
        adcs x4, x4, x9
        adcs x5, x5, xzr
        neg x10, x9
        adc x6, x6, x10
        stp x3, x4, [sp, #160]
        stp x5, x6, [sp, #176]
        ldp x5, x6, [sp, #64]
        ldp x4, x3, [sp]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #80]
        ldp x4, x3, [sp, #16]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        and x4, x3, #0xffffffff
        adcs x6, x6, x4
        adcs x7, x7, xzr
        and x4, x3, #0xffffffff00000001
        adc x8, x8, x4
        stp x5, x6, [sp, #64]
        stp x7, x8, [sp, #80]
        ldp x2, x3, [sp, #32]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [sp, #48]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        mov x5, #0xffffffff00000001
        adds x9, x9, x8, lsl #32
        lsr x3, x8, #32
        adcs x10, x10, x3
        mul x2, x8, x5
        umulh x8, x8, x5
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x3, x9, #32
        adcs x11, x11, x3
        mul x2, x9, x5
        umulh x9, x9, x5
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x3, x10, #32
        adcs x8, x8, x3
        mul x2, x10, x5
        umulh x10, x10, x5
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x3, x11, #32
        adcs x9, x9, x3
        mul x2, x11, x5
        umulh x11, x11, x5
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        cset x2, cs
        mov x3, #0xffffffff
        adds x12, x8, #0x1
        sbcs x13, x9, x3
        sbcs x14, x10, xzr
        sbcs x7, x11, x5
        sbcs xzr, x2, xzr
        csel x8, x8, x12, cc
        csel x9, x9, x13, cc
        csel x10, x10, x14, cc
        csel x11, x11, x7, cc
        stp x8, x9, [sp]
        stp x10, x11, [sp, #16]
        ldp x3, x4, [sp, #160]
        ldp x7, x8, [sp, #96]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [sp, #112]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #176]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #96]
        stp x14, x0, [sp, #112]
        ldp x5, x6, [sp, #64]
        ldp x4, x3, [sp, #32]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #80]
        ldp x4, x3, [sp, #48]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        and x4, x3, #0xffffffff
        adcs x6, x6, x4
        adcs x7, x7, xzr
        and x4, x3, #0xffffffff00000001
        adc x8, x8, x4
        stp x5, x6, [x15, #64]
        stp x7, x8, [x15, #80]
        ldp x1, x2, [sp, #128]
        lsl x0, x1, #2
        ldp x6, x7, [sp, #160]
        subs x0, x0, x6
        extr x1, x2, x1, #62
        sbcs x1, x1, x7
        ldp x3, x4, [sp, #144]
        extr x2, x3, x2, #62
        ldp x6, x7, [sp, #176]
        sbcs x2, x2, x6
        extr x3, x4, x3, #62
        sbcs x3, x3, x7
        lsr x4, x4, #62
        sbc x4, x4, xzr
        add x5, x4, #0x1
        lsl x8, x5, #32
        negs x6, x8
        ngcs x7, xzr
        sbc x8, x8, x5
        adds x0, x0, x5
        adcs x1, x1, x6
        adcs x2, x2, x7
        adcs x3, x3, x8
        csetm x5, cc
        adds x0, x0, x5
        and x6, x5, #0xffffffff
        adcs x1, x1, x6
        adcs x2, x2, xzr
        neg x7, x6
        adc x3, x3, x7
        stp x0, x1, [x15]
        stp x2, x3, [x15, #16]
        mov x1, #0x8
        mov x2, #0xffffffffffffffff
        ldp x9, x10, [sp]
        subs x9, x2, x9
        mov x2, #0xffffffff
        sbcs x10, x2, x10
        ldp x11, x12, [sp, #16]
        ngcs x11, x11
        mov x2, #0xffffffff00000001
        sbc x12, x2, x12
        lsl x3, x9, #3
        extr x4, x10, x9, #61
        extr x5, x11, x10, #61
        extr x6, x12, x11, #61
        lsr x7, x12, #61
        mov x1, #0x3
        ldp x9, x10, [sp, #96]
        mul x8, x9, x1
        umulh x9, x9, x1
        adds x3, x3, x8
        mul x8, x10, x1
        umulh x10, x10, x1
        adcs x4, x4, x8
        ldp x11, x12, [sp, #112]
        mul x8, x11, x1
        umulh x11, x11, x1
        adcs x5, x5, x8
        mul x8, x12, x1
        umulh x12, x12, x1
        adcs x6, x6, x8
        adc x7, x7, xzr
        adds x4, x4, x9
        adcs x5, x5, x10
        adcs x6, x6, x11
        adc x7, x7, x12
        add x8, x7, #0x1
        lsl x10, x8, #32
        adds x6, x6, x10
        adc x7, x7, xzr
        neg x9, x8
        sub x10, x10, #0x1
        subs x3, x3, x9
        sbcs x4, x4, x10
        sbcs x5, x5, xzr
        sbcs x6, x6, x8
        sbc x8, x7, x8
        adds x3, x3, x8
        and x9, x8, #0xffffffff
        adcs x4, x4, x9
        adcs x5, x5, xzr
        neg x10, x9
        adc x6, x6, x10
        stp x3, x4, [x15, #32]
        stp x5, x6, [x15, #48]
        mov x16, sp
        mov x17, #12
.Lrscrypto_p256_scalarmul_alt_clear_double_frame:
        stp xzr, xzr, [x16], #16
        subs x17, x17, #1
        b.ne .Lrscrypto_p256_scalarmul_alt_clear_double_frame
        add sp, sp, #(192 +0)
        .cfi_adjust_cfa_offset -192
        ret
        .cfi_endproc





Lp256_scalarmul_alt_local_p256_montjmixadd:
        .cfi_startproc
        sub sp, sp, #(192 +0)
        .cfi_adjust_cfa_offset 192
        mov x15, x0
        mov x16, x1
        mov x17, x2
        ldp x2, x3, [x16, #64]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [x16, #80]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        adds x9, x9, x8, lsl #32
        lsr x3, x8, #32
        adcs x10, x10, x3
        mov x3, #0xffffffff00000001
        mul x2, x8, x3
        umulh x8, x8, x3
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x3, x9, #32
        adcs x11, x11, x3
        mov x3, #0xffffffff00000001
        mul x2, x9, x3
        umulh x9, x9, x3
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x3, x10, #32
        adcs x8, x8, x3
        mov x3, #0xffffffff00000001
        mul x2, x10, x3
        umulh x10, x10, x3
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x3, x11, #32
        adcs x9, x9, x3
        mov x3, #0xffffffff00000001
        mul x2, x11, x3
        umulh x11, x11, x3
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        mov x2, #0xffffffffffffffff
        csel x2, xzr, x2, cc
        mov x3, #0xffffffff
        csel x3, xzr, x3, cc
        mov x5, #0xffffffff00000001
        csel x5, xzr, x5, cc
        subs x8, x8, x2
        sbcs x9, x9, x3
        sbcs x10, x10, xzr
        sbc x11, x11, x5
        stp x8, x9, [sp]
        stp x10, x11, [sp, #16]
        ldp x3, x4, [x16, #64]
        ldp x7, x8, [x17, #32]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [x17, #48]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [x16, #80]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #32]
        stp x14, x0, [sp, #48]
        ldp x3, x4, [sp]
        ldp x7, x8, [x17]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [x17, #16]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #16]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #64]
        stp x14, x0, [sp, #80]
        ldp x3, x4, [sp]
        ldp x7, x8, [sp, #32]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [sp, #48]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #16]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #32]
        stp x14, x0, [sp, #48]
        ldp x5, x6, [sp, #64]
        ldp x4, x3, [x16]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #80]
        ldp x4, x3, [x16, #16]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp, #160]
        stp x7, x8, [sp, #176]
        ldp x5, x6, [sp, #32]
        ldp x4, x3, [x16, #32]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #48]
        ldp x4, x3, [x16, #48]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp, #32]
        stp x7, x8, [sp, #48]
        ldp x2, x3, [sp, #160]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [sp, #176]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        adds x9, x9, x8, lsl #32
        lsr x3, x8, #32
        adcs x10, x10, x3
        mov x3, #0xffffffff00000001
        mul x2, x8, x3
        umulh x8, x8, x3
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x3, x9, #32
        adcs x11, x11, x3
        mov x3, #0xffffffff00000001
        mul x2, x9, x3
        umulh x9, x9, x3
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x3, x10, #32
        adcs x8, x8, x3
        mov x3, #0xffffffff00000001
        mul x2, x10, x3
        umulh x10, x10, x3
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x3, x11, #32
        adcs x9, x9, x3
        mov x3, #0xffffffff00000001
        mul x2, x11, x3
        umulh x11, x11, x3
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        mov x2, #0xffffffffffffffff
        csel x2, xzr, x2, cc
        mov x3, #0xffffffff
        csel x3, xzr, x3, cc
        mov x5, #0xffffffff00000001
        csel x5, xzr, x5, cc
        subs x8, x8, x2
        sbcs x9, x9, x3
        sbcs x10, x10, xzr
        sbc x11, x11, x5
        stp x8, x9, [sp, #96]
        stp x10, x11, [sp, #112]
        ldp x2, x3, [sp, #32]
        mul x9, x2, x3
        umulh x10, x2, x3
        ldp x4, x5, [sp, #48]
        mul x11, x2, x5
        umulh x12, x2, x5
        mul x6, x2, x4
        umulh x7, x2, x4
        adds x10, x10, x6
        adcs x11, x11, x7
        mul x6, x3, x4
        umulh x7, x3, x4
        adc x7, x7, xzr
        adds x11, x11, x6
        mul x13, x4, x5
        umulh x14, x4, x5
        adcs x12, x12, x7
        mul x6, x3, x5
        umulh x7, x3, x5
        adc x7, x7, xzr
        adds x12, x12, x6
        adcs x13, x13, x7
        adc x14, x14, xzr
        adds x9, x9, x9
        adcs x10, x10, x10
        adcs x11, x11, x11
        adcs x12, x12, x12
        adcs x13, x13, x13
        adcs x14, x14, x14
        cset x7, cs
        umulh x6, x2, x2
        mul x8, x2, x2
        adds x9, x9, x6
        mul x6, x3, x3
        adcs x10, x10, x6
        umulh x6, x3, x3
        adcs x11, x11, x6
        mul x6, x4, x4
        adcs x12, x12, x6
        umulh x6, x4, x4
        adcs x13, x13, x6
        mul x6, x5, x5
        adcs x14, x14, x6
        umulh x6, x5, x5
        adc x7, x7, x6
        adds x9, x9, x8, lsl #32
        lsr x3, x8, #32
        adcs x10, x10, x3
        mov x3, #0xffffffff00000001
        mul x2, x8, x3
        umulh x8, x8, x3
        adcs x11, x11, x2
        adc x8, x8, xzr
        adds x10, x10, x9, lsl #32
        lsr x3, x9, #32
        adcs x11, x11, x3
        mov x3, #0xffffffff00000001
        mul x2, x9, x3
        umulh x9, x9, x3
        adcs x8, x8, x2
        adc x9, x9, xzr
        adds x11, x11, x10, lsl #32
        lsr x3, x10, #32
        adcs x8, x8, x3
        mov x3, #0xffffffff00000001
        mul x2, x10, x3
        umulh x10, x10, x3
        adcs x9, x9, x2
        adc x10, x10, xzr
        adds x8, x8, x11, lsl #32
        lsr x3, x11, #32
        adcs x9, x9, x3
        mov x3, #0xffffffff00000001
        mul x2, x11, x3
        umulh x11, x11, x3
        adcs x10, x10, x2
        adc x11, x11, xzr
        adds x8, x8, x12
        adcs x9, x9, x13
        adcs x10, x10, x14
        adcs x11, x11, x7
        cset x2, cs
        mov x3, #0xffffffff
        mov x5, #0xffffffff00000001
        adds x12, x8, #0x1
        sbcs x13, x9, x3
        sbcs x14, x10, xzr
        sbcs x7, x11, x5
        sbcs xzr, x2, xzr
        csel x8, x8, x12, cc
        csel x9, x9, x13, cc
        csel x10, x10, x14, cc
        csel x11, x11, x7, cc
        stp x8, x9, [sp]
        stp x10, x11, [sp, #16]
        ldp x3, x4, [sp, #96]
        ldp x7, x8, [x16]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [x16, #16]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #112]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #128]
        stp x14, x0, [sp, #144]
        ldp x3, x4, [sp, #96]
        ldp x7, x8, [sp, #64]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [sp, #80]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #112]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #64]
        stp x14, x0, [sp, #80]
        ldp x5, x6, [sp]
        ldp x4, x3, [sp, #128]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #16]
        ldp x4, x3, [sp, #144]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp]
        stp x7, x8, [sp, #16]
        ldp x5, x6, [sp, #64]
        ldp x4, x3, [sp, #128]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #80]
        ldp x4, x3, [sp, #144]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp, #96]
        stp x7, x8, [sp, #112]
        ldp x3, x4, [sp, #160]
        ldp x7, x8, [x16, #64]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [x16, #80]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #176]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #160]
        stp x14, x0, [sp, #176]
        ldp x5, x6, [sp]
        ldp x4, x3, [sp, #64]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #16]
        ldp x4, x3, [sp, #80]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp]
        stp x7, x8, [sp, #16]
        ldp x5, x6, [sp, #128]
        ldp x4, x3, [sp]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #144]
        ldp x4, x3, [sp, #16]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp, #128]
        stp x7, x8, [sp, #144]
        ldp x3, x4, [sp, #96]
        ldp x7, x8, [x16, #32]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [x16, #48]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #112]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #96]
        stp x14, x0, [sp, #112]
        ldp x3, x4, [sp, #32]
        ldp x7, x8, [sp, #128]
        mul x12, x3, x7
        umulh x13, x3, x7
        mul x11, x3, x8
        umulh x14, x3, x8
        adds x13, x13, x11
        ldp x9, x10, [sp, #144]
        mul x11, x3, x9
        umulh x0, x3, x9
        adcs x14, x14, x11
        mul x11, x3, x10
        umulh x1, x3, x10
        adcs x0, x0, x11
        adc x1, x1, xzr
        ldp x5, x6, [sp, #48]
        mul x11, x4, x7
        adds x13, x13, x11
        mul x11, x4, x8
        adcs x14, x14, x11
        mul x11, x4, x9
        adcs x0, x0, x11
        mul x11, x4, x10
        adcs x1, x1, x11
        umulh x3, x4, x10
        adc x3, x3, xzr
        umulh x11, x4, x7
        adds x14, x14, x11
        umulh x11, x4, x8
        adcs x0, x0, x11
        umulh x11, x4, x9
        adcs x1, x1, x11
        adc x3, x3, xzr
        mul x11, x5, x7
        adds x14, x14, x11
        mul x11, x5, x8
        adcs x0, x0, x11
        mul x11, x5, x9
        adcs x1, x1, x11
        mul x11, x5, x10
        adcs x3, x3, x11
        umulh x4, x5, x10
        adc x4, x4, xzr
        umulh x11, x5, x7
        adds x0, x0, x11
        umulh x11, x5, x8
        adcs x1, x1, x11
        umulh x11, x5, x9
        adcs x3, x3, x11
        adc x4, x4, xzr
        mul x11, x6, x7
        adds x0, x0, x11
        mul x11, x6, x8
        adcs x1, x1, x11
        mul x11, x6, x9
        adcs x3, x3, x11
        mul x11, x6, x10
        adcs x4, x4, x11
        umulh x5, x6, x10
        adc x5, x5, xzr
        mov x10, #0xffffffff00000001
        adds x13, x13, x12, lsl #32
        lsr x11, x12, #32
        adcs x14, x14, x11
        mul x11, x12, x10
        umulh x12, x12, x10
        adcs x0, x0, x11
        adc x12, x12, xzr
        umulh x11, x6, x7
        adds x1, x1, x11
        umulh x11, x6, x8
        adcs x3, x3, x11
        umulh x11, x6, x9
        adcs x4, x4, x11
        adc x5, x5, xzr
        adds x14, x14, x13, lsl #32
        lsr x11, x13, #32
        adcs x0, x0, x11
        mul x11, x13, x10
        umulh x13, x13, x10
        adcs x12, x12, x11
        adc x13, x13, xzr
        adds x0, x0, x14, lsl #32
        lsr x11, x14, #32
        adcs x12, x12, x11
        mul x11, x14, x10
        umulh x14, x14, x10
        adcs x13, x13, x11
        adc x14, x14, xzr
        adds x12, x12, x0, lsl #32
        lsr x11, x0, #32
        adcs x13, x13, x11
        mul x11, x0, x10
        umulh x0, x0, x10
        adcs x14, x14, x11
        adc x0, x0, xzr
        adds x12, x12, x1
        adcs x13, x13, x3
        adcs x14, x14, x4
        adcs x0, x0, x5
        cset x8, cs
        mov x11, #0xffffffff
        adds x1, x12, #0x1
        sbcs x3, x13, x11
        sbcs x4, x14, xzr
        sbcs x5, x0, x10
        sbcs xzr, x8, xzr
        csel x12, x12, x1, cc
        csel x13, x13, x3, cc
        csel x14, x14, x4, cc
        csel x0, x0, x5, cc
        stp x12, x13, [sp, #128]
        stp x14, x0, [sp, #144]
        ldp x5, x6, [sp, #128]
        ldp x4, x3, [sp, #96]
        subs x5, x5, x4
        sbcs x6, x6, x3
        ldp x7, x8, [sp, #144]
        ldp x4, x3, [sp, #112]
        sbcs x7, x7, x4
        sbcs x8, x8, x3
        csetm x3, cc
        adds x5, x5, x3
        mov x4, #0xffffffff
        and x4, x4, x3
        adcs x6, x6, x4
        adcs x7, x7, xzr
        mov x4, #0xffffffff00000001
        and x4, x4, x3
        adc x8, x8, x4
        stp x5, x6, [sp, #128]
        stp x7, x8, [sp, #144]
        ldp x0, x1, [x16, #64]
        ldp x2, x3, [x16, #80]
        orr x4, x0, x1
        orr x5, x2, x3
        orr x4, x4, x5
        cmp x4, xzr
        ldp x0, x1, [sp]
        ldp x12, x13, [x17]
        csel x0, x0, x12, ne
        csel x1, x1, x13, ne
        ldp x2, x3, [sp, #16]
        ldp x12, x13, [x17, #16]
        csel x2, x2, x12, ne
        csel x3, x3, x13, ne
        ldp x4, x5, [sp, #128]
        ldp x12, x13, [x17, #32]
        csel x4, x4, x12, ne
        csel x5, x5, x13, ne
        ldp x6, x7, [sp, #144]
        ldp x12, x13, [x17, #48]
        csel x6, x6, x12, ne
        csel x7, x7, x13, ne
        ldp x8, x9, [sp, #160]
        mov x12, #0x1
        mov x13, #0xffffffff00000000
        csel x8, x8, x12, ne
        csel x9, x9, x13, ne
        ldp x10, x11, [sp, #176]
        mov x12, #0xffffffffffffffff
        mov x13, #0xfffffffe
        csel x10, x10, x12, ne
        csel x11, x11, x13, ne
        stp x0, x1, [x15]
        stp x2, x3, [x15, #16]
        stp x4, x5, [x15, #32]
        stp x6, x7, [x15, #48]
        stp x8, x9, [x15, #64]
        stp x10, x11, [x15, #80]
        mov x16, sp
        mov x17, #12
.Lrscrypto_p256_scalarmul_alt_clear_mixadd_frame:
        stp xzr, xzr, [x16], #16
        subs x17, x17, #1
        b.ne .Lrscrypto_p256_scalarmul_alt_clear_mixadd_frame
        add sp, sp, #(192 +0)
        .cfi_adjust_cfa_offset -192
        ret
        .cfi_endproc
