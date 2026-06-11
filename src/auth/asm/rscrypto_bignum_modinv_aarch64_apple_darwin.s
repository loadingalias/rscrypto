// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - generic/bignum_modinv.S
//
// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.

        .globl _rscrypto_bignum_modinv

        .private_extern _rscrypto_bignum_modinv
        .text
        .balign 4
_rscrypto_bignum_modinv:
        .cfi_startproc



        stp x19, x20, [sp, #-16]! %% .cfi_adjust_cfa_offset 16 %% .cfi_rel_offset x19, 0 %% .cfi_rel_offset x20, 8
        stp x21, x22, [sp, #-16]! %% .cfi_adjust_cfa_offset 16 %% .cfi_rel_offset x21, 0 %% .cfi_rel_offset x22, 8



        cbz x0, Lbignum_modinv_end



        lsl x10, x0, #3
        add x21, x4, x10
        add x22, x21, x10




        mov x10, xzr
Lbignum_modinv_copyloop:
        ldr x11, [x2, x10, lsl #3]
        ldr x12, [x3, x10, lsl #3]
        str x11, [x21, x10, lsl #3]
        str x12, [x22, x10, lsl #3]
        str x12, [x4, x10, lsl #3]
        str xzr, [x1, x10, lsl #3]
        add x10, x10, #1
        cmp x10, x0
        bcc Lbignum_modinv_copyloop
        ldr x11, [x4]
        sub x12, x11, #1
        str x12, [x4]




        lsl x20, x11, #2
        sub x20, x11, x20
        eor x20, x20, #2
        mov x12, #1
        madd x12, x11, x20, x12
        mul x11, x12, x12
        madd x20, x12, x20, x20
        mul x12, x11, x11
        madd x20, x11, x20, x20
        mul x11, x12, x12
        madd x20, x12, x20, x20
        madd x20, x11, x20, x20




        lsl x2, x0, #7



Lbignum_modinv_outerloop:






        add x10, x2, #63
        lsr x5, x10, #6
        cmp x5, x0
        csel x5, x0, x5, cs







        mov x13, xzr
        mov x15, xzr
        mov x14, xzr
        mov x16, xzr
        mov x19, xzr

        mov x10, xzr
Lbignum_modinv_toploop:
        ldr x11, [x21, x10, lsl #3]
        ldr x12, [x22, x10, lsl #3]
        orr x17, x11, x12
        cmp x17, xzr
        and x17, x19, x13
        csel x15, x17, x15, ne
        and x17, x19, x14
        csel x16, x17, x16, ne
        csel x13, x11, x13, ne
        csel x14, x12, x14, ne
        csetm x19, ne
        add x10, x10, #1
        cmp x10, x5
        bcc Lbignum_modinv_toploop

        orr x11, x13, x14
        clz x12, x11
        negs x17, x12
        lsl x13, x13, x12
        csel x15, x15, xzr, ne
        lsl x14, x14, x12
        csel x16, x16, xzr, ne
        lsr x15, x15, x17
        lsr x16, x16, x17
        orr x13, x13, x15
        orr x14, x14, x16

        ldr x15, [x21]
        ldr x16, [x22]
        mov x6, #1
        mov x7, xzr
        mov x8, xzr
        mov x9, #1

        mov x10, #58
        ands xzr, x15, #1

Lbignum_modinv_innerloop:




        csel x11, x14, xzr, ne
        csel x12, x16, xzr, ne
        csel x17, x8, xzr, ne
        csel x19, x9, xzr, ne
        ccmp x13, x14, #0x2, ne



        sub x11, x13, x11
        sub x12, x15, x12




        csel x14, x14, x13, cs
        cneg x11, x11, cc
        csel x16, x16, x15, cs
        cneg x15, x12, cc
        csel x8, x8, x6, cs
        csel x9, x9, x7, cs





        ands xzr, x12, #2
        add x6, x6, x17
        add x7, x7, x19
        lsr x13, x11, #1
        lsr x15, x15, #1
        add x8, x8, x8
        add x9, x9, x9



        sub x10, x10, #1
        cbnz x10, Lbignum_modinv_innerloop
        mov x13, xzr
        mov x14, xzr
        mov x17, xzr
        mov x19, xzr

        mov x10, xzr
Lbignum_modinv_congloop:
        ldr x11, [x4, x10, lsl #3]
        ldr x12, [x1, x10, lsl #3]

        mul x15, x6, x11
        mul x16, x7, x12
        adds x15, x15, x13
        umulh x13, x6, x11
        adc x13, x13, xzr
        adds x15, x15, x16
        extr x17, x15, x17, #58
        str x17, [x4, x10, lsl #3]
        mov x17, x15
        umulh x15, x7, x12
        adc x13, x13, x15

        mul x15, x8, x11
        mul x16, x9, x12
        adds x15, x15, x14
        umulh x14, x8, x11
        adc x14, x14, xzr
        adds x15, x15, x16
        extr x19, x15, x19, #58
        str x19, [x1, x10, lsl #3]
        mov x19, x15
        umulh x15, x9, x12
        adc x14, x14, x15

        add x10, x10, #1
        cmp x10, x0
        bcc Lbignum_modinv_congloop

        extr x13, x13, x17, #58
        extr x14, x14, x19, #58



        ldr x11, [x4]
        mul x17, x11, x20
        ldr x12, [x3]
        mul x15, x17, x12
        umulh x16, x17, x12
        adds x11, x11, x15

        mov x10, #1
        sub x11, x0, #1
        cbz x11, Lbignum_modinv_wmontend
Lbignum_modinv_wmontloop:
        ldr x11, [x3, x10, lsl #3]
        ldr x12, [x4, x10, lsl #3]
        mul x15, x17, x11
        adcs x12, x12, x16
        umulh x16, x17, x11
        adc x16, x16, xzr
        adds x12, x12, x15
        sub x15, x10, #1
        str x12, [x4, x15, lsl #3]
        add x10, x10, #1
        sub x11, x10, x0
        cbnz x11, Lbignum_modinv_wmontloop
Lbignum_modinv_wmontend:
        adcs x16, x16, x13
        adc x13, xzr, xzr
        sub x15, x10, #1
        str x16, [x4, x15, lsl #3]

        subs x10, xzr, xzr
Lbignum_modinv_wcmploop:
        ldr x11, [x4, x10, lsl #3]
        ldr x12, [x3, x10, lsl #3]
        sbcs xzr, x11, x12
        add x10, x10, #1
        sub x11, x10, x0
        cbnz x11, Lbignum_modinv_wcmploop

        sbcs xzr, x13, xzr
        csetm x13, cs

        subs x10, xzr, xzr
Lbignum_modinv_wcorrloop:
        ldr x11, [x4, x10, lsl #3]
        ldr x12, [x3, x10, lsl #3]
        and x12, x12, x13
        sbcs x11, x11, x12
        str x11, [x4, x10, lsl #3]
        add x10, x10, #1
        sub x11, x10, x0
        cbnz x11, Lbignum_modinv_wcorrloop



        ldr x11, [x1]
        mul x17, x11, x20
        ldr x12, [x3]
        mul x15, x17, x12
        umulh x16, x17, x12
        adds x11, x11, x15

        mov x10, #1
        sub x11, x0, #1
        cbz x11, Lbignum_modinv_zmontend
Lbignum_modinv_zmontloop:
        ldr x11, [x3, x10, lsl #3]
        ldr x12, [x1, x10, lsl #3]
        mul x15, x17, x11
        adcs x12, x12, x16
        umulh x16, x17, x11
        adc x16, x16, xzr
        adds x12, x12, x15
        sub x15, x10, #1
        str x12, [x1, x15, lsl #3]
        add x10, x10, #1
        sub x11, x10, x0
        cbnz x11, Lbignum_modinv_zmontloop
Lbignum_modinv_zmontend:
        adcs x16, x16, x14
        adc x14, xzr, xzr
        sub x15, x10, #1
        str x16, [x1, x15, lsl #3]

        subs x10, xzr, xzr
Lbignum_modinv_zcmploop:
        ldr x11, [x1, x10, lsl #3]
        ldr x12, [x3, x10, lsl #3]
        sbcs xzr, x11, x12
        add x10, x10, #1
        sub x11, x10, x0
        cbnz x11, Lbignum_modinv_zcmploop

        sbcs xzr, x14, xzr
        csetm x14, cs

        subs x10, xzr, xzr
Lbignum_modinv_zcorrloop:
        ldr x11, [x1, x10, lsl #3]
        ldr x12, [x3, x10, lsl #3]
        and x12, x12, x14
        sbcs x11, x11, x12
        str x11, [x1, x10, lsl #3]
        add x10, x10, #1
        sub x11, x10, x0
        cbnz x11, Lbignum_modinv_zcorrloop
        mov x13, xzr
        mov x14, xzr
        mov x17, xzr
        mov x19, xzr
        mov x10, xzr
Lbignum_modinv_crossloop:
        ldr x11, [x21, x10, lsl #3]
        ldr x12, [x22, x10, lsl #3]

        mul x15, x6, x11
        mul x16, x7, x12
        adds x15, x15, x13
        umulh x13, x6, x11
        adc x13, x13, xzr
        subs x15, x15, x16
        str x15, [x21, x10, lsl #3]
        umulh x15, x7, x12
        sub x17, x15, x17
        sbcs x13, x13, x17
        csetm x17, cc

        mul x15, x8, x11
        mul x16, x9, x12
        adds x15, x15, x14
        umulh x14, x8, x11
        adc x14, x14, xzr
        subs x15, x15, x16
        str x15, [x22, x10, lsl #3]
        umulh x15, x9, x12
        sub x19, x15, x19
        sbcs x14, x14, x19
        csetm x19, cc

        add x10, x10, #1
        cmp x10, x5
        bcc Lbignum_modinv_crossloop



        adds xzr, x17, x17

        ldr x15, [x21]
        mov x10, xzr
        sub x6, x5, #1
        cbz x6, Lbignum_modinv_negskip1

Lbignum_modinv_negloop1:
        add x11, x10, #8
        ldr x12, [x21, x11]
        extr x15, x12, x15, #58
        eor x15, x15, x17
        adcs x15, x15, xzr
        str x15, [x21, x10]
        mov x15, x12
        add x10, x10, #8
        sub x6, x6, #1
        cbnz x6, Lbignum_modinv_negloop1
Lbignum_modinv_negskip1:
        extr x15, x13, x15, #58
        eor x15, x15, x17
        adcs x15, x15, xzr
        str x15, [x21, x10]



        adds xzr, x19, x19

        ldr x15, [x22]
        mov x10, xzr
        sub x6, x5, #1
        cbz x6, Lbignum_modinv_negskip2
Lbignum_modinv_negloop2:
        add x11, x10, #8
        ldr x12, [x22, x11]
        extr x15, x12, x15, #58
        eor x15, x15, x19
        adcs x15, x15, xzr
        str x15, [x22, x10]
        mov x15, x12
        add x10, x10, #8
        sub x6, x6, #1
        cbnz x6, Lbignum_modinv_negloop2
Lbignum_modinv_negskip2:
        extr x15, x14, x15, #58
        eor x15, x15, x19
        adcs x15, x15, xzr
        str x15, [x22, x10]
        mov x10, xzr
        adds xzr, x17, x17
Lbignum_modinv_wfliploop:
        ldr x11, [x3, x10, lsl #3]
        ldr x12, [x4, x10, lsl #3]
        and x11, x11, x17
        eor x12, x12, x17
        adcs x11, x11, x12
        str x11, [x4, x10, lsl #3]
        add x10, x10, #1
        sub x11, x10, x0
        cbnz x11, Lbignum_modinv_wfliploop

        mvn x19, x19

        mov x10, xzr
        adds xzr, x19, x19
Lbignum_modinv_zfliploop:
        ldr x11, [x3, x10, lsl #3]
        ldr x12, [x1, x10, lsl #3]
        and x11, x11, x19
        eor x12, x12, x19
        adcs x11, x11, x12
        str x11, [x1, x10, lsl #3]
        add x10, x10, #1
        sub x11, x10, x0
        cbnz x11, Lbignum_modinv_zfliploop







        subs x2, x2, #58
        bhi Lbignum_modinv_outerloop

Lbignum_modinv_end:
        ldp x21, x22, [sp], #16 %% .cfi_adjust_cfa_offset -16 %% .cfi_restore x21 %% .cfi_restore x22
        ldp x19, x20, [sp], #16 %% .cfi_adjust_cfa_offset -16 %% .cfi_restore x19 %% .cfi_restore x20

        ret %% .cfi_endproc
