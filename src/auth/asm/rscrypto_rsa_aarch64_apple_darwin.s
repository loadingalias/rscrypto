// macOS AArch64 RSA Montgomery arithmetic for rscrypto.
//
// ABI:
//   x0 = out[32] mut u64
//   x1 = a[32] const u64
//   x2 = b[32] const u64
//   x3 = modulus[32] const u64
//   x4 = n0 Montgomery word
//   x5 = t[66] mut u64
//
// Computes out = a * b * R^-1 mod modulus using CIOS Montgomery reduction.

        .macro RSCRYPTO_RSA_CIOS32_MUL_STEP offset
        ldr x10, [x1, #\offset]
        ldr x11, [x15, #\offset]
        mul x12, x10, x7
        umulh x13, x10, x7
        adds x12, x12, x11
        adc x13, x13, xzr
        adds x12, x12, x8
        adc x8, x13, xzr
        str x12, [x15, #\offset]
        .endmacro

        .macro RSCRYPTO_RSA_CIOS32_REDC_STEP offset
        ldr x10, [x3, #\offset]
        ldr x11, [x15, #\offset]
        mul x12, x10, x7
        umulh x13, x10, x7
        adds x12, x12, x11
        adc x13, x13, xzr
        adds x12, x12, x8
        adc x8, x13, xzr
        str x12, [x15, #\offset]
        .endmacro

        .globl _rscrypto_rsa_mont_mul_cios_32_aarch64_apple_darwin
        .private_extern _rscrypto_rsa_mont_mul_cios_32_aarch64_apple_darwin
        .text
        .balign 4
_rscrypto_rsa_mont_mul_cios_32_aarch64_apple_darwin:
        stp xzr, xzr, [x5, #0]
        stp xzr, xzr, [x5, #16]
        stp xzr, xzr, [x5, #32]
        stp xzr, xzr, [x5, #48]
        stp xzr, xzr, [x5, #64]
        stp xzr, xzr, [x5, #80]
        stp xzr, xzr, [x5, #96]
        stp xzr, xzr, [x5, #112]
        stp xzr, xzr, [x5, #128]
        stp xzr, xzr, [x5, #144]
        stp xzr, xzr, [x5, #160]
        stp xzr, xzr, [x5, #176]
        stp xzr, xzr, [x5, #192]
        stp xzr, xzr, [x5, #208]
        stp xzr, xzr, [x5, #224]
        stp xzr, xzr, [x5, #240]
        stp xzr, xzr, [x5, #256]
        stp xzr, xzr, [x5, #272]
        stp xzr, xzr, [x5, #288]
        stp xzr, xzr, [x5, #304]
        stp xzr, xzr, [x5, #320]
        stp xzr, xzr, [x5, #336]
        stp xzr, xzr, [x5, #352]
        stp xzr, xzr, [x5, #368]
        stp xzr, xzr, [x5, #384]
        stp xzr, xzr, [x5, #400]
        stp xzr, xzr, [x5, #416]
        stp xzr, xzr, [x5, #432]
        stp xzr, xzr, [x5, #448]
        stp xzr, xzr, [x5, #464]
        stp xzr, xzr, [x5, #480]
        stp xzr, xzr, [x5, #496]
        str xzr, [x5, #512]
        str xzr, [x5, #520]

        mov x6, #0
        mov x15, x5
Lrscrypto_rsa_cios32_outer:
        ldr x7, [x2, x6, lsl #3]
        mov x8, #0
        RSCRYPTO_RSA_CIOS32_MUL_STEP 0
        RSCRYPTO_RSA_CIOS32_MUL_STEP 8
        RSCRYPTO_RSA_CIOS32_MUL_STEP 16
        RSCRYPTO_RSA_CIOS32_MUL_STEP 24
        RSCRYPTO_RSA_CIOS32_MUL_STEP 32
        RSCRYPTO_RSA_CIOS32_MUL_STEP 40
        RSCRYPTO_RSA_CIOS32_MUL_STEP 48
        RSCRYPTO_RSA_CIOS32_MUL_STEP 56
        RSCRYPTO_RSA_CIOS32_MUL_STEP 64
        RSCRYPTO_RSA_CIOS32_MUL_STEP 72
        RSCRYPTO_RSA_CIOS32_MUL_STEP 80
        RSCRYPTO_RSA_CIOS32_MUL_STEP 88
        RSCRYPTO_RSA_CIOS32_MUL_STEP 96
        RSCRYPTO_RSA_CIOS32_MUL_STEP 104
        RSCRYPTO_RSA_CIOS32_MUL_STEP 112
        RSCRYPTO_RSA_CIOS32_MUL_STEP 120
        RSCRYPTO_RSA_CIOS32_MUL_STEP 128
        RSCRYPTO_RSA_CIOS32_MUL_STEP 136
        RSCRYPTO_RSA_CIOS32_MUL_STEP 144
        RSCRYPTO_RSA_CIOS32_MUL_STEP 152
        RSCRYPTO_RSA_CIOS32_MUL_STEP 160
        RSCRYPTO_RSA_CIOS32_MUL_STEP 168
        RSCRYPTO_RSA_CIOS32_MUL_STEP 176
        RSCRYPTO_RSA_CIOS32_MUL_STEP 184
        RSCRYPTO_RSA_CIOS32_MUL_STEP 192
        RSCRYPTO_RSA_CIOS32_MUL_STEP 200
        RSCRYPTO_RSA_CIOS32_MUL_STEP 208
        RSCRYPTO_RSA_CIOS32_MUL_STEP 216
        RSCRYPTO_RSA_CIOS32_MUL_STEP 224
        RSCRYPTO_RSA_CIOS32_MUL_STEP 232
        RSCRYPTO_RSA_CIOS32_MUL_STEP 240
        RSCRYPTO_RSA_CIOS32_MUL_STEP 248

        ldr x10, [x15, #256]
        adds x10, x10, x8
        str x10, [x15, #256]
        cset x11, cs
        str x11, [x15, #264]

        ldr x7, [x15]
        mul x7, x7, x4
        mov x8, #0
        RSCRYPTO_RSA_CIOS32_REDC_STEP 0
        RSCRYPTO_RSA_CIOS32_REDC_STEP 8
        RSCRYPTO_RSA_CIOS32_REDC_STEP 16
        RSCRYPTO_RSA_CIOS32_REDC_STEP 24
        RSCRYPTO_RSA_CIOS32_REDC_STEP 32
        RSCRYPTO_RSA_CIOS32_REDC_STEP 40
        RSCRYPTO_RSA_CIOS32_REDC_STEP 48
        RSCRYPTO_RSA_CIOS32_REDC_STEP 56
        RSCRYPTO_RSA_CIOS32_REDC_STEP 64
        RSCRYPTO_RSA_CIOS32_REDC_STEP 72
        RSCRYPTO_RSA_CIOS32_REDC_STEP 80
        RSCRYPTO_RSA_CIOS32_REDC_STEP 88
        RSCRYPTO_RSA_CIOS32_REDC_STEP 96
        RSCRYPTO_RSA_CIOS32_REDC_STEP 104
        RSCRYPTO_RSA_CIOS32_REDC_STEP 112
        RSCRYPTO_RSA_CIOS32_REDC_STEP 120
        RSCRYPTO_RSA_CIOS32_REDC_STEP 128
        RSCRYPTO_RSA_CIOS32_REDC_STEP 136
        RSCRYPTO_RSA_CIOS32_REDC_STEP 144
        RSCRYPTO_RSA_CIOS32_REDC_STEP 152
        RSCRYPTO_RSA_CIOS32_REDC_STEP 160
        RSCRYPTO_RSA_CIOS32_REDC_STEP 168
        RSCRYPTO_RSA_CIOS32_REDC_STEP 176
        RSCRYPTO_RSA_CIOS32_REDC_STEP 184
        RSCRYPTO_RSA_CIOS32_REDC_STEP 192
        RSCRYPTO_RSA_CIOS32_REDC_STEP 200
        RSCRYPTO_RSA_CIOS32_REDC_STEP 208
        RSCRYPTO_RSA_CIOS32_REDC_STEP 216
        RSCRYPTO_RSA_CIOS32_REDC_STEP 224
        RSCRYPTO_RSA_CIOS32_REDC_STEP 232
        RSCRYPTO_RSA_CIOS32_REDC_STEP 240
        RSCRYPTO_RSA_CIOS32_REDC_STEP 248

        ldr x10, [x15, #256]
        adds x10, x10, x8
        str x10, [x15, #256]
        ldr x11, [x15, #264]
        adc x11, x11, xzr
        str x11, [x15, #264]

        add x6, x6, #1
        add x15, x15, #8
        cmp x6, #32
        b.ne Lrscrypto_rsa_cios32_outer

        ldp x10, x11, [x5, #256]
        stp x10, x11, [x0, #0]
        ldp x10, x11, [x5, #272]
        stp x10, x11, [x0, #16]
        ldp x10, x11, [x5, #288]
        stp x10, x11, [x0, #32]
        ldp x10, x11, [x5, #304]
        stp x10, x11, [x0, #48]
        ldp x10, x11, [x5, #320]
        stp x10, x11, [x0, #64]
        ldp x10, x11, [x5, #336]
        stp x10, x11, [x0, #80]
        ldp x10, x11, [x5, #352]
        stp x10, x11, [x0, #96]
        ldp x10, x11, [x5, #368]
        stp x10, x11, [x0, #112]
        ldp x10, x11, [x5, #384]
        stp x10, x11, [x0, #128]
        ldp x10, x11, [x5, #400]
        stp x10, x11, [x0, #144]
        ldp x10, x11, [x5, #416]
        stp x10, x11, [x0, #160]
        ldp x10, x11, [x5, #432]
        stp x10, x11, [x0, #176]
        ldp x10, x11, [x5, #448]
        stp x10, x11, [x0, #192]
        ldp x10, x11, [x5, #464]
        stp x10, x11, [x0, #208]
        ldp x10, x11, [x5, #480]
        stp x10, x11, [x0, #224]
        ldp x10, x11, [x5, #496]
        stp x10, x11, [x0, #240]

        ldr x10, [x5, #512]
        cbnz x10, Lrscrypto_rsa_cios32_subtract

        mov x9, #31
Lrscrypto_rsa_cios32_compare:
        ldr x10, [x0, x9, lsl #3]
        ldr x11, [x3, x9, lsl #3]
        cmp x10, x11
        b.hi Lrscrypto_rsa_cios32_subtract
        b.lo Lrscrypto_rsa_cios32_done
        subs x9, x9, #1
        b.pl Lrscrypto_rsa_cios32_compare

Lrscrypto_rsa_cios32_subtract:
        cmp xzr, xzr
        ldr x10, [x0, #0]
        ldr x11, [x3, #0]
        sbcs x10, x10, x11
        str x10, [x0, #0]
        ldr x10, [x0, #8]
        ldr x11, [x3, #8]
        sbcs x10, x10, x11
        str x10, [x0, #8]
        ldr x10, [x0, #16]
        ldr x11, [x3, #16]
        sbcs x10, x10, x11
        str x10, [x0, #16]
        ldr x10, [x0, #24]
        ldr x11, [x3, #24]
        sbcs x10, x10, x11
        str x10, [x0, #24]
        ldr x10, [x0, #32]
        ldr x11, [x3, #32]
        sbcs x10, x10, x11
        str x10, [x0, #32]
        ldr x10, [x0, #40]
        ldr x11, [x3, #40]
        sbcs x10, x10, x11
        str x10, [x0, #40]
        ldr x10, [x0, #48]
        ldr x11, [x3, #48]
        sbcs x10, x10, x11
        str x10, [x0, #48]
        ldr x10, [x0, #56]
        ldr x11, [x3, #56]
        sbcs x10, x10, x11
        str x10, [x0, #56]
        ldr x10, [x0, #64]
        ldr x11, [x3, #64]
        sbcs x10, x10, x11
        str x10, [x0, #64]
        ldr x10, [x0, #72]
        ldr x11, [x3, #72]
        sbcs x10, x10, x11
        str x10, [x0, #72]
        ldr x10, [x0, #80]
        ldr x11, [x3, #80]
        sbcs x10, x10, x11
        str x10, [x0, #80]
        ldr x10, [x0, #88]
        ldr x11, [x3, #88]
        sbcs x10, x10, x11
        str x10, [x0, #88]
        ldr x10, [x0, #96]
        ldr x11, [x3, #96]
        sbcs x10, x10, x11
        str x10, [x0, #96]
        ldr x10, [x0, #104]
        ldr x11, [x3, #104]
        sbcs x10, x10, x11
        str x10, [x0, #104]
        ldr x10, [x0, #112]
        ldr x11, [x3, #112]
        sbcs x10, x10, x11
        str x10, [x0, #112]
        ldr x10, [x0, #120]
        ldr x11, [x3, #120]
        sbcs x10, x10, x11
        str x10, [x0, #120]
        ldr x10, [x0, #128]
        ldr x11, [x3, #128]
        sbcs x10, x10, x11
        str x10, [x0, #128]
        ldr x10, [x0, #136]
        ldr x11, [x3, #136]
        sbcs x10, x10, x11
        str x10, [x0, #136]
        ldr x10, [x0, #144]
        ldr x11, [x3, #144]
        sbcs x10, x10, x11
        str x10, [x0, #144]
        ldr x10, [x0, #152]
        ldr x11, [x3, #152]
        sbcs x10, x10, x11
        str x10, [x0, #152]
        ldr x10, [x0, #160]
        ldr x11, [x3, #160]
        sbcs x10, x10, x11
        str x10, [x0, #160]
        ldr x10, [x0, #168]
        ldr x11, [x3, #168]
        sbcs x10, x10, x11
        str x10, [x0, #168]
        ldr x10, [x0, #176]
        ldr x11, [x3, #176]
        sbcs x10, x10, x11
        str x10, [x0, #176]
        ldr x10, [x0, #184]
        ldr x11, [x3, #184]
        sbcs x10, x10, x11
        str x10, [x0, #184]
        ldr x10, [x0, #192]
        ldr x11, [x3, #192]
        sbcs x10, x10, x11
        str x10, [x0, #192]
        ldr x10, [x0, #200]
        ldr x11, [x3, #200]
        sbcs x10, x10, x11
        str x10, [x0, #200]
        ldr x10, [x0, #208]
        ldr x11, [x3, #208]
        sbcs x10, x10, x11
        str x10, [x0, #208]
        ldr x10, [x0, #216]
        ldr x11, [x3, #216]
        sbcs x10, x10, x11
        str x10, [x0, #216]
        ldr x10, [x0, #224]
        ldr x11, [x3, #224]
        sbcs x10, x10, x11
        str x10, [x0, #224]
        ldr x10, [x0, #232]
        ldr x11, [x3, #232]
        sbcs x10, x10, x11
        str x10, [x0, #232]
        ldr x10, [x0, #240]
        ldr x11, [x3, #240]
        sbcs x10, x10, x11
        str x10, [x0, #240]
        ldr x10, [x0, #248]
        ldr x11, [x3, #248]
        sbcs x10, x10, x11
        str x10, [x0, #248]

Lrscrypto_rsa_cios32_done:
        ret

        .globl _rscrypto_rsa_mont_reduce_cios_32_aarch64_apple_darwin
        .private_extern _rscrypto_rsa_mont_reduce_cios_32_aarch64_apple_darwin
        .balign 4
_rscrypto_rsa_mont_reduce_cios_32_aarch64_apple_darwin:
        mov x5, x4
        mov x4, x3
        mov x3, x2

        ldp x10, x11, [x1, #0]
        stp x10, x11, [x5, #0]
        ldp x10, x11, [x1, #16]
        stp x10, x11, [x5, #16]
        ldp x10, x11, [x1, #32]
        stp x10, x11, [x5, #32]
        ldp x10, x11, [x1, #48]
        stp x10, x11, [x5, #48]
        ldp x10, x11, [x1, #64]
        stp x10, x11, [x5, #64]
        ldp x10, x11, [x1, #80]
        stp x10, x11, [x5, #80]
        ldp x10, x11, [x1, #96]
        stp x10, x11, [x5, #96]
        ldp x10, x11, [x1, #112]
        stp x10, x11, [x5, #112]
        ldp x10, x11, [x1, #128]
        stp x10, x11, [x5, #128]
        ldp x10, x11, [x1, #144]
        stp x10, x11, [x5, #144]
        ldp x10, x11, [x1, #160]
        stp x10, x11, [x5, #160]
        ldp x10, x11, [x1, #176]
        stp x10, x11, [x5, #176]
        ldp x10, x11, [x1, #192]
        stp x10, x11, [x5, #192]
        ldp x10, x11, [x1, #208]
        stp x10, x11, [x5, #208]
        ldp x10, x11, [x1, #224]
        stp x10, x11, [x5, #224]
        ldp x10, x11, [x1, #240]
        stp x10, x11, [x5, #240]
        stp xzr, xzr, [x5, #256]
        stp xzr, xzr, [x5, #272]
        stp xzr, xzr, [x5, #288]
        stp xzr, xzr, [x5, #304]
        stp xzr, xzr, [x5, #320]
        stp xzr, xzr, [x5, #336]
        stp xzr, xzr, [x5, #352]
        stp xzr, xzr, [x5, #368]
        stp xzr, xzr, [x5, #384]
        stp xzr, xzr, [x5, #400]
        stp xzr, xzr, [x5, #416]
        stp xzr, xzr, [x5, #432]
        stp xzr, xzr, [x5, #448]
        stp xzr, xzr, [x5, #464]
        stp xzr, xzr, [x5, #480]
        stp xzr, xzr, [x5, #496]
        str xzr, [x5, #512]
        str xzr, [x5, #520]

        mov x6, #0
        mov x15, x5
Lrscrypto_rsa_redc32_outer:
        ldr x7, [x15]
        mul x7, x7, x4
        mov x8, #0
        RSCRYPTO_RSA_CIOS32_REDC_STEP 0
        RSCRYPTO_RSA_CIOS32_REDC_STEP 8
        RSCRYPTO_RSA_CIOS32_REDC_STEP 16
        RSCRYPTO_RSA_CIOS32_REDC_STEP 24
        RSCRYPTO_RSA_CIOS32_REDC_STEP 32
        RSCRYPTO_RSA_CIOS32_REDC_STEP 40
        RSCRYPTO_RSA_CIOS32_REDC_STEP 48
        RSCRYPTO_RSA_CIOS32_REDC_STEP 56
        RSCRYPTO_RSA_CIOS32_REDC_STEP 64
        RSCRYPTO_RSA_CIOS32_REDC_STEP 72
        RSCRYPTO_RSA_CIOS32_REDC_STEP 80
        RSCRYPTO_RSA_CIOS32_REDC_STEP 88
        RSCRYPTO_RSA_CIOS32_REDC_STEP 96
        RSCRYPTO_RSA_CIOS32_REDC_STEP 104
        RSCRYPTO_RSA_CIOS32_REDC_STEP 112
        RSCRYPTO_RSA_CIOS32_REDC_STEP 120
        RSCRYPTO_RSA_CIOS32_REDC_STEP 128
        RSCRYPTO_RSA_CIOS32_REDC_STEP 136
        RSCRYPTO_RSA_CIOS32_REDC_STEP 144
        RSCRYPTO_RSA_CIOS32_REDC_STEP 152
        RSCRYPTO_RSA_CIOS32_REDC_STEP 160
        RSCRYPTO_RSA_CIOS32_REDC_STEP 168
        RSCRYPTO_RSA_CIOS32_REDC_STEP 176
        RSCRYPTO_RSA_CIOS32_REDC_STEP 184
        RSCRYPTO_RSA_CIOS32_REDC_STEP 192
        RSCRYPTO_RSA_CIOS32_REDC_STEP 200
        RSCRYPTO_RSA_CIOS32_REDC_STEP 208
        RSCRYPTO_RSA_CIOS32_REDC_STEP 216
        RSCRYPTO_RSA_CIOS32_REDC_STEP 224
        RSCRYPTO_RSA_CIOS32_REDC_STEP 232
        RSCRYPTO_RSA_CIOS32_REDC_STEP 240
        RSCRYPTO_RSA_CIOS32_REDC_STEP 248

        ldr x10, [x15, #256]
        adds x10, x10, x8
        str x10, [x15, #256]
        ldr x11, [x15, #264]
        adc x11, x11, xzr
        str x11, [x15, #264]

        add x6, x6, #1
        add x15, x15, #8
        cmp x6, #32
        b.ne Lrscrypto_rsa_redc32_outer

        ldp x10, x11, [x5, #256]
        stp x10, x11, [x0, #0]
        ldp x10, x11, [x5, #272]
        stp x10, x11, [x0, #16]
        ldp x10, x11, [x5, #288]
        stp x10, x11, [x0, #32]
        ldp x10, x11, [x5, #304]
        stp x10, x11, [x0, #48]
        ldp x10, x11, [x5, #320]
        stp x10, x11, [x0, #64]
        ldp x10, x11, [x5, #336]
        stp x10, x11, [x0, #80]
        ldp x10, x11, [x5, #352]
        stp x10, x11, [x0, #96]
        ldp x10, x11, [x5, #368]
        stp x10, x11, [x0, #112]
        ldp x10, x11, [x5, #384]
        stp x10, x11, [x0, #128]
        ldp x10, x11, [x5, #400]
        stp x10, x11, [x0, #144]
        ldp x10, x11, [x5, #416]
        stp x10, x11, [x0, #160]
        ldp x10, x11, [x5, #432]
        stp x10, x11, [x0, #176]
        ldp x10, x11, [x5, #448]
        stp x10, x11, [x0, #192]
        ldp x10, x11, [x5, #464]
        stp x10, x11, [x0, #208]
        ldp x10, x11, [x5, #480]
        stp x10, x11, [x0, #224]
        ldp x10, x11, [x5, #496]
        stp x10, x11, [x0, #240]

        ldr x10, [x5, #512]
        cbnz x10, Lrscrypto_rsa_redc32_subtract

        mov x9, #31
Lrscrypto_rsa_redc32_compare:
        ldr x10, [x0, x9, lsl #3]
        ldr x11, [x3, x9, lsl #3]
        cmp x10, x11
        b.hi Lrscrypto_rsa_redc32_subtract
        b.lo Lrscrypto_rsa_redc32_done
        subs x9, x9, #1
        b.pl Lrscrypto_rsa_redc32_compare

Lrscrypto_rsa_redc32_subtract:
        cmp xzr, xzr
        ldr x10, [x0, #0]
        ldr x11, [x3, #0]
        sbcs x10, x10, x11
        str x10, [x0, #0]
        ldr x10, [x0, #8]
        ldr x11, [x3, #8]
        sbcs x10, x10, x11
        str x10, [x0, #8]
        ldr x10, [x0, #16]
        ldr x11, [x3, #16]
        sbcs x10, x10, x11
        str x10, [x0, #16]
        ldr x10, [x0, #24]
        ldr x11, [x3, #24]
        sbcs x10, x10, x11
        str x10, [x0, #24]
        ldr x10, [x0, #32]
        ldr x11, [x3, #32]
        sbcs x10, x10, x11
        str x10, [x0, #32]
        ldr x10, [x0, #40]
        ldr x11, [x3, #40]
        sbcs x10, x10, x11
        str x10, [x0, #40]
        ldr x10, [x0, #48]
        ldr x11, [x3, #48]
        sbcs x10, x10, x11
        str x10, [x0, #48]
        ldr x10, [x0, #56]
        ldr x11, [x3, #56]
        sbcs x10, x10, x11
        str x10, [x0, #56]
        ldr x10, [x0, #64]
        ldr x11, [x3, #64]
        sbcs x10, x10, x11
        str x10, [x0, #64]
        ldr x10, [x0, #72]
        ldr x11, [x3, #72]
        sbcs x10, x10, x11
        str x10, [x0, #72]
        ldr x10, [x0, #80]
        ldr x11, [x3, #80]
        sbcs x10, x10, x11
        str x10, [x0, #80]
        ldr x10, [x0, #88]
        ldr x11, [x3, #88]
        sbcs x10, x10, x11
        str x10, [x0, #88]
        ldr x10, [x0, #96]
        ldr x11, [x3, #96]
        sbcs x10, x10, x11
        str x10, [x0, #96]
        ldr x10, [x0, #104]
        ldr x11, [x3, #104]
        sbcs x10, x10, x11
        str x10, [x0, #104]
        ldr x10, [x0, #112]
        ldr x11, [x3, #112]
        sbcs x10, x10, x11
        str x10, [x0, #112]
        ldr x10, [x0, #120]
        ldr x11, [x3, #120]
        sbcs x10, x10, x11
        str x10, [x0, #120]
        ldr x10, [x0, #128]
        ldr x11, [x3, #128]
        sbcs x10, x10, x11
        str x10, [x0, #128]
        ldr x10, [x0, #136]
        ldr x11, [x3, #136]
        sbcs x10, x10, x11
        str x10, [x0, #136]
        ldr x10, [x0, #144]
        ldr x11, [x3, #144]
        sbcs x10, x10, x11
        str x10, [x0, #144]
        ldr x10, [x0, #152]
        ldr x11, [x3, #152]
        sbcs x10, x10, x11
        str x10, [x0, #152]
        ldr x10, [x0, #160]
        ldr x11, [x3, #160]
        sbcs x10, x10, x11
        str x10, [x0, #160]
        ldr x10, [x0, #168]
        ldr x11, [x3, #168]
        sbcs x10, x10, x11
        str x10, [x0, #168]
        ldr x10, [x0, #176]
        ldr x11, [x3, #176]
        sbcs x10, x10, x11
        str x10, [x0, #176]
        ldr x10, [x0, #184]
        ldr x11, [x3, #184]
        sbcs x10, x10, x11
        str x10, [x0, #184]
        ldr x10, [x0, #192]
        ldr x11, [x3, #192]
        sbcs x10, x10, x11
        str x10, [x0, #192]
        ldr x10, [x0, #200]
        ldr x11, [x3, #200]
        sbcs x10, x10, x11
        str x10, [x0, #200]
        ldr x10, [x0, #208]
        ldr x11, [x3, #208]
        sbcs x10, x10, x11
        str x10, [x0, #208]
        ldr x10, [x0, #216]
        ldr x11, [x3, #216]
        sbcs x10, x10, x11
        str x10, [x0, #216]
        ldr x10, [x0, #224]
        ldr x11, [x3, #224]
        sbcs x10, x10, x11
        str x10, [x0, #224]
        ldr x10, [x0, #232]
        ldr x11, [x3, #232]
        sbcs x10, x10, x11
        str x10, [x0, #232]
        ldr x10, [x0, #240]
        ldr x11, [x3, #240]
        sbcs x10, x10, x11
        str x10, [x0, #240]
        ldr x10, [x0, #248]
        ldr x11, [x3, #248]
        sbcs x10, x10, x11
        str x10, [x0, #248]

Lrscrypto_rsa_redc32_done:
        ret

        .globl _rscrypto_rsa_mont_reduce_cios_words_aarch64_apple_darwin
        .private_extern _rscrypto_rsa_mont_reduce_cios_words_aarch64_apple_darwin
        .balign 4
_rscrypto_rsa_mont_reduce_cios_words_aarch64_apple_darwin:
        // x0 = out[words]
        // x1 = value[words]
        // x2 = modulus[words]
        // x3 = n0
        // x4 = t[2 * words + 2]
        // x5 = words
        mov x6, #0
Lrscrypto_rsa_redc_words_copy_value:
        ldr x10, [x1, x6, lsl #3]
        str x10, [x4, x6, lsl #3]
        add x6, x6, #1
        cmp x6, x5
        b.ne Lrscrypto_rsa_redc_words_copy_value

        add x14, x5, x5
        add x14, x14, #2
Lrscrypto_rsa_redc_words_zero_tail:
        str xzr, [x4, x6, lsl #3]
        add x6, x6, #1
        cmp x6, x14
        b.ne Lrscrypto_rsa_redc_words_zero_tail

        mov x6, #0
Lrscrypto_rsa_redc_words_outer:
        ldr x7, [x4, x6, lsl #3]
        mul x7, x7, x3
        mov x8, #0
        mov x9, #0

Lrscrypto_rsa_redc_words_inner:
        add x15, x6, x9
        ldr x10, [x2, x9, lsl #3]
        ldr x11, [x4, x15, lsl #3]
        mul x12, x10, x7
        umulh x13, x10, x7
        adds x12, x12, x11
        adc x13, x13, xzr
        adds x12, x12, x8
        adc x8, x13, xzr
        str x12, [x4, x15, lsl #3]
        add x9, x9, #1
        cmp x9, x5
        b.ne Lrscrypto_rsa_redc_words_inner

        add x15, x6, x5
        ldr x10, [x4, x15, lsl #3]
        adds x10, x10, x8
        str x10, [x4, x15, lsl #3]
        add x15, x15, #1
        ldr x10, [x4, x15, lsl #3]
        adc x10, x10, xzr
        str x10, [x4, x15, lsl #3]

        add x6, x6, #1
        cmp x6, x5
        b.ne Lrscrypto_rsa_redc_words_outer

        mov x6, #0
Lrscrypto_rsa_redc_words_copy_result:
        add x9, x5, x6
        ldr x10, [x4, x9, lsl #3]
        str x10, [x0, x6, lsl #3]
        add x6, x6, #1
        cmp x6, x5
        b.ne Lrscrypto_rsa_redc_words_copy_result

        cmp xzr, xzr
        mov x6, #0
        mov x16, x5
Lrscrypto_rsa_redc_words_subtract:
        ldr x10, [x0, x6, lsl #3]
        ldr x11, [x2, x6, lsl #3]
        sbcs x12, x10, x11
        str x12, [x4, x6, lsl #3]
        add x6, x6, #1
        sub x16, x16, #1
        cbnz x16, Lrscrypto_rsa_redc_words_subtract

        cset x14, cs
        add x9, x5, x5
        ldr x10, [x4, x9, lsl #3]
        cmp x10, #0
        cset x15, ne
        orr x14, x14, x15
        neg x14, x14
        mvn x15, x14

        mov x6, #0
Lrscrypto_rsa_redc_words_select:
        ldr x10, [x0, x6, lsl #3]
        ldr x11, [x4, x6, lsl #3]
        and x10, x10, x15
        and x11, x11, x14
        orr x10, x10, x11
        str x10, [x0, x6, lsl #3]
        add x6, x6, #1
        cmp x6, x5
        b.ne Lrscrypto_rsa_redc_words_select

        ret
