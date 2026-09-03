// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p256/bignum_tomont_p256_alt.S
// - x86_att/p256/bignum_montsqr_p256_alt.S
// - x86_att/p256/bignum_montmul_p256_alt.S
// - Microsoft x64 batch wrapper around target-native upstream field bodies
//
// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.

 .globl rscrypto_p256_curve_terms_alt
        .text
        .p2align 4
rscrypto_p256_curve_terms_alt:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rdi ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rdi, 0
        pushq %rsi ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rsi, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        subq $40, %rsp ; .cfi_adjust_cfa_offset 40
        movq %rcx, %r12
        movq %rdx, %r13
        movq %r12, %rdi
        movq %r13, %rsi
        callq .Lrscrypto_bignum_tomont_p256_alt_raw
        leaq 32(%r12), %rdi
        leaq 32(%r13), %rsi
        callq .Lrscrypto_bignum_tomont_p256_alt_raw
        leaq 64(%r12), %rdi
        leaq 32(%r12), %rsi
        callq .Lrscrypto_bignum_montsqr_p256_alt_raw
        movq %rsp, %rdi
        movq %r12, %rsi
        callq .Lrscrypto_bignum_montsqr_p256_alt_raw
        leaq 96(%r12), %rdi
        movq %rsp, %rsi
        movq %r12, %rdx
        callq .Lrscrypto_bignum_montmul_p256_alt_raw
        addq $40, %rsp ; .cfi_adjust_cfa_offset -40
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rsi ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rsi
        popq %rdi ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rdi
        retq ; .cfi_endproc

        .text
.Lrscrypto_bignum_tomont_p256_alt_raw:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        movl $0x0000000000000003, %ecx
        movq (%rsi), %rax
        mulq %rcx
        movq %rax, %r8
        movq %rdx, %r9
        movq 8(%rsi), %rax
        mulq %rcx
        xorl %r10d, %r10d
        addq %rax, %r9
        adcq %rdx, %r10
        movq 16(%rsi), %rax
        mulq %rcx
        xorl %r11d, %r11d
        addq %rax, %r10
        adcq %rdx, %r11
        movq 24(%rsi), %rax
        mulq %rcx
        xorl %r12d, %r12d
        addq %rax, %r11
        adcq %rdx, %r12
        movq $0xfffffffbffffffff, %rcx
        xorl %r13d, %r13d
        movq (%rsi), %rax ; mulq %rcx; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %r14, %r14
        movq 8(%rsi), %rax ; mulq %rcx; subq %r14, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %r14, %r14
        movq 16(%rsi), %rax ; mulq %rcx; subq %r14, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r14, %r14
        movq 24(%rsi), %rax ; mulq %rcx; subq %r14, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13
        xorl %r14d, %r14d
        movq $0x0000000100000000, %rcx
        movq %r8, %rax ; mulq %rcx; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %r15, %r15
        movq %r9, %rax ; mulq %rcx; subq %r15, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %r15, %r15
        notq %rcx
        leaq 2(%rcx), %rcx
        movq %r8, %rax ; mulq %rcx; subq %r15, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r15, %r15
        movq %r9, %rax ; mulq %rcx; subq %r15, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13
        adcq %r14, %r14
        movq $0xfffffffffffffffe, %rcx
        xorl %r15d, %r15d
        movq (%rsi), %rax ; mulq %rcx; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %r8, %r8
        movq 8(%rsi), %rax ; mulq %rcx; subq %r8, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r8, %r8
        movq 16(%rsi), %rax ; mulq %rcx; subq %r8, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %r8, %r8
        movq 24(%rsi), %rax ; mulq %rcx; subq %r8, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14
        adcq %r15, %r15
        movq $0x00000004fffffffd, %rcx
        xorl %r8d, %r8d
        movq (%rsi), %rax ; mulq %rcx; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r9, %r9
        movq 8(%rsi), %rax ; mulq %rcx; subq %r9, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %r9, %r9
        movq 16(%rsi), %rax ; mulq %rcx; subq %r9, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %r9, %r9
        movq 24(%rsi), %rax ; mulq %rcx; subq %r9, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15
        adcq %r8, %r8
        movq $0x0000000100000000, %rcx
        movq %r10, %rax ; mulq %rcx; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r9, %r9
        movq %r11, %rax ; mulq %rcx; subq %r9, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %r9, %r9
        notq %rcx
        leaq 2(%rcx), %rcx
        movq %r10, %rax ; mulq %rcx; subq %r9, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %r9, %r9
        movq %r11, %rax ; mulq %rcx; subq %r9, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15 ; sbbq %r9, %r9
        subq %r9, %r8
        xorl %edx, %edx
        leaq -1(%rdx), %r9
        incq %rdx
        addq %r12, %rdx
        decq %rcx
        adcq %r13, %rcx
        movq %r9, %rax
        adcq %r14, %r9
        movl $0x00000000fffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovcq %rdx, %r12
        cmovcq %rcx, %r13
        cmovcq %r9, %r14
        cmovcq %r11, %r15
        movq %r12, (%rdi)
        movq %r13, 8(%rdi)
        movq %r14, 16(%rdi)
        movq %r15, 24(%rdi)
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        retq ; .cfi_endproc

        .text
.Lrscrypto_bignum_montsqr_p256_alt_raw:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        movq (%rsi), %rax
        movq %rax, %rbx
        mulq %rax
        movq %rax, %r8
        movq %rdx, %r15
        movq 8(%rsi), %rax
        mulq %rbx
        movq %rax, %r9
        movq %rdx, %r10
        movq 24(%rsi), %rax
        movq %rax, %r13
        mulq %rbx
        movq %rax, %r11
        movq %rdx, %r12
        movq 16(%rsi), %rax
        movq %rax, %rbx
        mulq %r13
        movq %rax, %r13
        movq %rdx, %r14
        movq (%rsi), %rax ; mulq %rbx; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx
        movq 8(%rsi), %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %rcx, %rcx
        movq 24(%rsi), %rbx
        movq 8(%rsi), %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13
        adcq $0, %r14
        xorl %ecx, %ecx
        addq %r9, %r9
        adcq %r10, %r10
        adcq %r11, %r11
        adcq %r12, %r12
        adcq %r13, %r13
        adcq %r14, %r14
        adcq %rcx, %rcx
        movq 8(%rsi), %rax
        mulq %rax
        addq %r15, %r9
        adcq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        movq 16(%rsi), %rax
        mulq %rax
        negq %r15
        adcq %rax, %r12
        adcq %rdx, %r13
        sbbq %r15, %r15
        movq 24(%rsi), %rax
        mulq %rax
        negq %r15
        adcq %rax, %r14
        adcq %rcx, %rdx
        movq %rdx, %r15
        movq $0x0000000100000000, %rbx
        movq %r8, %rax ; mulq %rbx; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %rcx, %rcx
        movq %r9, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rcx, %rcx
        notq %rbx
        leaq 2(%rbx), %rbx
        movq %r8, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %rcx, %rcx
        xorl %r8d, %r8d
        movq %r9, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13
        adcq %r8, %r14
        adcq %r8, %r15
        adcq %r8, %r8
        movq $0x0000000100000000, %rbx
        movq %r10, %rax ; mulq %rbx; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %rcx, %rcx
        movq %r11, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %rcx, %rcx
        notq %rbx
        leaq 2(%rbx), %rbx
        movq %r10, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %rcx, %rcx
        xorl %r9d, %r9d
        movq %r11, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15
        adcq %r9, %r8
        movl $1, %ecx
        addq %r12, %rcx
        leaq -1(%rbx), %rbx
        adcq %r13, %rbx
        leaq -1(%r9), %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0x00000000fffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovcq %rcx, %r12
        cmovcq %rbx, %r13
        cmovcq %r9, %r14
        cmovcq %r11, %r15
        movq %r12, (%rdi)
        movq %r13, 8(%rdi)
        movq %r14, 16(%rdi)
        movq %r15, 24(%rdi)
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc

        .text
.Lrscrypto_bignum_montmul_p256_alt_raw:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        movq %rdx, %rcx
        movq (%rcx), %rbx
        movq (%rsi), %rax
        mulq %rbx
        movq %rax, %r8
        movq %rdx, %r9
        movq 8(%rsi), %rax
        mulq %rbx
        xorl %r10d, %r10d
        addq %rax, %r9
        adcq %rdx, %r10
        movq 16(%rsi), %rax
        mulq %rbx
        xorl %r11d, %r11d
        addq %rax, %r10
        adcq %rdx, %r11
        movq 24(%rsi), %rax
        mulq %rbx
        xorl %r12d, %r12d
        addq %rax, %r11
        adcq %rdx, %r12
        movq 8(%rcx), %rbx
        xorl %r13d, %r13d
        movq (%rsi), %rax ; mulq %rbx; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %r14, %r14
        movq 8(%rsi), %rax ; mulq %rbx; subq %r14, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %r14, %r14
        movq 16(%rsi), %rax ; mulq %rbx; subq %r14, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r14, %r14
        movq 24(%rsi), %rax ; mulq %rbx; subq %r14, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13
        xorl %r14d, %r14d
        movq $0x0000000100000000, %rbx
        movq %r8, %rax ; mulq %rbx; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %r15, %r15
        movq %r9, %rax ; mulq %rbx; subq %r15, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %r15, %r15
        notq %rbx
        leaq 2(%rbx), %rbx
        movq %r8, %rax ; mulq %rbx; subq %r15, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r15, %r15
        movq %r9, %rax ; mulq %rbx; subq %r15, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13
        adcq %r14, %r14
        movq 16(%rcx), %rbx
        xorl %r15d, %r15d
        movq (%rsi), %rax ; mulq %rbx; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %r8, %r8
        movq 8(%rsi), %rax ; mulq %rbx; subq %r8, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r8, %r8
        movq 16(%rsi), %rax ; mulq %rbx; subq %r8, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %r8, %r8
        movq 24(%rsi), %rax ; mulq %rbx; subq %r8, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14
        adcq %r15, %r15
        movq 24(%rcx), %rbx
        xorl %r8d, %r8d
        movq (%rsi), %rax ; mulq %rbx; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r9, %r9
        movq 8(%rsi), %rax ; mulq %rbx; subq %r9, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %r9, %r9
        movq 16(%rsi), %rax ; mulq %rbx; subq %r9, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %r9, %r9
        movq 24(%rsi), %rax ; mulq %rbx; subq %r9, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15
        adcq %r8, %r8
        xorl %r9d, %r9d
        movq $0x0000000100000000, %rbx
        movq %r10, %rax ; mulq %rbx; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %rcx, %rcx
        movq %r11, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %rcx, %rcx
        notq %rbx
        leaq 2(%rbx), %rbx
        movq %r10, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %rcx, %rcx
        movq %r11, %rax ; mulq %rbx; subq %rcx, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15
        adcq %r9, %r8
        movl $1, %ecx
        addq %r12, %rcx
        decq %rbx
        adcq %r13, %rbx
        decq %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0x00000000fffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovcq %rcx, %r12
        cmovcq %rbx, %r13
        cmovcq %r9, %r14
        cmovcq %r11, %r15
        movq %r12, (%rdi)
        movq %r13, 8(%rdi)
        movq %r14, 16(%rdi)
        movq %r15, 24(%rdi)
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
