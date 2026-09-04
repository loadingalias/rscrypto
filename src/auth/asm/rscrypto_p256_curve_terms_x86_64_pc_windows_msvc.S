// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p256/bignum_tomont_p256.S
// - x86_att/p256/bignum_montsqr_p256.S
// - x86_att/p256/bignum_montmul_p256.S
// - Microsoft x64 batch wrapper around target-native upstream field bodies
//
// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.

 .globl rscrypto_p256_curve_terms
        .text
        .p2align 4
rscrypto_p256_curve_terms:
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
        callq .Lrscrypto_bignum_tomont_p256_raw
        leaq 32(%r12), %rdi
        leaq 32(%r13), %rsi
        callq .Lrscrypto_bignum_tomont_p256_raw
        leaq 64(%r12), %rdi
        leaq 32(%r12), %rsi
        callq .Lrscrypto_bignum_montsqr_p256_raw
        movq %rsp, %rdi
        movq %r12, %rsi
        callq .Lrscrypto_bignum_montsqr_p256_raw
        leaq 96(%r12), %rdi
        movq %rsp, %rsi
        movq %r12, %rdx
        callq .Lrscrypto_bignum_montmul_p256_raw
        addq $40, %rsp ; .cfi_adjust_cfa_offset -40
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rsi ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rsi
        popq %rdi ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rdi
        retq ; .cfi_endproc

        .text
.Lrscrypto_bignum_tomont_p256_raw:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        xorq %r13, %r13
        movl $0x0000000000000003, %edx
        mulxq (%rsi), %r8, %r9
        mulxq 8(%rsi), %rcx, %r10
        adcxq %rcx, %r9
        mulxq 16(%rsi), %rcx, %r11
        adcxq %rcx, %r10
        mulxq 24(%rsi), %rcx, %r12
        adcxq %rcx, %r11
        adcxq %r13, %r12
        movq $0xfffffffbffffffff, %rdx
        xorq %r14, %r14
        mulxq (%rsi), %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10
        mulxq 8(%rsi), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11
        mulxq 16(%rsi), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12
        mulxq 24(%rsi), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13
        adcq %r14, %r13
        xorq %r15, %r15
        movq $0x0000000100000000, %rdx
        mulxq %r8, %rax, %rcx ; adcxq %rax, %r9 ; adoxq %rcx, %r10
        mulxq %r9, %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11
        movq $0xffffffff00000001, %rdx
        mulxq %r8, %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12
        mulxq %r9, %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13
        adcxq %r15, %r13
        adoxq %r15, %r14
        adcxq %r15, %r14
        movq $0xfffffffffffffffe, %rdx
        xorq %r8, %r8
        mulxq (%rsi), %rax, %rcx ; adcxq %rax, %r10 ; adoxq %rcx, %r11
        mulxq 8(%rsi), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12
        mulxq 16(%rsi), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13
        mulxq 24(%rsi), %rax, %rcx ; adcxq %rax, %r13 ; adoxq %rcx, %r14
        adcxq %r8, %r14
        adoxq %r8, %r15
        adcxq %r8, %r15
        movq $0x00000004fffffffd, %rdx
        xorq %r9, %r9
        mulxq (%rsi), %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12
        mulxq 8(%rsi), %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13
        mulxq 16(%rsi), %rax, %rcx ; adcxq %rax, %r13 ; adoxq %rcx, %r14
        mulxq 24(%rsi), %rax, %rcx ; adcxq %rax, %r14 ; adoxq %rcx, %r15
        adcxq %r9, %r15
        adoxq %r9, %r8
        adcxq %r9, %r8
        xorq %r9, %r9
        movq $0x0000000100000000, %rdx
        mulxq %r10, %rax, %rcx ; adcxq %rax, %r11 ; adoxq %rcx, %r12
        mulxq %r11, %rax, %rcx ; adcxq %rax, %r12 ; adoxq %rcx, %r13
        movq $0xffffffff00000001, %rdx
        mulxq %r10, %rax, %rcx ; adcxq %rax, %r13 ; adoxq %rcx, %r14
        mulxq %r11, %rax, %rcx ; adcxq %rax, %r14 ; adoxq %rcx, %r15
        adcxq %r9, %r15
        adoxq %r9, %r8
        adcxq %r9, %r8
        movl $0x00000000ffffffff, %edx
        movq $0xffffffff00000001, %rcx
        movq $-2, %rax
        subq %r12, %rax
        movq %rdx, %rax
        sbbq %r13, %rax
        movl $0, %eax
        sbbq %r14, %rax
        movq %rcx, %rax
        sbbq %r15, %rax
        movl $0, %eax
        sbbq %r8, %rax
        andq %rax, %rdx
        andq %rax, %rcx
        subq %rax, %r12
        sbbq %rdx, %r13
        sbbq $0, %r14
        sbbq %rcx, %r15
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
.Lrscrypto_bignum_montsqr_p256_raw:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        movq (%rsi), %rdx
        mulxq %rdx, %r8, %r15
        mulxq 8(%rsi), %r9, %r10
        mulxq 24(%rsi), %r11, %r12
        movq 16(%rsi), %rdx
        mulxq 24(%rsi), %r13, %r14
        xorl %ebp, %ebp
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        movq 24(%rsi), %rdx
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        adcxq %rbp, %r13
        adoxq %rbp, %r14
        adcq %rbp, %r14
        xorl %ebp, %ebp
        adcxq %r9, %r9
        adoxq %r15, %r9
        movq 8(%rsi), %rdx
        mulxq %rdx, %rax, %rdx
        adcxq %r10, %r10
        adoxq %rax, %r10
        adcxq %r11, %r11
        adoxq %rdx, %r11
        movq 16(%rsi), %rdx
        mulxq %rdx, %rax, %rdx
        adcxq %r12, %r12
        adoxq %rax, %r12
        adcxq %r13, %r13
        adoxq %rdx, %r13
        movq 24(%rsi), %rdx
        mulxq %rdx, %rax, %r15
        adcxq %r14, %r14
        adoxq %rax, %r14
        adcxq %rbp, %r15
        adoxq %rbp, %r15
        xorl %ebp, %ebp
        movq $0x0000000100000000, %rdx
        mulxq %r8, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10
        mulxq %r9, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        movq $0xffffffff00000001, %rdx
        mulxq %r8, %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq %r9, %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        adcxq %rbp, %r13
        movl %ebp, %r9d
        adoxq %rbp, %r9
        adcxq %rbp, %r9
        addq %r9, %r14
        adcq %rbp, %r15
        movl %ebp, %r8d
        adcq %rbp, %r8
        xorl %ebp, %ebp
        movq $0x0000000100000000, %rdx
        mulxq %r10, %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq %r11, %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        movq $0xffffffff00000001, %rdx
        mulxq %r10, %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        mulxq %r11, %rax, %rbx ; adcxq %rax, %r14 ; adoxq %rbx, %r15
        adcxq %rbp, %r15
        adoxq %rbp, %r8
        adcq %rbp, %r8
        movl $1, %ecx
        addq %r12, %rcx
        leaq -1(%rdx), %rdx
        adcq %r13, %rdx
        leaq -1(%rbp), %rbp
        movq %rbp, %rax
        adcq %r14, %rbp
        movl $0x00000000fffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovcq %rcx, %r12
        cmovcq %rdx, %r13
        cmovcq %rbp, %r14
        cmovcq %r11, %r15
        movq %r12, (%rdi)
        movq %r13, 8(%rdi)
        movq %r14, 16(%rdi)
        movq %r15, 24(%rdi)
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc

        .text
.Lrscrypto_bignum_montmul_p256_raw:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        movq %rdx, %rcx
        xorl %r13d, %r13d
        movq (%rcx), %rdx
        mulxq (%rsi), %r8, %r9
        mulxq 8(%rsi), %rbx, %r10
        adcq %rbx, %r9
        mulxq 16(%rsi), %rbx, %r11
        adcq %rbx, %r10
        mulxq 24(%rsi), %rbx, %r12
        adcq %rbx, %r11
        adcq %r13, %r12
        movq 8(%rcx), %rdx
        xorl %r14d, %r14d
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        mulxq 16(%rsi), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq 24(%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        adcq %r14, %r13
        xorl %r15d, %r15d
        movq $0x0000000100000000, %rdx
        mulxq %r8, %rax, %rbx ; adcxq %rax, %r9 ; adoxq %rbx, %r10
        mulxq %r9, %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        notq %rdx
        leaq 2(%rdx), %rdx
        mulxq %r8, %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq %r9, %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        adcxq %r15, %r13
        adoxq %r15, %r14
        adcq %r15, %r14
        movq 16(%rcx), %rdx
        xorl %r8d, %r8d
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r10 ; adoxq %rbx, %r11
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq 16(%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        adoxq %r8, %r14
        mulxq 24(%rsi), %rax, %rbx
        adcq %rax, %r13
        adcq %rbx, %r14
        adcq %r8, %r15
        movq 24(%rcx), %rdx
        xorl %r9d, %r9d
        mulxq (%rsi), %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq 8(%rsi), %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        mulxq 16(%rsi), %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        adoxq %r9, %r15
        mulxq 24(%rsi), %rax, %rbx
        adcq %rax, %r14
        adcq %rbx, %r15
        adcq %r9, %r8
        xorl %r9d, %r9d
        movq $0x0000000100000000, %rdx
        mulxq %r10, %rax, %rbx ; adcxq %rax, %r11 ; adoxq %rbx, %r12
        mulxq %r11, %rax, %rbx ; adcxq %rax, %r12 ; adoxq %rbx, %r13
        notq %rdx
        leaq 2(%rdx), %rdx
        mulxq %r10, %rax, %rbx ; adcxq %rax, %r13 ; adoxq %rbx, %r14
        mulxq %r11, %rax, %rbx ; adcxq %rax, %r14 ; adoxq %rbx, %r15
        adcxq %r9, %r15
        adoxq %r9, %r8
        adcq %r9, %r8
        movl $1, %ecx
        addq %r12, %rcx
        decq %rdx
        adcq %r13, %rdx
        decq %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0x00000000fffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovcq %rcx, %r12
        cmovcq %rdx, %r13
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
