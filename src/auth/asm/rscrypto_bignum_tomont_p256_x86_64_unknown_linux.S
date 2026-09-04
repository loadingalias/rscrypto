// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p256/bignum_tomont_p256.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_tomont_p256
        .type rscrypto_bignum_tomont_p256, %function
        .hidden rscrypto_bignum_tomont_p256
        .text
rscrypto_bignum_tomont_p256:
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
.size rscrypto_bignum_tomont_p256, .-rscrypto_bignum_tomont_p256
