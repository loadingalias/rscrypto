// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p384/bignum_montmul_p384_alt.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_montmul_p384_alt
        .type rscrypto_bignum_montmul_p384_alt, %function
        .hidden rscrypto_bignum_montmul_p384_alt
        .text
rscrypto_bignum_montmul_p384_alt:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
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
        movq 32(%rsi), %rax
        mulq %rbx
        xorl %r13d, %r13d
        addq %rax, %r12
        adcq %rdx, %r13
        movq 40(%rsi), %rax
        mulq %rbx
        xorl %r14d, %r14d
        addq %rax, %r13
        adcq %rdx, %r14
        xorl %r15d, %r15d
        movq %r8, %rbx ; shlq $32, %rbx ; addq %r8, %rbx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r8 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %r8, %rax ; adcq %rbx, %rdx ; adcl %ebp, %ebp ; subq %rax, %r9 ; sbbq %rdx, %r10 ; sbbq %rbp, %r11 ; sbbq $0, %r12 ; sbbq $0, %r13 ; sbbq $0, %rbx ; addq %rbx, %r14 ; adcq $0, %r15
        movq 8(%rcx), %rbx
        movq (%rsi), %rax ; mulq %rbx; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %r8, %r8
        movq 8(%rsi), %rax ; mulq %rbx; subq %r8, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %r8, %r8
        movq 16(%rsi), %rax ; mulq %rbx; subq %r8, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r8, %r8
        movq 24(%rsi), %rax ; mulq %rbx; subq %r8, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %r8, %r8
        movq 32(%rsi), %rax ; mulq %rbx; subq %r8, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %r8, %r8
        movq 40(%rsi), %rax ; mulq %rbx; subq %r8, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15 ; sbbq %r8, %r8
        negq %r8
        movq %r9, %rbx ; shlq $32, %rbx ; addq %r9, %rbx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r9 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %r9, %rax ; adcq %rbx, %rdx ; adcl %ebp, %ebp ; subq %rax, %r10 ; sbbq %rdx, %r11 ; sbbq %rbp, %r12 ; sbbq $0, %r13 ; sbbq $0, %r14 ; sbbq $0, %rbx ; addq %rbx, %r15 ; adcq $0, %r8
        movq 16(%rcx), %rbx
        movq (%rsi), %rax ; mulq %rbx; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %r9, %r9
        movq 8(%rsi), %rax ; mulq %rbx; subq %r9, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r9, %r9
        movq 16(%rsi), %rax ; mulq %rbx; subq %r9, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %r9, %r9
        movq 24(%rsi), %rax ; mulq %rbx; subq %r9, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %r9, %r9
        movq 32(%rsi), %rax ; mulq %rbx; subq %r9, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15 ; sbbq %r9, %r9
        movq 40(%rsi), %rax ; mulq %rbx; subq %r9, %rdx ; addq %rax, %r15 ; adcq %rdx, %r8 ; sbbq %r9, %r9
        negq %r9
        movq %r10, %rbx ; shlq $32, %rbx ; addq %r10, %rbx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r10 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %r10, %rax ; adcq %rbx, %rdx ; adcl %ebp, %ebp ; subq %rax, %r11 ; sbbq %rdx, %r12 ; sbbq %rbp, %r13 ; sbbq $0, %r14 ; sbbq $0, %r15 ; sbbq $0, %rbx ; addq %rbx, %r8 ; adcq $0, %r9
        movq 24(%rcx), %rbx
        movq (%rsi), %rax ; mulq %rbx; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %r10, %r10
        movq 8(%rsi), %rax ; mulq %rbx; subq %r10, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %r10, %r10
        movq 16(%rsi), %rax ; mulq %rbx; subq %r10, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %r10, %r10
        movq 24(%rsi), %rax ; mulq %rbx; subq %r10, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15 ; sbbq %r10, %r10
        movq 32(%rsi), %rax ; mulq %rbx; subq %r10, %rdx ; addq %rax, %r15 ; adcq %rdx, %r8 ; sbbq %r10, %r10
        movq 40(%rsi), %rax ; mulq %rbx; subq %r10, %rdx ; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %r10, %r10
        negq %r10
        movq %r11, %rbx ; shlq $32, %rbx ; addq %r11, %rbx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r11 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %r11, %rax ; adcq %rbx, %rdx ; adcl %ebp, %ebp ; subq %rax, %r12 ; sbbq %rdx, %r13 ; sbbq %rbp, %r14 ; sbbq $0, %r15 ; sbbq $0, %r8 ; sbbq $0, %rbx ; addq %rbx, %r9 ; adcq $0, %r10
        movq 32(%rcx), %rbx
        movq (%rsi), %rax ; mulq %rbx; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %r11, %r11
        movq 8(%rsi), %rax ; mulq %rbx; subq %r11, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %r11, %r11
        movq 16(%rsi), %rax ; mulq %rbx; subq %r11, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15 ; sbbq %r11, %r11
        movq 24(%rsi), %rax ; mulq %rbx; subq %r11, %rdx ; addq %rax, %r15 ; adcq %rdx, %r8 ; sbbq %r11, %r11
        movq 32(%rsi), %rax ; mulq %rbx; subq %r11, %rdx ; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %r11, %r11
        movq 40(%rsi), %rax ; mulq %rbx; subq %r11, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %r11, %r11
        negq %r11
        movq %r12, %rbx ; shlq $32, %rbx ; addq %r12, %rbx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r12 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %r12, %rax ; adcq %rbx, %rdx ; adcl %ebp, %ebp ; subq %rax, %r13 ; sbbq %rdx, %r14 ; sbbq %rbp, %r15 ; sbbq $0, %r8 ; sbbq $0, %r9 ; sbbq $0, %rbx ; addq %rbx, %r10 ; adcq $0, %r11
        movq 40(%rcx), %rbx
        movq (%rsi), %rax ; mulq %rbx; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %r12, %r12
        movq 8(%rsi), %rax ; mulq %rbx; subq %r12, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15 ; sbbq %r12, %r12
        movq 16(%rsi), %rax ; mulq %rbx; subq %r12, %rdx ; addq %rax, %r15 ; adcq %rdx, %r8 ; sbbq %r12, %r12
        movq 24(%rsi), %rax ; mulq %rbx; subq %r12, %rdx ; addq %rax, %r8 ; adcq %rdx, %r9 ; sbbq %r12, %r12
        movq 32(%rsi), %rax ; mulq %rbx; subq %r12, %rdx ; addq %rax, %r9 ; adcq %rdx, %r10 ; sbbq %r12, %r12
        movq 40(%rsi), %rax ; mulq %rbx; subq %r12, %rdx ; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %r12, %r12
        negq %r12
        movq %r13, %rbx ; shlq $32, %rbx ; addq %r13, %rbx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r13 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %r13, %rax ; adcq %rbx, %rdx ; adcl %ebp, %ebp ; subq %rax, %r14 ; sbbq %rdx, %r15 ; sbbq %rbp, %r8 ; sbbq $0, %r9 ; sbbq $0, %r10 ; sbbq $0, %rbx ; addq %rbx, %r11 ; adcq $0, %r12
        xorl %edx, %edx
        xorl %ebp, %ebp
        xorl %r13d, %r13d
        movq $0xffffffff00000001, %rax
        addq %r14, %rax
        movl $0x00000000ffffffff, %ebx
        adcq %r15, %rbx
        movl $0x0000000000000001, %ecx
        adcq %r8, %rcx
        adcq %r9, %rdx
        adcq %r10, %rbp
        adcq %r11, %r13
        adcq $0, %r12
        cmovnzq %rax, %r14
        cmovnzq %rbx, %r15
        cmovnzq %rcx, %r8
        cmovnzq %rdx, %r9
        cmovnzq %rbp, %r10
        cmovnzq %r13, %r11
        movq %r14, (%rdi)
        movq %r15, 8(%rdi)
        movq %r8, 16(%rdi)
        movq %r9, 24(%rdi)
        movq %r10, 32(%rdi)
        movq %r11, 40(%rdi)
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.size rscrypto_bignum_montmul_p384_alt, .-rscrypto_bignum_montmul_p384_alt
