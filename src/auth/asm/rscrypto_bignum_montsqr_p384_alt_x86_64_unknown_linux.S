// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p384/bignum_montsqr_p384_alt.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_montsqr_p384_alt
        .type rscrypto_bignum_montsqr_p384_alt, %function
        .hidden rscrypto_bignum_montsqr_p384_alt
        .text
rscrypto_bignum_montsqr_p384_alt:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        movq (%rsi), %rbx
        movq 8(%rsi), %rax
        mulq %rbx
        movq %rax, %r9
        movq %rdx, %r10
        movq 24(%rsi), %rax
        mulq %rbx
        movq %rax, %r11
        movq %rdx, %r12
        movq 40(%rsi), %rax
        mulq %rbx
        movq %rax, %r13
        movq %rdx, %r14
        movq 24(%rsi), %rax
        mulq 32(%rsi)
        movq %rax, %r15
        movq %rdx, %rcx
        movq 16(%rsi), %rbx
        movq (%rsi), %rax ; mulq %rbx; addq %rax, %r10 ; adcq %rdx, %r11 ; sbbq %rbp, %rbp
        movq 8(%rsi), %rax ; mulq %rbx; subq %rbp, %rdx ; addq %rax, %r11 ; adcq %rdx, %r12 ; sbbq %rbp, %rbp
        movq 8(%rsi), %rbx
        movq 24(%rsi), %rax ; mulq %rbx; subq %rbp, %rdx ; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %rbp, %rbp
        movq 32(%rsi), %rax ; mulq %rbx; subq %rbp, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %rbp, %rbp
        movq 40(%rsi), %rax ; mulq %rbx; subq %rbp, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15
        adcq $0, %rcx
        movq 32(%rsi), %rbx
        movq (%rsi), %rax ; mulq %rbx; addq %rax, %r12 ; adcq %rdx, %r13 ; sbbq %rbp, %rbp
        movq 16(%rsi), %rbx
        movq 24(%rsi), %rax ; mulq %rbx; subq %rbp, %rdx ; addq %rax, %r13 ; adcq %rdx, %r14 ; sbbq %rbp, %rbp
        movq 32(%rsi), %rax ; mulq %rbx; subq %rbp, %rdx ; addq %rax, %r14 ; adcq %rdx, %r15 ; sbbq %rbp, %rbp
        movq 40(%rsi), %rax ; mulq %rbx; subq %rbp, %rdx ; addq %rax, %r15 ; adcq %rdx, %rcx ; sbbq %rbp, %rbp
        xorl %ebx, %ebx
        movq 24(%rsi), %rax
        mulq 40(%rsi)
        subq %rbp, %rdx
        xorl %ebp, %ebp
        addq %rax, %rcx
        adcq %rdx, %rbx
        adcl %ebp, %ebp
        movq 32(%rsi), %rax
        mulq 40(%rsi)
        addq %rax, %rbx
        adcq %rdx, %rbp
        xorl %r8d, %r8d
        addq %r9, %r9
        adcq %r10, %r10
        adcq %r11, %r11
        adcq %r12, %r12
        adcq %r13, %r13
        adcq %r14, %r14
        adcq %r15, %r15
        adcq %rcx, %rcx
        adcq %rbx, %rbx
        adcq %rbp, %rbp
        adcl %r8d, %r8d
        movq (%rsi), %rax
        mulq %rax
        movq %r8, (%rdi)
        movq %rax, %r8
        movq 8(%rsi), %rax
        movq %rbp, 8(%rdi)
        addq %rdx, %r9
        sbbq %rbp, %rbp
        mulq %rax
        negq %rbp
        adcq %rax, %r10
        adcq %rdx, %r11
        sbbq %rbp, %rbp
        movq 16(%rsi), %rax
        mulq %rax
        negq %rbp
        adcq %rax, %r12
        adcq %rdx, %r13
        sbbq %rbp, %rbp
        movq 24(%rsi), %rax
        mulq %rax
        negq %rbp
        adcq %rax, %r14
        adcq %rdx, %r15
        sbbq %rbp, %rbp
        movq 32(%rsi), %rax
        mulq %rax
        negq %rbp
        adcq %rax, %rcx
        adcq %rdx, %rbx
        sbbq %rbp, %rbp
        movq 40(%rsi), %rax
        mulq %rax
        negq %rbp
        adcq 8(%rdi), %rax
        adcq (%rdi), %rdx
        movq %rax, %rbp
        movq %rdx, %rsi
        movq %rbx, (%rdi)
        movq %r8, %rbx ; shlq $32, %rbx ; addq %r8, %rbx ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r8 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %rax, %r8 ; movl $0, %eax ; adcq %rbx, %rdx ; adcl %eax, %eax ; subq %r8, %r9 ; sbbq %rdx, %r10 ; sbbq %rax, %r11 ; sbbq $0, %r12 ; sbbq $0, %r13 ; movq %rbx, %r8 ; sbbq $0, %r8
        movq %r9, %rbx ; shlq $32, %rbx ; addq %r9, %rbx ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r9 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %rax, %r9 ; movl $0, %eax ; adcq %rbx, %rdx ; adcl %eax, %eax ; subq %r9, %r10 ; sbbq %rdx, %r11 ; sbbq %rax, %r12 ; sbbq $0, %r13 ; sbbq $0, %r8 ; movq %rbx, %r9 ; sbbq $0, %r9
        movq %r10, %rbx ; shlq $32, %rbx ; addq %r10, %rbx ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r10 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %rax, %r10 ; movl $0, %eax ; adcq %rbx, %rdx ; adcl %eax, %eax ; subq %r10, %r11 ; sbbq %rdx, %r12 ; sbbq %rax, %r13 ; sbbq $0, %r8 ; sbbq $0, %r9 ; movq %rbx, %r10 ; sbbq $0, %r10
        movq %r11, %rbx ; shlq $32, %rbx ; addq %r11, %rbx ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r11 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %rax, %r11 ; movl $0, %eax ; adcq %rbx, %rdx ; adcl %eax, %eax ; subq %r11, %r12 ; sbbq %rdx, %r13 ; sbbq %rax, %r8 ; sbbq $0, %r9 ; sbbq $0, %r10 ; movq %rbx, %r11 ; sbbq $0, %r11
        movq %r12, %rbx ; shlq $32, %rbx ; addq %r12, %rbx ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r12 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %rax, %r12 ; movl $0, %eax ; adcq %rbx, %rdx ; adcl %eax, %eax ; subq %r12, %r13 ; sbbq %rdx, %r8 ; sbbq %rax, %r9 ; sbbq $0, %r10 ; sbbq $0, %r11 ; movq %rbx, %r12 ; sbbq $0, %r12
        movq %r13, %rbx ; shlq $32, %rbx ; addq %r13, %rbx ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r13 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %rax, %r13 ; movl $0, %eax ; adcq %rbx, %rdx ; adcl %eax, %eax ; subq %r13, %r8 ; sbbq %rdx, %r9 ; sbbq %rax, %r10 ; sbbq $0, %r11 ; sbbq $0, %r12 ; movq %rbx, %r13 ; sbbq $0, %r13
        movq (%rdi), %rbx
        addq %r8, %r14
        adcq %r9, %r15
        adcq %r10, %rcx
        adcq %r11, %rbx
        adcq %r12, %rbp
        adcq %r13, %rsi
        movl $0, %r8d
        adcq %r8, %r8
        xorq %r11, %r11
        xorq %r12, %r12
        xorq %r13, %r13
        movq $0xffffffff00000001, %rax
        addq %r14, %rax
        movl $0x00000000ffffffff, %r9d
        adcq %r15, %r9
        movl $0x0000000000000001, %r10d
        adcq %rcx, %r10
        adcq %rbx, %r11
        adcq %rbp, %r12
        adcq %rsi, %r13
        adcq $0, %r8
        cmovnzq %rax, %r14
        cmovnzq %r9, %r15
        cmovnzq %r10, %rcx
        cmovnzq %r11, %rbx
        cmovnzq %r12, %rbp
        cmovnzq %r13, %rsi
        movq %r14, (%rdi)
        movq %r15, 8(%rdi)
        movq %rcx, 16(%rdi)
        movq %rbx, 24(%rdi)
        movq %rbp, 32(%rdi)
        movq %rsi, 40(%rdi)
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.size rscrypto_bignum_montsqr_p384_alt, .-rscrypto_bignum_montsqr_p384_alt
