// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p384/bignum_montinv_p384.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_montinv_p384
        .type rscrypto_bignum_montinv_p384, %function
        .hidden rscrypto_bignum_montinv_p384
        .text
rscrypto_bignum_montinv_p384:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        subq $42*8, %rsp ; .cfi_adjust_cfa_offset 42*8
        movq %rdi, (40*8)(%rsp)
        movl $0xffffffff, %eax
        movq %rax, 0(%rsp)
        movq %rax, %rbx
        notq %rbx
        movq %rbx, 0 +8(%rsp)
        xorl %ebp, %ebp
        leaq -2(%rbp), %rcx
        movq %rcx, 0 +2*8(%rsp)
        leaq -1(%rbp), %rdx
        movq %rdx, 0 +3*8(%rsp)
        movq %rdx, 0 +4*8(%rsp)
        movq %rdx, 0 +5*8(%rsp)
        movq %rbp, 0 +6*8(%rsp)
        movq (%rsi), %r8
        subq %rax, %r8
        movq 8(%rsi), %r9
        sbbq %rbx, %r9
        movq 2*8(%rsi), %r10
        sbbq %rcx, %r10
        movq 3*8(%rsi), %r11
        sbbq %rdx, %r11
        movq 4*8(%rsi), %r12
        sbbq %rdx, %r12
        movq 5*8(%rsi), %r13
        sbbq %rdx, %r13
        cmovcq (%rsi), %r8
        cmovcq 8(%rsi), %r9
        cmovcq 2*8(%rsi), %r10
        cmovcq 3*8(%rsi), %r11
        cmovcq 4*8(%rsi), %r12
        cmovcq 5*8(%rsi), %r13
        movq %r8, (8*8)(%rsp)
        movq %r9, (8*8)+8(%rsp)
        movq %r10, (8*8)+2*8(%rsp)
        movq %r11, (8*8)+3*8(%rsp)
        movq %r12, (8*8)+4*8(%rsp)
        movq %r13, (8*8)+5*8(%rsp)
        movq %rbp, (8*8)+6*8(%rsp)
        xorl %eax, %eax
        movq %rax, (16*8)(%rsp)
        movq %rax, (16*8)+8(%rsp)
        movq %rax, (16*8)+2*8(%rsp)
        movq %rax, (16*8)+3*8(%rsp)
        movq %rax, (16*8)+4*8(%rsp)
        movq %rax, (16*8)+5*8(%rsp)
        movq %rax, (24*8)(%rsp)
        movq $0xfffff00000000800, %rcx
        movq %rcx, (24*8)+8(%rsp)
        movq $0x00001000000007ff, %rdx
        movq %rdx, (24*8)+2*8(%rsp)
        btr $11, %rcx
        movq %rcx, (24*8)+3*8(%rsp)
        movq %rdx, (24*8)+4*8(%rsp)
        bts $11, %rax
        movq %rax, (24*8)+5*8(%rsp)
        movq $15, (34*8)(%rsp)
        movq $1, (35*8)(%rsp)
        jmp Lrscrypto_bignum_montinv_p384_midloop
Lrscrypto_bignum_montinv_p384_loop:
        movq %r8, %r9
        sarq $63, %r9
        xorq %r9, %r8
        subq %r9, %r8
        movq %r10, %r11
        sarq $63, %r11
        xorq %r11, %r10
        subq %r11, %r10
        movq %r12, %r13
        sarq $63, %r13
        xorq %r13, %r12
        subq %r13, %r12
        movq %r14, %r15
        sarq $63, %r15
        xorq %r15, %r14
        subq %r15, %r14
        movq %r8, %rax
        andq %r9, %rax
        movq %r10, %rdi
        andq %r11, %rdi
        addq %rax, %rdi
        movq %rdi, (32*8)(%rsp)
        movq %r12, %rax
        andq %r13, %rax
        movq %r14, %rsi
        andq %r15, %rsi
        addq %rax, %rsi
        movq %rsi, (33*8)(%rsp)
        xorl %ebx, %ebx
        movq 0(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq (8*8)(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rdi
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq 0(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq (8*8)(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        xorl %ecx, %ecx
        movq 0 +8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq (8*8)+8(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        shrdq $59, %rbx, %rdi
        movq %rdi, 0(%rsp)
        xorl %edi, %edi
        movq 0 +8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rbp
        adcq %rdx, %rdi
        movq (8*8)+8(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rdi
        shrdq $59, %rbp, %rsi
        movq %rsi, (8*8)(%rsp)
        xorl %esi, %esi
        movq 0 +2*8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rsi
        movq (8*8)+2*8(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rsi
        shrdq $59, %rcx, %rbx
        movq %rbx, 0 +8(%rsp)
        xorl %ebx, %ebx
        movq 0 +2*8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq (8*8)+2*8(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rdi
        adcq %rdx, %rbx
        shrdq $59, %rdi, %rbp
        movq %rbp, (8*8)+8(%rsp)
        xorl %ebp, %ebp
        movq 0 +3*8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq (8*8)+3*8(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rsi
        adcq %rdx, %rbp
        shrdq $59, %rsi, %rcx
        movq %rcx, 0 +2*8(%rsp)
        xorl %ecx, %ecx
        movq 0 +3*8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq (8*8)+3*8(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbx
        adcq %rdx, %rcx
        shrdq $59, %rbx, %rdi
        movq %rdi, (8*8)+2*8(%rsp)
        xorl %edi, %edi
        movq 0 +4*8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbp
        adcq %rdx, %rdi
        movq (8*8)+4*8(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbp
        adcq %rdx, %rdi
        shrdq $59, %rbp, %rsi
        movq %rsi, 0 +3*8(%rsp)
        xorl %esi, %esi
        movq 0 +4*8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rcx
        adcq %rdx, %rsi
        movq (8*8)+4*8(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rcx
        adcq %rdx, %rsi
        shrdq $59, %rcx, %rbx
        movq %rbx, (8*8)+3*8(%rsp)
        movq 0 +5*8(%rsp), %rax
        xorq %r9, %rax
        movq 0 +6*8(%rsp), %rbx
        xorq %r9, %rbx
        andq %r8, %rbx
        negq %rbx
        mulq %r8
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq (8*8)+5*8(%rsp), %rax
        xorq %r11, %rax
        movq (8*8)+6*8(%rsp), %rdx
        xorq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %rbx
        mulq %r10
        addq %rax, %rdi
        adcq %rdx, %rbx
        shrdq $59, %rdi, %rbp
        movq %rbp, 0 +4*8(%rsp)
        shrdq $59, %rbx, %rdi
        sarq $59, %rbx
        movq 0 +5*8(%rsp), %rax
        movq %rdi, 0 +5*8(%rsp)
        movq 0 +6*8(%rsp), %rdi
        movq %rbx, 0 +6*8(%rsp)
        xorq %r13, %rax
        xorq %r13, %rdi
        andq %r12, %rdi
        negq %rdi
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rdi
        movq (8*8)+5*8(%rsp), %rax
        xorq %r15, %rax
        movq (8*8)+6*8(%rsp), %rdx
        xorq %r15, %rdx
        andq %r14, %rdx
        subq %rdx, %rdi
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rdi
        shrdq $59, %rsi, %rcx
        movq %rcx, (8*8)+4*8(%rsp)
        shrdq $59, %rdi, %rsi
        movq %rsi, (8*8)+5*8(%rsp)
        sarq $59, %rdi
        movq %rdi, (8*8)+6*8(%rsp)
        movq (32*8)(%rsp), %rbx
        movq (33*8)(%rsp), %rbp
        xorl %ecx, %ecx
        movq (16*8)(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq (24*8)(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq (16*8)(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, (16*8)(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq (24*8)(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, (24*8)(%rsp)
        xorl %ebx, %ebx
        movq (16*8)+8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq (24*8)+8(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq (16*8)+8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rcx, (16*8)+8(%rsp)
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq (24*8)+8(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq %rsi, (24*8)+8(%rsp)
        xorl %ecx, %ecx
        movq (16*8)+2*8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq (24*8)+2*8(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq (16*8)+2*8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, (16*8)+2*8(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq (24*8)+2*8(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, (24*8)+2*8(%rsp)
        xorl %ebx, %ebx
        movq (16*8)+3*8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq (24*8)+3*8(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq (16*8)+3*8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rcx, (16*8)+3*8(%rsp)
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq (24*8)+3*8(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq %rsi, (24*8)+3*8(%rsp)
        xorl %ecx, %ecx
        movq (16*8)+4*8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq (24*8)+4*8(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq (16*8)+4*8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, (16*8)+4*8(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq (24*8)+4*8(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, (24*8)+4*8(%rsp)
        movq (16*8)+5*8(%rsp), %rax
        xorq %r9, %rax
        movq %r9, %rbx
        andq %r8, %rbx
        negq %rbx
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq (24*8)+5*8(%rsp), %rax
        xorq %r11, %rax
        movq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %rbx
        mulq %r10
        addq %rax, %rcx
        adcq %rbx, %rdx
        movq (16*8)+5*8(%rsp), %rax
        movq %rcx, (16*8)+5*8(%rsp)
        movq %rdx, (16*8)+6*8(%rsp)
        xorq %r13, %rax
        movq %r13, %rcx
        andq %r12, %rcx
        negq %rcx
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rcx
        movq (24*8)+5*8(%rsp), %rax
        xorq %r15, %rax
        movq %r15, %rdx
        andq %r14, %rdx
        subq %rdx, %rcx
        mulq %r14
        addq %rax, %rsi
        adcq %rcx, %rdx
        movq %rsi, (24*8)+5*8(%rsp)
        movq %rdx, (24*8)+6*8(%rsp)
        movq $0xe000000000000000, %r8 ; xorl %eax, %eax ; addq (16*8)(%rsp), %r8 ; movq $0x000000001fffffff, %r9 ; leaq -1(%rax), %rax ; adcq 8 +(16*8)(%rsp), %r9 ; movq $0xdfffffffe0000000, %r10 ; adcq 2*8 +(16*8)(%rsp), %r10 ; movq 3*8 +(16*8)(%rsp), %r11 ; adcq %rax, %r11 ; movq 4*8 +(16*8)(%rsp), %r12 ; adcq %rax, %r12 ; movq 5*8 +(16*8)(%rsp), %r13 ; adcq %rax, %r13 ; movq $0x1fffffffffffffff, %r14 ; adcq 6*8 +(16*8)(%rsp), %r14 ; movq %r8, %rbx ; shlq $32, %rbx ; addq %r8, %rbx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r8 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %r8, %rax ; adcq %rbx, %rdx ; adcl %ebp, %ebp ; subq %rax, %r9 ; sbbq %rdx, %r10 ; sbbq %rbp, %r11 ; sbbq $0, %r12 ; sbbq $0, %r13 ; sbbq $0, %r14 ; sbbq %rax, %rax ; addq %rbx, %r14 ; adcq $0, %rax ; negq %rax; movq $0x00000000ffffffff, %rbx ; andq %rax, %rbx ; movq $0xffffffff00000000, %rcx ; andq %rax, %rcx ; movq $0xfffffffffffffffe, %rdx ; andq %rax, %rdx ; subq %rbx, %r9 ; movq %r9, (16*8)(%rsp) ; sbbq %rcx, %r10 ; movq %r10, 8 +(16*8)(%rsp) ; sbbq %rdx, %r11 ; movq %r11, 2*8 +(16*8)(%rsp) ; sbbq %rax, %r12 ; movq %r12, 3*8 +(16*8)(%rsp) ; sbbq %rax, %r13 ; movq %r13, 4*8 +(16*8)(%rsp) ; sbbq %rax, %r14 ; movq %r14, 5*8 +(16*8)(%rsp)
        movq $0xe000000000000000, %r8 ; xorl %eax, %eax ; addq (24*8)(%rsp), %r8 ; movq $0x000000001fffffff, %r9 ; leaq -1(%rax), %rax ; adcq 8 +(24*8)(%rsp), %r9 ; movq $0xdfffffffe0000000, %r10 ; adcq 2*8 +(24*8)(%rsp), %r10 ; movq 3*8 +(24*8)(%rsp), %r11 ; adcq %rax, %r11 ; movq 4*8 +(24*8)(%rsp), %r12 ; adcq %rax, %r12 ; movq 5*8 +(24*8)(%rsp), %r13 ; adcq %rax, %r13 ; movq $0x1fffffffffffffff, %r14 ; adcq 6*8 +(24*8)(%rsp), %r14 ; movq %r8, %rbx ; shlq $32, %rbx ; addq %r8, %rbx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r8 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %r8, %rax ; adcq %rbx, %rdx ; adcl %ebp, %ebp ; subq %rax, %r9 ; sbbq %rdx, %r10 ; sbbq %rbp, %r11 ; sbbq $0, %r12 ; sbbq $0, %r13 ; sbbq $0, %r14 ; sbbq %rax, %rax ; addq %rbx, %r14 ; adcq $0, %rax ; negq %rax; movq $0x00000000ffffffff, %rbx ; andq %rax, %rbx ; movq $0xffffffff00000000, %rcx ; andq %rax, %rcx ; movq $0xfffffffffffffffe, %rdx ; andq %rax, %rdx ; subq %rbx, %r9 ; movq %r9, (24*8)(%rsp) ; sbbq %rcx, %r10 ; movq %r10, 8 +(24*8)(%rsp) ; sbbq %rdx, %r11 ; movq %r11, 2*8 +(24*8)(%rsp) ; sbbq %rax, %r12 ; movq %r12, 3*8 +(24*8)(%rsp) ; sbbq %rax, %r13 ; movq %r13, 4*8 +(24*8)(%rsp) ; sbbq %rax, %r14 ; movq %r14, 5*8 +(24*8)(%rsp)
Lrscrypto_bignum_montinv_p384_midloop:
        movq (35*8)(%rsp), %rsi ; movq (%rsp), %rdx ; movq (8*8)(%rsp), %rcx ; movq %rdx, %rbx ; andq $0xfffff, %rbx ; movabsq $0xfffffe0000000000, %rax ; orq %rax, %rbx ; andq $0xfffff, %rcx ; movabsq $0xc000000000000000, %rax ; orq %rax, %rcx ; movq $0xfffffffffffffffe, %rax ; xorl %ebp, %ebp ; movl $0x2, %edx ; movq %rbx, %rdi ; movq %rax, %r8 ; testq %rsi, %rsi ; cmovs %rbp, %r8 ; testq $0x1, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; sarq $1, %rcx ; movl $0x100000, %eax ; leaq (%rbx,%rax), %rdx ; leaq (%rcx,%rax), %rdi ; shlq $0x16, %rdx ; shlq $0x16, %rdi ; sarq $0x2b, %rdx ; sarq $0x2b, %rdi ; movabsq $0x20000100000, %rax ; leaq (%rbx,%rax), %rbx ; leaq (%rcx,%rax), %rcx ; sarq $0x2a, %rbx ; sarq $0x2a, %rcx ; movq %rdx, (36*8)(%rsp) ; movq %rbx, (36*8)+0x8(%rsp) ; movq %rdi, (36*8)+0x10(%rsp) ; movq %rcx, (36*8)+0x18(%rsp) ; movq (%rsp), %r12 ; imulq %r12, %rdi ; imulq %rdx, %r12 ; movq (8*8)(%rsp), %r13 ; imulq %r13, %rbx ; imulq %rcx, %r13 ; addq %rbx, %r12 ; addq %rdi, %r13 ; sarq $0x14, %r12 ; sarq $0x14, %r13 ; movq %r12, %rbx ; andq $0xfffff, %rbx ; movabsq $0xfffffe0000000000, %rax ; orq %rax, %rbx ; movq %r13, %rcx ; andq $0xfffff, %rcx ; movabsq $0xc000000000000000, %rax ; orq %rax, %rcx ; movq $0xfffffffffffffffe, %rax ; movl $0x2, %edx ; movq %rbx, %rdi ; movq %rax, %r8 ; testq %rsi, %rsi ; cmovs %rbp, %r8 ; testq $0x1, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; sarq $1, %rcx ; movl $0x100000, %eax ; leaq (%rbx,%rax), %r8 ; leaq (%rcx,%rax), %r10 ; shlq $0x16, %r8 ; shlq $0x16, %r10 ; sarq $0x2b, %r8 ; sarq $0x2b, %r10 ; movabsq $0x20000100000, %rax ; leaq (%rbx,%rax), %r15 ; leaq (%rcx,%rax), %r11 ; sarq $0x2a, %r15 ; sarq $0x2a, %r11 ; movq %r13, %rbx ; movq %r12, %rcx ; imulq %r8, %r12 ; imulq %r15, %rbx ; addq %rbx, %r12 ; imulq %r11, %r13 ; imulq %r10, %rcx ; addq %rcx, %r13 ; sarq $0x14, %r12 ; sarq $0x14, %r13 ; movq %r12, %rbx ; andq $0xfffff, %rbx ; movabsq $0xfffffe0000000000, %rax ; orq %rax, %rbx ; movq %r13, %rcx ; andq $0xfffff, %rcx ; movabsq $0xc000000000000000, %rax ; orq %rax, %rcx ; movq (36*8)(%rsp), %rax ; imulq %r8, %rax ; movq (36*8)+0x10(%rsp), %rdx ; imulq %r15, %rdx ; imulq (36*8)+0x8(%rsp), %r8 ; imulq (36*8)+0x18(%rsp), %r15 ; addq %r8, %r15 ; leaq (%rax,%rdx), %r9 ; movq (36*8)(%rsp), %rax ; imulq %r10, %rax ; movq (36*8)+0x10(%rsp), %rdx ; imulq %r11, %rdx ; imulq (36*8)+0x8(%rsp), %r10 ; imulq (36*8)+0x18(%rsp), %r11 ; addq %r10, %r11 ; leaq (%rax,%rdx), %r13 ; movq $0xfffffffffffffffe, %rax ; movl $0x2, %edx ; movq %rbx, %rdi ; movq %rax, %r8 ; testq %rsi, %rsi ; cmovs %rbp, %r8 ; testq $0x1, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; cmovs %rbp, %r8 ; movq %rbx, %rdi ; testq %rdx, %rcx ; cmoveq %rbp, %r8 ; cmoveq %rbp, %rdi ; sarq $1, %rcx ; xorq %r8, %rdi ; xorq %r8, %rsi ; btq $0x3f, %r8 ; cmovbq %rcx, %rbx ; movq %rax, %r8 ; subq %rax, %rsi ; leaq (%rcx,%rdi), %rcx ; sarq $1, %rcx ; movl $0x100000, %eax ; leaq (%rbx,%rax), %r8 ; leaq (%rcx,%rax), %r12 ; shlq $0x15, %r8 ; shlq $0x15, %r12 ; sarq $0x2b, %r8 ; sarq $0x2b, %r12 ; movabsq $0x20000100000, %rax ; leaq (%rbx,%rax), %r10 ; leaq (%rcx,%rax), %r14 ; sarq $0x2b, %r10 ; sarq $0x2b, %r14 ; movq %r9, %rax ; imulq %r8, %rax ; movq %r13, %rdx ; imulq %r10, %rdx ; imulq %r15, %r8 ; imulq %r11, %r10 ; addq %r8, %r10 ; leaq (%rax,%rdx), %r8 ; movq %r9, %rax ; imulq %r12, %rax ; movq %r13, %rdx ; imulq %r14, %rdx ; imulq %r15, %r12 ; imulq %r11, %r14 ; addq %r12, %r14 ; leaq (%rax,%rdx), %r12
        movq %rsi, (35*8)(%rsp)
        decq (34*8)(%rsp)
        jnz Lrscrypto_bignum_montinv_p384_loop
        movq 0(%rsp), %rax
        movq (8*8)(%rsp), %rcx
        imulq %r8, %rax
        imulq %r10, %rcx
        addq %rcx, %rax
        sarq $63, %rax
        movq %r8, %r9
        sarq $63, %r9
        xorq %r9, %r8
        subq %r9, %r8
        xorq %rax, %r9
        movq %r10, %r11
        sarq $63, %r11
        xorq %r11, %r10
        subq %r11, %r10
        xorq %rax, %r11
        movq %r12, %r13
        sarq $63, %r13
        xorq %r13, %r12
        subq %r13, %r12
        xorq %rax, %r13
        movq %r14, %r15
        sarq $63, %r15
        xorq %r15, %r14
        subq %r15, %r14
        xorq %rax, %r15
        movq %r8, %rax
        andq %r9, %rax
        movq %r10, %r12
        andq %r11, %r12
        addq %rax, %r12
        xorl %r13d, %r13d
        movq (16*8)(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r12
        adcq %rdx, %r13
        movq (24*8)(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r12
        movq %r12, (16*8)(%rsp)
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movq (16*8)+8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r13
        adcq %rdx, %r14
        movq (24*8)+8(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r13
        movq %r13, (16*8)+8(%rsp)
        adcq %rdx, %r14
        xorl %r15d, %r15d
        movq (16*8)+2*8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r14
        adcq %rdx, %r15
        movq (24*8)+2*8(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r14
        movq %r14, (16*8)+2*8(%rsp)
        adcq %rdx, %r15
        xorl %r14d, %r14d
        movq (16*8)+3*8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r15
        adcq %rdx, %r14
        movq (24*8)+3*8(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r15
        movq %r15, (16*8)+3*8(%rsp)
        adcq %rdx, %r14
        xorl %r15d, %r15d
        movq (16*8)+4*8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r14
        adcq %rdx, %r15
        movq (24*8)+4*8(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r14
        movq %r14, (16*8)+4*8(%rsp)
        adcq %rdx, %r15
        movq (16*8)+5*8(%rsp), %rax
        xorq %r9, %rax
        andq %r8, %r9
        negq %r9
        mulq %r8
        addq %rax, %r15
        adcq %rdx, %r9
        movq (24*8)+5*8(%rsp), %rax
        xorq %r11, %rax
        movq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %r9
        mulq %r10
        addq %rax, %r15
        movq %r15, (16*8)+5*8(%rsp)
        adcq %rdx, %r9
        movq %r9, (16*8)+6*8(%rsp)
        movq $0xe000000000000000, %r8 ; xorl %eax, %eax ; addq (16*8)(%rsp), %r8 ; movq $0x000000001fffffff, %r9 ; leaq -1(%rax), %rax ; adcq 8 +(16*8)(%rsp), %r9 ; movq $0xdfffffffe0000000, %r10 ; adcq 2*8 +(16*8)(%rsp), %r10 ; movq 3*8 +(16*8)(%rsp), %r11 ; adcq %rax, %r11 ; movq 4*8 +(16*8)(%rsp), %r12 ; adcq %rax, %r12 ; movq 5*8 +(16*8)(%rsp), %r13 ; adcq %rax, %r13 ; movq $0x1fffffffffffffff, %r14 ; adcq 6*8 +(16*8)(%rsp), %r14 ; movq %r8, %rbx ; shlq $32, %rbx ; addq %r8, %rbx ; xorl %ebp, %ebp ; movq $0xffffffff00000001, %rax ; mulq %rbx; movq %rdx, %r8 ; movq $0x00000000ffffffff, %rax ; mulq %rbx; addq %r8, %rax ; adcq %rbx, %rdx ; adcl %ebp, %ebp ; subq %rax, %r9 ; sbbq %rdx, %r10 ; sbbq %rbp, %r11 ; sbbq $0, %r12 ; sbbq $0, %r13 ; sbbq $0, %r14 ; sbbq %rax, %rax ; addq %rbx, %r14 ; adcq $0, %rax ; negq %rax; movq $0x00000000ffffffff, %rbx ; andq %rax, %rbx ; movq $0xffffffff00000000, %rcx ; andq %rax, %rcx ; movq $0xfffffffffffffffe, %rdx ; andq %rax, %rdx ; subq %rbx, %r9 ; movq %r9, (16*8)(%rsp) ; sbbq %rcx, %r10 ; movq %r10, 8 +(16*8)(%rsp) ; sbbq %rdx, %r11 ; movq %r11, 2*8 +(16*8)(%rsp) ; sbbq %rax, %r12 ; movq %r12, 3*8 +(16*8)(%rsp) ; sbbq %rax, %r13 ; movq %r13, 4*8 +(16*8)(%rsp) ; sbbq %rax, %r14 ; movq %r14, 5*8 +(16*8)(%rsp)
        movl $0xffffffff, %eax
        movq %rax, %rbx
        notq %rbx
        xorl %ebp, %ebp
        leaq -2(%rbp), %rcx
        leaq -1(%rbp), %rdx
        movq (16*8)(%rsp), %r8
        subq %rax, %r8
        movq (16*8)+8(%rsp), %r9
        sbbq %rbx, %r9
        movq (16*8)+2*8(%rsp), %r10
        sbbq %rcx, %r10
        movq (16*8)+3*8(%rsp), %r11
        sbbq %rdx, %r11
        movq (16*8)+4*8(%rsp), %r12
        sbbq %rdx, %r12
        movq (16*8)+5*8(%rsp), %r13
        sbbq %rdx, %r13
        cmovcq (16*8)(%rsp), %r8
        cmovcq (16*8)+8(%rsp), %r9
        cmovcq (16*8)+2*8(%rsp), %r10
        cmovcq (16*8)+3*8(%rsp), %r11
        cmovcq (16*8)+4*8(%rsp), %r12
        cmovcq (16*8)+5*8(%rsp), %r13
        movq (40*8)(%rsp), %rdi
        movq %r8, (%rdi)
        movq %r9, 8(%rdi)
        movq %r10, 2*8(%rdi)
        movq %r11, 3*8(%rdi)
        movq %r12, 4*8(%rdi)
        movq %r13, 5*8(%rdi)
        addq $42*8, %rsp ; .cfi_adjust_cfa_offset -42*8
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.size rscrypto_bignum_montinv_p384, .-rscrypto_bignum_montinv_p384
