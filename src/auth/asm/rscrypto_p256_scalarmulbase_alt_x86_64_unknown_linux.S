// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/p256/p256_scalarmulbase_alt.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_p256_scalarmulbase_alt
        .type rscrypto_p256_scalarmulbase_alt, %function
        .hidden rscrypto_p256_scalarmulbase_alt
        .text
        .balign 4
rscrypto_p256_scalarmulbase_alt:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        subq $11*32, %rsp ; .cfi_adjust_cfa_offset 11*32
        movq %rdi, (9*32)(%rsp)
        movq %rdx, (9*32 +8)(%rsp)
        movq %rcx, (9*32 +16)(%rsp)
        movq $0xf3b9cac2fc632551, %r12
        movq $0xbce6faada7179e84, %r13
        movq $0xffffffffffffffff, %r14
        movq $0xffffffff00000000, %r15
        movq (%rsi), %r8
        subq %r12, %r8
        movq 8(%rsi), %r9
        sbbq %r13, %r9
        movq 16(%rsi), %r10
        sbbq %r14, %r10
        movq 24(%rsi), %r11
        sbbq %r15, %r11
        cmovcq (%rsi), %r8
        cmovcq 8(%rsi), %r9
        cmovcq 16(%rsi), %r10
        cmovcq 24(%rsi), %r11
        movq %r8, (0*32)(%rsp)
        movq %r9, (0*32)+8(%rsp)
        movq %r10, (0*32)+16(%rsp)
        movq %r11, (0*32)+24(%rsp)
        xorl %eax, %eax
        movq %rax, (1*32)(%rsp)
        movq %rax, (1*32)+8(%rsp)
        movq %rax, (1*32)+16(%rsp)
        movq %rax, (1*32)+24(%rsp)
        movq %rax, (1*32)+32(%rsp)
        movq %rax, (1*32)+40(%rsp)
        movq %rax, (1*32)+48(%rsp)
        movq %rax, (1*32)+56(%rsp)
        movq %rax, (1*32)+64(%rsp)
        movq %rax, (1*32)+72(%rsp)
        movq %rax, (1*32)+80(%rsp)
        movq %rax, (1*32)+88(%rsp)
        movq %rax, (9*32 +40)(%rsp)
        movq %rax, (9*32 +24)(%rsp)
Lrscrypto_p256_scalarmulbase_alt_loop:
        movq (0*32)(%rsp), %r8
        movq (0*32)+8(%rsp), %r9
        movq (0*32)+16(%rsp), %r10
        movq (0*32)+24(%rsp), %r11
        movq (9*32 +8)(%rsp), %rcx
        movl $1, %eax
        shlq %cl, %rax
        decq %rax
        andq %r8, %rax
        shrdq %cl, %r9, %r8
        shrdq %cl, %r10, %r9
        shrdq %cl, %r11, %r10
        shrq %cl, %r11
        addq (9*32 +40)(%rsp), %rax
        movq %rax, (9*32 +32)(%rsp)
        movq %r8, (0*32)(%rsp)
        movq %r9, (0*32)+8(%rsp)
        movq %r10, (0*32)+16(%rsp)
        movq %r11, (0*32)+24(%rsp)
        movl $1, %eax
        movq (9*32 +8)(%rsp), %rcx
        shlq %cl, %rax
        movq %rax, %rbx
        shrq $1, %rax
        subq (9*32 +32)(%rsp), %rbx
        cmpq (9*32 +32)(%rsp), %rax
        cmovncq (9*32 +32)(%rsp), %rbx
        sbbq %rax, %rax
        movq %rbx, (9*32 +48)(%rsp)
        negq %rax
        movq %rax, (9*32 +40)(%rsp)
        movq (9*32 +8)(%rsp), %rcx
        decq %rcx
        movl $1, %esi
        shlq %cl, %rsi
        movq (9*32 +48)(%rsp), %r12
        movq (9*32 +16)(%rsp), %rbp
Lrscrypto_p256_scalarmulbase_alt_tabloop:
        subq $1, %r12
        cmovzq (%rbp), %rax
        cmovzq 8(%rbp), %rbx
        cmovzq 16(%rbp), %rcx
        cmovzq 24(%rbp), %rdx
        cmovzq 32(%rbp), %r8
        cmovzq 40(%rbp), %r9
        cmovzq 48(%rbp), %r10
        cmovzq 56(%rbp), %r11
        addq $64, %rbp
        decq %rsi
        jnz Lrscrypto_p256_scalarmulbase_alt_tabloop
        movq %rbp, (9*32 +16)(%rsp)
        xorl %r14d, %r14d
        leaq -1(%r14), %r12
        movq $0x00000000ffffffff, %r15
        movq %r15, %r13
        negq %r15
        subq %r8, %r12
        sbbq %r9, %r13
        sbbq %r10, %r14
        sbbq %r11, %r15
        movq %rax, (7*32)(%rsp)
        movq %rbx, (7*32)+8(%rsp)
        movq %rcx, (7*32)+16(%rsp)
        movq %rdx, (7*32)+24(%rsp)
        movq (9*32 +40)(%rsp), %rax
        testq %rax, %rax
        cmovnzq %r12, %r8
        cmovnzq %r13, %r9
        cmovnzq %r14, %r10
        cmovnzq %r15, %r11
        movq %r8, (7*32)+32(%rsp)
        movq %r9, (7*32)+40(%rsp)
        movq %r10, (7*32)+48(%rsp)
        movq %r11, (7*32)+56(%rsp)
        leaq (4*32)(%rsp), %rdi
        leaq (1*32)(%rsp), %rsi
        leaq (7*32)(%rsp), %rdx
        callq Lrscrypto_p256_scalarmulbase_alt_local_p256_montjmixadd
        movq (9*32 +48)(%rsp), %rax
        testq %rax, %rax
        movq (1*32)(%rsp), %rax
        cmovnzq (4*32)(%rsp), %rax
        movq %rax, (1*32)(%rsp)
        movq (1*32)+8(%rsp), %rax
        cmovnzq (4*32)+8(%rsp), %rax
        movq %rax, (1*32)+8(%rsp)
        movq (1*32)+16(%rsp), %rax
        cmovnzq (4*32)+16(%rsp), %rax
        movq %rax, (1*32)+16(%rsp)
        movq (1*32)+24(%rsp), %rax
        cmovnzq (4*32)+24(%rsp), %rax
        movq %rax, (1*32)+24(%rsp)
        movq (1*32)+32(%rsp), %rax
        cmovnzq (4*32)+32(%rsp), %rax
        movq %rax, (1*32)+32(%rsp)
        movq (1*32)+40(%rsp), %rax
        cmovnzq (4*32)+40(%rsp), %rax
        movq %rax, (1*32)+40(%rsp)
        movq (1*32)+48(%rsp), %rax
        cmovnzq (4*32)+48(%rsp), %rax
        movq %rax, (1*32)+48(%rsp)
        movq (1*32)+56(%rsp), %rax
        cmovnzq (4*32)+56(%rsp), %rax
        movq %rax, (1*32)+56(%rsp)
        movq (1*32)+64(%rsp), %rax
        cmovnzq (4*32)+64(%rsp), %rax
        movq %rax, (1*32)+64(%rsp)
        movq (1*32)+72(%rsp), %rax
        cmovnzq (4*32)+72(%rsp), %rax
        movq %rax, (1*32)+72(%rsp)
        movq (1*32)+80(%rsp), %rax
        cmovnzq (4*32)+80(%rsp), %rax
        movq %rax, (1*32)+80(%rsp)
        movq (1*32)+88(%rsp), %rax
        cmovnzq (4*32)+88(%rsp), %rax
        movq %rax, (1*32)+88(%rsp)
        movq (9*32 +24)(%rsp), %rax
        incq %rax
        movq %rax, (9*32 +24)(%rsp)
        imulq (9*32 +8)(%rsp), %rax
        cmpq $257, %rax
        jc Lrscrypto_p256_scalarmulbase_alt_loop
        leaq (4*32)(%rsp), %rdi
        leaq (1*32)+64(%rsp), %rsi
        callq Lrscrypto_p256_scalarmulbase_alt_local_montsqr_p256
        leaq (5*32)(%rsp), %rdi
        leaq (1*32)+64(%rsp), %rsi
        leaq (4*32)(%rsp), %rdx
        callq Lrscrypto_p256_scalarmulbase_alt_local_montmul_p256
        leaq (4*32)(%rsp), %rdi
        leaq (5*32)(%rsp), %rsi
        callq Lrscrypto_p256_scalarmulbase_alt_local_demont_p256
        leaq (5*32)(%rsp), %rdi
        leaq (4*32)(%rsp), %rsi
        callq Lrscrypto_p256_scalarmulbase_alt_local_inv_p256
        leaq (4*32)(%rsp), %rdi
        leaq (1*32)+64(%rsp), %rsi
        leaq (5*32)(%rsp), %rdx
        callq Lrscrypto_p256_scalarmulbase_alt_local_montmul_p256
        movq (9*32)(%rsp), %rdi
        leaq (1*32)(%rsp), %rsi
        leaq (4*32)(%rsp), %rdx
        movq %rdi, %rbx
        callq Lrscrypto_p256_scalarmulbase_alt_local_montmul_p256
        leaq 32(%rbx), %rdi
        leaq (1*32)+32(%rsp), %rsi
        leaq (5*32)(%rsp), %rdx
        callq Lrscrypto_p256_scalarmulbase_alt_local_montmul_p256
        addq $11*32, %rsp ; .cfi_adjust_cfa_offset -11*32
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        retq ; .cfi_endproc
.size rscrypto_p256_scalarmulbase_alt, .-rscrypto_p256_scalarmulbase_alt
.type Lrscrypto_p256_scalarmulbase_alt_local_demont_p256, %function
Lrscrypto_p256_scalarmulbase_alt_local_demont_p256:
        .cfi_startproc
        movq (%rsi), %r8
        movq 0x8(%rsi), %r9
        movq 0x10(%rsi), %r10
        movq 0x18(%rsi), %r11
        movabsq $0x100000000, %rcx
        movq %r8, %rax
        mulq %rcx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %rsi, %rsi
        movq %r9, %rax
        mulq %rcx
        subq %rsi, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %rsi, %rsi
        negq %rcx
        negq %rsi
        incq %rcx
        movq %r8, %rax
        mulq %rcx
        addq %rax, %r11
        adcq %rdx, %rsi
        sbbq %r8, %r8
        negq %r8
        movq %r9, %rax
        mulq %rcx
        addq %rax, %rsi
        adcq %rdx, %r8
        negq %rcx
        incq %rcx
        movq %r10, %rax
        mulq %rcx
        addq %rax, %r11
        adcq %rdx, %rsi
        sbbq %r9, %r9
        movq %r11, %rax
        mulq %rcx
        subq %r9, %rdx
        addq %rax, %rsi
        adcq %rdx, %r8
        sbbq %r9, %r9
        negq %rcx
        negq %r9
        incq %rcx
        movq %r10, %rax
        mulq %rcx
        addq %rax, %r8
        adcq %rdx, %r9
        sbbq %r10, %r10
        negq %r10
        movq %r11, %rax
        mulq %rcx
        addq %rax, %r9
        adcq %rdx, %r10
        movq %rsi, (%rdi)
        movq %r8, 0x8(%rdi)
        movq %r9, 0x10(%rdi)
        movq %r10, 0x18(%rdi)
        retq ; .cfi_endproc
.size Lrscrypto_p256_scalarmulbase_alt_local_demont_p256, .-Lrscrypto_p256_scalarmulbase_alt_local_demont_p256
.type Lrscrypto_p256_scalarmulbase_alt_local_inv_p256, %function
Lrscrypto_p256_scalarmulbase_alt_local_inv_p256:
        .cfi_startproc
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        subq $240, %rsp ; .cfi_adjust_cfa_offset 240
        movq %rdi, 0xe0(%rsp)
        xorl %ecx, %ecx
        movl $0xffffffff, %edx
        movq %rdx, %rbx
        leaq -0x1(%rcx), %rax
        negq %rdx
        movq %rax, (%rsp)
        movq %rbx, 0x8(%rsp)
        movq %rcx, 0x10(%rsp)
        movq %rdx, 0x18(%rsp)
        movq %rcx, 0x20(%rsp)
        movq (%rsi), %r8
        movq 0x8(%rsi), %r9
        movq 0x10(%rsi), %r10
        movq 0x18(%rsi), %r11
        leaq 0x1(%rcx), %rax
        addq %r8, %rax
        leaq -0x1(%rdx), %rbx
        adcq %r9, %rbx
        notq %rcx
        adcq %r10, %rcx
        notq %rdx
        adcq %r11, %rdx
        cmovaeq %r8, %rax
        cmovaeq %r9, %rbx
        cmovaeq %r10, %rcx
        cmovaeq %r11, %rdx
        movq %rax, 0x28(%rsp)
        movq %rbx, 0x30(%rsp)
        movq %rcx, 0x38(%rsp)
        movq %rdx, 0x40(%rsp)
        xorl %eax, %eax
        movq %rax, 0x48(%rsp)
        xorl %eax, %eax
        movq %rax, 0x50(%rsp)
        movq %rax, 0x58(%rsp)
        movq %rax, 0x60(%rsp)
        movq %rax, 0x68(%rsp)
        movabsq $0x4000000000000, %rcx
        movq %rcx, 0x78(%rsp)
        movq %rax, 0x80(%rsp)
        movq %rax, 0x88(%rsp)
        movq %rax, 0x90(%rsp)
        movq $0xa, 0xb0(%rsp)
        movq $0x1, 0xb8(%rsp)
        jmp Lrscrypto_p256_scalarmulbase_alt_inv_midloop
Lrscrypto_p256_scalarmulbase_alt_inv_loop:
        movq %r8, %r9
        sarq $0x3f, %r9
        xorq %r9, %r8
        subq %r9, %r8
        movq %r10, %r11
        sarq $0x3f, %r11
        xorq %r11, %r10
        subq %r11, %r10
        movq %r12, %r13
        sarq $0x3f, %r13
        xorq %r13, %r12
        subq %r13, %r12
        movq %r14, %r15
        sarq $0x3f, %r15
        xorq %r15, %r14
        subq %r15, %r14
        movq %r8, %rax
        andq %r9, %rax
        movq %r10, %rdi
        andq %r11, %rdi
        addq %rax, %rdi
        movq %rdi, 0xa0(%rsp)
        movq %r12, %rax
        andq %r13, %rax
        movq %r14, %rsi
        andq %r15, %rsi
        addq %rax, %rsi
        movq %rsi, 0xa8(%rsp)
        xorl %ebx, %ebx
        movq (%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq 0x28(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rdi
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq (%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x28(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        xorl %ecx, %ecx
        movq 0x8(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x30(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        shrdq $0x3b, %rbx, %rdi
        movq %rdi, (%rsp)
        xorl %edi, %edi
        movq 0x8(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rbp
        adcq %rdx, %rdi
        movq 0x30(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rdi
        shrdq $0x3b, %rbp, %rsi
        movq %rsi, 0x28(%rsp)
        xorl %esi, %esi
        movq 0x10(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rsi
        movq 0x38(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rsi
        shrdq $0x3b, %rcx, %rbx
        movq %rbx, 0x8(%rsp)
        xorl %ebx, %ebx
        movq 0x10(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        addq %rax, %rdi
        adcq %rdx, %rbx
        movq 0x38(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rdi
        adcq %rdx, %rbx
        shrdq $0x3b, %rdi, %rbp
        movq %rbp, 0x30(%rsp)
        movq 0x18(%rsp), %rax
        xorq %r9, %rax
        movq 0x20(%rsp), %rbp
        xorq %r9, %rbp
        andq %r8, %rbp
        negq %rbp
        mulq %r8
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x40(%rsp), %rax
        xorq %r11, %rax
        movq 0x48(%rsp), %rdx
        xorq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %rbp
        mulq %r10
        addq %rax, %rsi
        adcq %rdx, %rbp
        shrdq $0x3b, %rsi, %rcx
        movq %rcx, 0x10(%rsp)
        shrdq $0x3b, %rbp, %rsi
        sarq $0x3b, %rbp
        movq 0x18(%rsp), %rax
        movq %rsi, 0x18(%rsp)
        movq 0x20(%rsp), %rsi
        movq %rbp, 0x20(%rsp)
        xorq %r13, %rax
        xorq %r13, %rsi
        andq %r12, %rsi
        negq %rsi
        mulq %r12
        addq %rax, %rbx
        adcq %rdx, %rsi
        movq 0x40(%rsp), %rax
        xorq %r15, %rax
        movq 0x48(%rsp), %rdx
        xorq %r15, %rdx
        andq %r14, %rdx
        subq %rdx, %rsi
        mulq %r14
        addq %rax, %rbx
        adcq %rdx, %rsi
        shrdq $0x3b, %rbx, %rdi
        movq %rdi, 0x38(%rsp)
        shrdq $0x3b, %rsi, %rbx
        movq %rbx, 0x40(%rsp)
        sarq $0x3b, %rsi
        movq %rsi, 0x48(%rsp)
        movq 0xa0(%rsp), %rbx
        movq 0xa8(%rsp), %rbp
        xorl %ecx, %ecx
        movq 0x50(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x78(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq 0x50(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, 0x50(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq 0x78(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, 0x78(%rsp)
        xorl %ebx, %ebx
        movq 0x58(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq 0x80(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rcx
        adcq %rdx, %rbx
        xorl %ebp, %ebp
        movq 0x58(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rcx, 0x58(%rsp)
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq 0x80(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rsi
        adcq %rdx, %rbp
        movq %rsi, 0x80(%rsp)
        xorl %ecx, %ecx
        movq 0x60(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %rbx
        adcq %rdx, %rcx
        movq 0x88(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %rbx
        adcq %rdx, %rcx
        xorl %esi, %esi
        movq 0x60(%rsp), %rax
        xorq %r13, %rax
        mulq %r12
        movq %rbx, 0x60(%rsp)
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq 0x88(%rsp), %rax
        xorq %r15, %rax
        mulq %r14
        addq %rax, %rbp
        adcq %rdx, %rsi
        movq %rbp, 0x88(%rsp)
        movq 0x68(%rsp), %rax
        xorq %r9, %rax
        movq %r9, %rbx
        andq %r8, %rbx
        negq %rbx
        mulq %r8
        addq %rax, %rcx
        adcq %rdx, %rbx
        movq 0x90(%rsp), %rax
        xorq %r11, %rax
        movq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %rbx
        mulq %r10
        addq %rax, %rcx
        adcq %rbx, %rdx
        movq 0x68(%rsp), %rax
        movq %rcx, 0x68(%rsp)
        movq %rdx, 0x70(%rsp)
        xorq %r13, %rax
        movq %r13, %rcx
        andq %r12, %rcx
        negq %rcx
        mulq %r12
        addq %rax, %rsi
        adcq %rdx, %rcx
        movq 0x90(%rsp), %rax
        xorq %r15, %rax
        movq %r15, %rdx
        andq %r14, %rdx
        subq %rdx, %rcx
        mulq %r14
        addq %rax, %rsi
        adcq %rcx, %rdx
        movq %rsi, 0x90(%rsp)
        movq %rdx, 0x98(%rsp)
        movabsq $0xe000000000000000, %r8
        addq 0x50(%rsp), %r8
        movq $0xffffffffffffffff, %r9
        adcq 0x58(%rsp), %r9
        movq $0x1fffffff, %r10
        adcq 0x60(%rsp), %r10
        movabsq $0x2000000000000000, %r11
        adcq 0x68(%rsp), %r11
        movabsq $0x1fffffffe0000000, %r12
        adcq 0x70(%rsp), %r12
        movq %r8, %rbx
        shlq $0x20, %rbx
        movabsq $0xffffffff00000001, %rax
        mulq %r8
        shrq $0x20, %r8
        addq %rbx, %r9
        adcq %r8, %r10
        adcq %rax, %r11
        adcq %rdx, %r12
        sbbq %rax, %rax
        movl $0xffffffff, %ebx
        andq %rax, %rbx
        movabsq $0xffffffff00000001, %rdx
        andq %rax, %rdx
        subq %rax, %r9
        movq %r9, 0x50(%rsp)
        sbbq %rbx, %r10
        movq %r10, 0x58(%rsp)
        sbbq $0x0, %r11
        movq %r11, 0x60(%rsp)
        sbbq %rdx, %r12
        movq %r12, 0x68(%rsp)
        movabsq $0xe000000000000000, %r8
        addq 0x78(%rsp), %r8
        movq $0xffffffffffffffff, %r9
        adcq 0x80(%rsp), %r9
        movq $0x1fffffff, %r10
        adcq 0x88(%rsp), %r10
        movabsq $0x2000000000000000, %r11
        adcq 0x90(%rsp), %r11
        movabsq $0x1fffffffe0000000, %r12
        adcq 0x98(%rsp), %r12
        movq %r8, %rbx
        shlq $0x20, %rbx
        movabsq $0xffffffff00000001, %rax
        mulq %r8
        shrq $0x20, %r8
        addq %rbx, %r9
        adcq %r8, %r10
        adcq %rax, %r11
        adcq %rdx, %r12
        sbbq %rax, %rax
        movl $0xffffffff, %ebx
        andq %rax, %rbx
        movabsq $0xffffffff00000001, %rdx
        andq %rax, %rdx
        subq %rax, %r9
        movq %r9, 0x78(%rsp)
        sbbq %rbx, %r10
        movq %r10, 0x80(%rsp)
        sbbq $0x0, %r11
        movq %r11, 0x88(%rsp)
        sbbq %rdx, %r12
        movq %r12, 0x90(%rsp)
Lrscrypto_p256_scalarmulbase_alt_inv_midloop:
        movq 0xb8(%rsp), %rsi
        movq (%rsp), %rdx
        movq 0x28(%rsp), %rcx
        movq %rdx, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq $0xfffffffffffffffe, %rax
        xorl %ebp, %ebp
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %rdx
        leaq (%rcx,%rax), %rdi
        shlq $0x16, %rdx
        shlq $0x16, %rdi
        sarq $0x2b, %rdx
        sarq $0x2b, %rdi
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %rbx
        leaq (%rcx,%rax), %rcx
        sarq $0x2a, %rbx
        sarq $0x2a, %rcx
        movq %rdx, 0xc0(%rsp)
        movq %rbx, 0xc8(%rsp)
        movq %rdi, 0xd0(%rsp)
        movq %rcx, 0xd8(%rsp)
        movq (%rsp), %r12
        imulq %r12, %rdi
        imulq %rdx, %r12
        movq 0x28(%rsp), %r13
        imulq %r13, %rbx
        imulq %rcx, %r13
        addq %rbx, %r12
        addq %rdi, %r13
        sarq $0x14, %r12
        sarq $0x14, %r13
        movq %r12, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        movq %r13, %rcx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq $0xfffffffffffffffe, %rax
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %r8
        leaq (%rcx,%rax), %r10
        shlq $0x16, %r8
        shlq $0x16, %r10
        sarq $0x2b, %r8
        sarq $0x2b, %r10
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %r15
        leaq (%rcx,%rax), %r11
        sarq $0x2a, %r15
        sarq $0x2a, %r11
        movq %r13, %rbx
        movq %r12, %rcx
        imulq %r8, %r12
        imulq %r15, %rbx
        addq %rbx, %r12
        imulq %r11, %r13
        imulq %r10, %rcx
        addq %rcx, %r13
        sarq $0x14, %r12
        sarq $0x14, %r13
        movq %r12, %rbx
        andq $0xfffff, %rbx
        movabsq $0xfffffe0000000000, %rax
        orq %rax, %rbx
        movq %r13, %rcx
        andq $0xfffff, %rcx
        movabsq $0xc000000000000000, %rax
        orq %rax, %rcx
        movq 0xc0(%rsp), %rax
        imulq %r8, %rax
        movq 0xd0(%rsp), %rdx
        imulq %r15, %rdx
        imulq 0xc8(%rsp), %r8
        imulq 0xd8(%rsp), %r15
        addq %r8, %r15
        leaq (%rax,%rdx), %r9
        movq 0xc0(%rsp), %rax
        imulq %r10, %rax
        movq 0xd0(%rsp), %rdx
        imulq %r11, %rdx
        imulq 0xc8(%rsp), %r10
        imulq 0xd8(%rsp), %r11
        addq %r10, %r11
        leaq (%rax,%rdx), %r13
        movq $0xfffffffffffffffe, %rax
        movl $0x2, %edx
        movq %rbx, %rdi
        movq %rax, %r8
        testq %rsi, %rsi
        cmovs %rbp, %r8
        testq $0x1, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        cmovs %rbp, %r8
        movq %rbx, %rdi
        testq %rdx, %rcx
        cmoveq %rbp, %r8
        cmoveq %rbp, %rdi
        sarq $1, %rcx
        xorq %r8, %rdi
        xorq %r8, %rsi
        btq $0x3f, %r8
        cmovbq %rcx, %rbx
        movq %rax, %r8
        subq %rax, %rsi
        leaq (%rcx,%rdi), %rcx
        sarq $1, %rcx
        movl $0x100000, %eax
        leaq (%rbx,%rax), %r8
        leaq (%rcx,%rax), %r12
        shlq $0x15, %r8
        shlq $0x15, %r12
        sarq $0x2b, %r8
        sarq $0x2b, %r12
        movabsq $0x20000100000, %rax
        leaq (%rbx,%rax), %r10
        leaq (%rcx,%rax), %r14
        sarq $0x2b, %r10
        sarq $0x2b, %r14
        movq %r9, %rax
        imulq %r8, %rax
        movq %r13, %rdx
        imulq %r10, %rdx
        imulq %r15, %r8
        imulq %r11, %r10
        addq %r8, %r10
        leaq (%rax,%rdx), %r8
        movq %r9, %rax
        imulq %r12, %rax
        movq %r13, %rdx
        imulq %r14, %rdx
        imulq %r15, %r12
        imulq %r11, %r14
        addq %r12, %r14
        leaq (%rax,%rdx), %r12
        movq %rsi, 0xb8(%rsp)
        decq 0xb0(%rsp)
        jne Lrscrypto_p256_scalarmulbase_alt_inv_loop
        movq (%rsp), %rax
        movq 0x28(%rsp), %rcx
        imulq %r8, %rax
        imulq %r10, %rcx
        addq %rcx, %rax
        sarq $0x3f, %rax
        movq %r8, %r9
        sarq $0x3f, %r9
        xorq %r9, %r8
        subq %r9, %r8
        xorq %rax, %r9
        movq %r10, %r11
        sarq $0x3f, %r11
        xorq %r11, %r10
        subq %r11, %r10
        xorq %rax, %r11
        movq %r12, %r13
        sarq $0x3f, %r13
        xorq %r13, %r12
        subq %r13, %r12
        xorq %rax, %r13
        movq %r14, %r15
        sarq $0x3f, %r15
        xorq %r15, %r14
        subq %r15, %r14
        xorq %rax, %r15
        movq %r8, %rax
        andq %r9, %rax
        movq %r10, %r12
        andq %r11, %r12
        addq %rax, %r12
        xorl %r13d, %r13d
        movq 0x50(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r12
        adcq %rdx, %r13
        movq 0x78(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movq 0x58(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r13
        adcq %rdx, %r14
        movq 0x80(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r13
        adcq %rdx, %r14
        xorl %r15d, %r15d
        movq 0x60(%rsp), %rax
        xorq %r9, %rax
        mulq %r8
        addq %rax, %r14
        adcq %rdx, %r15
        movq 0x88(%rsp), %rax
        xorq %r11, %rax
        mulq %r10
        addq %rax, %r14
        adcq %rdx, %r15
        movq 0x68(%rsp), %rax
        xorq %r9, %rax
        andq %r8, %r9
        negq %r9
        mulq %r8
        addq %rax, %r15
        adcq %rdx, %r9
        movq 0x90(%rsp), %rax
        xorq %r11, %rax
        movq %r11, %rdx
        andq %r10, %rdx
        subq %rdx, %r9
        mulq %r10
        addq %rax, %r15
        adcq %rdx, %r9
        movq %r12, 0x50(%rsp)
        movq %r13, 0x58(%rsp)
        movq %r14, 0x60(%rsp)
        movq %r15, 0x68(%rsp)
        movq %r9, 0x70(%rsp)
        movabsq $0xe000000000000000, %r8
        addq 0x50(%rsp), %r8
        movq $0xffffffffffffffff, %r9
        adcq 0x58(%rsp), %r9
        movq $0x1fffffff, %r10
        adcq 0x60(%rsp), %r10
        movabsq $0x2000000000000000, %r11
        adcq 0x68(%rsp), %r11
        movabsq $0x1fffffffe0000000, %r12
        adcq 0x70(%rsp), %r12
        movq %r8, %rbx
        shlq $0x20, %rbx
        movabsq $0xffffffff00000001, %rax
        mulq %r8
        shrq $0x20, %r8
        addq %rbx, %r9
        adcq %r8, %r10
        adcq %rax, %r11
        adcq %rdx, %r12
        sbbq %rax, %rax
        movl $0xffffffff, %ebx
        andq %rax, %rbx
        movabsq $0xffffffff00000001, %rdx
        andq %rax, %rdx
        subq %rax, %r9
        movq %r9, 0x50(%rsp)
        sbbq %rbx, %r10
        movq %r10, 0x58(%rsp)
        sbbq $0x0, %r11
        movq %r11, 0x60(%rsp)
        sbbq %rdx, %r12
        movq %r12, 0x68(%rsp)
        movq 0x50(%rsp), %r8
        movq 0x58(%rsp), %r9
        movq 0x60(%rsp), %r10
        movq 0x68(%rsp), %r11
        movl $0x1, %eax
        movl $0xffffffff, %ebx
        leaq -0x2(%rax), %rcx
        leaq -0x1(%rbx), %rdx
        notq %rbx
        addq %r8, %rax
        adcq %r9, %rbx
        adcq %r10, %rcx
        adcq %r11, %rdx
        cmovaeq %r8, %rax
        cmovaeq %r9, %rbx
        cmovaeq %r10, %rcx
        cmovaeq %r11, %rdx
        movq 0xe0(%rsp), %rdi
        movq %rax, (%rdi)
        movq %rbx, 0x8(%rdi)
        movq %rcx, 0x10(%rdi)
        movq %rdx, 0x18(%rdi)
        addq $240, %rsp ; .cfi_adjust_cfa_offset -240
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.size Lrscrypto_p256_scalarmulbase_alt_local_inv_p256, .-Lrscrypto_p256_scalarmulbase_alt_local_inv_p256
.type Lrscrypto_p256_scalarmulbase_alt_local_montmul_p256, %function
Lrscrypto_p256_scalarmulbase_alt_local_montmul_p256:
        .cfi_startproc
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
        movq 0x8(%rsi), %rax
        mulq %rbx
        xorl %r10d, %r10d
        addq %rax, %r9
        adcq %rdx, %r10
        movq 0x10(%rsi), %rax
        mulq %rbx
        xorl %r11d, %r11d
        addq %rax, %r10
        adcq %rdx, %r11
        movq 0x18(%rsi), %rax
        mulq %rbx
        xorl %r12d, %r12d
        addq %rax, %r11
        adcq %rdx, %r12
        movq 0x8(%rcx), %rbx
        xorl %r13d, %r13d
        movq (%rsi), %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r14, %r14
        movq 0x8(%rsi), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r14, %r14
        movq 0x10(%rsi), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r14, %r14
        movq 0x18(%rsi), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r14, %r14
        movq 0x10(%rcx), %rbx
        xorl %r15d, %r15d
        movq (%rsi), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r8, %r8
        movq 0x8(%rsi), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r8, %r8
        movq 0x10(%rsi), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r8, %r8
        movq 0x18(%rsi), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        adcq %r15, %r15
        movq 0x18(%rcx), %rbx
        xorl %r8d, %r8d
        movq (%rsi), %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r9, %r9
        movq 0x8(%rsi), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r9, %r9
        movq 0x10(%rsi), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %r9, %r9
        movq 0x18(%rsi), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r8, %r8
        xorl %r9d, %r9d
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        decq %rbx
        adcq %r13, %rbx
        decq %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, (%rdi)
        movq %r13, 0x8(%rdi)
        movq %r14, 0x10(%rdi)
        movq %r15, 0x18(%rdi)
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.size Lrscrypto_p256_scalarmulbase_alt_local_montmul_p256, .-Lrscrypto_p256_scalarmulbase_alt_local_montmul_p256
.type Lrscrypto_p256_scalarmulbase_alt_local_montsqr_p256, %function
Lrscrypto_p256_scalarmulbase_alt_local_montsqr_p256:
        .cfi_startproc
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
        movq 0x8(%rsi), %rax
        mulq %rbx
        movq %rax, %r9
        movq %rdx, %r10
        movq 0x18(%rsi), %rax
        movq %rax, %r13
        mulq %rbx
        movq %rax, %r11
        movq %rdx, %r12
        movq 0x10(%rsi), %rax
        movq %rax, %rbx
        mulq %r13
        movq %rax, %r13
        movq %rdx, %r14
        movq (%rsi), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %rcx, %rcx
        movq 0x8(%rsi), %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq 0x18(%rsi), %rbx
        movq 0x8(%rsi), %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq $0x0, %r14
        xorl %ecx, %ecx
        addq %r9, %r9
        adcq %r10, %r10
        adcq %r11, %r11
        adcq %r12, %r12
        adcq %r13, %r13
        adcq %r14, %r14
        adcq %rcx, %rcx
        movq 0x8(%rsi), %rax
        mulq %rax
        addq %r15, %r9
        adcq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        movq 0x10(%rsi), %rax
        mulq %rax
        negq %r15
        adcq %rax, %r12
        adcq %rdx, %r13
        sbbq %r15, %r15
        movq 0x18(%rsi), %rax
        mulq %rax
        negq %r15
        adcq %rax, %r14
        adcq %rcx, %rdx
        movq %rdx, %r15
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %rcx, %rcx
        movq %r9, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        xorl %r8d, %r8d
        movq %r9, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r8, %r14
        adcq %r8, %r15
        adcq %r8, %r8
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        xorl %r9d, %r9d
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        leaq -0x1(%rbx), %rbx
        adcq %r13, %rbx
        leaq -0x1(%r9), %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, (%rdi)
        movq %r13, 0x8(%rdi)
        movq %r14, 0x10(%rdi)
        movq %r15, 0x18(%rdi)
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.size Lrscrypto_p256_scalarmulbase_alt_local_montsqr_p256, .-Lrscrypto_p256_scalarmulbase_alt_local_montsqr_p256
.type Lrscrypto_p256_scalarmulbase_alt_local_p256_montjmixadd, %function
Lrscrypto_p256_scalarmulbase_alt_local_p256_montjmixadd:
        .cfi_startproc
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        subq $192, %rsp ; .cfi_adjust_cfa_offset 192
        movq %rdx, %rbp
        movq 0x40(%rsi), %rax
        movq %rax, %rbx
        mulq %rax
        movq %rax, %r8
        movq %rdx, %r15
        movq 0x48(%rsi), %rax
        mulq %rbx
        movq %rax, %r9
        movq %rdx, %r10
        movq 0x58(%rsi), %rax
        movq %rax, %r13
        mulq %rbx
        movq %rax, %r11
        movq %rdx, %r12
        movq 0x50(%rsi), %rax
        movq %rax, %rbx
        mulq %r13
        movq %rax, %r13
        movq %rdx, %r14
        movq 0x40(%rsi), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %rcx, %rcx
        movq 0x48(%rsi), %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq 0x58(%rsi), %rbx
        movq 0x48(%rsi), %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq $0x0, %r14
        xorl %ecx, %ecx
        addq %r9, %r9
        adcq %r10, %r10
        adcq %r11, %r11
        adcq %r12, %r12
        adcq %r13, %r13
        adcq %r14, %r14
        adcq %rcx, %rcx
        movq 0x48(%rsi), %rax
        mulq %rax
        addq %r15, %r9
        adcq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        movq 0x50(%rsi), %rax
        mulq %rax
        negq %r15
        adcq %rax, %r12
        adcq %rdx, %r13
        sbbq %r15, %r15
        movq 0x58(%rsi), %rax
        mulq %rax
        negq %r15
        adcq %rax, %r14
        adcq %rcx, %rdx
        movq %rdx, %r15
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %rcx, %rcx
        movq %r9, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        xorl %r8d, %r8d
        movq %r9, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r8, %r14
        adcq %r8, %r15
        adcq %r8, %r8
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        xorl %r9d, %r9d
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        leaq -0x1(%rbx), %rbx
        adcq %r13, %rbx
        leaq -0x1(%r9), %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, (%rsp)
        movq %r13, 0x8(%rsp)
        movq %r14, 0x10(%rsp)
        movq %r15, 0x18(%rsp)
        movq 0x20(%rbp), %rbx
        movq 0x40(%rsi), %rax
        mulq %rbx
        movq %rax, %r8
        movq %rdx, %r9
        movq 0x48(%rsi), %rax
        mulq %rbx
        xorl %r10d, %r10d
        addq %rax, %r9
        adcq %rdx, %r10
        movq 0x50(%rsi), %rax
        mulq %rbx
        xorl %r11d, %r11d
        addq %rax, %r10
        adcq %rdx, %r11
        movq 0x58(%rsi), %rax
        mulq %rbx
        xorl %r12d, %r12d
        addq %rax, %r11
        adcq %rdx, %r12
        movq 0x28(%rbp), %rbx
        xorl %r13d, %r13d
        movq 0x40(%rsi), %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r14, %r14
        movq 0x48(%rsi), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r14, %r14
        movq 0x50(%rsi), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r14, %r14
        movq 0x58(%rsi), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r14, %r14
        movq 0x30(%rbp), %rbx
        xorl %r15d, %r15d
        movq 0x40(%rsi), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r8, %r8
        movq 0x48(%rsi), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r8, %r8
        movq 0x50(%rsi), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r8, %r8
        movq 0x58(%rsi), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        adcq %r15, %r15
        movq 0x38(%rbp), %rbx
        xorl %r8d, %r8d
        movq 0x40(%rsi), %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r9, %r9
        movq 0x48(%rsi), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r9, %r9
        movq 0x50(%rsi), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %r9, %r9
        movq 0x58(%rsi), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r8, %r8
        xorl %r9d, %r9d
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        decq %rbx
        adcq %r13, %rbx
        decq %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, 0x20(%rsp)
        movq %r13, 0x28(%rsp)
        movq %r14, 0x30(%rsp)
        movq %r15, 0x38(%rsp)
        movq 0x0(%rbp), %rbx
        movq (%rsp), %rax
        mulq %rbx
        movq %rax, %r8
        movq %rdx, %r9
        movq 0x8(%rsp), %rax
        mulq %rbx
        xorl %r10d, %r10d
        addq %rax, %r9
        adcq %rdx, %r10
        movq 0x10(%rsp), %rax
        mulq %rbx
        xorl %r11d, %r11d
        addq %rax, %r10
        adcq %rdx, %r11
        movq 0x18(%rsp), %rax
        mulq %rbx
        xorl %r12d, %r12d
        addq %rax, %r11
        adcq %rdx, %r12
        movq 0x8(%rbp), %rbx
        xorl %r13d, %r13d
        movq (%rsp), %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r14, %r14
        movq 0x8(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r14, %r14
        movq 0x10(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r14, %r14
        movq 0x18(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r14, %r14
        movq 0x10(%rbp), %rbx
        xorl %r15d, %r15d
        movq (%rsp), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r8, %r8
        movq 0x8(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r8, %r8
        movq 0x10(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r8, %r8
        movq 0x18(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        adcq %r15, %r15
        movq 0x18(%rbp), %rbx
        xorl %r8d, %r8d
        movq (%rsp), %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r9, %r9
        movq 0x8(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r9, %r9
        movq 0x10(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %r9, %r9
        movq 0x18(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r8, %r8
        xorl %r9d, %r9d
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        decq %rbx
        adcq %r13, %rbx
        decq %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, 0x40(%rsp)
        movq %r13, 0x48(%rsp)
        movq %r14, 0x50(%rsp)
        movq %r15, 0x58(%rsp)
        movq 0x20(%rsp), %rbx
        movq (%rsp), %rax
        mulq %rbx
        movq %rax, %r8
        movq %rdx, %r9
        movq 0x8(%rsp), %rax
        mulq %rbx
        xorl %r10d, %r10d
        addq %rax, %r9
        adcq %rdx, %r10
        movq 0x10(%rsp), %rax
        mulq %rbx
        xorl %r11d, %r11d
        addq %rax, %r10
        adcq %rdx, %r11
        movq 0x18(%rsp), %rax
        mulq %rbx
        xorl %r12d, %r12d
        addq %rax, %r11
        adcq %rdx, %r12
        movq 0x28(%rsp), %rbx
        xorl %r13d, %r13d
        movq (%rsp), %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r14, %r14
        movq 0x8(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r14, %r14
        movq 0x10(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r14, %r14
        movq 0x18(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r14, %r14
        movq 0x30(%rsp), %rbx
        xorl %r15d, %r15d
        movq (%rsp), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r8, %r8
        movq 0x8(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r8, %r8
        movq 0x10(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r8, %r8
        movq 0x18(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        adcq %r15, %r15
        movq 0x38(%rsp), %rbx
        xorl %r8d, %r8d
        movq (%rsp), %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r9, %r9
        movq 0x8(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r9, %r9
        movq 0x10(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %r9, %r9
        movq 0x18(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r8, %r8
        xorl %r9d, %r9d
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        decq %rbx
        adcq %r13, %rbx
        decq %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, 0x20(%rsp)
        movq %r13, 0x28(%rsp)
        movq %r14, 0x30(%rsp)
        movq %r15, 0x38(%rsp)
        movq 0x40(%rsp), %rax
        subq (%rsi), %rax
        movq 0x48(%rsp), %rcx
        sbbq 0x8(%rsi), %rcx
        movq 0x50(%rsp), %r8
        sbbq 0x10(%rsi), %r8
        movq 0x58(%rsp), %r9
        sbbq 0x18(%rsi), %r9
        movl $0xffffffff, %r10d
        sbbq %r11, %r11
        xorq %rdx, %rdx
        andq %r11, %r10
        subq %r10, %rdx
        addq %r11, %rax
        movq %rax, 0xa0(%rsp)
        adcq %r10, %rcx
        movq %rcx, 0xa8(%rsp)
        adcq $0x0, %r8
        movq %r8, 0xb0(%rsp)
        adcq %rdx, %r9
        movq %r9, 0xb8(%rsp)
        movq 0x20(%rsp), %rax
        subq 0x20(%rsi), %rax
        movq 0x28(%rsp), %rcx
        sbbq 0x28(%rsi), %rcx
        movq 0x30(%rsp), %r8
        sbbq 0x30(%rsi), %r8
        movq 0x38(%rsp), %r9
        sbbq 0x38(%rsi), %r9
        movl $0xffffffff, %r10d
        sbbq %r11, %r11
        xorq %rdx, %rdx
        andq %r11, %r10
        subq %r10, %rdx
        addq %r11, %rax
        movq %rax, 0x20(%rsp)
        adcq %r10, %rcx
        movq %rcx, 0x28(%rsp)
        adcq $0x0, %r8
        movq %r8, 0x30(%rsp)
        adcq %rdx, %r9
        movq %r9, 0x38(%rsp)
        movq 0xa0(%rsp), %rax
        movq %rax, %rbx
        mulq %rax
        movq %rax, %r8
        movq %rdx, %r15
        movq 0xa8(%rsp), %rax
        mulq %rbx
        movq %rax, %r9
        movq %rdx, %r10
        movq 0xb8(%rsp), %rax
        movq %rax, %r13
        mulq %rbx
        movq %rax, %r11
        movq %rdx, %r12
        movq 0xb0(%rsp), %rax
        movq %rax, %rbx
        mulq %r13
        movq %rax, %r13
        movq %rdx, %r14
        movq 0xa0(%rsp), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %rcx, %rcx
        movq 0xa8(%rsp), %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq 0xb8(%rsp), %rbx
        movq 0xa8(%rsp), %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq $0x0, %r14
        xorl %ecx, %ecx
        addq %r9, %r9
        adcq %r10, %r10
        adcq %r11, %r11
        adcq %r12, %r12
        adcq %r13, %r13
        adcq %r14, %r14
        adcq %rcx, %rcx
        movq 0xa8(%rsp), %rax
        mulq %rax
        addq %r15, %r9
        adcq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        movq 0xb0(%rsp), %rax
        mulq %rax
        negq %r15
        adcq %rax, %r12
        adcq %rdx, %r13
        sbbq %r15, %r15
        movq 0xb8(%rsp), %rax
        mulq %rax
        negq %r15
        adcq %rax, %r14
        adcq %rcx, %rdx
        movq %rdx, %r15
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %rcx, %rcx
        movq %r9, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        xorl %r8d, %r8d
        movq %r9, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r8, %r14
        adcq %r8, %r15
        adcq %r8, %r8
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        xorl %r9d, %r9d
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        leaq -0x1(%rbx), %rbx
        adcq %r13, %rbx
        leaq -0x1(%r9), %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, 0x60(%rsp)
        movq %r13, 0x68(%rsp)
        movq %r14, 0x70(%rsp)
        movq %r15, 0x78(%rsp)
        movq 0x20(%rsp), %rax
        movq %rax, %rbx
        mulq %rax
        movq %rax, %r8
        movq %rdx, %r15
        movq 0x28(%rsp), %rax
        mulq %rbx
        movq %rax, %r9
        movq %rdx, %r10
        movq 0x38(%rsp), %rax
        movq %rax, %r13
        mulq %rbx
        movq %rax, %r11
        movq %rdx, %r12
        movq 0x30(%rsp), %rax
        movq %rax, %rbx
        mulq %r13
        movq %rax, %r13
        movq %rdx, %r14
        movq 0x20(%rsp), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %rcx, %rcx
        movq 0x28(%rsp), %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq 0x38(%rsp), %rbx
        movq 0x28(%rsp), %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq $0x0, %r14
        xorl %ecx, %ecx
        addq %r9, %r9
        adcq %r10, %r10
        adcq %r11, %r11
        adcq %r12, %r12
        adcq %r13, %r13
        adcq %r14, %r14
        adcq %rcx, %rcx
        movq 0x28(%rsp), %rax
        mulq %rax
        addq %r15, %r9
        adcq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        movq 0x30(%rsp), %rax
        mulq %rax
        negq %r15
        adcq %rax, %r12
        adcq %rdx, %r13
        sbbq %r15, %r15
        movq 0x38(%rsp), %rax
        mulq %rax
        negq %r15
        adcq %rax, %r14
        adcq %rcx, %rdx
        movq %rdx, %r15
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %rcx, %rcx
        movq %r9, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        xorl %r8d, %r8d
        movq %r9, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r8, %r14
        adcq %r8, %r15
        adcq %r8, %r8
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        xorl %r9d, %r9d
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        leaq -0x1(%rbx), %rbx
        adcq %r13, %rbx
        leaq -0x1(%r9), %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, (%rsp)
        movq %r13, 0x8(%rsp)
        movq %r14, 0x10(%rsp)
        movq %r15, 0x18(%rsp)
        movq (%rsi), %rbx
        movq 0x60(%rsp), %rax
        mulq %rbx
        movq %rax, %r8
        movq %rdx, %r9
        movq 0x68(%rsp), %rax
        mulq %rbx
        xorl %r10d, %r10d
        addq %rax, %r9
        adcq %rdx, %r10
        movq 0x70(%rsp), %rax
        mulq %rbx
        xorl %r11d, %r11d
        addq %rax, %r10
        adcq %rdx, %r11
        movq 0x78(%rsp), %rax
        mulq %rbx
        xorl %r12d, %r12d
        addq %rax, %r11
        adcq %rdx, %r12
        movq 0x8(%rsi), %rbx
        xorl %r13d, %r13d
        movq 0x60(%rsp), %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r14, %r14
        movq 0x68(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r14, %r14
        movq 0x70(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r14, %r14
        movq 0x78(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r14, %r14
        movq 0x10(%rsi), %rbx
        xorl %r15d, %r15d
        movq 0x60(%rsp), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r8, %r8
        movq 0x68(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r8, %r8
        movq 0x70(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r8, %r8
        movq 0x78(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        adcq %r15, %r15
        movq 0x18(%rsi), %rbx
        xorl %r8d, %r8d
        movq 0x60(%rsp), %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r9, %r9
        movq 0x68(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r9, %r9
        movq 0x70(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %r9, %r9
        movq 0x78(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r8, %r8
        xorl %r9d, %r9d
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        decq %rbx
        adcq %r13, %rbx
        decq %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, 0x80(%rsp)
        movq %r13, 0x88(%rsp)
        movq %r14, 0x90(%rsp)
        movq %r15, 0x98(%rsp)
        movq 0x40(%rsp), %rbx
        movq 0x60(%rsp), %rax
        mulq %rbx
        movq %rax, %r8
        movq %rdx, %r9
        movq 0x68(%rsp), %rax
        mulq %rbx
        xorl %r10d, %r10d
        addq %rax, %r9
        adcq %rdx, %r10
        movq 0x70(%rsp), %rax
        mulq %rbx
        xorl %r11d, %r11d
        addq %rax, %r10
        adcq %rdx, %r11
        movq 0x78(%rsp), %rax
        mulq %rbx
        xorl %r12d, %r12d
        addq %rax, %r11
        adcq %rdx, %r12
        movq 0x48(%rsp), %rbx
        xorl %r13d, %r13d
        movq 0x60(%rsp), %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r14, %r14
        movq 0x68(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r14, %r14
        movq 0x70(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r14, %r14
        movq 0x78(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r14, %r14
        movq 0x50(%rsp), %rbx
        xorl %r15d, %r15d
        movq 0x60(%rsp), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r8, %r8
        movq 0x68(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r8, %r8
        movq 0x70(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r8, %r8
        movq 0x78(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        adcq %r15, %r15
        movq 0x58(%rsp), %rbx
        xorl %r8d, %r8d
        movq 0x60(%rsp), %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r9, %r9
        movq 0x68(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r9, %r9
        movq 0x70(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %r9, %r9
        movq 0x78(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r8, %r8
        xorl %r9d, %r9d
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        decq %rbx
        adcq %r13, %rbx
        decq %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, 0x40(%rsp)
        movq %r13, 0x48(%rsp)
        movq %r14, 0x50(%rsp)
        movq %r15, 0x58(%rsp)
        movq (%rsp), %rax
        subq 0x80(%rsp), %rax
        movq 0x8(%rsp), %rcx
        sbbq 0x88(%rsp), %rcx
        movq 0x10(%rsp), %r8
        sbbq 0x90(%rsp), %r8
        movq 0x18(%rsp), %r9
        sbbq 0x98(%rsp), %r9
        movl $0xffffffff, %r10d
        sbbq %r11, %r11
        xorq %rdx, %rdx
        andq %r11, %r10
        subq %r10, %rdx
        addq %r11, %rax
        movq %rax, (%rsp)
        adcq %r10, %rcx
        movq %rcx, 0x8(%rsp)
        adcq $0x0, %r8
        movq %r8, 0x10(%rsp)
        adcq %rdx, %r9
        movq %r9, 0x18(%rsp)
        movq 0x40(%rsp), %rax
        subq 0x80(%rsp), %rax
        movq 0x48(%rsp), %rcx
        sbbq 0x88(%rsp), %rcx
        movq 0x50(%rsp), %r8
        sbbq 0x90(%rsp), %r8
        movq 0x58(%rsp), %r9
        sbbq 0x98(%rsp), %r9
        movl $0xffffffff, %r10d
        sbbq %r11, %r11
        xorq %rdx, %rdx
        andq %r11, %r10
        subq %r10, %rdx
        addq %r11, %rax
        movq %rax, 0x60(%rsp)
        adcq %r10, %rcx
        movq %rcx, 0x68(%rsp)
        adcq $0x0, %r8
        movq %r8, 0x70(%rsp)
        adcq %rdx, %r9
        movq %r9, 0x78(%rsp)
        movq 0x40(%rsi), %rbx
        movq 0xa0(%rsp), %rax
        mulq %rbx
        movq %rax, %r8
        movq %rdx, %r9
        movq 0xa8(%rsp), %rax
        mulq %rbx
        xorl %r10d, %r10d
        addq %rax, %r9
        adcq %rdx, %r10
        movq 0xb0(%rsp), %rax
        mulq %rbx
        xorl %r11d, %r11d
        addq %rax, %r10
        adcq %rdx, %r11
        movq 0xb8(%rsp), %rax
        mulq %rbx
        xorl %r12d, %r12d
        addq %rax, %r11
        adcq %rdx, %r12
        movq 0x48(%rsi), %rbx
        xorl %r13d, %r13d
        movq 0xa0(%rsp), %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r14, %r14
        movq 0xa8(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r14, %r14
        movq 0xb0(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r14, %r14
        movq 0xb8(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r14, %r14
        movq 0x50(%rsi), %rbx
        xorl %r15d, %r15d
        movq 0xa0(%rsp), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r8, %r8
        movq 0xa8(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r8, %r8
        movq 0xb0(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r8, %r8
        movq 0xb8(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        adcq %r15, %r15
        movq 0x58(%rsi), %rbx
        xorl %r8d, %r8d
        movq 0xa0(%rsp), %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r9, %r9
        movq 0xa8(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r9, %r9
        movq 0xb0(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %r9, %r9
        movq 0xb8(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r8, %r8
        xorl %r9d, %r9d
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        decq %rbx
        adcq %r13, %rbx
        decq %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, 0xa0(%rsp)
        movq %r13, 0xa8(%rsp)
        movq %r14, 0xb0(%rsp)
        movq %r15, 0xb8(%rsp)
        movq (%rsp), %rax
        subq 0x40(%rsp), %rax
        movq 0x8(%rsp), %rcx
        sbbq 0x48(%rsp), %rcx
        movq 0x10(%rsp), %r8
        sbbq 0x50(%rsp), %r8
        movq 0x18(%rsp), %r9
        sbbq 0x58(%rsp), %r9
        movl $0xffffffff, %r10d
        sbbq %r11, %r11
        xorq %rdx, %rdx
        andq %r11, %r10
        subq %r10, %rdx
        addq %r11, %rax
        movq %rax, (%rsp)
        adcq %r10, %rcx
        movq %rcx, 0x8(%rsp)
        adcq $0x0, %r8
        movq %r8, 0x10(%rsp)
        adcq %rdx, %r9
        movq %r9, 0x18(%rsp)
        movq 0x80(%rsp), %rax
        subq (%rsp), %rax
        movq 0x88(%rsp), %rcx
        sbbq 0x8(%rsp), %rcx
        movq 0x90(%rsp), %r8
        sbbq 0x10(%rsp), %r8
        movq 0x98(%rsp), %r9
        sbbq 0x18(%rsp), %r9
        movl $0xffffffff, %r10d
        sbbq %r11, %r11
        xorq %rdx, %rdx
        andq %r11, %r10
        subq %r10, %rdx
        addq %r11, %rax
        movq %rax, 0x80(%rsp)
        adcq %r10, %rcx
        movq %rcx, 0x88(%rsp)
        adcq $0x0, %r8
        movq %r8, 0x90(%rsp)
        adcq %rdx, %r9
        movq %r9, 0x98(%rsp)
        movq 0x20(%rsi), %rbx
        movq 0x60(%rsp), %rax
        mulq %rbx
        movq %rax, %r8
        movq %rdx, %r9
        movq 0x68(%rsp), %rax
        mulq %rbx
        xorl %r10d, %r10d
        addq %rax, %r9
        adcq %rdx, %r10
        movq 0x70(%rsp), %rax
        mulq %rbx
        xorl %r11d, %r11d
        addq %rax, %r10
        adcq %rdx, %r11
        movq 0x78(%rsp), %rax
        mulq %rbx
        xorl %r12d, %r12d
        addq %rax, %r11
        adcq %rdx, %r12
        movq 0x28(%rsi), %rbx
        xorl %r13d, %r13d
        movq 0x60(%rsp), %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r14, %r14
        movq 0x68(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r14, %r14
        movq 0x70(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r14, %r14
        movq 0x78(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r14, %r14
        movq 0x30(%rsi), %rbx
        xorl %r15d, %r15d
        movq 0x60(%rsp), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r8, %r8
        movq 0x68(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r8, %r8
        movq 0x70(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r8, %r8
        movq 0x78(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        adcq %r15, %r15
        movq 0x38(%rsi), %rbx
        xorl %r8d, %r8d
        movq 0x60(%rsp), %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r9, %r9
        movq 0x68(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r9, %r9
        movq 0x70(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %r9, %r9
        movq 0x78(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r8, %r8
        xorl %r9d, %r9d
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        decq %rbx
        adcq %r13, %rbx
        decq %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, 0x60(%rsp)
        movq %r13, 0x68(%rsp)
        movq %r14, 0x70(%rsp)
        movq %r15, 0x78(%rsp)
        movq 0x80(%rsp), %rbx
        movq 0x20(%rsp), %rax
        mulq %rbx
        movq %rax, %r8
        movq %rdx, %r9
        movq 0x28(%rsp), %rax
        mulq %rbx
        xorl %r10d, %r10d
        addq %rax, %r9
        adcq %rdx, %r10
        movq 0x30(%rsp), %rax
        mulq %rbx
        xorl %r11d, %r11d
        addq %rax, %r10
        adcq %rdx, %r11
        movq 0x38(%rsp), %rax
        mulq %rbx
        xorl %r12d, %r12d
        addq %rax, %r11
        adcq %rdx, %r12
        movq 0x88(%rsp), %rbx
        xorl %r13d, %r13d
        movq 0x20(%rsp), %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r14, %r14
        movq 0x28(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r14, %r14
        movq 0x30(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r14, %r14
        movq 0x38(%rsp), %rax
        mulq %rbx
        subq %r14, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        xorl %r14d, %r14d
        movabsq $0x100000000, %rbx
        movq %r8, %rax
        mulq %rbx
        addq %rax, %r9
        adcq %rdx, %r10
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r15, %r15
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r8, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r15, %r15
        movq %r9, %rax
        mulq %rbx
        subq %r15, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        adcq %r14, %r14
        movq 0x90(%rsp), %rbx
        xorl %r15d, %r15d
        movq 0x20(%rsp), %rax
        mulq %rbx
        addq %rax, %r10
        adcq %rdx, %r11
        sbbq %r8, %r8
        movq 0x28(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r8, %r8
        movq 0x30(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r8, %r8
        movq 0x38(%rsp), %rax
        mulq %rbx
        subq %r8, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        adcq %r15, %r15
        movq 0x98(%rsp), %rbx
        xorl %r8d, %r8d
        movq 0x20(%rsp), %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %r9, %r9
        movq 0x28(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %r9, %r9
        movq 0x30(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %r9, %r9
        movq 0x38(%rsp), %rax
        mulq %rbx
        subq %r9, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r8, %r8
        xorl %r9d, %r9d
        movabsq $0x100000000, %rbx
        movq %r10, %rax
        mulq %rbx
        addq %rax, %r11
        adcq %rdx, %r12
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r12
        adcq %rdx, %r13
        sbbq %rcx, %rcx
        notq %rbx
        leaq 0x2(%rbx), %rbx
        movq %r10, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r13
        adcq %rdx, %r14
        sbbq %rcx, %rcx
        movq %r11, %rax
        mulq %rbx
        subq %rcx, %rdx
        addq %rax, %r14
        adcq %rdx, %r15
        adcq %r9, %r8
        movl $0x1, %ecx
        addq %r12, %rcx
        decq %rbx
        adcq %r13, %rbx
        decq %r9
        movq %r9, %rax
        adcq %r14, %r9
        movl $0xfffffffe, %r11d
        adcq %r15, %r11
        adcq %r8, %rax
        cmovbq %rcx, %r12
        cmovbq %rbx, %r13
        cmovbq %r9, %r14
        cmovbq %r11, %r15
        movq %r12, 0x80(%rsp)
        movq %r13, 0x88(%rsp)
        movq %r14, 0x90(%rsp)
        movq %r15, 0x98(%rsp)
        movq 0x80(%rsp), %rax
        subq 0x60(%rsp), %rax
        movq 0x88(%rsp), %rcx
        sbbq 0x68(%rsp), %rcx
        movq 0x90(%rsp), %r8
        sbbq 0x70(%rsp), %r8
        movq 0x98(%rsp), %r9
        sbbq 0x78(%rsp), %r9
        movl $0xffffffff, %r10d
        sbbq %r11, %r11
        xorq %rdx, %rdx
        andq %r11, %r10
        subq %r10, %rdx
        addq %r11, %rax
        movq %rax, 0x80(%rsp)
        adcq %r10, %rcx
        movq %rcx, 0x88(%rsp)
        adcq $0x0, %r8
        movq %r8, 0x90(%rsp)
        adcq %rdx, %r9
        movq %r9, 0x98(%rsp)
        movq 0x40(%rsi), %rax
        movq 0x48(%rsi), %rdx
        orq 0x50(%rsi), %rax
        orq 0x58(%rsi), %rdx
        orq %rdx, %rax
        movq (%rsp), %r8
        movq 0x0(%rbp), %rax
        cmoveq %rax, %r8
        movq 0x8(%rsp), %r9
        movq 0x8(%rbp), %rax
        cmoveq %rax, %r9
        movq 0x10(%rsp), %r10
        movq 0x10(%rbp), %rax
        cmoveq %rax, %r10
        movq 0x18(%rsp), %r11
        movq 0x18(%rbp), %rax
        cmoveq %rax, %r11
        movq 0x80(%rsp), %r12
        movq 0x20(%rbp), %rax
        cmoveq %rax, %r12
        movq 0x88(%rsp), %r13
        movq 0x28(%rbp), %rax
        cmoveq %rax, %r13
        movq 0x90(%rsp), %r14
        movq 0x30(%rbp), %rax
        cmoveq %rax, %r14
        movq 0x98(%rsp), %r15
        movq 0x38(%rbp), %rax
        cmoveq %rax, %r15
        movq %r8, (%rdi)
        movq %r9, 0x8(%rdi)
        movq %r10, 0x10(%rdi)
        movq %r11, 0x18(%rdi)
        movq %r12, 0x20(%rdi)
        movq %r13, 0x28(%rdi)
        movq %r14, 0x30(%rdi)
        movq %r15, 0x38(%rdi)
        movq 0xa0(%rsp), %r8
        movq 0xa8(%rsp), %r9
        movq 0xb0(%rsp), %r10
        movq 0xb8(%rsp), %r11
        movl $0x1, %eax
        cmoveq %rax, %r8
        movabsq $0xffffffff00000000, %rax
        cmoveq %rax, %r9
        movq $0xffffffffffffffff, %rax
        cmoveq %rax, %r10
        movl $0xfffffffe, %eax
        cmoveq %rax, %r11
        movq %r8, 0x40(%rdi)
        movq %r9, 0x48(%rdi)
        movq %r10, 0x50(%rdi)
        movq %r11, 0x58(%rdi)
        addq $192, %rsp ; .cfi_adjust_cfa_offset -192
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        retq ; .cfi_endproc
.size Lrscrypto_p256_scalarmulbase_alt_local_p256_montjmixadd, .-Lrscrypto_p256_scalarmulbase_alt_local_p256_montjmixadd
