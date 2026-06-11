// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0
//
// Adapted for rscrypto from s2n-bignum:
// - x86_att/generic/bignum_modinv.S
//
// Public symbol is renamed and embedded with Rust global_asm!.

 .globl rscrypto_bignum_modinv
        .type rscrypto_bignum_modinv, %function
        .hidden rscrypto_bignum_modinv
        .text
rscrypto_bignum_modinv:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %rbp ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbp, 0
        pushq %rbx ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %rbx, 0
        pushq %r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r12, 0
        pushq %r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r13, 0
        pushq %r14 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r14, 0
        pushq %r15 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %r15, 0
        subq $80, %rsp ; .cfi_adjust_cfa_offset 80
        testq %rdi, %rdi
        jz Lrscrypto_bignum_modinv_end
        movq %rsi, 64(%rsp)
        movq %r8, 56(%rsp)
        movq %rcx, 72(%rsp)
        leaq (%r8,%rdi,8), %r10
        movq %r10, 48(%rsp)
        leaq (%r10,%rdi,8), %r15
        xorq %r11, %r11
        xorq %r9, %r9
Lrscrypto_bignum_modinv_copyloop:
        movq (%rdx,%r9,8), %rax
        movq (%rcx,%r9,8), %rbx
        movq %rax, (%r10,%r9,8)
        movq %rbx, (%r15,%r9,8)
        movq %rbx, (%r8,%r9,8)
        movq %r11, (%rsi,%r9,8)
        incq %r9
        cmpq %rdi, %r9
        jc Lrscrypto_bignum_modinv_copyloop
        movq (%r8), %rax
        movq %rax, %rbx
        decq %rbx
        movq %rbx, (%r8)
        movq %rax, %rbp
        movq %rax, %r12
        shlq $2, %rbp
        subq %rbp, %r12
        xorq $2, %r12
        movq %r12, %rbp
        imulq %rax, %rbp
        movl $2, %eax
        addq %rbp, %rax
        addq $1, %rbp
        imulq %rax, %r12
        imulq %rbp, %rbp
        movl $1, %eax
        addq %rbp, %rax
        imulq %rax, %r12
        imulq %rbp, %rbp
        movl $1, %eax
        addq %rbp, %rax
        imulq %rax, %r12
        imulq %rbp, %rbp
        movl $1, %eax
        addq %rbp, %rax
        imulq %rax, %r12
        movq %r12, 40(%rsp)
        movq %rdi, %rax
        shlq $7, %rax
        movq %rax, 32(%rsp)
Lrscrypto_bignum_modinv_outerloop:
        movq 32(%rsp), %r13
        addq $63, %r13
        shrq $6, %r13
        cmpq %rdi, %r13
        cmovncq %rdi, %r13
        xorq %r12, %r12
        xorq %r14, %r14
        xorq %rbp, %rbp
        xorq %rsi, %rsi
        xorq %r11, %r11
        movq 48(%rsp), %r8
        leaq (%r8,%rdi,8), %r15
        xorq %r9, %r9
Lrscrypto_bignum_modinv_toploop:
        movq (%r8,%r9,8), %rbx
        movq (%r15,%r9,8), %rcx
        movq %r11, %r10
        andq %r12, %r10
        andq %rbp, %r11
        movq %rbx, %rax
        orq %rcx, %rax
        negq %rax
        cmovcq %r10, %r14
        cmovcq %r11, %rsi
        cmovcq %rbx, %r12
        cmovcq %rcx, %rbp
        sbbq %r11, %r11
        incq %r9
        cmpq %r13, %r9
        jc Lrscrypto_bignum_modinv_toploop
        movq %r12, %rax
        orq %rbp, %rax
        bsrq %rax, %rcx
        xorq $63, %rcx
        shldq %cl, %r14, %r12
        shldq %cl, %rsi, %rbp
        movq (%r8), %rax
        movq %rax, %r14
        movq (%r15), %rax
        movq %rax, %rsi
        movl $1, %r10d
        movl $0, %r11d
        movl $0, %ecx
        movl $1, %edx
        movl $58, %r9d
        movq %rdi, 8(%rsp)
        movq %r13, 16(%rsp)
        movq %r8, (%rsp)
        movq %r15, 24(%rsp)
Lrscrypto_bignum_modinv_innerloop:
        xorl %eax, %eax
        xorl %ebx, %ebx
        xorq %r8, %r8
        xorq %r15, %r15
        btq $0, %r14
        cmovcq %rbp, %rax
        cmovcq %rsi, %rbx
        cmovcq %rcx, %r8
        cmovcq %rdx, %r15
        movq %r14, %r13
        subq %rbx, %r14
        subq %r13, %rbx
        movq %r12, %rdi
        subq %rax, %rdi
        cmovcq %r12, %rbp
        leaq -1(%rdi), %r12
        cmovcq %rbx, %r14
        cmovcq %r13, %rsi
        notq %r12
        cmovcq %r10, %rcx
        cmovcq %r11, %rdx
        cmovncq %rdi, %r12
        shrq $1, %r14
        addq %r8, %r10
        addq %r15, %r11
        shrq $1, %r12
        addq %rcx, %rcx
        addq %rdx, %rdx
        decq %r9
        jnz Lrscrypto_bignum_modinv_innerloop
        movq 8(%rsp), %rdi
        movq 16(%rsp), %r13
        movq (%rsp), %r8
        movq 24(%rsp), %r15
        movq %r10, (%rsp)
        movq %r11, 8(%rsp)
        movq %rcx, 16(%rsp)
        movq %rdx, 24(%rsp)
        movq 56(%rsp), %r8
        movq 64(%rsp), %r15
        xorq %r14, %r14
        xorq %rsi, %rsi
        xorq %r10, %r10
        xorq %r11, %r11
        xorq %r9, %r9
Lrscrypto_bignum_modinv_congloop:
        movq (%r8,%r9,8), %rcx
        movq (%rsp), %rax
        mulq %rcx
        addq %rax, %r14
        adcq $0, %rdx
        movq %rdx, %r12
        movq 16(%rsp), %rax
        mulq %rcx
        addq %rax, %rsi
        adcq $0, %rdx
        movq %rdx, %rbp
        movq (%r15,%r9,8), %rcx
        movq 8(%rsp), %rax
        mulq %rcx
        addq %rax, %r14
        adcq %rdx, %r12
        shrdq $58, %r14, %r10
        movq %r10, (%r8,%r9,8)
        movq %r14, %r10
        movq %r12, %r14
        movq 24(%rsp), %rax
        mulq %rcx
        addq %rax, %rsi
        adcq %rdx, %rbp
        shrdq $58, %rsi, %r11
        movq %r11, (%r15,%r9,8)
        movq %rsi, %r11
        movq %rbp, %rsi
        incq %r9
        cmpq %rdi, %r9
        jc Lrscrypto_bignum_modinv_congloop
        shldq $64-58, %r10, %r14
        shldq $64-58, %r11, %rsi
        movq 72(%rsp), %r15
        movq (%r8), %rbx
        movq 40(%rsp), %r12
        imulq %rbx, %r12
        movq (%r15), %rax
        mulq %r12
        addq %rbx, %rax
        movq %rdx, %r10
        movl $1, %r9d
        movq %rdi, %rcx
        decq %rcx
        jz Lrscrypto_bignum_modinv_wmontend
Lrscrypto_bignum_modinv_wmontloop:
        adcq (%r8,%r9,8), %r10
        sbbq %rbx, %rbx
        movq (%r15,%r9,8), %rax
        mulq %r12
        subq %rbx, %rdx
        addq %r10, %rax
        movq %rax, -8(%r8,%r9,8)
        movq %rdx, %r10
        incq %r9
        decq %rcx
        jnz Lrscrypto_bignum_modinv_wmontloop
Lrscrypto_bignum_modinv_wmontend:
        adcq %r14, %r10
        movq %r10, -8(%r8,%rdi,8)
        sbbq %r10, %r10
        negq %r10
        movq %rdi, %rcx
        xorq %r9, %r9
Lrscrypto_bignum_modinv_wcmploop:
        movq (%r8,%r9,8), %rax
        sbbq (%r15,%r9,8), %rax
        incq %r9
        decq %rcx
        jnz Lrscrypto_bignum_modinv_wcmploop
        sbbq $0, %r10
        sbbq %r10, %r10
        notq %r10
        xorq %rcx, %rcx
        xorq %r9, %r9
Lrscrypto_bignum_modinv_wcorrloop:
        movq (%r8,%r9,8), %rax
        movq (%r15,%r9,8), %rbx
        andq %r10, %rbx
        negq %rcx
        sbbq %rbx, %rax
        sbbq %rcx, %rcx
        movq %rax, (%r8,%r9,8)
        incq %r9
        cmpq %rdi, %r9
        jc Lrscrypto_bignum_modinv_wcorrloop
        movq 64(%rsp), %r8
        movq (%r8), %rbx
        movq 40(%rsp), %rbp
        imulq %rbx, %rbp
        movq (%r15), %rax
        mulq %rbp
        addq %rbx, %rax
        movq %rdx, %r11
        movl $1, %r9d
        movq %rdi, %rcx
        decq %rcx
        jz Lrscrypto_bignum_modinv_zmontend
Lrscrypto_bignum_modinv_zmontloop:
        adcq (%r8,%r9,8), %r11
        sbbq %rbx, %rbx
        movq (%r15,%r9,8), %rax
        mulq %rbp
        subq %rbx, %rdx
        addq %r11, %rax
        movq %rax, -8(%r8,%r9,8)
        movq %rdx, %r11
        incq %r9
        decq %rcx
        jnz Lrscrypto_bignum_modinv_zmontloop
Lrscrypto_bignum_modinv_zmontend:
        adcq %rsi, %r11
        movq %r11, -8(%r8,%rdi,8)
        sbbq %r11, %r11
        negq %r11
        movq %rdi, %rcx
        xorq %r9, %r9
Lrscrypto_bignum_modinv_zcmploop:
        movq (%r8,%r9,8), %rax
        sbbq (%r15,%r9,8), %rax
        incq %r9
        decq %rcx
        jnz Lrscrypto_bignum_modinv_zcmploop
        sbbq $0, %r11
        sbbq %r11, %r11
        notq %r11
        xorq %rcx, %rcx
        xorq %r9, %r9
Lrscrypto_bignum_modinv_zcorrloop:
        movq (%r8,%r9,8), %rax
        movq (%r15,%r9,8), %rbx
        andq %r11, %rbx
        negq %rcx
        sbbq %rbx, %rax
        sbbq %rcx, %rcx
        movq %rax, (%r8,%r9,8)
        incq %r9
        cmpq %rdi, %r9
        jc Lrscrypto_bignum_modinv_zcorrloop
        movq 48(%rsp), %r8
        leaq (%r8,%rdi,8), %r15
        xorq %r9, %r9
        xorq %r12, %r12
        xorq %r14, %r14
        xorq %rbp, %rbp
        xorq %rsi, %rsi
Lrscrypto_bignum_modinv_crossloop:
        movq (%r8,%r9,8), %rcx
        movq (%rsp), %rax
        mulq %rcx
        addq %rax, %r14
        adcq $0, %rdx
        movq %rdx, %r10
        movq 16(%rsp), %rax
        mulq %rcx
        addq %rax, %rsi
        adcq $0, %rdx
        movq %rdx, %r11
        movq (%r15,%r9,8), %rcx
        movq 8(%rsp), %rax
        mulq %rcx
        subq %r12, %rdx
        subq %rax, %r14
        sbbq %rdx, %r10
        sbbq %r12, %r12
        movq %r14, (%r8,%r9,8)
        movq %r10, %r14
        movq 24(%rsp), %rax
        mulq %rcx
        subq %rbp, %rdx
        subq %rax, %rsi
        sbbq %rdx, %r11
        sbbq %rbp, %rbp
        movq %rsi, (%r15,%r9,8)
        movq %r11, %rsi
        incq %r9
        cmpq %r13, %r9
        jc Lrscrypto_bignum_modinv_crossloop
        xorq %r9, %r9
        movq %r12, %r10
        movq %rbp, %r11
        xorq %r12, %r14
        xorq %rbp, %rsi
Lrscrypto_bignum_modinv_optnegloop:
        movq (%r8,%r9,8), %rax
        xorq %r12, %rax
        negq %r10
        adcq $0, %rax
        sbbq %r10, %r10
        movq %rax, (%r8,%r9,8)
        movq (%r15,%r9,8), %rax
        xorq %rbp, %rax
        negq %r11
        adcq $0, %rax
        sbbq %r11, %r11
        movq %rax, (%r15,%r9,8)
        incq %r9
        cmpq %r13, %r9
        jc Lrscrypto_bignum_modinv_optnegloop
        subq %r10, %r14
        subq %r11, %rsi
        movq %r13, %r9
Lrscrypto_bignum_modinv_shiftloop:
        movq -8(%r8,%r9,8), %rax
        movq %rax, %r10
        shrdq $58, %r14, %rax
        movq %rax, -8(%r8,%r9,8)
        movq %r10, %r14
        movq -8(%r15,%r9,8), %rax
        movq %rax, %r11
        shrdq $58, %rsi, %rax
        movq %rax, -8(%r15,%r9,8)
        movq %r11, %rsi
        decq %r9
        jnz Lrscrypto_bignum_modinv_shiftloop
        notq %rbp
        movq 72(%rsp), %rcx
        movq 56(%rsp), %r8
        movq 64(%rsp), %r15
        movq %r12, %r10
        movq %rbp, %r11
        xorq %r9, %r9
Lrscrypto_bignum_modinv_fliploop:
        movq %rbp, %rdx
        movq (%rcx,%r9,8), %rax
        andq %rax, %rdx
        andq %r12, %rax
        movq (%r8,%r9,8), %rbx
        xorq %r12, %rbx
        negq %r10
        adcq %rbx, %rax
        sbbq %r10, %r10
        movq %rax, (%r8,%r9,8)
        movq (%r15,%r9,8), %rbx
        xorq %rbp, %rbx
        negq %r11
        adcq %rbx, %rdx
        sbbq %r11, %r11
        movq %rdx, (%r15,%r9,8)
        incq %r9
        cmpq %rdi, %r9
        jc Lrscrypto_bignum_modinv_fliploop
        subq $58, 32(%rsp)
        jnbe Lrscrypto_bignum_modinv_outerloop
Lrscrypto_bignum_modinv_end:
        addq $80, %rsp ; .cfi_adjust_cfa_offset -80
        popq %r15 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r15
        popq %r14 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r14
        popq %r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r13
        popq %r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %r12
        popq %rbx ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbx
        popq %rbp ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %rbp
        retq ; .cfi_endproc
.size rscrypto_bignum_modinv, .-rscrypto_bignum_modinv
