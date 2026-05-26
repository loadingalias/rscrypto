// Copyright 2005-2016 The OpenSSL Project Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted for rscrypto from AWS-LC/BoringSSL generated x86_64-mont.S and x86_64-mont5.S.
// Public symbols are renamed into the rscrypto namespace and embedded with Rust global_asm!.

// This file is generated from a similarly-named Perl script in the BoringSSL
// source tree. Do not edit by hand.


.text	

.globl	rscrypto_rsa_bn_mul_mont_nohw_x86_64_elf
.hidden rscrypto_rsa_bn_mul_mont_nohw_x86_64_elf
.type	rscrypto_rsa_bn_mul_mont_nohw_x86_64_elf,@function
.align	16
rscrypto_rsa_bn_mul_mont_nohw_x86_64_elf:
.cfi_startproc	
	movl	%r9d,%r9d
	movq	%rsp,%rax
.cfi_def_cfa_register	%rax
	pushq	%rbx
.cfi_offset	%rbx,-16
	pushq	%rbp
.cfi_offset	%rbp,-24
	pushq	%r12
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_offset	%r15,-56

	negq	%r9
	movq	%rsp,%r11
	leaq	-16(%rsp,%r9,8),%r10
	negq	%r9
	andq	$-1024,%r10









	subq	%r10,%r11
	andq	$-4096,%r11
	leaq	(%r10,%r11,1),%rsp
	movq	(%rsp),%r11
	cmpq	%r10,%rsp
	ja	.Lmul_page_walk
	jmp	.Lmul_page_walk_done

.align	16
.Lmul_page_walk:
	leaq	-4096(%rsp),%rsp
	movq	(%rsp),%r11
	cmpq	%r10,%rsp
	ja	.Lmul_page_walk
.Lmul_page_walk_done:

	movq	%rax,8(%rsp,%r9,8)
.cfi_escape	0x0f,0x0a,0x77,0x08,0x79,0x00,0x38,0x1e,0x22,0x06,0x23,0x08
.Lmul_body:
	movq	%rdx,%r12
	movq	(%r8),%r8
	movq	(%r12),%rbx
	movq	(%rsi),%rax

	xorq	%r14,%r14
	xorq	%r15,%r15

	movq	%r8,%rbp
	mulq	%rbx
	movq	%rax,%r10
	movq	(%rcx),%rax

	imulq	%r10,%rbp
	movq	%rdx,%r11

	mulq	%rbp
	addq	%rax,%r10
	movq	8(%rsi),%rax
	adcq	$0,%rdx
	movq	%rdx,%r13

	leaq	1(%r15),%r15
	jmp	.L1st_enter

.align	16
.L1st:
	addq	%rax,%r13
	movq	(%rsi,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r11,%r13
	movq	%r10,%r11
	adcq	$0,%rdx
	movq	%r13,-16(%rsp,%r15,8)
	movq	%rdx,%r13

.L1st_enter:
	mulq	%rbx
	addq	%rax,%r11
	movq	(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	leaq	1(%r15),%r15
	movq	%rdx,%r10

	mulq	%rbp
	cmpq	%r9,%r15
	jne	.L1st

	addq	%rax,%r13
	movq	(%rsi),%rax
	adcq	$0,%rdx
	addq	%r11,%r13
	adcq	$0,%rdx
	movq	%r13,-16(%rsp,%r15,8)
	movq	%rdx,%r13
	movq	%r10,%r11

	xorq	%rdx,%rdx
	addq	%r11,%r13
	adcq	$0,%rdx
	movq	%r13,-8(%rsp,%r9,8)
	movq	%rdx,(%rsp,%r9,8)

	leaq	1(%r14),%r14
	jmp	.Louter
.align	16
.Louter:
	movq	(%r12,%r14,8),%rbx
	xorq	%r15,%r15
	movq	%r8,%rbp
	movq	(%rsp),%r10
	mulq	%rbx
	addq	%rax,%r10
	movq	(%rcx),%rax
	adcq	$0,%rdx

	imulq	%r10,%rbp
	movq	%rdx,%r11

	mulq	%rbp
	addq	%rax,%r10
	movq	8(%rsi),%rax
	adcq	$0,%rdx
	movq	8(%rsp),%r10
	movq	%rdx,%r13

	leaq	1(%r15),%r15
	jmp	.Linner_enter

.align	16
.Linner:
	addq	%rax,%r13
	movq	(%rsi,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r10,%r13
	movq	(%rsp,%r15,8),%r10
	adcq	$0,%rdx
	movq	%r13,-16(%rsp,%r15,8)
	movq	%rdx,%r13

.Linner_enter:
	mulq	%rbx
	addq	%rax,%r11
	movq	(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r11,%r10
	movq	%rdx,%r11
	adcq	$0,%r11
	leaq	1(%r15),%r15

	mulq	%rbp
	cmpq	%r9,%r15
	jne	.Linner

	addq	%rax,%r13
	movq	(%rsi),%rax
	adcq	$0,%rdx
	addq	%r10,%r13
	movq	(%rsp,%r15,8),%r10
	adcq	$0,%rdx
	movq	%r13,-16(%rsp,%r15,8)
	movq	%rdx,%r13

	xorq	%rdx,%rdx
	addq	%r11,%r13
	adcq	$0,%rdx
	addq	%r10,%r13
	adcq	$0,%rdx
	movq	%r13,-8(%rsp,%r9,8)
	movq	%rdx,(%rsp,%r9,8)

	leaq	1(%r14),%r14
	cmpq	%r9,%r14
	jb	.Louter

	xorq	%r14,%r14
	movq	(%rsp),%rax
	movq	%r9,%r15

.align	16
.Lsub:	sbbq	(%rcx,%r14,8),%rax
	movq	%rax,(%rdi,%r14,8)
	movq	8(%rsp,%r14,8),%rax
	leaq	1(%r14),%r14
	decq	%r15
	jnz	.Lsub

	sbbq	$0,%rax
	movq	$-1,%rbx
	xorq	%rax,%rbx
	xorq	%r14,%r14
	movq	%r9,%r15

.Lcopy:
	movq	(%rdi,%r14,8),%rcx
	movq	(%rsp,%r14,8),%rdx
	andq	%rbx,%rcx
	andq	%rax,%rdx
	movq	%r9,(%rsp,%r14,8)
	orq	%rcx,%rdx
	movq	%rdx,(%rdi,%r14,8)
	leaq	1(%r14),%r14
	subq	$1,%r15
	jnz	.Lcopy

	movq	8(%rsp,%r9,8),%rsi
.cfi_def_cfa	%rsi,8
	movq	$1,%rax
	movq	-48(%rsi),%r15
.cfi_restore	%r15
	movq	-40(%rsi),%r14
.cfi_restore	%r14
	movq	-32(%rsi),%r13
.cfi_restore	%r13
	movq	-24(%rsi),%r12
.cfi_restore	%r12
	movq	-16(%rsi),%rbp
.cfi_restore	%rbp
	movq	-8(%rsi),%rbx
.cfi_restore	%rbx
	leaq	(%rsi),%rsp
.cfi_def_cfa_register	%rsp
.Lmul_epilogue:
	.byte	0xf3,0xc3
.cfi_endproc	
.size	rscrypto_rsa_bn_mul_mont_nohw_x86_64_elf,.-rscrypto_rsa_bn_mul_mont_nohw_x86_64_elf
.globl	rscrypto_rsa_bn_mul4x_mont_x86_64_elf
.hidden rscrypto_rsa_bn_mul4x_mont_x86_64_elf
.type	rscrypto_rsa_bn_mul4x_mont_x86_64_elf,@function
.align	16
rscrypto_rsa_bn_mul4x_mont_x86_64_elf:
.cfi_startproc	
	movl	%r9d,%r9d
	movq	%rsp,%rax
.cfi_def_cfa_register	%rax
	pushq	%rbx
.cfi_offset	%rbx,-16
	pushq	%rbp
.cfi_offset	%rbp,-24
	pushq	%r12
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_offset	%r15,-56

	negq	%r9
	movq	%rsp,%r11
	leaq	-32(%rsp,%r9,8),%r10
	negq	%r9
	andq	$-1024,%r10

	subq	%r10,%r11
	andq	$-4096,%r11
	leaq	(%r10,%r11,1),%rsp
	movq	(%rsp),%r11
	cmpq	%r10,%rsp
	ja	.Lmul4x_page_walk
	jmp	.Lmul4x_page_walk_done

.Lmul4x_page_walk:
	leaq	-4096(%rsp),%rsp
	movq	(%rsp),%r11
	cmpq	%r10,%rsp
	ja	.Lmul4x_page_walk
.Lmul4x_page_walk_done:

	movq	%rax,8(%rsp,%r9,8)
.cfi_escape	0x0f,0x0a,0x77,0x08,0x79,0x00,0x38,0x1e,0x22,0x06,0x23,0x08
.Lmul4x_body:
	movq	%rdi,16(%rsp,%r9,8)
	movq	%rdx,%r12
	movq	(%r8),%r8
	movq	(%r12),%rbx
	movq	(%rsi),%rax

	xorq	%r14,%r14
	xorq	%r15,%r15

	movq	%r8,%rbp
	mulq	%rbx
	movq	%rax,%r10
	movq	(%rcx),%rax

	imulq	%r10,%rbp
	movq	%rdx,%r11

	mulq	%rbp
	addq	%rax,%r10
	movq	8(%rsi),%rax
	adcq	$0,%rdx
	movq	%rdx,%rdi

	mulq	%rbx
	addq	%rax,%r11
	movq	8(%rcx),%rax
	adcq	$0,%rdx
	movq	%rdx,%r10

	mulq	%rbp
	addq	%rax,%rdi
	movq	16(%rsi),%rax
	adcq	$0,%rdx
	addq	%r11,%rdi
	leaq	4(%r15),%r15
	adcq	$0,%rdx
	movq	%rdi,(%rsp)
	movq	%rdx,%r13
	jmp	.L1st4x
.align	16
.L1st4x:
	mulq	%rbx
	addq	%rax,%r10
	movq	-16(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	movq	%rdx,%r11

	mulq	%rbp
	addq	%rax,%r13
	movq	-8(%rsi,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r10,%r13
	adcq	$0,%rdx
	movq	%r13,-24(%rsp,%r15,8)
	movq	%rdx,%rdi

	mulq	%rbx
	addq	%rax,%r11
	movq	-8(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	movq	%rdx,%r10

	mulq	%rbp
	addq	%rax,%rdi
	movq	(%rsi,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r11,%rdi
	adcq	$0,%rdx
	movq	%rdi,-16(%rsp,%r15,8)
	movq	%rdx,%r13

	mulq	%rbx
	addq	%rax,%r10
	movq	(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	movq	%rdx,%r11

	mulq	%rbp
	addq	%rax,%r13
	movq	8(%rsi,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r10,%r13
	adcq	$0,%rdx
	movq	%r13,-8(%rsp,%r15,8)
	movq	%rdx,%rdi

	mulq	%rbx
	addq	%rax,%r11
	movq	8(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	leaq	4(%r15),%r15
	movq	%rdx,%r10

	mulq	%rbp
	addq	%rax,%rdi
	movq	-16(%rsi,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r11,%rdi
	adcq	$0,%rdx
	movq	%rdi,-32(%rsp,%r15,8)
	movq	%rdx,%r13
	cmpq	%r9,%r15
	jb	.L1st4x

	mulq	%rbx
	addq	%rax,%r10
	movq	-16(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	movq	%rdx,%r11

	mulq	%rbp
	addq	%rax,%r13
	movq	-8(%rsi,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r10,%r13
	adcq	$0,%rdx
	movq	%r13,-24(%rsp,%r15,8)
	movq	%rdx,%rdi

	mulq	%rbx
	addq	%rax,%r11
	movq	-8(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	movq	%rdx,%r10

	mulq	%rbp
	addq	%rax,%rdi
	movq	(%rsi),%rax
	adcq	$0,%rdx
	addq	%r11,%rdi
	adcq	$0,%rdx
	movq	%rdi,-16(%rsp,%r15,8)
	movq	%rdx,%r13

	xorq	%rdi,%rdi
	addq	%r10,%r13
	adcq	$0,%rdi
	movq	%r13,-8(%rsp,%r15,8)
	movq	%rdi,(%rsp,%r15,8)

	leaq	1(%r14),%r14
.align	4
.Louter4x:
	movq	(%r12,%r14,8),%rbx
	xorq	%r15,%r15
	movq	(%rsp),%r10
	movq	%r8,%rbp
	mulq	%rbx
	addq	%rax,%r10
	movq	(%rcx),%rax
	adcq	$0,%rdx

	imulq	%r10,%rbp
	movq	%rdx,%r11

	mulq	%rbp
	addq	%rax,%r10
	movq	8(%rsi),%rax
	adcq	$0,%rdx
	movq	%rdx,%rdi

	mulq	%rbx
	addq	%rax,%r11
	movq	8(%rcx),%rax
	adcq	$0,%rdx
	addq	8(%rsp),%r11
	adcq	$0,%rdx
	movq	%rdx,%r10

	mulq	%rbp
	addq	%rax,%rdi
	movq	16(%rsi),%rax
	adcq	$0,%rdx
	addq	%r11,%rdi
	leaq	4(%r15),%r15
	adcq	$0,%rdx
	movq	%rdi,(%rsp)
	movq	%rdx,%r13
	jmp	.Linner4x
.align	16
.Linner4x:
	mulq	%rbx
	addq	%rax,%r10
	movq	-16(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	addq	-16(%rsp,%r15,8),%r10
	adcq	$0,%rdx
	movq	%rdx,%r11

	mulq	%rbp
	addq	%rax,%r13
	movq	-8(%rsi,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r10,%r13
	adcq	$0,%rdx
	movq	%r13,-24(%rsp,%r15,8)
	movq	%rdx,%rdi

	mulq	%rbx
	addq	%rax,%r11
	movq	-8(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	addq	-8(%rsp,%r15,8),%r11
	adcq	$0,%rdx
	movq	%rdx,%r10

	mulq	%rbp
	addq	%rax,%rdi
	movq	(%rsi,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r11,%rdi
	adcq	$0,%rdx
	movq	%rdi,-16(%rsp,%r15,8)
	movq	%rdx,%r13

	mulq	%rbx
	addq	%rax,%r10
	movq	(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	addq	(%rsp,%r15,8),%r10
	adcq	$0,%rdx
	movq	%rdx,%r11

	mulq	%rbp
	addq	%rax,%r13
	movq	8(%rsi,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r10,%r13
	adcq	$0,%rdx
	movq	%r13,-8(%rsp,%r15,8)
	movq	%rdx,%rdi

	mulq	%rbx
	addq	%rax,%r11
	movq	8(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	addq	8(%rsp,%r15,8),%r11
	adcq	$0,%rdx
	leaq	4(%r15),%r15
	movq	%rdx,%r10

	mulq	%rbp
	addq	%rax,%rdi
	movq	-16(%rsi,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r11,%rdi
	adcq	$0,%rdx
	movq	%rdi,-32(%rsp,%r15,8)
	movq	%rdx,%r13
	cmpq	%r9,%r15
	jb	.Linner4x

	mulq	%rbx
	addq	%rax,%r10
	movq	-16(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	addq	-16(%rsp,%r15,8),%r10
	adcq	$0,%rdx
	movq	%rdx,%r11

	mulq	%rbp
	addq	%rax,%r13
	movq	-8(%rsi,%r15,8),%rax
	adcq	$0,%rdx
	addq	%r10,%r13
	adcq	$0,%rdx
	movq	%r13,-24(%rsp,%r15,8)
	movq	%rdx,%rdi

	mulq	%rbx
	addq	%rax,%r11
	movq	-8(%rcx,%r15,8),%rax
	adcq	$0,%rdx
	addq	-8(%rsp,%r15,8),%r11
	adcq	$0,%rdx
	leaq	1(%r14),%r14
	movq	%rdx,%r10

	mulq	%rbp
	addq	%rax,%rdi
	movq	(%rsi),%rax
	adcq	$0,%rdx
	addq	%r11,%rdi
	adcq	$0,%rdx
	movq	%rdi,-16(%rsp,%r15,8)
	movq	%rdx,%r13

	xorq	%rdi,%rdi
	addq	%r10,%r13
	adcq	$0,%rdi
	addq	(%rsp,%r9,8),%r13
	adcq	$0,%rdi
	movq	%r13,-8(%rsp,%r15,8)
	movq	%rdi,(%rsp,%r15,8)

	cmpq	%r9,%r14
	jb	.Louter4x
	movq	16(%rsp,%r9,8),%rdi
	leaq	-4(%r9),%r15
	movq	0(%rsp),%rax
	movq	8(%rsp),%rdx
	shrq	$2,%r15
	leaq	(%rsp),%rsi
	xorq	%r14,%r14

	subq	0(%rcx),%rax
	movq	16(%rsi),%rbx
	movq	24(%rsi),%rbp
	sbbq	8(%rcx),%rdx

.Lsub4x:
	movq	%rax,0(%rdi,%r14,8)
	movq	%rdx,8(%rdi,%r14,8)
	sbbq	16(%rcx,%r14,8),%rbx
	movq	32(%rsi,%r14,8),%rax
	movq	40(%rsi,%r14,8),%rdx
	sbbq	24(%rcx,%r14,8),%rbp
	movq	%rbx,16(%rdi,%r14,8)
	movq	%rbp,24(%rdi,%r14,8)
	sbbq	32(%rcx,%r14,8),%rax
	movq	48(%rsi,%r14,8),%rbx
	movq	56(%rsi,%r14,8),%rbp
	sbbq	40(%rcx,%r14,8),%rdx
	leaq	4(%r14),%r14
	decq	%r15
	jnz	.Lsub4x

	movq	%rax,0(%rdi,%r14,8)
	movq	32(%rsi,%r14,8),%rax
	sbbq	16(%rcx,%r14,8),%rbx
	movq	%rdx,8(%rdi,%r14,8)
	sbbq	24(%rcx,%r14,8),%rbp
	movq	%rbx,16(%rdi,%r14,8)

	sbbq	$0,%rax
	movq	%rbp,24(%rdi,%r14,8)
	pxor	%xmm0,%xmm0
.byte	102,72,15,110,224
	pcmpeqd	%xmm5,%xmm5
	pshufd	$0,%xmm4,%xmm4
	movq	%r9,%r15
	pxor	%xmm4,%xmm5
	shrq	$2,%r15
	xorl	%eax,%eax

	jmp	.Lcopy4x
.align	16
.Lcopy4x:
	movdqa	(%rsp,%rax,1),%xmm1
	movdqu	(%rdi,%rax,1),%xmm2
	pand	%xmm4,%xmm1
	pand	%xmm5,%xmm2
	movdqa	16(%rsp,%rax,1),%xmm3
	movdqa	%xmm0,(%rsp,%rax,1)
	por	%xmm2,%xmm1
	movdqu	16(%rdi,%rax,1),%xmm2
	movdqu	%xmm1,(%rdi,%rax,1)
	pand	%xmm4,%xmm3
	pand	%xmm5,%xmm2
	movdqa	%xmm0,16(%rsp,%rax,1)
	por	%xmm2,%xmm3
	movdqu	%xmm3,16(%rdi,%rax,1)
	leaq	32(%rax),%rax
	decq	%r15
	jnz	.Lcopy4x
	movq	8(%rsp,%r9,8),%rsi
.cfi_def_cfa	%rsi, 8
	movq	$1,%rax
	movq	-48(%rsi),%r15
.cfi_restore	%r15
	movq	-40(%rsi),%r14
.cfi_restore	%r14
	movq	-32(%rsi),%r13
.cfi_restore	%r13
	movq	-24(%rsi),%r12
.cfi_restore	%r12
	movq	-16(%rsi),%rbp
.cfi_restore	%rbp
	movq	-8(%rsi),%rbx
.cfi_restore	%rbx
	leaq	(%rsi),%rsp
.cfi_def_cfa_register	%rsp
.Lmul4x_epilogue:
	.byte	0xf3,0xc3
.cfi_endproc	
.size	rscrypto_rsa_bn_mul4x_mont_x86_64_elf,.-rscrypto_rsa_bn_mul4x_mont_x86_64_elf
.extern	rscrypto_rsa_bn_sqrx8x_internal_x86_64_elf
.hidden rscrypto_rsa_bn_sqrx8x_internal_x86_64_elf
.extern	rscrypto_rsa_bn_sqr8x_internal_x86_64_elf
.hidden rscrypto_rsa_bn_sqr8x_internal_x86_64_elf

.globl	rscrypto_rsa_bn_sqr8x_mont_x86_64_elf
.hidden rscrypto_rsa_bn_sqr8x_mont_x86_64_elf
.type	rscrypto_rsa_bn_sqr8x_mont_x86_64_elf,@function
.align	32
rscrypto_rsa_bn_sqr8x_mont_x86_64_elf:
.cfi_startproc	
	movl	%r9d,%r9d
	movq	%rsp,%rax
.cfi_def_cfa_register	%rax
	pushq	%rbx
.cfi_offset	%rbx,-16
	pushq	%rbp
.cfi_offset	%rbp,-24
	pushq	%r12
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_offset	%r15,-56
.Lsqr8x_prologue:

	movl	%r9d,%r10d
	shll	$3,%r9d
	shlq	$3+2,%r10
	negq	%r9






	leaq	-64(%rsp,%r9,2),%r11
	movq	%rsp,%rbp
	movq	(%r8),%r8
	subq	%rsi,%r11
	andq	$4095,%r11
	cmpq	%r11,%r10
	jb	.Lsqr8x_sp_alt
	subq	%r11,%rbp
	leaq	-64(%rbp,%r9,2),%rbp
	jmp	.Lsqr8x_sp_done

.align	32
.Lsqr8x_sp_alt:
	leaq	4096-64(,%r9,2),%r10
	leaq	-64(%rbp,%r9,2),%rbp
	subq	%r10,%r11
	movq	$0,%r10
	cmovcq	%r10,%r11
	subq	%r11,%rbp
.Lsqr8x_sp_done:
	andq	$-64,%rbp
	movq	%rsp,%r11
	subq	%rbp,%r11
	andq	$-4096,%r11
	leaq	(%r11,%rbp,1),%rsp
	movq	(%rsp),%r10
	cmpq	%rbp,%rsp
	ja	.Lsqr8x_page_walk
	jmp	.Lsqr8x_page_walk_done

.align	16
.Lsqr8x_page_walk:
	leaq	-4096(%rsp),%rsp
	movq	(%rsp),%r10
	cmpq	%rbp,%rsp
	ja	.Lsqr8x_page_walk
.Lsqr8x_page_walk_done:

	movq	%r9,%r10
	negq	%r9

	movq	%r8,32(%rsp)
	movq	%rax,40(%rsp)
.cfi_escape	0x0f,0x05,0x77,0x28,0x06,0x23,0x08
.Lsqr8x_body:

.byte	102,72,15,110,209
	pxor	%xmm0,%xmm0
.byte	102,72,15,110,207
.byte	102,73,15,110,218
	testq	%rdx,%rdx
	jz	.Lsqr8x_nox

	call	rscrypto_rsa_bn_sqrx8x_internal_x86_64_elf




	leaq	(%r8,%rcx,1),%rbx
	movq	%rcx,%r9
	movq	%rcx,%rdx
.byte	102,72,15,126,207
	sarq	$3+2,%rcx
	jmp	.Lsqr8x_sub

.align	32
.Lsqr8x_nox:
	call	rscrypto_rsa_bn_sqr8x_internal_x86_64_elf




	leaq	(%rdi,%r9,1),%rbx
	movq	%r9,%rcx
	movq	%r9,%rdx
.byte	102,72,15,126,207
	sarq	$3+2,%rcx
	jmp	.Lsqr8x_sub

.align	32
.Lsqr8x_sub:
	movq	0(%rbx),%r12
	movq	8(%rbx),%r13
	movq	16(%rbx),%r14
	movq	24(%rbx),%r15
	leaq	32(%rbx),%rbx
	sbbq	0(%rbp),%r12
	sbbq	8(%rbp),%r13
	sbbq	16(%rbp),%r14
	sbbq	24(%rbp),%r15
	leaq	32(%rbp),%rbp
	movq	%r12,0(%rdi)
	movq	%r13,8(%rdi)
	movq	%r14,16(%rdi)
	movq	%r15,24(%rdi)
	leaq	32(%rdi),%rdi
	incq	%rcx
	jnz	.Lsqr8x_sub

	sbbq	$0,%rax
	leaq	(%rbx,%r9,1),%rbx
	leaq	(%rdi,%r9,1),%rdi

.byte	102,72,15,110,200
	pxor	%xmm0,%xmm0
	pshufd	$0,%xmm1,%xmm1
	movq	40(%rsp),%rsi
.cfi_def_cfa	%rsi,8
	jmp	.Lsqr8x_cond_copy

.align	32
.Lsqr8x_cond_copy:
	movdqa	0(%rbx),%xmm2
	movdqa	16(%rbx),%xmm3
	leaq	32(%rbx),%rbx
	movdqu	0(%rdi),%xmm4
	movdqu	16(%rdi),%xmm5
	leaq	32(%rdi),%rdi
	movdqa	%xmm0,-32(%rbx)
	movdqa	%xmm0,-16(%rbx)
	movdqa	%xmm0,-32(%rbx,%rdx,1)
	movdqa	%xmm0,-16(%rbx,%rdx,1)
	pcmpeqd	%xmm1,%xmm0
	pand	%xmm1,%xmm2
	pand	%xmm1,%xmm3
	pand	%xmm0,%xmm4
	pand	%xmm0,%xmm5
	pxor	%xmm0,%xmm0
	por	%xmm2,%xmm4
	por	%xmm3,%xmm5
	movdqu	%xmm4,-32(%rdi)
	movdqu	%xmm5,-16(%rdi)
	addq	$32,%r9
	jnz	.Lsqr8x_cond_copy

	movq	$1,%rax
	movq	-48(%rsi),%r15
.cfi_restore	%r15
	movq	-40(%rsi),%r14
.cfi_restore	%r14
	movq	-32(%rsi),%r13
.cfi_restore	%r13
	movq	-24(%rsi),%r12
.cfi_restore	%r12
	movq	-16(%rsi),%rbp
.cfi_restore	%rbp
	movq	-8(%rsi),%rbx
.cfi_restore	%rbx
	leaq	(%rsi),%rsp
.cfi_def_cfa_register	%rsp
.Lsqr8x_epilogue:
	.byte	0xf3,0xc3
.cfi_endproc	
.size	rscrypto_rsa_bn_sqr8x_mont_x86_64_elf,.-rscrypto_rsa_bn_sqr8x_mont_x86_64_elf
.globl	rscrypto_rsa_bn_mulx4x_mont_x86_64_elf
.hidden rscrypto_rsa_bn_mulx4x_mont_x86_64_elf
.type	rscrypto_rsa_bn_mulx4x_mont_x86_64_elf,@function
.align	32
rscrypto_rsa_bn_mulx4x_mont_x86_64_elf:
.cfi_startproc	
	movq	%rsp,%rax
.cfi_def_cfa_register	%rax
	pushq	%rbx
.cfi_offset	%rbx,-16
	pushq	%rbp
.cfi_offset	%rbp,-24
	pushq	%r12
.cfi_offset	%r12,-32
	pushq	%r13
.cfi_offset	%r13,-40
	pushq	%r14
.cfi_offset	%r14,-48
	pushq	%r15
.cfi_offset	%r15,-56
.Lmulx4x_prologue:

	shll	$3,%r9d
	xorq	%r10,%r10
	subq	%r9,%r10
	movq	(%r8),%r8
	leaq	-72(%rsp,%r10,1),%rbp
	andq	$-128,%rbp
	movq	%rsp,%r11
	subq	%rbp,%r11
	andq	$-4096,%r11
	leaq	(%r11,%rbp,1),%rsp
	movq	(%rsp),%r10
	cmpq	%rbp,%rsp
	ja	.Lmulx4x_page_walk
	jmp	.Lmulx4x_page_walk_done

.align	16
.Lmulx4x_page_walk:
	leaq	-4096(%rsp),%rsp
	movq	(%rsp),%r10
	cmpq	%rbp,%rsp
	ja	.Lmulx4x_page_walk
.Lmulx4x_page_walk_done:

	leaq	(%rdx,%r9,1),%r10












	movq	%r9,0(%rsp)
	shrq	$5,%r9
	movq	%r10,16(%rsp)
	subq	$1,%r9
	movq	%r8,24(%rsp)
	movq	%rdi,32(%rsp)
	movq	%rax,40(%rsp)
.cfi_escape	0x0f,0x05,0x77,0x28,0x06,0x23,0x08
	movq	%r9,48(%rsp)
	jmp	.Lmulx4x_body

.align	32
.Lmulx4x_body:
	leaq	8(%rdx),%rdi
	movq	(%rdx),%rdx
	leaq	64+32(%rsp),%rbx
	movq	%rdx,%r9

	mulxq	0(%rsi),%r8,%rax
	mulxq	8(%rsi),%r11,%r14
	addq	%rax,%r11
	movq	%rdi,8(%rsp)
	mulxq	16(%rsi),%r12,%r13
	adcq	%r14,%r12
	adcq	$0,%r13

	movq	%r8,%rdi
	imulq	24(%rsp),%r8
	xorq	%rbp,%rbp

	mulxq	24(%rsi),%rax,%r14
	movq	%r8,%rdx
	leaq	32(%rsi),%rsi
	adcxq	%rax,%r13
	adcxq	%rbp,%r14

	mulxq	0(%rcx),%rax,%r10
	adcxq	%rax,%rdi
	adoxq	%r11,%r10
	mulxq	8(%rcx),%rax,%r11
	adcxq	%rax,%r10
	adoxq	%r12,%r11
.byte	0xc4,0x62,0xfb,0xf6,0xa1,0x10,0x00,0x00,0x00
	movq	48(%rsp),%rdi
	movq	%r10,-32(%rbx)
	adcxq	%rax,%r11
	adoxq	%r13,%r12
	mulxq	24(%rcx),%rax,%r15
	movq	%r9,%rdx
	movq	%r11,-24(%rbx)
	adcxq	%rax,%r12
	adoxq	%rbp,%r15
	leaq	32(%rcx),%rcx
	movq	%r12,-16(%rbx)

	jmp	.Lmulx4x_1st

.align	32
.Lmulx4x_1st:
	adcxq	%rbp,%r15
	mulxq	0(%rsi),%r10,%rax
	adcxq	%r14,%r10
	mulxq	8(%rsi),%r11,%r14
	adcxq	%rax,%r11
	mulxq	16(%rsi),%r12,%rax
	adcxq	%r14,%r12
	mulxq	24(%rsi),%r13,%r14
.byte	0x67,0x67
	movq	%r8,%rdx
	adcxq	%rax,%r13
	adcxq	%rbp,%r14
	leaq	32(%rsi),%rsi
	leaq	32(%rbx),%rbx

	adoxq	%r15,%r10
	mulxq	0(%rcx),%rax,%r15
	adcxq	%rax,%r10
	adoxq	%r15,%r11
	mulxq	8(%rcx),%rax,%r15
	adcxq	%rax,%r11
	adoxq	%r15,%r12
	mulxq	16(%rcx),%rax,%r15
	movq	%r10,-40(%rbx)
	adcxq	%rax,%r12
	movq	%r11,-32(%rbx)
	adoxq	%r15,%r13
	mulxq	24(%rcx),%rax,%r15
	movq	%r9,%rdx
	movq	%r12,-24(%rbx)
	adcxq	%rax,%r13
	adoxq	%rbp,%r15
	leaq	32(%rcx),%rcx
	movq	%r13,-16(%rbx)

	decq	%rdi
	jnz	.Lmulx4x_1st

	movq	0(%rsp),%rax
	movq	8(%rsp),%rdi
	adcq	%rbp,%r15
	addq	%r15,%r14
	sbbq	%r15,%r15
	movq	%r14,-8(%rbx)
	jmp	.Lmulx4x_outer

.align	32
.Lmulx4x_outer:
	movq	(%rdi),%rdx
	leaq	8(%rdi),%rdi
	subq	%rax,%rsi
	movq	%r15,(%rbx)
	leaq	64+32(%rsp),%rbx
	subq	%rax,%rcx

	mulxq	0(%rsi),%r8,%r11
	xorl	%ebp,%ebp
	movq	%rdx,%r9
	mulxq	8(%rsi),%r14,%r12
	adoxq	-32(%rbx),%r8
	adcxq	%r14,%r11
	mulxq	16(%rsi),%r15,%r13
	adoxq	-24(%rbx),%r11
	adcxq	%r15,%r12
	adoxq	-16(%rbx),%r12
	adcxq	%rbp,%r13
	adoxq	%rbp,%r13

	movq	%rdi,8(%rsp)
	movq	%r8,%r15
	imulq	24(%rsp),%r8
	xorl	%ebp,%ebp

	mulxq	24(%rsi),%rax,%r14
	movq	%r8,%rdx
	adcxq	%rax,%r13
	adoxq	-8(%rbx),%r13
	adcxq	%rbp,%r14
	leaq	32(%rsi),%rsi
	adoxq	%rbp,%r14

	mulxq	0(%rcx),%rax,%r10
	adcxq	%rax,%r15
	adoxq	%r11,%r10
	mulxq	8(%rcx),%rax,%r11
	adcxq	%rax,%r10
	adoxq	%r12,%r11
	mulxq	16(%rcx),%rax,%r12
	movq	%r10,-32(%rbx)
	adcxq	%rax,%r11
	adoxq	%r13,%r12
	mulxq	24(%rcx),%rax,%r15
	movq	%r9,%rdx
	movq	%r11,-24(%rbx)
	leaq	32(%rcx),%rcx
	adcxq	%rax,%r12
	adoxq	%rbp,%r15
	movq	48(%rsp),%rdi
	movq	%r12,-16(%rbx)

	jmp	.Lmulx4x_inner

.align	32
.Lmulx4x_inner:
	mulxq	0(%rsi),%r10,%rax
	adcxq	%rbp,%r15
	adoxq	%r14,%r10
	mulxq	8(%rsi),%r11,%r14
	adcxq	0(%rbx),%r10
	adoxq	%rax,%r11
	mulxq	16(%rsi),%r12,%rax
	adcxq	8(%rbx),%r11
	adoxq	%r14,%r12
	mulxq	24(%rsi),%r13,%r14
	movq	%r8,%rdx
	adcxq	16(%rbx),%r12
	adoxq	%rax,%r13
	adcxq	24(%rbx),%r13
	adoxq	%rbp,%r14
	leaq	32(%rsi),%rsi
	leaq	32(%rbx),%rbx
	adcxq	%rbp,%r14

	adoxq	%r15,%r10
	mulxq	0(%rcx),%rax,%r15
	adcxq	%rax,%r10
	adoxq	%r15,%r11
	mulxq	8(%rcx),%rax,%r15
	adcxq	%rax,%r11
	adoxq	%r15,%r12
	mulxq	16(%rcx),%rax,%r15
	movq	%r10,-40(%rbx)
	adcxq	%rax,%r12
	adoxq	%r15,%r13
	mulxq	24(%rcx),%rax,%r15
	movq	%r9,%rdx
	movq	%r11,-32(%rbx)
	movq	%r12,-24(%rbx)
	adcxq	%rax,%r13
	adoxq	%rbp,%r15
	leaq	32(%rcx),%rcx
	movq	%r13,-16(%rbx)

	decq	%rdi
	jnz	.Lmulx4x_inner

	movq	0(%rsp),%rax
	movq	8(%rsp),%rdi
	adcq	%rbp,%r15
	subq	0(%rbx),%rbp
	adcq	%r15,%r14
	sbbq	%r15,%r15
	movq	%r14,-8(%rbx)

	cmpq	16(%rsp),%rdi
	jne	.Lmulx4x_outer

	leaq	64(%rsp),%rbx
	subq	%rax,%rcx
	negq	%r15
	movq	%rax,%rdx
	shrq	$3+2,%rax
	movq	32(%rsp),%rdi
	jmp	.Lmulx4x_sub

.align	32
.Lmulx4x_sub:
	movq	0(%rbx),%r11
	movq	8(%rbx),%r12
	movq	16(%rbx),%r13
	movq	24(%rbx),%r14
	leaq	32(%rbx),%rbx
	sbbq	0(%rcx),%r11
	sbbq	8(%rcx),%r12
	sbbq	16(%rcx),%r13
	sbbq	24(%rcx),%r14
	leaq	32(%rcx),%rcx
	movq	%r11,0(%rdi)
	movq	%r12,8(%rdi)
	movq	%r13,16(%rdi)
	movq	%r14,24(%rdi)
	leaq	32(%rdi),%rdi
	decq	%rax
	jnz	.Lmulx4x_sub

	sbbq	$0,%r15
	leaq	64(%rsp),%rbx
	subq	%rdx,%rdi

.byte	102,73,15,110,207
	pxor	%xmm0,%xmm0
	pshufd	$0,%xmm1,%xmm1
	movq	40(%rsp),%rsi
.cfi_def_cfa	%rsi,8
	jmp	.Lmulx4x_cond_copy

.align	32
.Lmulx4x_cond_copy:
	movdqa	0(%rbx),%xmm2
	movdqa	16(%rbx),%xmm3
	leaq	32(%rbx),%rbx
	movdqu	0(%rdi),%xmm4
	movdqu	16(%rdi),%xmm5
	leaq	32(%rdi),%rdi
	movdqa	%xmm0,-32(%rbx)
	movdqa	%xmm0,-16(%rbx)
	pcmpeqd	%xmm1,%xmm0
	pand	%xmm1,%xmm2
	pand	%xmm1,%xmm3
	pand	%xmm0,%xmm4
	pand	%xmm0,%xmm5
	pxor	%xmm0,%xmm0
	por	%xmm2,%xmm4
	por	%xmm3,%xmm5
	movdqu	%xmm4,-32(%rdi)
	movdqu	%xmm5,-16(%rdi)
	subq	$32,%rdx
	jnz	.Lmulx4x_cond_copy

	movq	%rdx,(%rbx)

	movq	$1,%rax
	movq	-48(%rsi),%r15
.cfi_restore	%r15
	movq	-40(%rsi),%r14
.cfi_restore	%r14
	movq	-32(%rsi),%r13
.cfi_restore	%r13
	movq	-24(%rsi),%r12
.cfi_restore	%r12
	movq	-16(%rsi),%rbp
.cfi_restore	%rbp
	movq	-8(%rsi),%rbx
.cfi_restore	%rbx
	leaq	(%rsi),%rsp
.cfi_def_cfa_register	%rsp
.Lmulx4x_epilogue:
	.byte	0xf3,0xc3
.cfi_endproc	
.size	rscrypto_rsa_bn_mulx4x_mont_x86_64_elf,.-rscrypto_rsa_bn_mulx4x_mont_x86_64_elf
.byte	77,111,110,116,103,111,109,101,114,121,32,77,117,108,116,105,112,108,105,99,97,116,105,111,110,32,102,111,114,32,120,56,54,95,54,52,44,32,67,82,89,80,84,79,71,65,77,83,32,98,121,32,60,97,112,112,114,111,64,111,112,101,110,115,115,108,46,111,114,103,62,0
.align	16

// Square internals required by rscrypto_rsa_bn_sqr8x_mont_x86_64_elf.
.globl	rscrypto_rsa_bn_sqr8x_internal_x86_64_elf
.hidden rscrypto_rsa_bn_sqr8x_internal_x86_64_elf
.hidden	rscrypto_rsa_bn_sqr8x_internal_x86_64_elf
.type	rscrypto_rsa_bn_sqr8x_internal_x86_64_elf,@function
.align	32
rscrypto_rsa_bn_sqr8x_internal_x86_64_elf:
.cfi_startproc	
	leaq	32(%r10),%rbp
	leaq	(%rsi,%r9,1),%rsi

	movq	%r9,%rcx


	movq	-32(%rsi,%rbp,1),%r14
	leaq	48+8(%rsp,%r9,2),%rdi
	movq	-24(%rsi,%rbp,1),%rax
	leaq	-32(%rdi,%rbp,1),%rdi
	movq	-16(%rsi,%rbp,1),%rbx
	movq	%rax,%r15

	mulq	%r14
	movq	%rax,%r10
	movq	%rbx,%rax
	movq	%rdx,%r11
	movq	%r10,-24(%rdi,%rbp,1)

	mulq	%r14
	addq	%rax,%r11
	movq	%rbx,%rax
	adcq	$0,%rdx
	movq	%r11,-16(%rdi,%rbp,1)
	movq	%rdx,%r10


	movq	-8(%rsi,%rbp,1),%rbx
	mulq	%r15
	movq	%rax,%r12
	movq	%rbx,%rax
	movq	%rdx,%r13

	leaq	(%rbp),%rcx
	mulq	%r14
	addq	%rax,%r10
	movq	%rbx,%rax
	movq	%rdx,%r11
	adcq	$0,%r11
	addq	%r12,%r10
	adcq	$0,%r11
	movq	%r10,-8(%rdi,%rcx,1)
	jmp	.Lsqr4x_1st

.align	32
.Lsqr4x_1st:
	movq	(%rsi,%rcx,1),%rbx
	mulq	%r15
	addq	%rax,%r13
	movq	%rbx,%rax
	movq	%rdx,%r12
	adcq	$0,%r12

	mulq	%r14
	addq	%rax,%r11
	movq	%rbx,%rax
	movq	8(%rsi,%rcx,1),%rbx
	movq	%rdx,%r10
	adcq	$0,%r10
	addq	%r13,%r11
	adcq	$0,%r10


	mulq	%r15
	addq	%rax,%r12
	movq	%rbx,%rax
	movq	%r11,(%rdi,%rcx,1)
	movq	%rdx,%r13
	adcq	$0,%r13

	mulq	%r14
	addq	%rax,%r10
	movq	%rbx,%rax
	movq	16(%rsi,%rcx,1),%rbx
	movq	%rdx,%r11
	adcq	$0,%r11
	addq	%r12,%r10
	adcq	$0,%r11

	mulq	%r15
	addq	%rax,%r13
	movq	%rbx,%rax
	movq	%r10,8(%rdi,%rcx,1)
	movq	%rdx,%r12
	adcq	$0,%r12

	mulq	%r14
	addq	%rax,%r11
	movq	%rbx,%rax
	movq	24(%rsi,%rcx,1),%rbx
	movq	%rdx,%r10
	adcq	$0,%r10
	addq	%r13,%r11
	adcq	$0,%r10


	mulq	%r15
	addq	%rax,%r12
	movq	%rbx,%rax
	movq	%r11,16(%rdi,%rcx,1)
	movq	%rdx,%r13
	adcq	$0,%r13
	leaq	32(%rcx),%rcx

	mulq	%r14
	addq	%rax,%r10
	movq	%rbx,%rax
	movq	%rdx,%r11
	adcq	$0,%r11
	addq	%r12,%r10
	adcq	$0,%r11
	movq	%r10,-8(%rdi,%rcx,1)

	cmpq	$0,%rcx
	jne	.Lsqr4x_1st

	mulq	%r15
	addq	%rax,%r13
	leaq	16(%rbp),%rbp
	adcq	$0,%rdx
	addq	%r11,%r13
	adcq	$0,%rdx

	movq	%r13,(%rdi)
	movq	%rdx,%r12
	movq	%rdx,8(%rdi)
	jmp	.Lsqr4x_outer

.align	32
.Lsqr4x_outer:
	movq	-32(%rsi,%rbp,1),%r14
	leaq	48+8(%rsp,%r9,2),%rdi
	movq	-24(%rsi,%rbp,1),%rax
	leaq	-32(%rdi,%rbp,1),%rdi
	movq	-16(%rsi,%rbp,1),%rbx
	movq	%rax,%r15

	mulq	%r14
	movq	-24(%rdi,%rbp,1),%r10
	addq	%rax,%r10
	movq	%rbx,%rax
	adcq	$0,%rdx
	movq	%r10,-24(%rdi,%rbp,1)
	movq	%rdx,%r11

	mulq	%r14
	addq	%rax,%r11
	movq	%rbx,%rax
	adcq	$0,%rdx
	addq	-16(%rdi,%rbp,1),%r11
	movq	%rdx,%r10
	adcq	$0,%r10
	movq	%r11,-16(%rdi,%rbp,1)

	xorq	%r12,%r12

	movq	-8(%rsi,%rbp,1),%rbx
	mulq	%r15
	addq	%rax,%r12
	movq	%rbx,%rax
	adcq	$0,%rdx
	addq	-8(%rdi,%rbp,1),%r12
	movq	%rdx,%r13
	adcq	$0,%r13

	mulq	%r14
	addq	%rax,%r10
	movq	%rbx,%rax
	adcq	$0,%rdx
	addq	%r12,%r10
	movq	%rdx,%r11
	adcq	$0,%r11
	movq	%r10,-8(%rdi,%rbp,1)

	leaq	(%rbp),%rcx
	jmp	.Lsqr4x_inner

.align	32
.Lsqr4x_inner:
	movq	(%rsi,%rcx,1),%rbx
	mulq	%r15
	addq	%rax,%r13
	movq	%rbx,%rax
	movq	%rdx,%r12
	adcq	$0,%r12
	addq	(%rdi,%rcx,1),%r13
	adcq	$0,%r12

.byte	0x67
	mulq	%r14
	addq	%rax,%r11
	movq	%rbx,%rax
	movq	8(%rsi,%rcx,1),%rbx
	movq	%rdx,%r10
	adcq	$0,%r10
	addq	%r13,%r11
	adcq	$0,%r10

	mulq	%r15
	addq	%rax,%r12
	movq	%r11,(%rdi,%rcx,1)
	movq	%rbx,%rax
	movq	%rdx,%r13
	adcq	$0,%r13
	addq	8(%rdi,%rcx,1),%r12
	leaq	16(%rcx),%rcx
	adcq	$0,%r13

	mulq	%r14
	addq	%rax,%r10
	movq	%rbx,%rax
	adcq	$0,%rdx
	addq	%r12,%r10
	movq	%rdx,%r11
	adcq	$0,%r11
	movq	%r10,-8(%rdi,%rcx,1)

	cmpq	$0,%rcx
	jne	.Lsqr4x_inner

.byte	0x67
	mulq	%r15
	addq	%rax,%r13
	adcq	$0,%rdx
	addq	%r11,%r13
	adcq	$0,%rdx

	movq	%r13,(%rdi)
	movq	%rdx,%r12
	movq	%rdx,8(%rdi)

	addq	$16,%rbp
	jnz	.Lsqr4x_outer


	movq	-32(%rsi),%r14
	leaq	48+8(%rsp,%r9,2),%rdi
	movq	-24(%rsi),%rax
	leaq	-32(%rdi,%rbp,1),%rdi
	movq	-16(%rsi),%rbx
	movq	%rax,%r15

	mulq	%r14
	addq	%rax,%r10
	movq	%rbx,%rax
	movq	%rdx,%r11
	adcq	$0,%r11

	mulq	%r14
	addq	%rax,%r11
	movq	%rbx,%rax
	movq	%r10,-24(%rdi)
	movq	%rdx,%r10
	adcq	$0,%r10
	addq	%r13,%r11
	movq	-8(%rsi),%rbx
	adcq	$0,%r10

	mulq	%r15
	addq	%rax,%r12
	movq	%rbx,%rax
	movq	%r11,-16(%rdi)
	movq	%rdx,%r13
	adcq	$0,%r13

	mulq	%r14
	addq	%rax,%r10
	movq	%rbx,%rax
	movq	%rdx,%r11
	adcq	$0,%r11
	addq	%r12,%r10
	adcq	$0,%r11
	movq	%r10,-8(%rdi)

	mulq	%r15
	addq	%rax,%r13
	movq	-16(%rsi),%rax
	adcq	$0,%rdx
	addq	%r11,%r13
	adcq	$0,%rdx

	movq	%r13,(%rdi)
	movq	%rdx,%r12
	movq	%rdx,8(%rdi)

	mulq	%rbx
	addq	$16,%rbp
	xorq	%r14,%r14
	subq	%r9,%rbp
	xorq	%r15,%r15

	addq	%r12,%rax
	adcq	$0,%rdx
	movq	%rax,8(%rdi)
	movq	%rdx,16(%rdi)
	movq	%r15,24(%rdi)

	movq	-16(%rsi,%rbp,1),%rax
	leaq	48+8(%rsp),%rdi
	xorq	%r10,%r10
	movq	8(%rdi),%r11

	leaq	(%r14,%r10,2),%r12
	shrq	$63,%r10
	leaq	(%rcx,%r11,2),%r13
	shrq	$63,%r11
	orq	%r10,%r13
	movq	16(%rdi),%r10
	movq	%r11,%r14
	mulq	%rax
	negq	%r15
	movq	24(%rdi),%r11
	adcq	%rax,%r12
	movq	-8(%rsi,%rbp,1),%rax
	movq	%r12,(%rdi)
	adcq	%rdx,%r13

	leaq	(%r14,%r10,2),%rbx
	movq	%r13,8(%rdi)
	sbbq	%r15,%r15
	shrq	$63,%r10
	leaq	(%rcx,%r11,2),%r8
	shrq	$63,%r11
	orq	%r10,%r8
	movq	32(%rdi),%r10
	movq	%r11,%r14
	mulq	%rax
	negq	%r15
	movq	40(%rdi),%r11
	adcq	%rax,%rbx
	movq	0(%rsi,%rbp,1),%rax
	movq	%rbx,16(%rdi)
	adcq	%rdx,%r8
	leaq	16(%rbp),%rbp
	movq	%r8,24(%rdi)
	sbbq	%r15,%r15
	leaq	64(%rdi),%rdi
	jmp	.Lsqr4x_shift_n_add

.align	32
.Lsqr4x_shift_n_add:
	leaq	(%r14,%r10,2),%r12
	shrq	$63,%r10
	leaq	(%rcx,%r11,2),%r13
	shrq	$63,%r11
	orq	%r10,%r13
	movq	-16(%rdi),%r10
	movq	%r11,%r14
	mulq	%rax
	negq	%r15
	movq	-8(%rdi),%r11
	adcq	%rax,%r12
	movq	-8(%rsi,%rbp,1),%rax
	movq	%r12,-32(%rdi)
	adcq	%rdx,%r13

	leaq	(%r14,%r10,2),%rbx
	movq	%r13,-24(%rdi)
	sbbq	%r15,%r15
	shrq	$63,%r10
	leaq	(%rcx,%r11,2),%r8
	shrq	$63,%r11
	orq	%r10,%r8
	movq	0(%rdi),%r10
	movq	%r11,%r14
	mulq	%rax
	negq	%r15
	movq	8(%rdi),%r11
	adcq	%rax,%rbx
	movq	0(%rsi,%rbp,1),%rax
	movq	%rbx,-16(%rdi)
	adcq	%rdx,%r8

	leaq	(%r14,%r10,2),%r12
	movq	%r8,-8(%rdi)
	sbbq	%r15,%r15
	shrq	$63,%r10
	leaq	(%rcx,%r11,2),%r13
	shrq	$63,%r11
	orq	%r10,%r13
	movq	16(%rdi),%r10
	movq	%r11,%r14
	mulq	%rax
	negq	%r15
	movq	24(%rdi),%r11
	adcq	%rax,%r12
	movq	8(%rsi,%rbp,1),%rax
	movq	%r12,0(%rdi)
	adcq	%rdx,%r13

	leaq	(%r14,%r10,2),%rbx
	movq	%r13,8(%rdi)
	sbbq	%r15,%r15
	shrq	$63,%r10
	leaq	(%rcx,%r11,2),%r8
	shrq	$63,%r11
	orq	%r10,%r8
	movq	32(%rdi),%r10
	movq	%r11,%r14
	mulq	%rax
	negq	%r15
	movq	40(%rdi),%r11
	adcq	%rax,%rbx
	movq	16(%rsi,%rbp,1),%rax
	movq	%rbx,16(%rdi)
	adcq	%rdx,%r8
	movq	%r8,24(%rdi)
	sbbq	%r15,%r15
	leaq	64(%rdi),%rdi
	addq	$32,%rbp
	jnz	.Lsqr4x_shift_n_add

	leaq	(%r14,%r10,2),%r12
.byte	0x67
	shrq	$63,%r10
	leaq	(%rcx,%r11,2),%r13
	shrq	$63,%r11
	orq	%r10,%r13
	movq	-16(%rdi),%r10
	movq	%r11,%r14
	mulq	%rax
	negq	%r15
	movq	-8(%rdi),%r11
	adcq	%rax,%r12
	movq	-8(%rsi),%rax
	movq	%r12,-32(%rdi)
	adcq	%rdx,%r13

	leaq	(%r14,%r10,2),%rbx
	movq	%r13,-24(%rdi)
	sbbq	%r15,%r15
	shrq	$63,%r10
	leaq	(%rcx,%r11,2),%r8
	shrq	$63,%r11
	orq	%r10,%r8
	mulq	%rax
	negq	%r15
	adcq	%rax,%rbx
	adcq	%rdx,%r8
	movq	%rbx,-16(%rdi)
	movq	%r8,-8(%rdi)
.byte	102,72,15,126,213
rscrypto_rsa_bn_sqr8x_reduction_x86_64_elf:
	xorq	%rax,%rax
	leaq	(%r9,%rbp,1),%rcx
	leaq	48+8(%rsp,%r9,2),%rdx
	movq	%rcx,0+8(%rsp)
	leaq	48+8(%rsp,%r9,1),%rdi
	movq	%rdx,8+8(%rsp)
	negq	%r9
	jmp	.L8x_reduction_loop

.align	32
.L8x_reduction_loop:
	leaq	(%rdi,%r9,1),%rdi
.byte	0x66
	movq	0(%rdi),%rbx
	movq	8(%rdi),%r9
	movq	16(%rdi),%r10
	movq	24(%rdi),%r11
	movq	32(%rdi),%r12
	movq	40(%rdi),%r13
	movq	48(%rdi),%r14
	movq	56(%rdi),%r15
	movq	%rax,(%rdx)
	leaq	64(%rdi),%rdi

.byte	0x67
	movq	%rbx,%r8
	imulq	32+8(%rsp),%rbx
	movq	0(%rbp),%rax
	movl	$8,%ecx
	jmp	.L8x_reduce

.align	32
.L8x_reduce:
	mulq	%rbx
	movq	8(%rbp),%rax
	negq	%r8
	movq	%rdx,%r8
	adcq	$0,%r8

	mulq	%rbx
	addq	%rax,%r9
	movq	16(%rbp),%rax
	adcq	$0,%rdx
	addq	%r9,%r8
	movq	%rbx,48-8+8(%rsp,%rcx,8)
	movq	%rdx,%r9
	adcq	$0,%r9

	mulq	%rbx
	addq	%rax,%r10
	movq	24(%rbp),%rax
	adcq	$0,%rdx
	addq	%r10,%r9
	movq	32+8(%rsp),%rsi
	movq	%rdx,%r10
	adcq	$0,%r10

	mulq	%rbx
	addq	%rax,%r11
	movq	32(%rbp),%rax
	adcq	$0,%rdx
	imulq	%r8,%rsi
	addq	%r11,%r10
	movq	%rdx,%r11
	adcq	$0,%r11

	mulq	%rbx
	addq	%rax,%r12
	movq	40(%rbp),%rax
	adcq	$0,%rdx
	addq	%r12,%r11
	movq	%rdx,%r12
	adcq	$0,%r12

	mulq	%rbx
	addq	%rax,%r13
	movq	48(%rbp),%rax
	adcq	$0,%rdx
	addq	%r13,%r12
	movq	%rdx,%r13
	adcq	$0,%r13

	mulq	%rbx
	addq	%rax,%r14
	movq	56(%rbp),%rax
	adcq	$0,%rdx
	addq	%r14,%r13
	movq	%rdx,%r14
	adcq	$0,%r14

	mulq	%rbx
	movq	%rsi,%rbx
	addq	%rax,%r15
	movq	0(%rbp),%rax
	adcq	$0,%rdx
	addq	%r15,%r14
	movq	%rdx,%r15
	adcq	$0,%r15

	decl	%ecx
	jnz	.L8x_reduce

	leaq	64(%rbp),%rbp
	xorq	%rax,%rax
	movq	8+8(%rsp),%rdx
	cmpq	0+8(%rsp),%rbp
	jae	.L8x_no_tail

.byte	0x66
	addq	0(%rdi),%r8
	adcq	8(%rdi),%r9
	adcq	16(%rdi),%r10
	adcq	24(%rdi),%r11
	adcq	32(%rdi),%r12
	adcq	40(%rdi),%r13
	adcq	48(%rdi),%r14
	adcq	56(%rdi),%r15
	sbbq	%rsi,%rsi

	movq	48+56+8(%rsp),%rbx
	movl	$8,%ecx
	movq	0(%rbp),%rax
	jmp	.L8x_tail

.align	32
.L8x_tail:
	mulq	%rbx
	addq	%rax,%r8
	movq	8(%rbp),%rax
	movq	%r8,(%rdi)
	movq	%rdx,%r8
	adcq	$0,%r8

	mulq	%rbx
	addq	%rax,%r9
	movq	16(%rbp),%rax
	adcq	$0,%rdx
	addq	%r9,%r8
	leaq	8(%rdi),%rdi
	movq	%rdx,%r9
	adcq	$0,%r9

	mulq	%rbx
	addq	%rax,%r10
	movq	24(%rbp),%rax
	adcq	$0,%rdx
	addq	%r10,%r9
	movq	%rdx,%r10
	adcq	$0,%r10

	mulq	%rbx
	addq	%rax,%r11
	movq	32(%rbp),%rax
	adcq	$0,%rdx
	addq	%r11,%r10
	movq	%rdx,%r11
	adcq	$0,%r11

	mulq	%rbx
	addq	%rax,%r12
	movq	40(%rbp),%rax
	adcq	$0,%rdx
	addq	%r12,%r11
	movq	%rdx,%r12
	adcq	$0,%r12

	mulq	%rbx
	addq	%rax,%r13
	movq	48(%rbp),%rax
	adcq	$0,%rdx
	addq	%r13,%r12
	movq	%rdx,%r13
	adcq	$0,%r13

	mulq	%rbx
	addq	%rax,%r14
	movq	56(%rbp),%rax
	adcq	$0,%rdx
	addq	%r14,%r13
	movq	%rdx,%r14
	adcq	$0,%r14

	mulq	%rbx
	movq	48-16+8(%rsp,%rcx,8),%rbx
	addq	%rax,%r15
	adcq	$0,%rdx
	addq	%r15,%r14
	movq	0(%rbp),%rax
	movq	%rdx,%r15
	adcq	$0,%r15

	decl	%ecx
	jnz	.L8x_tail

	leaq	64(%rbp),%rbp
	movq	8+8(%rsp),%rdx
	cmpq	0+8(%rsp),%rbp
	jae	.L8x_tail_done

	movq	48+56+8(%rsp),%rbx
	negq	%rsi
	movq	0(%rbp),%rax
	adcq	0(%rdi),%r8
	adcq	8(%rdi),%r9
	adcq	16(%rdi),%r10
	adcq	24(%rdi),%r11
	adcq	32(%rdi),%r12
	adcq	40(%rdi),%r13
	adcq	48(%rdi),%r14
	adcq	56(%rdi),%r15
	sbbq	%rsi,%rsi

	movl	$8,%ecx
	jmp	.L8x_tail

.align	32
.L8x_tail_done:
	xorq	%rax,%rax
	addq	(%rdx),%r8
	adcq	$0,%r9
	adcq	$0,%r10
	adcq	$0,%r11
	adcq	$0,%r12
	adcq	$0,%r13
	adcq	$0,%r14
	adcq	$0,%r15
	adcq	$0,%rax

	negq	%rsi
.L8x_no_tail:
	adcq	0(%rdi),%r8
	adcq	8(%rdi),%r9
	adcq	16(%rdi),%r10
	adcq	24(%rdi),%r11
	adcq	32(%rdi),%r12
	adcq	40(%rdi),%r13
	adcq	48(%rdi),%r14
	adcq	56(%rdi),%r15
	adcq	$0,%rax
	movq	-8(%rbp),%rcx
	xorq	%rsi,%rsi

.byte	102,72,15,126,213

	movq	%r8,0(%rdi)
	movq	%r9,8(%rdi)
.byte	102,73,15,126,217
	movq	%r10,16(%rdi)
	movq	%r11,24(%rdi)
	movq	%r12,32(%rdi)
	movq	%r13,40(%rdi)
	movq	%r14,48(%rdi)
	movq	%r15,56(%rdi)
	leaq	64(%rdi),%rdi

	cmpq	%rdx,%rdi
	jb	.L8x_reduction_loop
	.byte	0xf3,0xc3
.cfi_endproc	
.size	rscrypto_rsa_bn_sqr8x_internal_x86_64_elf,.-rscrypto_rsa_bn_sqr8x_internal_x86_64_elf
.globl	rscrypto_rsa_bn_sqrx8x_internal_x86_64_elf
.hidden rscrypto_rsa_bn_sqrx8x_internal_x86_64_elf
.hidden	rscrypto_rsa_bn_sqrx8x_internal_x86_64_elf
.type	rscrypto_rsa_bn_sqrx8x_internal_x86_64_elf,@function
.align	32
rscrypto_rsa_bn_sqrx8x_internal_x86_64_elf:
.cfi_startproc	
	leaq	48+8(%rsp),%rdi
	leaq	(%rsi,%r9,1),%rbp
	movq	%r9,0+8(%rsp)
	movq	%rbp,8+8(%rsp)
	jmp	.Lsqr8x_zero_start

.align	32
.byte	0x66,0x66,0x66,0x2e,0x0f,0x1f,0x84,0x00,0x00,0x00,0x00,0x00
.Lsqrx8x_zero:
.byte	0x3e
	movdqa	%xmm0,0(%rdi)
	movdqa	%xmm0,16(%rdi)
	movdqa	%xmm0,32(%rdi)
	movdqa	%xmm0,48(%rdi)
.Lsqr8x_zero_start:
	movdqa	%xmm0,64(%rdi)
	movdqa	%xmm0,80(%rdi)
	movdqa	%xmm0,96(%rdi)
	movdqa	%xmm0,112(%rdi)
	leaq	128(%rdi),%rdi
	subq	$64,%r9
	jnz	.Lsqrx8x_zero

	movq	0(%rsi),%rdx

	xorq	%r10,%r10
	xorq	%r11,%r11
	xorq	%r12,%r12
	xorq	%r13,%r13
	xorq	%r14,%r14
	xorq	%r15,%r15
	leaq	48+8(%rsp),%rdi
	xorq	%rbp,%rbp
	jmp	.Lsqrx8x_outer_loop

.align	32
.Lsqrx8x_outer_loop:
	mulxq	8(%rsi),%r8,%rax
	adcxq	%r9,%r8
	adoxq	%rax,%r10
	mulxq	16(%rsi),%r9,%rax
	adcxq	%r10,%r9
	adoxq	%rax,%r11
.byte	0xc4,0xe2,0xab,0xf6,0x86,0x18,0x00,0x00,0x00
	adcxq	%r11,%r10
	adoxq	%rax,%r12
.byte	0xc4,0xe2,0xa3,0xf6,0x86,0x20,0x00,0x00,0x00
	adcxq	%r12,%r11
	adoxq	%rax,%r13
	mulxq	40(%rsi),%r12,%rax
	adcxq	%r13,%r12
	adoxq	%rax,%r14
	mulxq	48(%rsi),%r13,%rax
	adcxq	%r14,%r13
	adoxq	%r15,%rax
	mulxq	56(%rsi),%r14,%r15
	movq	8(%rsi),%rdx
	adcxq	%rax,%r14
	adoxq	%rbp,%r15
	adcq	64(%rdi),%r15
	movq	%r8,8(%rdi)
	movq	%r9,16(%rdi)
	sbbq	%rcx,%rcx
	xorq	%rbp,%rbp


	mulxq	16(%rsi),%r8,%rbx
	mulxq	24(%rsi),%r9,%rax
	adcxq	%r10,%r8
	adoxq	%rbx,%r9
	mulxq	32(%rsi),%r10,%rbx
	adcxq	%r11,%r9
	adoxq	%rax,%r10
.byte	0xc4,0xe2,0xa3,0xf6,0x86,0x28,0x00,0x00,0x00
	adcxq	%r12,%r10
	adoxq	%rbx,%r11
.byte	0xc4,0xe2,0x9b,0xf6,0x9e,0x30,0x00,0x00,0x00
	adcxq	%r13,%r11
	adoxq	%r14,%r12
.byte	0xc4,0x62,0x93,0xf6,0xb6,0x38,0x00,0x00,0x00
	movq	16(%rsi),%rdx
	adcxq	%rax,%r12
	adoxq	%rbx,%r13
	adcxq	%r15,%r13
	adoxq	%rbp,%r14
	adcxq	%rbp,%r14

	movq	%r8,24(%rdi)
	movq	%r9,32(%rdi)

	mulxq	24(%rsi),%r8,%rbx
	mulxq	32(%rsi),%r9,%rax
	adcxq	%r10,%r8
	adoxq	%rbx,%r9
	mulxq	40(%rsi),%r10,%rbx
	adcxq	%r11,%r9
	adoxq	%rax,%r10
.byte	0xc4,0xe2,0xa3,0xf6,0x86,0x30,0x00,0x00,0x00
	adcxq	%r12,%r10
	adoxq	%r13,%r11
.byte	0xc4,0x62,0x9b,0xf6,0xae,0x38,0x00,0x00,0x00
.byte	0x3e
	movq	24(%rsi),%rdx
	adcxq	%rbx,%r11
	adoxq	%rax,%r12
	adcxq	%r14,%r12
	movq	%r8,40(%rdi)
	movq	%r9,48(%rdi)
	mulxq	32(%rsi),%r8,%rax
	adoxq	%rbp,%r13
	adcxq	%rbp,%r13

	mulxq	40(%rsi),%r9,%rbx
	adcxq	%r10,%r8
	adoxq	%rax,%r9
	mulxq	48(%rsi),%r10,%rax
	adcxq	%r11,%r9
	adoxq	%r12,%r10
	mulxq	56(%rsi),%r11,%r12
	movq	32(%rsi),%rdx
	movq	40(%rsi),%r14
	adcxq	%rbx,%r10
	adoxq	%rax,%r11
	movq	48(%rsi),%r15
	adcxq	%r13,%r11
	adoxq	%rbp,%r12
	adcxq	%rbp,%r12

	movq	%r8,56(%rdi)
	movq	%r9,64(%rdi)

	mulxq	%r14,%r9,%rax
	movq	56(%rsi),%r8
	adcxq	%r10,%r9
	mulxq	%r15,%r10,%rbx
	adoxq	%rax,%r10
	adcxq	%r11,%r10
	mulxq	%r8,%r11,%rax
	movq	%r14,%rdx
	adoxq	%rbx,%r11
	adcxq	%r12,%r11

	adcxq	%rbp,%rax

	mulxq	%r15,%r14,%rbx
	mulxq	%r8,%r12,%r13
	movq	%r15,%rdx
	leaq	64(%rsi),%rsi
	adcxq	%r14,%r11
	adoxq	%rbx,%r12
	adcxq	%rax,%r12
	adoxq	%rbp,%r13

.byte	0x67,0x67
	mulxq	%r8,%r8,%r14
	adcxq	%r8,%r13
	adcxq	%rbp,%r14

	cmpq	8+8(%rsp),%rsi
	je	.Lsqrx8x_outer_break

	negq	%rcx
	movq	$-8,%rcx
	movq	%rbp,%r15
	movq	64(%rdi),%r8
	adcxq	72(%rdi),%r9
	adcxq	80(%rdi),%r10
	adcxq	88(%rdi),%r11
	adcq	96(%rdi),%r12
	adcq	104(%rdi),%r13
	adcq	112(%rdi),%r14
	adcq	120(%rdi),%r15
	leaq	(%rsi),%rbp
	leaq	128(%rdi),%rdi
	sbbq	%rax,%rax

	movq	-64(%rsi),%rdx
	movq	%rax,16+8(%rsp)
	movq	%rdi,24+8(%rsp)


	xorl	%eax,%eax
	jmp	.Lsqrx8x_loop

.align	32
.Lsqrx8x_loop:
	movq	%r8,%rbx
	mulxq	0(%rbp),%rax,%r8
	adcxq	%rax,%rbx
	adoxq	%r9,%r8

	mulxq	8(%rbp),%rax,%r9
	adcxq	%rax,%r8
	adoxq	%r10,%r9

	mulxq	16(%rbp),%rax,%r10
	adcxq	%rax,%r9
	adoxq	%r11,%r10

	mulxq	24(%rbp),%rax,%r11
	adcxq	%rax,%r10
	adoxq	%r12,%r11

.byte	0xc4,0x62,0xfb,0xf6,0xa5,0x20,0x00,0x00,0x00
	adcxq	%rax,%r11
	adoxq	%r13,%r12

	mulxq	40(%rbp),%rax,%r13
	adcxq	%rax,%r12
	adoxq	%r14,%r13

	mulxq	48(%rbp),%rax,%r14
	movq	%rbx,(%rdi,%rcx,8)
	movl	$0,%ebx
	adcxq	%rax,%r13
	adoxq	%r15,%r14

.byte	0xc4,0x62,0xfb,0xf6,0xbd,0x38,0x00,0x00,0x00
	movq	8(%rsi,%rcx,8),%rdx
	adcxq	%rax,%r14
	adoxq	%rbx,%r15
	adcxq	%rbx,%r15

.byte	0x67
	incq	%rcx
	jnz	.Lsqrx8x_loop

	leaq	64(%rbp),%rbp
	movq	$-8,%rcx
	cmpq	8+8(%rsp),%rbp
	je	.Lsqrx8x_break

	subq	16+8(%rsp),%rbx
.byte	0x66
	movq	-64(%rsi),%rdx
	adcxq	0(%rdi),%r8
	adcxq	8(%rdi),%r9
	adcq	16(%rdi),%r10
	adcq	24(%rdi),%r11
	adcq	32(%rdi),%r12
	adcq	40(%rdi),%r13
	adcq	48(%rdi),%r14
	adcq	56(%rdi),%r15
	leaq	64(%rdi),%rdi
.byte	0x67
	sbbq	%rax,%rax
	xorl	%ebx,%ebx
	movq	%rax,16+8(%rsp)
	jmp	.Lsqrx8x_loop

.align	32
.Lsqrx8x_break:
	xorq	%rbp,%rbp
	subq	16+8(%rsp),%rbx
	adcxq	%rbp,%r8
	movq	24+8(%rsp),%rcx
	adcxq	%rbp,%r9
	movq	0(%rsi),%rdx
	adcq	$0,%r10
	movq	%r8,0(%rdi)
	adcq	$0,%r11
	adcq	$0,%r12
	adcq	$0,%r13
	adcq	$0,%r14
	adcq	$0,%r15
	cmpq	%rcx,%rdi
	je	.Lsqrx8x_outer_loop

	movq	%r9,8(%rdi)
	movq	8(%rcx),%r9
	movq	%r10,16(%rdi)
	movq	16(%rcx),%r10
	movq	%r11,24(%rdi)
	movq	24(%rcx),%r11
	movq	%r12,32(%rdi)
	movq	32(%rcx),%r12
	movq	%r13,40(%rdi)
	movq	40(%rcx),%r13
	movq	%r14,48(%rdi)
	movq	48(%rcx),%r14
	movq	%r15,56(%rdi)
	movq	56(%rcx),%r15
	movq	%rcx,%rdi
	jmp	.Lsqrx8x_outer_loop

.align	32
.Lsqrx8x_outer_break:
	movq	%r9,72(%rdi)
.byte	102,72,15,126,217
	movq	%r10,80(%rdi)
	movq	%r11,88(%rdi)
	movq	%r12,96(%rdi)
	movq	%r13,104(%rdi)
	movq	%r14,112(%rdi)
	leaq	48+8(%rsp),%rdi
	movq	(%rsi,%rcx,1),%rdx

	movq	8(%rdi),%r11
	xorq	%r10,%r10
	movq	0+8(%rsp),%r9
	adoxq	%r11,%r11
	movq	16(%rdi),%r12
	movq	24(%rdi),%r13


.align	32
.Lsqrx4x_shift_n_add:
	mulxq	%rdx,%rax,%rbx
	adoxq	%r12,%r12
	adcxq	%r10,%rax
.byte	0x48,0x8b,0x94,0x0e,0x08,0x00,0x00,0x00
.byte	0x4c,0x8b,0x97,0x20,0x00,0x00,0x00
	adoxq	%r13,%r13
	adcxq	%r11,%rbx
	movq	40(%rdi),%r11
	movq	%rax,0(%rdi)
	movq	%rbx,8(%rdi)

	mulxq	%rdx,%rax,%rbx
	adoxq	%r10,%r10
	adcxq	%r12,%rax
	movq	16(%rsi,%rcx,1),%rdx
	movq	48(%rdi),%r12
	adoxq	%r11,%r11
	adcxq	%r13,%rbx
	movq	56(%rdi),%r13
	movq	%rax,16(%rdi)
	movq	%rbx,24(%rdi)

	mulxq	%rdx,%rax,%rbx
	adoxq	%r12,%r12
	adcxq	%r10,%rax
	movq	24(%rsi,%rcx,1),%rdx
	leaq	32(%rcx),%rcx
	movq	64(%rdi),%r10
	adoxq	%r13,%r13
	adcxq	%r11,%rbx
	movq	72(%rdi),%r11
	movq	%rax,32(%rdi)
	movq	%rbx,40(%rdi)

	mulxq	%rdx,%rax,%rbx
	adoxq	%r10,%r10
	adcxq	%r12,%rax
	jrcxz	.Lsqrx4x_shift_n_add_break
.byte	0x48,0x8b,0x94,0x0e,0x00,0x00,0x00,0x00
	adoxq	%r11,%r11
	adcxq	%r13,%rbx
	movq	80(%rdi),%r12
	movq	88(%rdi),%r13
	movq	%rax,48(%rdi)
	movq	%rbx,56(%rdi)
	leaq	64(%rdi),%rdi
	nop
	jmp	.Lsqrx4x_shift_n_add

.align	32
.Lsqrx4x_shift_n_add_break:
	adcxq	%r13,%rbx
	movq	%rax,48(%rdi)
	movq	%rbx,56(%rdi)
	leaq	64(%rdi),%rdi
.byte	102,72,15,126,213
rscrypto_rsa_bn_sqrx8x_reduction_x86_64_elf:
	xorl	%eax,%eax
	movq	32+8(%rsp),%rbx
	movq	48+8(%rsp),%rdx
	leaq	-64(%rbp,%r9,1),%rcx

	movq	%rcx,0+8(%rsp)
	movq	%rdi,8+8(%rsp)

	leaq	48+8(%rsp),%rdi
	jmp	.Lsqrx8x_reduction_loop

.align	32
.Lsqrx8x_reduction_loop:
	movq	8(%rdi),%r9
	movq	16(%rdi),%r10
	movq	24(%rdi),%r11
	movq	32(%rdi),%r12
	movq	%rdx,%r8
	imulq	%rbx,%rdx
	movq	40(%rdi),%r13
	movq	48(%rdi),%r14
	movq	56(%rdi),%r15
	movq	%rax,24+8(%rsp)

	leaq	64(%rdi),%rdi
	xorq	%rsi,%rsi
	movq	$-8,%rcx
	jmp	.Lsqrx8x_reduce

.align	32
.Lsqrx8x_reduce:
	movq	%r8,%rbx
	mulxq	0(%rbp),%rax,%r8
	adcxq	%rbx,%rax
	adoxq	%r9,%r8

	mulxq	8(%rbp),%rbx,%r9
	adcxq	%rbx,%r8
	adoxq	%r10,%r9

	mulxq	16(%rbp),%rbx,%r10
	adcxq	%rbx,%r9
	adoxq	%r11,%r10

	mulxq	24(%rbp),%rbx,%r11
	adcxq	%rbx,%r10
	adoxq	%r12,%r11

.byte	0xc4,0x62,0xe3,0xf6,0xa5,0x20,0x00,0x00,0x00
	movq	%rdx,%rax
	movq	%r8,%rdx
	adcxq	%rbx,%r11
	adoxq	%r13,%r12

	mulxq	32+8(%rsp),%rbx,%rdx
	movq	%rax,%rdx
	movq	%rax,64+48+8(%rsp,%rcx,8)

	mulxq	40(%rbp),%rax,%r13
	adcxq	%rax,%r12
	adoxq	%r14,%r13

	mulxq	48(%rbp),%rax,%r14
	adcxq	%rax,%r13
	adoxq	%r15,%r14

	mulxq	56(%rbp),%rax,%r15
	movq	%rbx,%rdx
	adcxq	%rax,%r14
	adoxq	%rsi,%r15
	adcxq	%rsi,%r15

.byte	0x67,0x67,0x67
	incq	%rcx
	jnz	.Lsqrx8x_reduce

	movq	%rsi,%rax
	cmpq	0+8(%rsp),%rbp
	jae	.Lsqrx8x_no_tail

	movq	48+8(%rsp),%rdx
	addq	0(%rdi),%r8
	leaq	64(%rbp),%rbp
	movq	$-8,%rcx
	adcxq	8(%rdi),%r9
	adcxq	16(%rdi),%r10
	adcq	24(%rdi),%r11
	adcq	32(%rdi),%r12
	adcq	40(%rdi),%r13
	adcq	48(%rdi),%r14
	adcq	56(%rdi),%r15
	leaq	64(%rdi),%rdi
	sbbq	%rax,%rax

	xorq	%rsi,%rsi
	movq	%rax,16+8(%rsp)
	jmp	.Lsqrx8x_tail

.align	32
.Lsqrx8x_tail:
	movq	%r8,%rbx
	mulxq	0(%rbp),%rax,%r8
	adcxq	%rax,%rbx
	adoxq	%r9,%r8

	mulxq	8(%rbp),%rax,%r9
	adcxq	%rax,%r8
	adoxq	%r10,%r9

	mulxq	16(%rbp),%rax,%r10
	adcxq	%rax,%r9
	adoxq	%r11,%r10

	mulxq	24(%rbp),%rax,%r11
	adcxq	%rax,%r10
	adoxq	%r12,%r11

.byte	0xc4,0x62,0xfb,0xf6,0xa5,0x20,0x00,0x00,0x00
	adcxq	%rax,%r11
	adoxq	%r13,%r12

	mulxq	40(%rbp),%rax,%r13
	adcxq	%rax,%r12
	adoxq	%r14,%r13

	mulxq	48(%rbp),%rax,%r14
	adcxq	%rax,%r13
	adoxq	%r15,%r14

	mulxq	56(%rbp),%rax,%r15
	movq	72+48+8(%rsp,%rcx,8),%rdx
	adcxq	%rax,%r14
	adoxq	%rsi,%r15
	movq	%rbx,(%rdi,%rcx,8)
	movq	%r8,%rbx
	adcxq	%rsi,%r15

	incq	%rcx
	jnz	.Lsqrx8x_tail

	cmpq	0+8(%rsp),%rbp
	jae	.Lsqrx8x_tail_done

	subq	16+8(%rsp),%rsi
	movq	48+8(%rsp),%rdx
	leaq	64(%rbp),%rbp
	adcq	0(%rdi),%r8
	adcq	8(%rdi),%r9
	adcq	16(%rdi),%r10
	adcq	24(%rdi),%r11
	adcq	32(%rdi),%r12
	adcq	40(%rdi),%r13
	adcq	48(%rdi),%r14
	adcq	56(%rdi),%r15
	leaq	64(%rdi),%rdi
	sbbq	%rax,%rax
	subq	$8,%rcx

	xorq	%rsi,%rsi
	movq	%rax,16+8(%rsp)
	jmp	.Lsqrx8x_tail

.align	32
.Lsqrx8x_tail_done:
	xorq	%rax,%rax
	addq	24+8(%rsp),%r8
	adcq	$0,%r9
	adcq	$0,%r10
	adcq	$0,%r11
	adcq	$0,%r12
	adcq	$0,%r13
	adcq	$0,%r14
	adcq	$0,%r15
	adcq	$0,%rax

	subq	16+8(%rsp),%rsi
.Lsqrx8x_no_tail:
	adcq	0(%rdi),%r8
.byte	102,72,15,126,217
	adcq	8(%rdi),%r9
	movq	56(%rbp),%rsi
.byte	102,72,15,126,213
	adcq	16(%rdi),%r10
	adcq	24(%rdi),%r11
	adcq	32(%rdi),%r12
	adcq	40(%rdi),%r13
	adcq	48(%rdi),%r14
	adcq	56(%rdi),%r15
	adcq	$0,%rax

	movq	32+8(%rsp),%rbx
	movq	64(%rdi,%rcx,1),%rdx

	movq	%r8,0(%rdi)
	leaq	64(%rdi),%r8
	movq	%r9,8(%rdi)
	movq	%r10,16(%rdi)
	movq	%r11,24(%rdi)
	movq	%r12,32(%rdi)
	movq	%r13,40(%rdi)
	movq	%r14,48(%rdi)
	movq	%r15,56(%rdi)

	leaq	64(%rdi,%rcx,1),%rdi
	cmpq	8+8(%rsp),%r8
	jb	.Lsqrx8x_reduction_loop
	.byte	0xf3,0xc3
.cfi_endproc	
.size	rscrypto_rsa_bn_sqrx8x_internal_x86_64_elf,.-rscrypto_rsa_bn_sqrx8x_internal_x86_64_elf

.section .note.GNU-stack,"",@progbits
