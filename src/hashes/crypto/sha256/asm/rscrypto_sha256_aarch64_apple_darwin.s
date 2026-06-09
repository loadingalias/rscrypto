// Copyright 2014-2020 The OpenSSL Project Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted for rscrypto from the AWS-LC/BoringSSL/OpenSSL AArch64 SHA-256 backend:
// - crypto/fipsmodule/sha/asm/sha512-armv8.pl (SHA-256 Armv8 section)
//
// The symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.

.section	__TEXT,__const
.align	6

Lrscrypto_sha256_K256:
.long	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5
.long	0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5
.long	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3
.long	0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174
.long	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc
.long	0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da
.long	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7
.long	0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967
.long	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13
.long	0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85
.long	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3
.long	0xd192e819,0xd6990624,0xf40e3585,0x106aa070
.long	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5
.long	0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3
.long	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208
.long	0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
.long	0	// terminator

.byte	83,72,65,50,53,54,32,98,108,111,99,107,32,116,114,97,110,115,102,111,114,109,32,102,111,114,32,65,82,77,118,56,44,32,67,82,89,80,84,79,71,65,77,83,32,98,121,32,60,97,112,112,114,111,64,111,112,101,110,115,115,108,46,111,114,103,62,0
.align	2
.align	2
.text
.globl	_rscrypto_sha256_block_data_order_hw
.private_extern	_rscrypto_sha256_block_data_order_hw

.align	6
_rscrypto_sha256_block_data_order_hw:
	// Armv8.3-A PAuth: even though x30 is pushed to stack it is not popped later.
	stp	x29,x30,[sp,#-16]!
	add	x29,sp,#0

	ld1	{{v0.4s,v1.4s}},[x0]
	adrp	x3,Lrscrypto_sha256_K256@PAGE
	add	x3,x3,Lrscrypto_sha256_K256@PAGEOFF

Lrscrypto_sha256_loop_hw:
	ld1	{{v4.16b,v5.16b,v6.16b,v7.16b}},[x1],#64
	sub	x2,x2,#1
	ld1	{{v16.4s}},[x3],#16
	rev32	v4.16b,v4.16b
	rev32	v5.16b,v5.16b
	rev32	v6.16b,v6.16b
	rev32	v7.16b,v7.16b
	orr	v18.16b,v0.16b,v0.16b
	orr	v19.16b,v1.16b,v1.16b
	ld1	{{v17.4s}},[x3],#16
	add	v16.4s,v16.4s,v4.4s
.long	0x5e2828a4	// sha256su0 v4.16b,v5.16b
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e104020	// sha256h v0.16b,v1.16b,v16.4s
.long	0x5e105041	// sha256h2 v1.16b,v2.16b,v16.4s
.long	0x5e0760c4	// sha256su1 v4.16b,v6.16b,v7.16b
	ld1	{{v16.4s}},[x3],#16
	add	v17.4s,v17.4s,v5.4s
.long	0x5e2828c5	// sha256su0 v5.16b,v6.16b
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e114020	// sha256h v0.16b,v1.16b,v17.4s
.long	0x5e115041	// sha256h2 v1.16b,v2.16b,v17.4s
.long	0x5e0460e5	// sha256su1 v5.16b,v7.16b,v4.16b
	ld1	{{v17.4s}},[x3],#16
	add	v16.4s,v16.4s,v6.4s
.long	0x5e2828e6	// sha256su0 v6.16b,v7.16b
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e104020	// sha256h v0.16b,v1.16b,v16.4s
.long	0x5e105041	// sha256h2 v1.16b,v2.16b,v16.4s
.long	0x5e056086	// sha256su1 v6.16b,v4.16b,v5.16b
	ld1	{{v16.4s}},[x3],#16
	add	v17.4s,v17.4s,v7.4s
.long	0x5e282887	// sha256su0 v7.16b,v4.16b
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e114020	// sha256h v0.16b,v1.16b,v17.4s
.long	0x5e115041	// sha256h2 v1.16b,v2.16b,v17.4s
.long	0x5e0660a7	// sha256su1 v7.16b,v5.16b,v6.16b
	ld1	{{v17.4s}},[x3],#16
	add	v16.4s,v16.4s,v4.4s
.long	0x5e2828a4	// sha256su0 v4.16b,v5.16b
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e104020	// sha256h v0.16b,v1.16b,v16.4s
.long	0x5e105041	// sha256h2 v1.16b,v2.16b,v16.4s
.long	0x5e0760c4	// sha256su1 v4.16b,v6.16b,v7.16b
	ld1	{{v16.4s}},[x3],#16
	add	v17.4s,v17.4s,v5.4s
.long	0x5e2828c5	// sha256su0 v5.16b,v6.16b
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e114020	// sha256h v0.16b,v1.16b,v17.4s
.long	0x5e115041	// sha256h2 v1.16b,v2.16b,v17.4s
.long	0x5e0460e5	// sha256su1 v5.16b,v7.16b,v4.16b
	ld1	{{v17.4s}},[x3],#16
	add	v16.4s,v16.4s,v6.4s
.long	0x5e2828e6	// sha256su0 v6.16b,v7.16b
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e104020	// sha256h v0.16b,v1.16b,v16.4s
.long	0x5e105041	// sha256h2 v1.16b,v2.16b,v16.4s
.long	0x5e056086	// sha256su1 v6.16b,v4.16b,v5.16b
	ld1	{{v16.4s}},[x3],#16
	add	v17.4s,v17.4s,v7.4s
.long	0x5e282887	// sha256su0 v7.16b,v4.16b
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e114020	// sha256h v0.16b,v1.16b,v17.4s
.long	0x5e115041	// sha256h2 v1.16b,v2.16b,v17.4s
.long	0x5e0660a7	// sha256su1 v7.16b,v5.16b,v6.16b
	ld1	{{v17.4s}},[x3],#16
	add	v16.4s,v16.4s,v4.4s
.long	0x5e2828a4	// sha256su0 v4.16b,v5.16b
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e104020	// sha256h v0.16b,v1.16b,v16.4s
.long	0x5e105041	// sha256h2 v1.16b,v2.16b,v16.4s
.long	0x5e0760c4	// sha256su1 v4.16b,v6.16b,v7.16b
	ld1	{{v16.4s}},[x3],#16
	add	v17.4s,v17.4s,v5.4s
.long	0x5e2828c5	// sha256su0 v5.16b,v6.16b
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e114020	// sha256h v0.16b,v1.16b,v17.4s
.long	0x5e115041	// sha256h2 v1.16b,v2.16b,v17.4s
.long	0x5e0460e5	// sha256su1 v5.16b,v7.16b,v4.16b
	ld1	{{v17.4s}},[x3],#16
	add	v16.4s,v16.4s,v6.4s
.long	0x5e2828e6	// sha256su0 v6.16b,v7.16b
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e104020	// sha256h v0.16b,v1.16b,v16.4s
.long	0x5e105041	// sha256h2 v1.16b,v2.16b,v16.4s
.long	0x5e056086	// sha256su1 v6.16b,v4.16b,v5.16b
	ld1	{{v16.4s}},[x3],#16
	add	v17.4s,v17.4s,v7.4s
.long	0x5e282887	// sha256su0 v7.16b,v4.16b
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e114020	// sha256h v0.16b,v1.16b,v17.4s
.long	0x5e115041	// sha256h2 v1.16b,v2.16b,v17.4s
.long	0x5e0660a7	// sha256su1 v7.16b,v5.16b,v6.16b
	ld1	{{v17.4s}},[x3],#16
	add	v16.4s,v16.4s,v4.4s
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e104020	// sha256h v0.16b,v1.16b,v16.4s
.long	0x5e105041	// sha256h2 v1.16b,v2.16b,v16.4s

	ld1	{{v16.4s}},[x3],#16
	add	v17.4s,v17.4s,v5.4s
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e114020	// sha256h v0.16b,v1.16b,v17.4s
.long	0x5e115041	// sha256h2 v1.16b,v2.16b,v17.4s

	ld1	{{v17.4s}},[x3]
	add	v16.4s,v16.4s,v6.4s
	sub	x3,x3,#64*4-16
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e104020	// sha256h v0.16b,v1.16b,v16.4s
.long	0x5e105041	// sha256h2 v1.16b,v2.16b,v16.4s

	add	v17.4s,v17.4s,v7.4s
	orr	v2.16b,v0.16b,v0.16b
.long	0x5e114020	// sha256h v0.16b,v1.16b,v17.4s
.long	0x5e115041	// sha256h2 v1.16b,v2.16b,v17.4s

	add	v0.4s,v0.4s,v18.4s
	add	v1.4s,v1.4s,v19.4s

	cbnz	x2,Lrscrypto_sha256_loop_hw

	st1	{{v0.4s,v1.4s}},[x0]

	ldr	x29,[sp],#16
	ret
