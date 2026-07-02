// Copyright (c) 2026 rscrypto contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// rscrypto-owned Linux AArch64 SHA3 Keccak-f[1600] x3 kernel.
// Processes states A/B with NEON SHA3 lanes and state C with scalar
// integer lanes in the same 24-round loop.

.text
.arch armv8.2-a+sha3

.global rscrypto_keccakf1600_aarch64_sha3_x3
.type rscrypto_keccakf1600_aarch64_sha3_x3, %function
rscrypto_keccakf1600_aarch64_sha3_x3:
  sub sp, sp, #192
  stp x19, x20, [sp, #0]
  stp x21, x22, [sp, #16]
  stp x23, x24, [sp, #32]
  stp x25, x26, [sp, #48]
  stp x27, x28, [sp, #64]
  stp x29, x30, [sp, #80]
  stp d8, d9, [sp, #96]
  stp d10, d11, [sp, #112]
  stp d12, d13, [sp, #128]
  stp d14, d15, [sp, #144]
  stp x0, x1, [sp, #160]
  str x2, [sp, #176]

  // Load A/B into NEON lane pairs: lane 0 = A, lane 1 = B.
  ldr d0, [x0, #0]
  ldr d31, [x1, #0]
  mov v0.d[1], v31.d[0]
  ldr d1, [x0, #8]
  ldr d31, [x1, #8]
  mov v1.d[1], v31.d[0]
  ldr d2, [x0, #16]
  ldr d31, [x1, #16]
  mov v2.d[1], v31.d[0]
  ldr d3, [x0, #24]
  ldr d31, [x1, #24]
  mov v3.d[1], v31.d[0]
  ldr d4, [x0, #32]
  ldr d31, [x1, #32]
  mov v4.d[1], v31.d[0]
  ldr d5, [x0, #40]
  ldr d31, [x1, #40]
  mov v5.d[1], v31.d[0]
  ldr d6, [x0, #48]
  ldr d31, [x1, #48]
  mov v6.d[1], v31.d[0]
  ldr d7, [x0, #56]
  ldr d31, [x1, #56]
  mov v7.d[1], v31.d[0]
  ldr d8, [x0, #64]
  ldr d31, [x1, #64]
  mov v8.d[1], v31.d[0]
  ldr d9, [x0, #72]
  ldr d31, [x1, #72]
  mov v9.d[1], v31.d[0]
  ldr d10, [x0, #80]
  ldr d31, [x1, #80]
  mov v10.d[1], v31.d[0]
  ldr d11, [x0, #88]
  ldr d31, [x1, #88]
  mov v11.d[1], v31.d[0]
  ldr d12, [x0, #96]
  ldr d31, [x1, #96]
  mov v12.d[1], v31.d[0]
  ldr d13, [x0, #104]
  ldr d31, [x1, #104]
  mov v13.d[1], v31.d[0]
  ldr d14, [x0, #112]
  ldr d31, [x1, #112]
  mov v14.d[1], v31.d[0]
  ldr d15, [x0, #120]
  ldr d31, [x1, #120]
  mov v15.d[1], v31.d[0]
  ldr d16, [x0, #128]
  ldr d31, [x1, #128]
  mov v16.d[1], v31.d[0]
  ldr d17, [x0, #136]
  ldr d31, [x1, #136]
  mov v17.d[1], v31.d[0]
  ldr d18, [x0, #144]
  ldr d31, [x1, #144]
  mov v18.d[1], v31.d[0]
  ldr d19, [x0, #152]
  ldr d31, [x1, #152]
  mov v19.d[1], v31.d[0]
  ldr d20, [x0, #160]
  ldr d31, [x1, #160]
  mov v20.d[1], v31.d[0]
  ldr d21, [x0, #168]
  ldr d31, [x1, #168]
  mov v21.d[1], v31.d[0]
  ldr d22, [x0, #176]
  ldr d31, [x1, #176]
  mov v22.d[1], v31.d[0]
  ldr d23, [x0, #184]
  ldr d31, [x1, #184]
  mov v23.d[1], v31.d[0]
  ldr d24, [x0, #192]
  ldr d31, [x1, #192]
  mov v24.d[1], v31.d[0]

  // Load C into scalar lanes x0..x24.
  ldr x25, [sp, #176]
  ldp x0, x1, [x25, #0]
  ldp x2, x3, [x25, #16]
  ldp x4, x5, [x25, #32]
  ldp x6, x7, [x25, #48]
  ldp x8, x9, [x25, #64]
  ldp x10, x11, [x25, #80]
  ldp x12, x13, [x25, #96]
  ldp x14, x15, [x25, #112]
  ldp x16, x17, [x25, #128]
  ldp x18, x19, [x25, #144]
  ldp x20, x21, [x25, #160]
  ldp x22, x23, [x25, #176]
  ldr x24, [x25, #192]

  adrp x30, .Lrscrypto_keccakf1600_aarch64_sha3_x3_rc
  add x30, x30, :lo12:.Lrscrypto_keccakf1600_aarch64_sha3_x3_rc

.Lrscrypto_keccakf1600_aarch64_sha3_x3_round:
  // A/B theta column parity.
  mov v25.16b, v0.16b
  eor3 v25.16b, v25.16b, v5.16b, v10.16b
  eor3 v25.16b, v25.16b, v15.16b, v20.16b
  mov v26.16b, v1.16b
  eor3 v26.16b, v26.16b, v6.16b, v11.16b
  eor3 v26.16b, v26.16b, v16.16b, v21.16b
  mov v27.16b, v2.16b
  eor3 v27.16b, v27.16b, v7.16b, v12.16b
  eor3 v27.16b, v27.16b, v17.16b, v22.16b
  mov v28.16b, v3.16b
  eor3 v28.16b, v28.16b, v8.16b, v13.16b
  eor3 v28.16b, v28.16b, v18.16b, v23.16b
  mov v29.16b, v4.16b
  eor3 v29.16b, v29.16b, v9.16b, v14.16b
  eor3 v29.16b, v29.16b, v19.16b, v24.16b

  // A/B theta diffusion plus rho.
  mov v30.16b, v29.16b
  rax1 v30.2d, v30.2d, v26.2d
  xar v0.2d, v0.2d, v30.2d, #0
  xar v5.2d, v5.2d, v30.2d, #28
  xar v10.2d, v10.2d, v30.2d, #61
  xar v15.2d, v15.2d, v30.2d, #23
  xar v20.2d, v20.2d, v30.2d, #46
  mov v30.16b, v25.16b
  rax1 v30.2d, v30.2d, v27.2d
  xar v1.2d, v1.2d, v30.2d, #63
  xar v6.2d, v6.2d, v30.2d, #20
  xar v11.2d, v11.2d, v30.2d, #54
  xar v16.2d, v16.2d, v30.2d, #19
  xar v21.2d, v21.2d, v30.2d, #62
  mov v30.16b, v26.16b
  rax1 v30.2d, v30.2d, v28.2d
  xar v2.2d, v2.2d, v30.2d, #2
  xar v7.2d, v7.2d, v30.2d, #58
  xar v12.2d, v12.2d, v30.2d, #21
  xar v17.2d, v17.2d, v30.2d, #49
  xar v22.2d, v22.2d, v30.2d, #3
  mov v30.16b, v27.16b
  rax1 v30.2d, v30.2d, v29.2d
  xar v3.2d, v3.2d, v30.2d, #36
  xar v8.2d, v8.2d, v30.2d, #9
  xar v13.2d, v13.2d, v30.2d, #39
  xar v18.2d, v18.2d, v30.2d, #43
  xar v23.2d, v23.2d, v30.2d, #8
  mov v30.16b, v28.16b
  rax1 v30.2d, v30.2d, v25.2d
  xar v4.2d, v4.2d, v30.2d, #37
  xar v9.2d, v9.2d, v30.2d, #44
  xar v14.2d, v14.2d, v30.2d, #25
  xar v19.2d, v19.2d, v30.2d, #56
  xar v24.2d, v24.2d, v30.2d, #50

  // A/B pi permutation cycle.
  mov v25.16b, v1.16b
  mov v26.16b, v10.16b
  mov v10.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v7.16b
  mov v7.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v11.16b
  mov v11.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v17.16b
  mov v17.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v18.16b
  mov v18.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v3.16b
  mov v3.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v5.16b
  mov v5.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v16.16b
  mov v16.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v8.16b
  mov v8.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v21.16b
  mov v21.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v24.16b
  mov v24.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v4.16b
  mov v4.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v15.16b
  mov v15.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v23.16b
  mov v23.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v19.16b
  mov v19.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v13.16b
  mov v13.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v12.16b
  mov v12.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v2.16b
  mov v2.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v20.16b
  mov v20.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v14.16b
  mov v14.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v22.16b
  mov v22.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v9.16b
  mov v9.16b, v25.16b
  mov v25.16b, v26.16b
  mov v26.16b, v6.16b
  mov v6.16b, v25.16b
  mov v25.16b, v26.16b
  mov v1.16b, v25.16b

  // A/B chi rows.
  bcax v25.16b, v0.16b, v2.16b, v1.16b
  bcax v26.16b, v1.16b, v3.16b, v2.16b
  bcax v27.16b, v2.16b, v4.16b, v3.16b
  bcax v28.16b, v3.16b, v0.16b, v4.16b
  bcax v29.16b, v4.16b, v1.16b, v0.16b
  mov v0.16b, v25.16b
  mov v1.16b, v26.16b
  mov v2.16b, v27.16b
  mov v3.16b, v28.16b
  mov v4.16b, v29.16b
  bcax v25.16b, v5.16b, v7.16b, v6.16b
  bcax v26.16b, v6.16b, v8.16b, v7.16b
  bcax v27.16b, v7.16b, v9.16b, v8.16b
  bcax v28.16b, v8.16b, v5.16b, v9.16b
  bcax v29.16b, v9.16b, v6.16b, v5.16b
  mov v5.16b, v25.16b
  mov v6.16b, v26.16b
  mov v7.16b, v27.16b
  mov v8.16b, v28.16b
  mov v9.16b, v29.16b
  bcax v25.16b, v10.16b, v12.16b, v11.16b
  bcax v26.16b, v11.16b, v13.16b, v12.16b
  bcax v27.16b, v12.16b, v14.16b, v13.16b
  bcax v28.16b, v13.16b, v10.16b, v14.16b
  bcax v29.16b, v14.16b, v11.16b, v10.16b
  mov v10.16b, v25.16b
  mov v11.16b, v26.16b
  mov v12.16b, v27.16b
  mov v13.16b, v28.16b
  mov v14.16b, v29.16b
  bcax v25.16b, v15.16b, v17.16b, v16.16b
  bcax v26.16b, v16.16b, v18.16b, v17.16b
  bcax v27.16b, v17.16b, v19.16b, v18.16b
  bcax v28.16b, v18.16b, v15.16b, v19.16b
  bcax v29.16b, v19.16b, v16.16b, v15.16b
  mov v15.16b, v25.16b
  mov v16.16b, v26.16b
  mov v17.16b, v27.16b
  mov v18.16b, v28.16b
  mov v19.16b, v29.16b
  bcax v25.16b, v20.16b, v22.16b, v21.16b
  bcax v26.16b, v21.16b, v23.16b, v22.16b
  bcax v27.16b, v22.16b, v24.16b, v23.16b
  bcax v28.16b, v23.16b, v20.16b, v24.16b
  bcax v29.16b, v24.16b, v21.16b, v20.16b
  mov v20.16b, v25.16b
  mov v21.16b, v26.16b
  mov v22.16b, v27.16b
  mov v23.16b, v28.16b
  mov v24.16b, v29.16b

  // Scalar C theta.
  str x30, [sp, #184]
  eor x25, x0, x5
  eor x25, x25, x10
  eor x25, x25, x15
  eor x25, x25, x20
  eor x26, x1, x6
  eor x26, x26, x11
  eor x26, x26, x16
  eor x26, x26, x21
  eor x27, x2, x7
  eor x27, x27, x12
  eor x27, x27, x17
  eor x27, x27, x22
  eor x28, x3, x8
  eor x28, x28, x13
  eor x28, x28, x18
  eor x28, x28, x23
  eor x29, x4, x9
  eor x29, x29, x14
  eor x29, x29, x19
  eor x29, x29, x24

  // Scalar C theta diffusion plus rho.
  eor x30, x29, x26, ror #63
  eor x0, x0, x30
  eor x5, x5, x30
  ror x5, x5, #28
  eor x10, x10, x30
  ror x10, x10, #61
  eor x15, x15, x30
  ror x15, x15, #23
  eor x20, x20, x30
  ror x20, x20, #46
  eor x30, x25, x27, ror #63
  eor x1, x1, x30
  ror x1, x1, #63
  eor x6, x6, x30
  ror x6, x6, #20
  eor x11, x11, x30
  ror x11, x11, #54
  eor x16, x16, x30
  ror x16, x16, #19
  eor x21, x21, x30
  ror x21, x21, #62
  eor x30, x26, x28, ror #63
  eor x2, x2, x30
  ror x2, x2, #2
  eor x7, x7, x30
  ror x7, x7, #58
  eor x12, x12, x30
  ror x12, x12, #21
  eor x17, x17, x30
  ror x17, x17, #49
  eor x22, x22, x30
  ror x22, x22, #3
  eor x30, x27, x29, ror #63
  eor x3, x3, x30
  ror x3, x3, #36
  eor x8, x8, x30
  ror x8, x8, #9
  eor x13, x13, x30
  ror x13, x13, #39
  eor x18, x18, x30
  ror x18, x18, #43
  eor x23, x23, x30
  ror x23, x23, #8
  eor x30, x28, x25, ror #63
  eor x4, x4, x30
  ror x4, x4, #37
  eor x9, x9, x30
  ror x9, x9, #44
  eor x14, x14, x30
  ror x14, x14, #25
  eor x19, x19, x30
  ror x19, x19, #56
  eor x24, x24, x30
  ror x24, x24, #50

  // Scalar C pi permutation cycle.
  mov x25, x1
  mov x26, x10
  mov x10, x25
  mov x25, x26
  mov x26, x7
  mov x7, x25
  mov x25, x26
  mov x26, x11
  mov x11, x25
  mov x25, x26
  mov x26, x17
  mov x17, x25
  mov x25, x26
  mov x26, x18
  mov x18, x25
  mov x25, x26
  mov x26, x3
  mov x3, x25
  mov x25, x26
  mov x26, x5
  mov x5, x25
  mov x25, x26
  mov x26, x16
  mov x16, x25
  mov x25, x26
  mov x26, x8
  mov x8, x25
  mov x25, x26
  mov x26, x21
  mov x21, x25
  mov x25, x26
  mov x26, x24
  mov x24, x25
  mov x25, x26
  mov x26, x4
  mov x4, x25
  mov x25, x26
  mov x26, x15
  mov x15, x25
  mov x25, x26
  mov x26, x23
  mov x23, x25
  mov x25, x26
  mov x26, x19
  mov x19, x25
  mov x25, x26
  mov x26, x13
  mov x13, x25
  mov x25, x26
  mov x26, x12
  mov x12, x25
  mov x25, x26
  mov x26, x2
  mov x2, x25
  mov x25, x26
  mov x26, x20
  mov x20, x25
  mov x25, x26
  mov x26, x14
  mov x14, x25
  mov x25, x26
  mov x26, x22
  mov x22, x25
  mov x25, x26
  mov x26, x9
  mov x9, x25
  mov x25, x26
  mov x26, x6
  mov x6, x25
  mov x25, x26
  mov x1, x25

  // Scalar C chi rows.
  bic x25, x2, x1
  eor x25, x0, x25
  bic x26, x3, x2
  eor x26, x1, x26
  bic x27, x4, x3
  eor x27, x2, x27
  bic x28, x0, x4
  eor x28, x3, x28
  bic x29, x1, x0
  eor x29, x4, x29
  mov x0, x25
  mov x1, x26
  mov x2, x27
  mov x3, x28
  mov x4, x29
  bic x25, x7, x6
  eor x25, x5, x25
  bic x26, x8, x7
  eor x26, x6, x26
  bic x27, x9, x8
  eor x27, x7, x27
  bic x28, x5, x9
  eor x28, x8, x28
  bic x29, x6, x5
  eor x29, x9, x29
  mov x5, x25
  mov x6, x26
  mov x7, x27
  mov x8, x28
  mov x9, x29
  bic x25, x12, x11
  eor x25, x10, x25
  bic x26, x13, x12
  eor x26, x11, x26
  bic x27, x14, x13
  eor x27, x12, x27
  bic x28, x10, x14
  eor x28, x13, x28
  bic x29, x11, x10
  eor x29, x14, x29
  mov x10, x25
  mov x11, x26
  mov x12, x27
  mov x13, x28
  mov x14, x29
  bic x25, x17, x16
  eor x25, x15, x25
  bic x26, x18, x17
  eor x26, x16, x26
  bic x27, x19, x18
  eor x27, x17, x27
  bic x28, x15, x19
  eor x28, x18, x28
  bic x29, x16, x15
  eor x29, x19, x29
  mov x15, x25
  mov x16, x26
  mov x17, x27
  mov x18, x28
  mov x19, x29
  bic x25, x22, x21
  eor x25, x20, x25
  bic x26, x23, x22
  eor x26, x21, x26
  bic x27, x24, x23
  eor x27, x22, x27
  bic x28, x20, x24
  eor x28, x23, x28
  bic x29, x21, x20
  eor x29, x24, x29
  mov x20, x25
  mov x21, x26
  mov x22, x27
  mov x23, x28
  mov x24, x29

  // Iota for all three states and loop control.
  ldr x30, [sp, #184]
  ldr x25, [x30], #8
  dup v31.2d, x25
  eor v0.16b, v0.16b, v31.16b
  eor x0, x0, x25
  adrp x26, .Lrscrypto_keccakf1600_aarch64_sha3_x3_rc_end
  add x26, x26, :lo12:.Lrscrypto_keccakf1600_aarch64_sha3_x3_rc_end
  cmp x30, x26
  b.ne .Lrscrypto_keccakf1600_aarch64_sha3_x3_round

  // Store A/B from NEON lane pairs.
  ldp x25, x26, [sp, #160]
  ldr x27, [sp, #176]
  str d0, [x25, #0]
  mov d31, v0.d[1]
  str d31, [x26, #0]
  str d1, [x25, #8]
  mov d31, v1.d[1]
  str d31, [x26, #8]
  str d2, [x25, #16]
  mov d31, v2.d[1]
  str d31, [x26, #16]
  str d3, [x25, #24]
  mov d31, v3.d[1]
  str d31, [x26, #24]
  str d4, [x25, #32]
  mov d31, v4.d[1]
  str d31, [x26, #32]
  str d5, [x25, #40]
  mov d31, v5.d[1]
  str d31, [x26, #40]
  str d6, [x25, #48]
  mov d31, v6.d[1]
  str d31, [x26, #48]
  str d7, [x25, #56]
  mov d31, v7.d[1]
  str d31, [x26, #56]
  str d8, [x25, #64]
  mov d31, v8.d[1]
  str d31, [x26, #64]
  str d9, [x25, #72]
  mov d31, v9.d[1]
  str d31, [x26, #72]
  str d10, [x25, #80]
  mov d31, v10.d[1]
  str d31, [x26, #80]
  str d11, [x25, #88]
  mov d31, v11.d[1]
  str d31, [x26, #88]
  str d12, [x25, #96]
  mov d31, v12.d[1]
  str d31, [x26, #96]
  str d13, [x25, #104]
  mov d31, v13.d[1]
  str d31, [x26, #104]
  str d14, [x25, #112]
  mov d31, v14.d[1]
  str d31, [x26, #112]
  str d15, [x25, #120]
  mov d31, v15.d[1]
  str d31, [x26, #120]
  str d16, [x25, #128]
  mov d31, v16.d[1]
  str d31, [x26, #128]
  str d17, [x25, #136]
  mov d31, v17.d[1]
  str d31, [x26, #136]
  str d18, [x25, #144]
  mov d31, v18.d[1]
  str d31, [x26, #144]
  str d19, [x25, #152]
  mov d31, v19.d[1]
  str d31, [x26, #152]
  str d20, [x25, #160]
  mov d31, v20.d[1]
  str d31, [x26, #160]
  str d21, [x25, #168]
  mov d31, v21.d[1]
  str d31, [x26, #168]
  str d22, [x25, #176]
  mov d31, v22.d[1]
  str d31, [x26, #176]
  str d23, [x25, #184]
  mov d31, v23.d[1]
  str d31, [x26, #184]
  str d24, [x25, #192]
  mov d31, v24.d[1]
  str d31, [x26, #192]

  // Store C scalar lanes.
  stp x0, x1, [x27, #0]
  stp x2, x3, [x27, #16]
  stp x4, x5, [x27, #32]
  stp x6, x7, [x27, #48]
  stp x8, x9, [x27, #64]
  stp x10, x11, [x27, #80]
  stp x12, x13, [x27, #96]
  stp x14, x15, [x27, #112]
  stp x16, x17, [x27, #128]
  stp x18, x19, [x27, #144]
  stp x20, x21, [x27, #160]
  stp x22, x23, [x27, #176]
  str x24, [x27, #192]

  ldp d8, d9, [sp, #96]
  ldp d10, d11, [sp, #112]
  ldp d12, d13, [sp, #128]
  ldp d14, d15, [sp, #144]
  ldp x29, x30, [sp, #80]
  ldp x27, x28, [sp, #64]
  ldp x25, x26, [sp, #48]
  ldp x23, x24, [sp, #32]
  ldp x21, x22, [sp, #16]
  ldp x19, x20, [sp, #0]
  add sp, sp, #192
  ret

.p2align 3
.Lrscrypto_keccakf1600_aarch64_sha3_x3_rc:
  .quad 0x0000000000000001
  .quad 0x0000000000008082
  .quad 0x800000000000808a
  .quad 0x8000000080008000
  .quad 0x000000000000808b
  .quad 0x0000000080000001
  .quad 0x8000000080008081
  .quad 0x8000000000008009
  .quad 0x000000000000008a
  .quad 0x0000000000000088
  .quad 0x0000000080008009
  .quad 0x000000008000000a
  .quad 0x000000008000808b
  .quad 0x800000000000008b
  .quad 0x8000000000008089
  .quad 0x8000000000008003
  .quad 0x8000000000008002
  .quad 0x8000000000000080
  .quad 0x000000000000800a
  .quad 0x800000008000000a
  .quad 0x8000000080008081
  .quad 0x8000000000008080
  .quad 0x0000000080000001
  .quad 0x8000000080008008
.Lrscrypto_keccakf1600_aarch64_sha3_x3_rc_end:

.size rscrypto_keccakf1600_aarch64_sha3_x3, .Lrscrypto_keccakf1600_aarch64_sha3_x3_rc_end-rscrypto_keccakf1600_aarch64_sha3_x3
