.text
.arch armv9.2-a+sve2+sve-sha3

.global rscrypto_keccakf1600_aarch64_sve2_sha3_x4
.type rscrypto_keccakf1600_aarch64_sve2_sha3_x4, %function
rscrypto_keccakf1600_aarch64_sve2_sha3_x4:
  cntd x4
  cmp x4, #4
  b.lo .Lsve2_sha3_x4_unavailable

  sub sp, sp, #96
  stp d8, d9, [sp]
  stp d10, d11, [sp, #16]
  stp d12, d13, [sp, #32]
  stp d14, d15, [sp, #48]
  stp x0, x1, [sp, #64]
  stp x2, x3, [sp, #80]

  ptrue p0.d, vl4
  add x6, sp, #64
  ld1d { z30.d }, p0/z, [x6]

  ld1d { z0.d }, p0/z, [z30.d, #0]
  ld1d { z1.d }, p0/z, [z30.d, #8]
  ld1d { z2.d }, p0/z, [z30.d, #16]
  ld1d { z3.d }, p0/z, [z30.d, #24]
  ld1d { z4.d }, p0/z, [z30.d, #32]
  ld1d { z5.d }, p0/z, [z30.d, #40]
  ld1d { z6.d }, p0/z, [z30.d, #48]
  ld1d { z7.d }, p0/z, [z30.d, #56]
  ld1d { z8.d }, p0/z, [z30.d, #64]
  ld1d { z9.d }, p0/z, [z30.d, #72]
  ld1d { z10.d }, p0/z, [z30.d, #80]
  ld1d { z11.d }, p0/z, [z30.d, #88]
  ld1d { z12.d }, p0/z, [z30.d, #96]
  ld1d { z13.d }, p0/z, [z30.d, #104]
  ld1d { z14.d }, p0/z, [z30.d, #112]
  ld1d { z15.d }, p0/z, [z30.d, #120]
  ld1d { z16.d }, p0/z, [z30.d, #128]
  ld1d { z17.d }, p0/z, [z30.d, #136]
  ld1d { z18.d }, p0/z, [z30.d, #144]
  ld1d { z19.d }, p0/z, [z30.d, #152]
  ld1d { z20.d }, p0/z, [z30.d, #160]
  ld1d { z21.d }, p0/z, [z30.d, #168]
  ld1d { z22.d }, p0/z, [z30.d, #176]
  ld1d { z23.d }, p0/z, [z30.d, #184]
  ld1d { z24.d }, p0/z, [z30.d, #192]

  adrp x4, .Lrscrypto_keccakf1600_sve2_sha3_rc
  add x4, x4, :lo12:.Lrscrypto_keccakf1600_sve2_sha3_rc
  mov x6, #24

.Lsve2_sha3_x4_round:
  mov z25.d, z0.d
  eor3 z25.d, z25.d, z5.d, z10.d
  eor3 z25.d, z25.d, z15.d, z20.d
  mov z26.d, z1.d
  eor3 z26.d, z26.d, z6.d, z11.d
  eor3 z26.d, z26.d, z16.d, z21.d
  mov z27.d, z2.d
  eor3 z27.d, z27.d, z7.d, z12.d
  eor3 z27.d, z27.d, z17.d, z22.d
  mov z28.d, z3.d
  eor3 z28.d, z28.d, z8.d, z13.d
  eor3 z28.d, z28.d, z18.d, z23.d
  mov z29.d, z4.d
  eor3 z29.d, z29.d, z9.d, z14.d
  eor3 z29.d, z29.d, z19.d, z24.d

  mov z30.d, z29.d
  rax1 z30.d, z30.d, z26.d
  xar z0.d, z0.d, z30.d, #64
  xar z5.d, z5.d, z30.d, #28
  xar z10.d, z10.d, z30.d, #61
  xar z15.d, z15.d, z30.d, #23
  xar z20.d, z20.d, z30.d, #46
  mov z30.d, z25.d
  rax1 z30.d, z30.d, z27.d
  xar z1.d, z1.d, z30.d, #63
  xar z6.d, z6.d, z30.d, #20
  xar z11.d, z11.d, z30.d, #54
  xar z16.d, z16.d, z30.d, #19
  xar z21.d, z21.d, z30.d, #62
  mov z30.d, z26.d
  rax1 z30.d, z30.d, z28.d
  xar z2.d, z2.d, z30.d, #2
  xar z7.d, z7.d, z30.d, #58
  xar z12.d, z12.d, z30.d, #21
  xar z17.d, z17.d, z30.d, #49
  xar z22.d, z22.d, z30.d, #3
  mov z30.d, z27.d
  rax1 z30.d, z30.d, z29.d
  xar z3.d, z3.d, z30.d, #36
  xar z8.d, z8.d, z30.d, #9
  xar z13.d, z13.d, z30.d, #39
  xar z18.d, z18.d, z30.d, #43
  xar z23.d, z23.d, z30.d, #8
  mov z30.d, z28.d
  rax1 z30.d, z30.d, z25.d
  xar z4.d, z4.d, z30.d, #37
  xar z9.d, z9.d, z30.d, #44
  xar z14.d, z14.d, z30.d, #25
  xar z19.d, z19.d, z30.d, #56
  xar z24.d, z24.d, z30.d, #50

  mov z25.d, z1.d
  mov z26.d, z10.d
  mov z10.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z7.d
  mov z7.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z11.d
  mov z11.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z17.d
  mov z17.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z18.d
  mov z18.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z3.d
  mov z3.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z5.d
  mov z5.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z16.d
  mov z16.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z8.d
  mov z8.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z21.d
  mov z21.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z24.d
  mov z24.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z4.d
  mov z4.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z15.d
  mov z15.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z23.d
  mov z23.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z19.d
  mov z19.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z13.d
  mov z13.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z12.d
  mov z12.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z2.d
  mov z2.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z20.d
  mov z20.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z14.d
  mov z14.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z22.d
  mov z22.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z9.d
  mov z9.d, z25.d
  mov z25.d, z26.d
  mov z26.d, z6.d
  mov z6.d, z25.d
  mov z25.d, z26.d
  mov z1.d, z25.d

  mov z25.d, z0.d
  bcax z25.d, z25.d, z2.d, z1.d
  mov z26.d, z1.d
  bcax z26.d, z26.d, z3.d, z2.d
  mov z27.d, z2.d
  bcax z27.d, z27.d, z4.d, z3.d
  mov z28.d, z3.d
  bcax z28.d, z28.d, z0.d, z4.d
  mov z29.d, z4.d
  bcax z29.d, z29.d, z1.d, z0.d
  mov z0.d, z25.d
  mov z1.d, z26.d
  mov z2.d, z27.d
  mov z3.d, z28.d
  mov z4.d, z29.d
  mov z25.d, z5.d
  bcax z25.d, z25.d, z7.d, z6.d
  mov z26.d, z6.d
  bcax z26.d, z26.d, z8.d, z7.d
  mov z27.d, z7.d
  bcax z27.d, z27.d, z9.d, z8.d
  mov z28.d, z8.d
  bcax z28.d, z28.d, z5.d, z9.d
  mov z29.d, z9.d
  bcax z29.d, z29.d, z6.d, z5.d
  mov z5.d, z25.d
  mov z6.d, z26.d
  mov z7.d, z27.d
  mov z8.d, z28.d
  mov z9.d, z29.d
  mov z25.d, z10.d
  bcax z25.d, z25.d, z12.d, z11.d
  mov z26.d, z11.d
  bcax z26.d, z26.d, z13.d, z12.d
  mov z27.d, z12.d
  bcax z27.d, z27.d, z14.d, z13.d
  mov z28.d, z13.d
  bcax z28.d, z28.d, z10.d, z14.d
  mov z29.d, z14.d
  bcax z29.d, z29.d, z11.d, z10.d
  mov z10.d, z25.d
  mov z11.d, z26.d
  mov z12.d, z27.d
  mov z13.d, z28.d
  mov z14.d, z29.d
  mov z25.d, z15.d
  bcax z25.d, z25.d, z17.d, z16.d
  mov z26.d, z16.d
  bcax z26.d, z26.d, z18.d, z17.d
  mov z27.d, z17.d
  bcax z27.d, z27.d, z19.d, z18.d
  mov z28.d, z18.d
  bcax z28.d, z28.d, z15.d, z19.d
  mov z29.d, z19.d
  bcax z29.d, z29.d, z16.d, z15.d
  mov z15.d, z25.d
  mov z16.d, z26.d
  mov z17.d, z27.d
  mov z18.d, z28.d
  mov z19.d, z29.d
  mov z25.d, z20.d
  bcax z25.d, z25.d, z22.d, z21.d
  mov z26.d, z21.d
  bcax z26.d, z26.d, z23.d, z22.d
  mov z27.d, z22.d
  bcax z27.d, z27.d, z24.d, z23.d
  mov z28.d, z23.d
  bcax z28.d, z28.d, z20.d, z24.d
  mov z29.d, z24.d
  bcax z29.d, z29.d, z21.d, z20.d
  mov z20.d, z25.d
  mov z21.d, z26.d
  mov z22.d, z27.d
  mov z23.d, z28.d
  mov z24.d, z29.d

  ldr x5, [x4], #8
  dup z25.d, x5
  eor z0.d, z0.d, z25.d

  subs x6, x6, #1
  b.ne .Lsve2_sha3_x4_round

  add x6, sp, #64
  ld1d { z30.d }, p0/z, [x6]
  st1d { z0.d }, p0, [z30.d, #0]
  st1d { z1.d }, p0, [z30.d, #8]
  st1d { z2.d }, p0, [z30.d, #16]
  st1d { z3.d }, p0, [z30.d, #24]
  st1d { z4.d }, p0, [z30.d, #32]
  st1d { z5.d }, p0, [z30.d, #40]
  st1d { z6.d }, p0, [z30.d, #48]
  st1d { z7.d }, p0, [z30.d, #56]
  st1d { z8.d }, p0, [z30.d, #64]
  st1d { z9.d }, p0, [z30.d, #72]
  st1d { z10.d }, p0, [z30.d, #80]
  st1d { z11.d }, p0, [z30.d, #88]
  st1d { z12.d }, p0, [z30.d, #96]
  st1d { z13.d }, p0, [z30.d, #104]
  st1d { z14.d }, p0, [z30.d, #112]
  st1d { z15.d }, p0, [z30.d, #120]
  st1d { z16.d }, p0, [z30.d, #128]
  st1d { z17.d }, p0, [z30.d, #136]
  st1d { z18.d }, p0, [z30.d, #144]
  st1d { z19.d }, p0, [z30.d, #152]
  st1d { z20.d }, p0, [z30.d, #160]
  st1d { z21.d }, p0, [z30.d, #168]
  st1d { z22.d }, p0, [z30.d, #176]
  st1d { z23.d }, p0, [z30.d, #184]
  st1d { z24.d }, p0, [z30.d, #192]

  ldp d8, d9, [sp]
  ldp d10, d11, [sp, #16]
  ldp d12, d13, [sp, #32]
  ldp d14, d15, [sp, #48]
  add sp, sp, #96
  mov w0, #1
  ret

.Lsve2_sha3_x4_unavailable:
  mov w0, #0
  ret

.p2align 3
.Lrscrypto_keccakf1600_sve2_sha3_rc:
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

.size rscrypto_keccakf1600_aarch64_sve2_sha3_x4, .-rscrypto_keccakf1600_aarch64_sve2_sha3_x4
