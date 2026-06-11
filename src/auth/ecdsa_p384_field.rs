//! Fiat-Crypto P-384 field arithmetic used by ECDSA verification.
//!
//! This file contains a narrow extraction of generated Fiat-Crypto code
//! from the RustCrypto `p384` crate field backend. The generated source
//! is distributed under MIT, Apache-2.0, or BSD-1-Clause terms by the
//! Fiat-Crypto authors. It is kept private to this module and wrapped by
//! small `mul`/`square` helpers below.

#![allow(
  non_camel_case_types,
  clippy::identity_op,
  clippy::too_many_arguments,
  clippy::unnecessary_cast,
  clippy::unused_unit,
  unused_assignments,
  unused_variables,
  dead_code,
  unused_parens
)]

type fiat_p384_u1 = u8;
type fiat_p384_i1 = i8;
type fiat_p384_u2 = u8;
type fiat_p384_i2 = i8;
type fiat_p384_montgomery_domain_field_element = [u64; 6];

const fn fiat_p384_addcarryx_u64(arg1: fiat_p384_u1, arg2: u64, arg3: u64) -> (u64, fiat_p384_u1) {
  let mut out1: u64 = 0;
  let mut out2: fiat_p384_u1 = 0;
  let x1: u128 = (((arg1 as u128) + (arg2 as u128)) + (arg3 as u128));
  let x2: u64 = ((x1 & (0xffffffffffffffff as u128)) as u64);
  let x3: fiat_p384_u1 = ((x1 >> 64) as fiat_p384_u1);
  out1 = x2;
  out2 = x3;
  (out1, out2)
}

const fn fiat_p384_subborrowx_u64(arg1: fiat_p384_u1, arg2: u64, arg3: u64) -> (u64, fiat_p384_u1) {
  let mut out1: u64 = 0;
  let mut out2: fiat_p384_u1 = 0;
  let x1: i128 = (((arg2 as i128) - (arg1 as i128)) - (arg3 as i128));
  let x2: fiat_p384_i1 = ((x1 >> 64) as fiat_p384_i1);
  let x3: u64 = ((x1 & (0xffffffffffffffff as i128)) as u64);
  out1 = x3;
  out2 = (((0x0 as fiat_p384_i2) - (x2 as fiat_p384_i2)) as fiat_p384_u1);
  (out1, out2)
}

const fn fiat_p384_mulx_u64(arg1: u64, arg2: u64) -> (u64, u64) {
  let mut out1: u64 = 0;
  let mut out2: u64 = 0;
  let x1: u128 = ((arg1 as u128) * (arg2 as u128));
  let x2: u64 = ((x1 & (0xffffffffffffffff as u128)) as u64);
  let x3: u64 = ((x1 >> 64) as u64);
  out1 = x2;
  out2 = x3;
  (out1, out2)
}

const fn fiat_p384_cmovznz_u64(arg1: fiat_p384_u1, arg2: u64, arg3: u64) -> u64 {
  let mut out1: u64 = 0;
  let x1: fiat_p384_u1 = (!(!arg1));
  let x2: u64 = ((((((0x0 as fiat_p384_i2) - (x1 as fiat_p384_i2)) as fiat_p384_i1) as i128)
    & (0xffffffffffffffff as i128)) as u64);
  let x3: u64 = ((x2 & arg3) | ((!x2) & arg2));
  out1 = x3;
  out1
}

const fn fiat_p384_mul(
  arg1: &fiat_p384_montgomery_domain_field_element,
  arg2: &fiat_p384_montgomery_domain_field_element,
) -> fiat_p384_montgomery_domain_field_element {
  let mut out1: fiat_p384_montgomery_domain_field_element = [0; 6];
  let x1: u64 = (arg1[1]);
  let x2: u64 = (arg1[2]);
  let x3: u64 = (arg1[3]);
  let x4: u64 = (arg1[4]);
  let x5: u64 = (arg1[5]);
  let x6: u64 = (arg1[0]);
  let (x7, x8) = fiat_p384_mulx_u64(x6, (arg2[5]));
  let (x9, x10) = fiat_p384_mulx_u64(x6, (arg2[4]));
  let (x11, x12) = fiat_p384_mulx_u64(x6, (arg2[3]));
  let (x13, x14) = fiat_p384_mulx_u64(x6, (arg2[2]));
  let (x15, x16) = fiat_p384_mulx_u64(x6, (arg2[1]));
  let (x17, x18) = fiat_p384_mulx_u64(x6, (arg2[0]));
  let (x19, x20) = fiat_p384_addcarryx_u64(0x0, x18, x15);
  let (x21, x22) = fiat_p384_addcarryx_u64(x20, x16, x13);
  let (x23, x24) = fiat_p384_addcarryx_u64(x22, x14, x11);
  let (x25, x26) = fiat_p384_addcarryx_u64(x24, x12, x9);
  let (x27, x28) = fiat_p384_addcarryx_u64(x26, x10, x7);
  let x29: u64 = ((x28 as u64) + x8);
  let (x30, x31) = fiat_p384_mulx_u64(x17, 0x100000001);
  let (x32, x33) = fiat_p384_mulx_u64(x30, 0xffffffffffffffff);
  let (x34, x35) = fiat_p384_mulx_u64(x30, 0xffffffffffffffff);
  let (x36, x37) = fiat_p384_mulx_u64(x30, 0xffffffffffffffff);
  let (x38, x39) = fiat_p384_mulx_u64(x30, 0xfffffffffffffffe);
  let (x40, x41) = fiat_p384_mulx_u64(x30, 0xffffffff00000000);
  let (x42, x43) = fiat_p384_mulx_u64(x30, 0xffffffff);
  let (x44, x45) = fiat_p384_addcarryx_u64(0x0, x43, x40);
  let (x46, x47) = fiat_p384_addcarryx_u64(x45, x41, x38);
  let (x48, x49) = fiat_p384_addcarryx_u64(x47, x39, x36);
  let (x50, x51) = fiat_p384_addcarryx_u64(x49, x37, x34);
  let (x52, x53) = fiat_p384_addcarryx_u64(x51, x35, x32);
  let x54: u64 = ((x53 as u64) + x33);
  let (x55, x56) = fiat_p384_addcarryx_u64(0x0, x17, x42);
  let (x57, x58) = fiat_p384_addcarryx_u64(x56, x19, x44);
  let (x59, x60) = fiat_p384_addcarryx_u64(x58, x21, x46);
  let (x61, x62) = fiat_p384_addcarryx_u64(x60, x23, x48);
  let (x63, x64) = fiat_p384_addcarryx_u64(x62, x25, x50);
  let (x65, x66) = fiat_p384_addcarryx_u64(x64, x27, x52);
  let (x67, x68) = fiat_p384_addcarryx_u64(x66, x29, x54);
  let (x69, x70) = fiat_p384_mulx_u64(x1, (arg2[5]));
  let (x71, x72) = fiat_p384_mulx_u64(x1, (arg2[4]));
  let (x73, x74) = fiat_p384_mulx_u64(x1, (arg2[3]));
  let (x75, x76) = fiat_p384_mulx_u64(x1, (arg2[2]));
  let (x77, x78) = fiat_p384_mulx_u64(x1, (arg2[1]));
  let (x79, x80) = fiat_p384_mulx_u64(x1, (arg2[0]));
  let (x81, x82) = fiat_p384_addcarryx_u64(0x0, x80, x77);
  let (x83, x84) = fiat_p384_addcarryx_u64(x82, x78, x75);
  let (x85, x86) = fiat_p384_addcarryx_u64(x84, x76, x73);
  let (x87, x88) = fiat_p384_addcarryx_u64(x86, x74, x71);
  let (x89, x90) = fiat_p384_addcarryx_u64(x88, x72, x69);
  let x91: u64 = ((x90 as u64) + x70);
  let (x92, x93) = fiat_p384_addcarryx_u64(0x0, x57, x79);
  let (x94, x95) = fiat_p384_addcarryx_u64(x93, x59, x81);
  let (x96, x97) = fiat_p384_addcarryx_u64(x95, x61, x83);
  let (x98, x99) = fiat_p384_addcarryx_u64(x97, x63, x85);
  let (x100, x101) = fiat_p384_addcarryx_u64(x99, x65, x87);
  let (x102, x103) = fiat_p384_addcarryx_u64(x101, x67, x89);
  let (x104, x105) = fiat_p384_addcarryx_u64(x103, (x68 as u64), x91);
  let (x106, x107) = fiat_p384_mulx_u64(x92, 0x100000001);
  let (x108, x109) = fiat_p384_mulx_u64(x106, 0xffffffffffffffff);
  let (x110, x111) = fiat_p384_mulx_u64(x106, 0xffffffffffffffff);
  let (x112, x113) = fiat_p384_mulx_u64(x106, 0xffffffffffffffff);
  let (x114, x115) = fiat_p384_mulx_u64(x106, 0xfffffffffffffffe);
  let (x116, x117) = fiat_p384_mulx_u64(x106, 0xffffffff00000000);
  let (x118, x119) = fiat_p384_mulx_u64(x106, 0xffffffff);
  let (x120, x121) = fiat_p384_addcarryx_u64(0x0, x119, x116);
  let (x122, x123) = fiat_p384_addcarryx_u64(x121, x117, x114);
  let (x124, x125) = fiat_p384_addcarryx_u64(x123, x115, x112);
  let (x126, x127) = fiat_p384_addcarryx_u64(x125, x113, x110);
  let (x128, x129) = fiat_p384_addcarryx_u64(x127, x111, x108);
  let x130: u64 = ((x129 as u64) + x109);
  let (x131, x132) = fiat_p384_addcarryx_u64(0x0, x92, x118);
  let (x133, x134) = fiat_p384_addcarryx_u64(x132, x94, x120);
  let (x135, x136) = fiat_p384_addcarryx_u64(x134, x96, x122);
  let (x137, x138) = fiat_p384_addcarryx_u64(x136, x98, x124);
  let (x139, x140) = fiat_p384_addcarryx_u64(x138, x100, x126);
  let (x141, x142) = fiat_p384_addcarryx_u64(x140, x102, x128);
  let (x143, x144) = fiat_p384_addcarryx_u64(x142, x104, x130);
  let x145: u64 = ((x144 as u64) + (x105 as u64));
  let (x146, x147) = fiat_p384_mulx_u64(x2, (arg2[5]));
  let (x148, x149) = fiat_p384_mulx_u64(x2, (arg2[4]));
  let (x150, x151) = fiat_p384_mulx_u64(x2, (arg2[3]));
  let (x152, x153) = fiat_p384_mulx_u64(x2, (arg2[2]));
  let (x154, x155) = fiat_p384_mulx_u64(x2, (arg2[1]));
  let (x156, x157) = fiat_p384_mulx_u64(x2, (arg2[0]));
  let (x158, x159) = fiat_p384_addcarryx_u64(0x0, x157, x154);
  let (x160, x161) = fiat_p384_addcarryx_u64(x159, x155, x152);
  let (x162, x163) = fiat_p384_addcarryx_u64(x161, x153, x150);
  let (x164, x165) = fiat_p384_addcarryx_u64(x163, x151, x148);
  let (x166, x167) = fiat_p384_addcarryx_u64(x165, x149, x146);
  let x168: u64 = ((x167 as u64) + x147);
  let (x169, x170) = fiat_p384_addcarryx_u64(0x0, x133, x156);
  let (x171, x172) = fiat_p384_addcarryx_u64(x170, x135, x158);
  let (x173, x174) = fiat_p384_addcarryx_u64(x172, x137, x160);
  let (x175, x176) = fiat_p384_addcarryx_u64(x174, x139, x162);
  let (x177, x178) = fiat_p384_addcarryx_u64(x176, x141, x164);
  let (x179, x180) = fiat_p384_addcarryx_u64(x178, x143, x166);
  let (x181, x182) = fiat_p384_addcarryx_u64(x180, x145, x168);
  let (x183, x184) = fiat_p384_mulx_u64(x169, 0x100000001);
  let (x185, x186) = fiat_p384_mulx_u64(x183, 0xffffffffffffffff);
  let (x187, x188) = fiat_p384_mulx_u64(x183, 0xffffffffffffffff);
  let (x189, x190) = fiat_p384_mulx_u64(x183, 0xffffffffffffffff);
  let (x191, x192) = fiat_p384_mulx_u64(x183, 0xfffffffffffffffe);
  let (x193, x194) = fiat_p384_mulx_u64(x183, 0xffffffff00000000);
  let (x195, x196) = fiat_p384_mulx_u64(x183, 0xffffffff);
  let (x197, x198) = fiat_p384_addcarryx_u64(0x0, x196, x193);
  let (x199, x200) = fiat_p384_addcarryx_u64(x198, x194, x191);
  let (x201, x202) = fiat_p384_addcarryx_u64(x200, x192, x189);
  let (x203, x204) = fiat_p384_addcarryx_u64(x202, x190, x187);
  let (x205, x206) = fiat_p384_addcarryx_u64(x204, x188, x185);
  let x207: u64 = ((x206 as u64) + x186);
  let (x208, x209) = fiat_p384_addcarryx_u64(0x0, x169, x195);
  let (x210, x211) = fiat_p384_addcarryx_u64(x209, x171, x197);
  let (x212, x213) = fiat_p384_addcarryx_u64(x211, x173, x199);
  let (x214, x215) = fiat_p384_addcarryx_u64(x213, x175, x201);
  let (x216, x217) = fiat_p384_addcarryx_u64(x215, x177, x203);
  let (x218, x219) = fiat_p384_addcarryx_u64(x217, x179, x205);
  let (x220, x221) = fiat_p384_addcarryx_u64(x219, x181, x207);
  let x222: u64 = ((x221 as u64) + (x182 as u64));
  let (x223, x224) = fiat_p384_mulx_u64(x3, (arg2[5]));
  let (x225, x226) = fiat_p384_mulx_u64(x3, (arg2[4]));
  let (x227, x228) = fiat_p384_mulx_u64(x3, (arg2[3]));
  let (x229, x230) = fiat_p384_mulx_u64(x3, (arg2[2]));
  let (x231, x232) = fiat_p384_mulx_u64(x3, (arg2[1]));
  let (x233, x234) = fiat_p384_mulx_u64(x3, (arg2[0]));
  let (x235, x236) = fiat_p384_addcarryx_u64(0x0, x234, x231);
  let (x237, x238) = fiat_p384_addcarryx_u64(x236, x232, x229);
  let (x239, x240) = fiat_p384_addcarryx_u64(x238, x230, x227);
  let (x241, x242) = fiat_p384_addcarryx_u64(x240, x228, x225);
  let (x243, x244) = fiat_p384_addcarryx_u64(x242, x226, x223);
  let x245: u64 = ((x244 as u64) + x224);
  let (x246, x247) = fiat_p384_addcarryx_u64(0x0, x210, x233);
  let (x248, x249) = fiat_p384_addcarryx_u64(x247, x212, x235);
  let (x250, x251) = fiat_p384_addcarryx_u64(x249, x214, x237);
  let (x252, x253) = fiat_p384_addcarryx_u64(x251, x216, x239);
  let (x254, x255) = fiat_p384_addcarryx_u64(x253, x218, x241);
  let (x256, x257) = fiat_p384_addcarryx_u64(x255, x220, x243);
  let (x258, x259) = fiat_p384_addcarryx_u64(x257, x222, x245);
  let (x260, x261) = fiat_p384_mulx_u64(x246, 0x100000001);
  let (x262, x263) = fiat_p384_mulx_u64(x260, 0xffffffffffffffff);
  let (x264, x265) = fiat_p384_mulx_u64(x260, 0xffffffffffffffff);
  let (x266, x267) = fiat_p384_mulx_u64(x260, 0xffffffffffffffff);
  let (x268, x269) = fiat_p384_mulx_u64(x260, 0xfffffffffffffffe);
  let (x270, x271) = fiat_p384_mulx_u64(x260, 0xffffffff00000000);
  let (x272, x273) = fiat_p384_mulx_u64(x260, 0xffffffff);
  let (x274, x275) = fiat_p384_addcarryx_u64(0x0, x273, x270);
  let (x276, x277) = fiat_p384_addcarryx_u64(x275, x271, x268);
  let (x278, x279) = fiat_p384_addcarryx_u64(x277, x269, x266);
  let (x280, x281) = fiat_p384_addcarryx_u64(x279, x267, x264);
  let (x282, x283) = fiat_p384_addcarryx_u64(x281, x265, x262);
  let x284: u64 = ((x283 as u64) + x263);
  let (x285, x286) = fiat_p384_addcarryx_u64(0x0, x246, x272);
  let (x287, x288) = fiat_p384_addcarryx_u64(x286, x248, x274);
  let (x289, x290) = fiat_p384_addcarryx_u64(x288, x250, x276);
  let (x291, x292) = fiat_p384_addcarryx_u64(x290, x252, x278);
  let (x293, x294) = fiat_p384_addcarryx_u64(x292, x254, x280);
  let (x295, x296) = fiat_p384_addcarryx_u64(x294, x256, x282);
  let (x297, x298) = fiat_p384_addcarryx_u64(x296, x258, x284);
  let x299: u64 = ((x298 as u64) + (x259 as u64));
  let (x300, x301) = fiat_p384_mulx_u64(x4, (arg2[5]));
  let (x302, x303) = fiat_p384_mulx_u64(x4, (arg2[4]));
  let (x304, x305) = fiat_p384_mulx_u64(x4, (arg2[3]));
  let (x306, x307) = fiat_p384_mulx_u64(x4, (arg2[2]));
  let (x308, x309) = fiat_p384_mulx_u64(x4, (arg2[1]));
  let (x310, x311) = fiat_p384_mulx_u64(x4, (arg2[0]));
  let (x312, x313) = fiat_p384_addcarryx_u64(0x0, x311, x308);
  let (x314, x315) = fiat_p384_addcarryx_u64(x313, x309, x306);
  let (x316, x317) = fiat_p384_addcarryx_u64(x315, x307, x304);
  let (x318, x319) = fiat_p384_addcarryx_u64(x317, x305, x302);
  let (x320, x321) = fiat_p384_addcarryx_u64(x319, x303, x300);
  let x322: u64 = ((x321 as u64) + x301);
  let (x323, x324) = fiat_p384_addcarryx_u64(0x0, x287, x310);
  let (x325, x326) = fiat_p384_addcarryx_u64(x324, x289, x312);
  let (x327, x328) = fiat_p384_addcarryx_u64(x326, x291, x314);
  let (x329, x330) = fiat_p384_addcarryx_u64(x328, x293, x316);
  let (x331, x332) = fiat_p384_addcarryx_u64(x330, x295, x318);
  let (x333, x334) = fiat_p384_addcarryx_u64(x332, x297, x320);
  let (x335, x336) = fiat_p384_addcarryx_u64(x334, x299, x322);
  let (x337, x338) = fiat_p384_mulx_u64(x323, 0x100000001);
  let (x339, x340) = fiat_p384_mulx_u64(x337, 0xffffffffffffffff);
  let (x341, x342) = fiat_p384_mulx_u64(x337, 0xffffffffffffffff);
  let (x343, x344) = fiat_p384_mulx_u64(x337, 0xffffffffffffffff);
  let (x345, x346) = fiat_p384_mulx_u64(x337, 0xfffffffffffffffe);
  let (x347, x348) = fiat_p384_mulx_u64(x337, 0xffffffff00000000);
  let (x349, x350) = fiat_p384_mulx_u64(x337, 0xffffffff);
  let (x351, x352) = fiat_p384_addcarryx_u64(0x0, x350, x347);
  let (x353, x354) = fiat_p384_addcarryx_u64(x352, x348, x345);
  let (x355, x356) = fiat_p384_addcarryx_u64(x354, x346, x343);
  let (x357, x358) = fiat_p384_addcarryx_u64(x356, x344, x341);
  let (x359, x360) = fiat_p384_addcarryx_u64(x358, x342, x339);
  let x361: u64 = ((x360 as u64) + x340);
  let (x362, x363) = fiat_p384_addcarryx_u64(0x0, x323, x349);
  let (x364, x365) = fiat_p384_addcarryx_u64(x363, x325, x351);
  let (x366, x367) = fiat_p384_addcarryx_u64(x365, x327, x353);
  let (x368, x369) = fiat_p384_addcarryx_u64(x367, x329, x355);
  let (x370, x371) = fiat_p384_addcarryx_u64(x369, x331, x357);
  let (x372, x373) = fiat_p384_addcarryx_u64(x371, x333, x359);
  let (x374, x375) = fiat_p384_addcarryx_u64(x373, x335, x361);
  let x376: u64 = ((x375 as u64) + (x336 as u64));
  let (x377, x378) = fiat_p384_mulx_u64(x5, (arg2[5]));
  let (x379, x380) = fiat_p384_mulx_u64(x5, (arg2[4]));
  let (x381, x382) = fiat_p384_mulx_u64(x5, (arg2[3]));
  let (x383, x384) = fiat_p384_mulx_u64(x5, (arg2[2]));
  let (x385, x386) = fiat_p384_mulx_u64(x5, (arg2[1]));
  let (x387, x388) = fiat_p384_mulx_u64(x5, (arg2[0]));
  let (x389, x390) = fiat_p384_addcarryx_u64(0x0, x388, x385);
  let (x391, x392) = fiat_p384_addcarryx_u64(x390, x386, x383);
  let (x393, x394) = fiat_p384_addcarryx_u64(x392, x384, x381);
  let (x395, x396) = fiat_p384_addcarryx_u64(x394, x382, x379);
  let (x397, x398) = fiat_p384_addcarryx_u64(x396, x380, x377);
  let x399: u64 = ((x398 as u64) + x378);
  let (x400, x401) = fiat_p384_addcarryx_u64(0x0, x364, x387);
  let (x402, x403) = fiat_p384_addcarryx_u64(x401, x366, x389);
  let (x404, x405) = fiat_p384_addcarryx_u64(x403, x368, x391);
  let (x406, x407) = fiat_p384_addcarryx_u64(x405, x370, x393);
  let (x408, x409) = fiat_p384_addcarryx_u64(x407, x372, x395);
  let (x410, x411) = fiat_p384_addcarryx_u64(x409, x374, x397);
  let (x412, x413) = fiat_p384_addcarryx_u64(x411, x376, x399);
  let (x414, x415) = fiat_p384_mulx_u64(x400, 0x100000001);
  let (x416, x417) = fiat_p384_mulx_u64(x414, 0xffffffffffffffff);
  let (x418, x419) = fiat_p384_mulx_u64(x414, 0xffffffffffffffff);
  let (x420, x421) = fiat_p384_mulx_u64(x414, 0xffffffffffffffff);
  let (x422, x423) = fiat_p384_mulx_u64(x414, 0xfffffffffffffffe);
  let (x424, x425) = fiat_p384_mulx_u64(x414, 0xffffffff00000000);
  let (x426, x427) = fiat_p384_mulx_u64(x414, 0xffffffff);
  let (x428, x429) = fiat_p384_addcarryx_u64(0x0, x427, x424);
  let (x430, x431) = fiat_p384_addcarryx_u64(x429, x425, x422);
  let (x432, x433) = fiat_p384_addcarryx_u64(x431, x423, x420);
  let (x434, x435) = fiat_p384_addcarryx_u64(x433, x421, x418);
  let (x436, x437) = fiat_p384_addcarryx_u64(x435, x419, x416);
  let x438: u64 = ((x437 as u64) + x417);
  let (x439, x440) = fiat_p384_addcarryx_u64(0x0, x400, x426);
  let (x441, x442) = fiat_p384_addcarryx_u64(x440, x402, x428);
  let (x443, x444) = fiat_p384_addcarryx_u64(x442, x404, x430);
  let (x445, x446) = fiat_p384_addcarryx_u64(x444, x406, x432);
  let (x447, x448) = fiat_p384_addcarryx_u64(x446, x408, x434);
  let (x449, x450) = fiat_p384_addcarryx_u64(x448, x410, x436);
  let (x451, x452) = fiat_p384_addcarryx_u64(x450, x412, x438);
  let x453: u64 = ((x452 as u64) + (x413 as u64));
  let (x454, x455) = fiat_p384_subborrowx_u64(0x0, x441, 0xffffffff);
  let (x456, x457) = fiat_p384_subborrowx_u64(x455, x443, 0xffffffff00000000);
  let (x458, x459) = fiat_p384_subborrowx_u64(x457, x445, 0xfffffffffffffffe);
  let (x460, x461) = fiat_p384_subborrowx_u64(x459, x447, 0xffffffffffffffff);
  let (x462, x463) = fiat_p384_subborrowx_u64(x461, x449, 0xffffffffffffffff);
  let (x464, x465) = fiat_p384_subborrowx_u64(x463, x451, 0xffffffffffffffff);
  let (x466, x467) = fiat_p384_subborrowx_u64(x465, x453, (0x0 as u64));
  let (x468) = fiat_p384_cmovznz_u64(x467, x454, x441);
  let (x469) = fiat_p384_cmovznz_u64(x467, x456, x443);
  let (x470) = fiat_p384_cmovznz_u64(x467, x458, x445);
  let (x471) = fiat_p384_cmovznz_u64(x467, x460, x447);
  let (x472) = fiat_p384_cmovznz_u64(x467, x462, x449);
  let (x473) = fiat_p384_cmovznz_u64(x467, x464, x451);
  out1[0] = x468;
  out1[1] = x469;
  out1[2] = x470;
  out1[3] = x471;
  out1[4] = x472;
  out1[5] = x473;
  out1
}

const fn fiat_p384_square(
  arg1: &fiat_p384_montgomery_domain_field_element,
) -> fiat_p384_montgomery_domain_field_element {
  let mut out1: fiat_p384_montgomery_domain_field_element = [0; 6];
  let x1: u64 = (arg1[1]);
  let x2: u64 = (arg1[2]);
  let x3: u64 = (arg1[3]);
  let x4: u64 = (arg1[4]);
  let x5: u64 = (arg1[5]);
  let x6: u64 = (arg1[0]);
  let (x7, x8) = fiat_p384_mulx_u64(x6, (arg1[5]));
  let (x9, x10) = fiat_p384_mulx_u64(x6, (arg1[4]));
  let (x11, x12) = fiat_p384_mulx_u64(x6, (arg1[3]));
  let (x13, x14) = fiat_p384_mulx_u64(x6, (arg1[2]));
  let (x15, x16) = fiat_p384_mulx_u64(x6, (arg1[1]));
  let (x17, x18) = fiat_p384_mulx_u64(x6, (arg1[0]));
  let (x19, x20) = fiat_p384_addcarryx_u64(0x0, x18, x15);
  let (x21, x22) = fiat_p384_addcarryx_u64(x20, x16, x13);
  let (x23, x24) = fiat_p384_addcarryx_u64(x22, x14, x11);
  let (x25, x26) = fiat_p384_addcarryx_u64(x24, x12, x9);
  let (x27, x28) = fiat_p384_addcarryx_u64(x26, x10, x7);
  let x29: u64 = ((x28 as u64) + x8);
  let (x30, x31) = fiat_p384_mulx_u64(x17, 0x100000001);
  let (x32, x33) = fiat_p384_mulx_u64(x30, 0xffffffffffffffff);
  let (x34, x35) = fiat_p384_mulx_u64(x30, 0xffffffffffffffff);
  let (x36, x37) = fiat_p384_mulx_u64(x30, 0xffffffffffffffff);
  let (x38, x39) = fiat_p384_mulx_u64(x30, 0xfffffffffffffffe);
  let (x40, x41) = fiat_p384_mulx_u64(x30, 0xffffffff00000000);
  let (x42, x43) = fiat_p384_mulx_u64(x30, 0xffffffff);
  let (x44, x45) = fiat_p384_addcarryx_u64(0x0, x43, x40);
  let (x46, x47) = fiat_p384_addcarryx_u64(x45, x41, x38);
  let (x48, x49) = fiat_p384_addcarryx_u64(x47, x39, x36);
  let (x50, x51) = fiat_p384_addcarryx_u64(x49, x37, x34);
  let (x52, x53) = fiat_p384_addcarryx_u64(x51, x35, x32);
  let x54: u64 = ((x53 as u64) + x33);
  let (x55, x56) = fiat_p384_addcarryx_u64(0x0, x17, x42);
  let (x57, x58) = fiat_p384_addcarryx_u64(x56, x19, x44);
  let (x59, x60) = fiat_p384_addcarryx_u64(x58, x21, x46);
  let (x61, x62) = fiat_p384_addcarryx_u64(x60, x23, x48);
  let (x63, x64) = fiat_p384_addcarryx_u64(x62, x25, x50);
  let (x65, x66) = fiat_p384_addcarryx_u64(x64, x27, x52);
  let (x67, x68) = fiat_p384_addcarryx_u64(x66, x29, x54);
  let (x69, x70) = fiat_p384_mulx_u64(x1, (arg1[5]));
  let (x71, x72) = fiat_p384_mulx_u64(x1, (arg1[4]));
  let (x73, x74) = fiat_p384_mulx_u64(x1, (arg1[3]));
  let (x75, x76) = fiat_p384_mulx_u64(x1, (arg1[2]));
  let (x77, x78) = fiat_p384_mulx_u64(x1, (arg1[1]));
  let (x79, x80) = fiat_p384_mulx_u64(x1, (arg1[0]));
  let (x81, x82) = fiat_p384_addcarryx_u64(0x0, x80, x77);
  let (x83, x84) = fiat_p384_addcarryx_u64(x82, x78, x75);
  let (x85, x86) = fiat_p384_addcarryx_u64(x84, x76, x73);
  let (x87, x88) = fiat_p384_addcarryx_u64(x86, x74, x71);
  let (x89, x90) = fiat_p384_addcarryx_u64(x88, x72, x69);
  let x91: u64 = ((x90 as u64) + x70);
  let (x92, x93) = fiat_p384_addcarryx_u64(0x0, x57, x79);
  let (x94, x95) = fiat_p384_addcarryx_u64(x93, x59, x81);
  let (x96, x97) = fiat_p384_addcarryx_u64(x95, x61, x83);
  let (x98, x99) = fiat_p384_addcarryx_u64(x97, x63, x85);
  let (x100, x101) = fiat_p384_addcarryx_u64(x99, x65, x87);
  let (x102, x103) = fiat_p384_addcarryx_u64(x101, x67, x89);
  let (x104, x105) = fiat_p384_addcarryx_u64(x103, (x68 as u64), x91);
  let (x106, x107) = fiat_p384_mulx_u64(x92, 0x100000001);
  let (x108, x109) = fiat_p384_mulx_u64(x106, 0xffffffffffffffff);
  let (x110, x111) = fiat_p384_mulx_u64(x106, 0xffffffffffffffff);
  let (x112, x113) = fiat_p384_mulx_u64(x106, 0xffffffffffffffff);
  let (x114, x115) = fiat_p384_mulx_u64(x106, 0xfffffffffffffffe);
  let (x116, x117) = fiat_p384_mulx_u64(x106, 0xffffffff00000000);
  let (x118, x119) = fiat_p384_mulx_u64(x106, 0xffffffff);
  let (x120, x121) = fiat_p384_addcarryx_u64(0x0, x119, x116);
  let (x122, x123) = fiat_p384_addcarryx_u64(x121, x117, x114);
  let (x124, x125) = fiat_p384_addcarryx_u64(x123, x115, x112);
  let (x126, x127) = fiat_p384_addcarryx_u64(x125, x113, x110);
  let (x128, x129) = fiat_p384_addcarryx_u64(x127, x111, x108);
  let x130: u64 = ((x129 as u64) + x109);
  let (x131, x132) = fiat_p384_addcarryx_u64(0x0, x92, x118);
  let (x133, x134) = fiat_p384_addcarryx_u64(x132, x94, x120);
  let (x135, x136) = fiat_p384_addcarryx_u64(x134, x96, x122);
  let (x137, x138) = fiat_p384_addcarryx_u64(x136, x98, x124);
  let (x139, x140) = fiat_p384_addcarryx_u64(x138, x100, x126);
  let (x141, x142) = fiat_p384_addcarryx_u64(x140, x102, x128);
  let (x143, x144) = fiat_p384_addcarryx_u64(x142, x104, x130);
  let x145: u64 = ((x144 as u64) + (x105 as u64));
  let (x146, x147) = fiat_p384_mulx_u64(x2, (arg1[5]));
  let (x148, x149) = fiat_p384_mulx_u64(x2, (arg1[4]));
  let (x150, x151) = fiat_p384_mulx_u64(x2, (arg1[3]));
  let (x152, x153) = fiat_p384_mulx_u64(x2, (arg1[2]));
  let (x154, x155) = fiat_p384_mulx_u64(x2, (arg1[1]));
  let (x156, x157) = fiat_p384_mulx_u64(x2, (arg1[0]));
  let (x158, x159) = fiat_p384_addcarryx_u64(0x0, x157, x154);
  let (x160, x161) = fiat_p384_addcarryx_u64(x159, x155, x152);
  let (x162, x163) = fiat_p384_addcarryx_u64(x161, x153, x150);
  let (x164, x165) = fiat_p384_addcarryx_u64(x163, x151, x148);
  let (x166, x167) = fiat_p384_addcarryx_u64(x165, x149, x146);
  let x168: u64 = ((x167 as u64) + x147);
  let (x169, x170) = fiat_p384_addcarryx_u64(0x0, x133, x156);
  let (x171, x172) = fiat_p384_addcarryx_u64(x170, x135, x158);
  let (x173, x174) = fiat_p384_addcarryx_u64(x172, x137, x160);
  let (x175, x176) = fiat_p384_addcarryx_u64(x174, x139, x162);
  let (x177, x178) = fiat_p384_addcarryx_u64(x176, x141, x164);
  let (x179, x180) = fiat_p384_addcarryx_u64(x178, x143, x166);
  let (x181, x182) = fiat_p384_addcarryx_u64(x180, x145, x168);
  let (x183, x184) = fiat_p384_mulx_u64(x169, 0x100000001);
  let (x185, x186) = fiat_p384_mulx_u64(x183, 0xffffffffffffffff);
  let (x187, x188) = fiat_p384_mulx_u64(x183, 0xffffffffffffffff);
  let (x189, x190) = fiat_p384_mulx_u64(x183, 0xffffffffffffffff);
  let (x191, x192) = fiat_p384_mulx_u64(x183, 0xfffffffffffffffe);
  let (x193, x194) = fiat_p384_mulx_u64(x183, 0xffffffff00000000);
  let (x195, x196) = fiat_p384_mulx_u64(x183, 0xffffffff);
  let (x197, x198) = fiat_p384_addcarryx_u64(0x0, x196, x193);
  let (x199, x200) = fiat_p384_addcarryx_u64(x198, x194, x191);
  let (x201, x202) = fiat_p384_addcarryx_u64(x200, x192, x189);
  let (x203, x204) = fiat_p384_addcarryx_u64(x202, x190, x187);
  let (x205, x206) = fiat_p384_addcarryx_u64(x204, x188, x185);
  let x207: u64 = ((x206 as u64) + x186);
  let (x208, x209) = fiat_p384_addcarryx_u64(0x0, x169, x195);
  let (x210, x211) = fiat_p384_addcarryx_u64(x209, x171, x197);
  let (x212, x213) = fiat_p384_addcarryx_u64(x211, x173, x199);
  let (x214, x215) = fiat_p384_addcarryx_u64(x213, x175, x201);
  let (x216, x217) = fiat_p384_addcarryx_u64(x215, x177, x203);
  let (x218, x219) = fiat_p384_addcarryx_u64(x217, x179, x205);
  let (x220, x221) = fiat_p384_addcarryx_u64(x219, x181, x207);
  let x222: u64 = ((x221 as u64) + (x182 as u64));
  let (x223, x224) = fiat_p384_mulx_u64(x3, (arg1[5]));
  let (x225, x226) = fiat_p384_mulx_u64(x3, (arg1[4]));
  let (x227, x228) = fiat_p384_mulx_u64(x3, (arg1[3]));
  let (x229, x230) = fiat_p384_mulx_u64(x3, (arg1[2]));
  let (x231, x232) = fiat_p384_mulx_u64(x3, (arg1[1]));
  let (x233, x234) = fiat_p384_mulx_u64(x3, (arg1[0]));
  let (x235, x236) = fiat_p384_addcarryx_u64(0x0, x234, x231);
  let (x237, x238) = fiat_p384_addcarryx_u64(x236, x232, x229);
  let (x239, x240) = fiat_p384_addcarryx_u64(x238, x230, x227);
  let (x241, x242) = fiat_p384_addcarryx_u64(x240, x228, x225);
  let (x243, x244) = fiat_p384_addcarryx_u64(x242, x226, x223);
  let x245: u64 = ((x244 as u64) + x224);
  let (x246, x247) = fiat_p384_addcarryx_u64(0x0, x210, x233);
  let (x248, x249) = fiat_p384_addcarryx_u64(x247, x212, x235);
  let (x250, x251) = fiat_p384_addcarryx_u64(x249, x214, x237);
  let (x252, x253) = fiat_p384_addcarryx_u64(x251, x216, x239);
  let (x254, x255) = fiat_p384_addcarryx_u64(x253, x218, x241);
  let (x256, x257) = fiat_p384_addcarryx_u64(x255, x220, x243);
  let (x258, x259) = fiat_p384_addcarryx_u64(x257, x222, x245);
  let (x260, x261) = fiat_p384_mulx_u64(x246, 0x100000001);
  let (x262, x263) = fiat_p384_mulx_u64(x260, 0xffffffffffffffff);
  let (x264, x265) = fiat_p384_mulx_u64(x260, 0xffffffffffffffff);
  let (x266, x267) = fiat_p384_mulx_u64(x260, 0xffffffffffffffff);
  let (x268, x269) = fiat_p384_mulx_u64(x260, 0xfffffffffffffffe);
  let (x270, x271) = fiat_p384_mulx_u64(x260, 0xffffffff00000000);
  let (x272, x273) = fiat_p384_mulx_u64(x260, 0xffffffff);
  let (x274, x275) = fiat_p384_addcarryx_u64(0x0, x273, x270);
  let (x276, x277) = fiat_p384_addcarryx_u64(x275, x271, x268);
  let (x278, x279) = fiat_p384_addcarryx_u64(x277, x269, x266);
  let (x280, x281) = fiat_p384_addcarryx_u64(x279, x267, x264);
  let (x282, x283) = fiat_p384_addcarryx_u64(x281, x265, x262);
  let x284: u64 = ((x283 as u64) + x263);
  let (x285, x286) = fiat_p384_addcarryx_u64(0x0, x246, x272);
  let (x287, x288) = fiat_p384_addcarryx_u64(x286, x248, x274);
  let (x289, x290) = fiat_p384_addcarryx_u64(x288, x250, x276);
  let (x291, x292) = fiat_p384_addcarryx_u64(x290, x252, x278);
  let (x293, x294) = fiat_p384_addcarryx_u64(x292, x254, x280);
  let (x295, x296) = fiat_p384_addcarryx_u64(x294, x256, x282);
  let (x297, x298) = fiat_p384_addcarryx_u64(x296, x258, x284);
  let x299: u64 = ((x298 as u64) + (x259 as u64));
  let (x300, x301) = fiat_p384_mulx_u64(x4, (arg1[5]));
  let (x302, x303) = fiat_p384_mulx_u64(x4, (arg1[4]));
  let (x304, x305) = fiat_p384_mulx_u64(x4, (arg1[3]));
  let (x306, x307) = fiat_p384_mulx_u64(x4, (arg1[2]));
  let (x308, x309) = fiat_p384_mulx_u64(x4, (arg1[1]));
  let (x310, x311) = fiat_p384_mulx_u64(x4, (arg1[0]));
  let (x312, x313) = fiat_p384_addcarryx_u64(0x0, x311, x308);
  let (x314, x315) = fiat_p384_addcarryx_u64(x313, x309, x306);
  let (x316, x317) = fiat_p384_addcarryx_u64(x315, x307, x304);
  let (x318, x319) = fiat_p384_addcarryx_u64(x317, x305, x302);
  let (x320, x321) = fiat_p384_addcarryx_u64(x319, x303, x300);
  let x322: u64 = ((x321 as u64) + x301);
  let (x323, x324) = fiat_p384_addcarryx_u64(0x0, x287, x310);
  let (x325, x326) = fiat_p384_addcarryx_u64(x324, x289, x312);
  let (x327, x328) = fiat_p384_addcarryx_u64(x326, x291, x314);
  let (x329, x330) = fiat_p384_addcarryx_u64(x328, x293, x316);
  let (x331, x332) = fiat_p384_addcarryx_u64(x330, x295, x318);
  let (x333, x334) = fiat_p384_addcarryx_u64(x332, x297, x320);
  let (x335, x336) = fiat_p384_addcarryx_u64(x334, x299, x322);
  let (x337, x338) = fiat_p384_mulx_u64(x323, 0x100000001);
  let (x339, x340) = fiat_p384_mulx_u64(x337, 0xffffffffffffffff);
  let (x341, x342) = fiat_p384_mulx_u64(x337, 0xffffffffffffffff);
  let (x343, x344) = fiat_p384_mulx_u64(x337, 0xffffffffffffffff);
  let (x345, x346) = fiat_p384_mulx_u64(x337, 0xfffffffffffffffe);
  let (x347, x348) = fiat_p384_mulx_u64(x337, 0xffffffff00000000);
  let (x349, x350) = fiat_p384_mulx_u64(x337, 0xffffffff);
  let (x351, x352) = fiat_p384_addcarryx_u64(0x0, x350, x347);
  let (x353, x354) = fiat_p384_addcarryx_u64(x352, x348, x345);
  let (x355, x356) = fiat_p384_addcarryx_u64(x354, x346, x343);
  let (x357, x358) = fiat_p384_addcarryx_u64(x356, x344, x341);
  let (x359, x360) = fiat_p384_addcarryx_u64(x358, x342, x339);
  let x361: u64 = ((x360 as u64) + x340);
  let (x362, x363) = fiat_p384_addcarryx_u64(0x0, x323, x349);
  let (x364, x365) = fiat_p384_addcarryx_u64(x363, x325, x351);
  let (x366, x367) = fiat_p384_addcarryx_u64(x365, x327, x353);
  let (x368, x369) = fiat_p384_addcarryx_u64(x367, x329, x355);
  let (x370, x371) = fiat_p384_addcarryx_u64(x369, x331, x357);
  let (x372, x373) = fiat_p384_addcarryx_u64(x371, x333, x359);
  let (x374, x375) = fiat_p384_addcarryx_u64(x373, x335, x361);
  let x376: u64 = ((x375 as u64) + (x336 as u64));
  let (x377, x378) = fiat_p384_mulx_u64(x5, (arg1[5]));
  let (x379, x380) = fiat_p384_mulx_u64(x5, (arg1[4]));
  let (x381, x382) = fiat_p384_mulx_u64(x5, (arg1[3]));
  let (x383, x384) = fiat_p384_mulx_u64(x5, (arg1[2]));
  let (x385, x386) = fiat_p384_mulx_u64(x5, (arg1[1]));
  let (x387, x388) = fiat_p384_mulx_u64(x5, (arg1[0]));
  let (x389, x390) = fiat_p384_addcarryx_u64(0x0, x388, x385);
  let (x391, x392) = fiat_p384_addcarryx_u64(x390, x386, x383);
  let (x393, x394) = fiat_p384_addcarryx_u64(x392, x384, x381);
  let (x395, x396) = fiat_p384_addcarryx_u64(x394, x382, x379);
  let (x397, x398) = fiat_p384_addcarryx_u64(x396, x380, x377);
  let x399: u64 = ((x398 as u64) + x378);
  let (x400, x401) = fiat_p384_addcarryx_u64(0x0, x364, x387);
  let (x402, x403) = fiat_p384_addcarryx_u64(x401, x366, x389);
  let (x404, x405) = fiat_p384_addcarryx_u64(x403, x368, x391);
  let (x406, x407) = fiat_p384_addcarryx_u64(x405, x370, x393);
  let (x408, x409) = fiat_p384_addcarryx_u64(x407, x372, x395);
  let (x410, x411) = fiat_p384_addcarryx_u64(x409, x374, x397);
  let (x412, x413) = fiat_p384_addcarryx_u64(x411, x376, x399);
  let (x414, x415) = fiat_p384_mulx_u64(x400, 0x100000001);
  let (x416, x417) = fiat_p384_mulx_u64(x414, 0xffffffffffffffff);
  let (x418, x419) = fiat_p384_mulx_u64(x414, 0xffffffffffffffff);
  let (x420, x421) = fiat_p384_mulx_u64(x414, 0xffffffffffffffff);
  let (x422, x423) = fiat_p384_mulx_u64(x414, 0xfffffffffffffffe);
  let (x424, x425) = fiat_p384_mulx_u64(x414, 0xffffffff00000000);
  let (x426, x427) = fiat_p384_mulx_u64(x414, 0xffffffff);
  let (x428, x429) = fiat_p384_addcarryx_u64(0x0, x427, x424);
  let (x430, x431) = fiat_p384_addcarryx_u64(x429, x425, x422);
  let (x432, x433) = fiat_p384_addcarryx_u64(x431, x423, x420);
  let (x434, x435) = fiat_p384_addcarryx_u64(x433, x421, x418);
  let (x436, x437) = fiat_p384_addcarryx_u64(x435, x419, x416);
  let x438: u64 = ((x437 as u64) + x417);
  let (x439, x440) = fiat_p384_addcarryx_u64(0x0, x400, x426);
  let (x441, x442) = fiat_p384_addcarryx_u64(x440, x402, x428);
  let (x443, x444) = fiat_p384_addcarryx_u64(x442, x404, x430);
  let (x445, x446) = fiat_p384_addcarryx_u64(x444, x406, x432);
  let (x447, x448) = fiat_p384_addcarryx_u64(x446, x408, x434);
  let (x449, x450) = fiat_p384_addcarryx_u64(x448, x410, x436);
  let (x451, x452) = fiat_p384_addcarryx_u64(x450, x412, x438);
  let x453: u64 = ((x452 as u64) + (x413 as u64));
  let (x454, x455) = fiat_p384_subborrowx_u64(0x0, x441, 0xffffffff);
  let (x456, x457) = fiat_p384_subborrowx_u64(x455, x443, 0xffffffff00000000);
  let (x458, x459) = fiat_p384_subborrowx_u64(x457, x445, 0xfffffffffffffffe);
  let (x460, x461) = fiat_p384_subborrowx_u64(x459, x447, 0xffffffffffffffff);
  let (x462, x463) = fiat_p384_subborrowx_u64(x461, x449, 0xffffffffffffffff);
  let (x464, x465) = fiat_p384_subborrowx_u64(x463, x451, 0xffffffffffffffff);
  let (x466, x467) = fiat_p384_subborrowx_u64(x465, x453, (0x0 as u64));
  let (x468) = fiat_p384_cmovznz_u64(x467, x454, x441);
  let (x469) = fiat_p384_cmovznz_u64(x467, x456, x443);
  let (x470) = fiat_p384_cmovznz_u64(x467, x458, x445);
  let (x471) = fiat_p384_cmovznz_u64(x467, x460, x447);
  let (x472) = fiat_p384_cmovznz_u64(x467, x462, x449);
  let (x473) = fiat_p384_cmovznz_u64(x467, x464, x451);
  out1[0] = x468;
  out1[1] = x469;
  out1[2] = x470;
  out1[3] = x471;
  out1[4] = x472;
  out1[5] = x473;
  out1
}

#[inline]
pub(crate) const fn mul(lhs: [u64; 6], rhs: [u64; 6]) -> [u64; 6] {
  fiat_p384_mul(&lhs, &rhs)
}

#[inline]
pub(crate) const fn square(value: [u64; 6]) -> [u64; 6] {
  fiat_p384_square(&value)
}
