//! ECDSA signing and verification for NIST P-256 and P-384.
//!
//! This module exposes strict, typed APIs for the ECDSA profiles most
//! protocols use today: P-256/SHA-256 and P-384/SHA-384.

use core::{
  fmt,
  hash::{Hash, Hasher},
};

use super::hmac::{HmacSha256, HmacSha384};
use crate::{
  SecretBytes,
  hashes::crypto::{Sha256, Sha384},
  secret::ZeroizingBytes,
  traits::{Mac, VerificationError, ct},
};

#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
#[path = "ecdsa_aarch64_asm.rs"]
mod ecdsa_aarch64_asm;
#[path = "ecdsa_generator_tables.rs"]
mod ecdsa_generator_tables;
#[path = "ecdsa_p384_field.rs"]
mod ecdsa_p384_field;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[path = "ecdsa_x86_64_asm.rs"]
mod ecdsa_x86_64_asm;

use ecdsa_generator_tables::{
  P256_SIGNING_COMB_WIDTH, P256_SIGNING_GENERATOR_COMB_X, P256_SIGNING_GENERATOR_COMB_Y, P384_SIGNING_COMB_WIDTH,
  P384_SIGNING_GENERATOR_COMB_X, P384_SIGNING_GENERATOR_COMB_Y,
};

#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
use self::ecdsa_aarch64_asm as ecdsa_platform_asm;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
use self::ecdsa_x86_64_asm as ecdsa_platform_asm;

const TAG_SEQUENCE: u8 = 0x30;
const TAG_INTEGER: u8 = 0x02;
const TAG_BIT_STRING: u8 = 0x03;
const TAG_OBJECT_IDENTIFIER: u8 = 0x06;
const COMB_WIDTH: usize = 4;
const COMB_TABLE_SIZE: usize = 1 << COMB_WIDTH;
const P256_SIGNING_COMB_ROWS: usize = 37;
const P384_SIGNING_COMB_ROWS: usize = 48;
const _: () = assert!(P256_SIGNING_COMB_ROWS * P256_SIGNING_COMB_WIDTH == 259);

const ID_EC_PUBLIC_KEY_OID: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
const SECP256R1_OID: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];
const SECP384R1_OID: &[u8] = &[0x2b, 0x81, 0x04, 0x00, 0x22];
const P256_NONCE_DOMAIN: &[u8] = b"rscrypto-ecdsa-p256-sha256-sign-v1";
const P384_NONCE_DOMAIN: &[u8] = b"rscrypto-ecdsa-p384-sha384-sign-v1";
const P256_PUBKEY_BLIND_DOMAIN: &[u8] = b"rscrypto-ecdsa-p256-pubkey-blind-v1";
const P384_PUBKEY_BLIND_DOMAIN: &[u8] = b"rscrypto-ecdsa-p384-pubkey-blind-v1";
const ECDSA_KEY_GENERATION_ATTEMPTS: usize = 32;

const P256_FIELD: Uint<4> = Uint([
  0xffff_ffff_ffff_ffff,
  0x0000_0000_ffff_ffff,
  0x0000_0000_0000_0000,
  0xffff_ffff_0000_0001,
]);
const P256_ORDER: Uint<4> = Uint([
  0xf3b9_cac2_fc63_2551,
  0xbce6_faad_a717_9e84,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_0000_0000,
]);
const P256_ORDER_MINUS_TWO: Uint<4> = Uint([
  0xf3b9_cac2_fc63_254f,
  0xbce6_faad_a717_9e84,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_0000_0000,
]);
const P256_ORDER_HALF: Uint<4> = Uint([
  0x79dc_e561_7e31_92a8,
  0xde73_7d56_d38b_cf42,
  0x7fff_ffff_ffff_ffff,
  0x7fff_ffff_8000_0000,
]);
const P256_FIELD_MINUS_TWO: Uint<4> = Uint([
  0xffff_ffff_ffff_fffd,
  0x0000_0000_ffff_ffff,
  0x0000_0000_0000_0000,
  0xffff_ffff_0000_0001,
]);
const P256_B: Uint<4> = Uint([
  0x3bce_3c3e_27d2_604b,
  0x651d_06b0_cc53_b0f6,
  0xb3eb_bd55_7698_86bc,
  0x5ac6_35d8_aa3a_93e7,
]);
const P256_GX: Uint<4> = Uint([
  0xf4a1_3945_d898_c296,
  0x7703_7d81_2deb_33a0,
  0xf8bc_e6e5_63a4_40f2,
  0x6b17_d1f2_e12c_4247,
]);
const P256_GY: Uint<4> = Uint([
  0xcbb6_4068_37bf_51f5,
  0x2bce_3357_6b31_5ece,
  0x8ee7_eb4a_7c0f_9e16,
  0x4fe3_42e2_fe1a_7f9b,
]);

const P384_FIELD: Uint<6> = Uint([
  0x0000_0000_ffff_ffff,
  0xffff_ffff_0000_0000,
  0xffff_ffff_ffff_fffe,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
]);
const P384_ORDER: Uint<6> = Uint([
  0xecec_196a_ccc5_2973,
  0x581a_0db2_48b0_a77a,
  0xc763_4d81_f437_2ddf,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
]);
const P384_ORDER_MINUS_TWO: Uint<6> = Uint([
  0xecec_196a_ccc5_2971,
  0x581a_0db2_48b0_a77a,
  0xc763_4d81_f437_2ddf,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
]);
const P384_ORDER_HALF: Uint<6> = Uint([
  0x7676_0cb5_6662_94b9,
  0xac0d_06d9_2458_53bd,
  0xe3b1_a6c0_fa1b_96ef,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
  0x7fff_ffff_ffff_ffff,
]);
const P384_FIELD_MINUS_TWO: Uint<6> = Uint([
  0x0000_0000_ffff_fffd,
  0xffff_ffff_0000_0000,
  0xffff_ffff_ffff_fffe,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
  0xffff_ffff_ffff_ffff,
]);
#[cfg(test)]
const P384_FIELD_MONTGOMERY_ONE: Uint<6> = Uint([
  0xffff_ffff_0000_0001,
  0x0000_0000_ffff_ffff,
  0x0000_0000_0000_0001,
  0x0000_0000_0000_0000,
  0x0000_0000_0000_0000,
  0x0000_0000_0000_0000,
]);
const P384_B: Uint<6> = Uint([
  0x2a85_c8ed_d3ec_2aef,
  0xc656_398d_8a2e_d19d,
  0x0314_088f_5013_875a,
  0x181d_9c6e_fe81_4112,
  0x988e_056b_e3f8_2d19,
  0xb331_2fa7_e23e_e7e4,
]);
const P384_GX: Uint<6> = Uint([
  0x3a54_5e38_7276_0ab7,
  0x5502_f25d_bf55_296c,
  0x59f7_41e0_8254_2a38,
  0x6e1d_3b62_8ba7_9b98,
  0x8eb1_c71e_f320_ad74,
  0xaa87_ca22_be8b_0537,
]);
const P384_GY: Uint<6> = Uint([
  0x7a43_1d7c_90ea_0e5f,
  0x0a60_b1ce_1d7e_819d,
  0xe9da_3113_b5f0_b8c0,
  0xf8f4_1dbd_289a_147c,
  0x5d9e_98bf_9292_dc29,
  0x3617_de4a_9626_2c6f,
]);

// Fixed generator comb-table coordinates are stored in Montgomery form so
// verification does not rebuild them through field multiplications.
const P256_GENERATOR_COMB_X: [Uint<4>; COMB_TABLE_SIZE] = [
  Uint([
    0x79e7_30d4_18a9_143c,
    0x75ba_95fc_5fed_b601,
    0x79fb_732b_7762_2510,
    0x1890_5f76_a537_55c6,
  ]),
  Uint([
    0x79e7_30d4_18a9_143c,
    0x75ba_95fc_5fed_b601,
    0x79fb_732b_7762_2510,
    0x1890_5f76_a537_55c6,
  ]),
  Uint([
    0x4f92_2fc5_16a0_d2bb,
    0x0d5c_c16c_1a62_3499,
    0x9241_cf3a_57c6_2c8b,
    0x2f5e_6961_fd1b_667f,
  ]),
  Uint([
    0x9e56_6847_e137_bbbc,
    0xe434_469e_8a6a_0bec,
    0xb1c4_2761_79d7_3463,
    0x5abe_0285_133d_0015,
  ]),
  Uint([
    0x62a8_c244_bfe2_0925,
    0x91c1_9ac3_8fdc_e867,
    0x5a96_a5d5_dd38_7063,
    0x61d5_87d4_21d3_24f6,
  ]),
  Uint([
    0x1c89_1f2b_2cb1_9ffd,
    0x01ba_8d5b_b192_3c23,
    0xb6d0_3d67_8ac5_ca8e,
    0x586e_b04c_1f13_bedc,
  ]),
  Uint([
    0x6257_7734_d2b5_33d5,
    0x673b_8af6_a1bd_ddc0,
    0x577e_7c9a_a79e_c293,
    0xbb6d_e651_c3b2_66b1,
  ]),
  Uint([
    0xbd6a_38e1_1ae5_aa1c,
    0xb8b7_652b_49e7_3658,
    0x0b13_0014_ee5f_87ed,
    0x9d0f_27b2_aeeb_ffcd,
  ]),
  Uint([
    0x56f8_410e_f4f8_b16a,
    0x9724_1afe_c47b_266a,
    0x0a40_6b8e_6d9c_87c1,
    0x803f_3e02_cd42_ab1b,
  ]),
  Uint([
    0x846a_56f2_c379_ab34,
    0xa8ee_068b_841d_f8d1,
    0x2031_4459_176c_68ef,
    0xf1af_32d5_915f_1f30,
  ]),
  Uint([
    0xed93_e225_d5be_5a2b,
    0x6fe7_9983_5934_f3c6,
    0x4314_0926_2262_6ffc,
    0x50bb_b4d9_7990_216a,
  ]),
  Uint([
    0xfc68_b5c5_9b39_1593,
    0xc385_f5a2_5982_70fc,
    0x7144_f3aa_d19a_dcbb,
    0xdd55_8999_83fb_ae0c,
  ]),
  Uint([
    0x5fe1_4bfe_80ec_21fe,
    0xf6ce_116a_c255_be82,
    0x98bc_5a07_2f4a_5d67,
    0xfad2_7148_db7e_63af,
  ]),
  Uint([
    0x1e9e_cc49_a56c_0dd7,
    0xa5cf_fcd8_4608_6c74,
    0x8f7a_1408_f505_aece,
    0xb37b_85c0_bef0_c47e,
  ]),
  Uint([
    0x0a1c_7294_95c8_f8be,
    0x2961_c480_3bf3_62bf,
    0x9e41_8403_df63_d4ac,
    0xc109_f9cb_91ec_e900,
  ]),
  Uint([
    0x0d5a_e356_4291_3074,
    0x5549_1b27_48a5_42b1,
    0x469c_a665_b310_732a,
    0x2959_1d52_5f1a_4cc1,
  ]),
];
const P256_GENERATOR_COMB_Y: [Uint<4>; COMB_TABLE_SIZE] = [
  Uint([
    0xddf2_5357_ce95_560a,
    0x8b4a_b8e4_ba19_e45c,
    0xd2e8_8688_dd21_f325,
    0x8571_ff18_2588_5d85,
  ]),
  Uint([
    0xddf2_5357_ce95_560a,
    0x8b4a_b8e4_ba19_e45c,
    0xd2e8_8688_dd21_f325,
    0x8571_ff18_2588_5d85,
  ]),
  Uint([
    0x5c15_c70b_f5a0_1797,
    0x3d20_b44d_6095_6192,
    0x0491_1b37_071f_db52,
    0xf648_f916_8d6f_0f7b,
  ]),
  Uint([
    0x92aa_837c_c04c_7dab,
    0x573d_9f4c_4326_0c07,
    0x0c93_1562_78e6_cc37,
    0x94bb_725b_6b6f_7383,
  ]),
  Uint([
    0xe876_73a2_a371_73ea,
    0x2384_8008_5377_8b65,
    0x10f8_441e_05ba_b43e,
    0xfa11_fe12_4621_efbe,
  ]),
  Uint([
    0x0c35_c6e5_27e8_ed09,
    0x1e81_a33c_1819_ede2,
    0x278f_d6c0_56c6_52fa,
    0x19d5_ac08_7086_4f11,
  ]),
  Uint([
    0xe7e9_303a_b652_59b3,
    0xd6a0_afd3_d03a_7480,
    0xc5ac_83d1_9b3c_fc27,
    0x60b4_619a_5d18_b99b,
  ]),
  Uint([
    0xca92_4631_7a73_0a55,
    0x9c95_5b2f_ddbb_c83a,
    0x07c1_dfe0_ac01_9a71,
    0x244a_566d_356e_c48d,
  ]),
  Uint([
    0x7f03_09a8_04db_ec69,
    0xa83b_85f7_3bba_d05f,
    0xc609_7273_ad8e_197f,
    0xc097_440e_5067_adc1,
  ]),
  Uint([
    0x99c3_7531_5d75_bd50,
    0x837c_ffba_f72f_67bc,
    0x0613_a418_48d7_723f,
    0x23d0_f130_e2d4_1c8b,
  ]),
  Uint([
    0x3781_91c6_e57e_c63e,
    0x6542_2c40_181d_cdb2,
    0x41a8_099b_0236_e0f6,
    0x2b10_0118_01fe_49c3,
  ]),
  Uint([
    0x93b8_8b8e_74b8_2ff4,
    0xd2e0_3c40_71e7_34c9,
    0x9a7a_9eaf_43c0_322a,
    0xe6e4_c551_149d_6041,
  ]),
  Uint([
    0x90c0_b6ac_29ab_05b3,
    0x37a9_a83c_4e25_1ae6,
    0x0a7d_c875_c2aa_de7d,
    0x7738_7de3_9f0e_1a84,
  ]),
  Uint([
    0x3596_b6e4_cc0e_6a8f,
    0xfd6d_4bbf_6b38_8f23,
    0xaba4_53fa_c39c_ef4e,
    0x9c13_5ac8_f9f6_28d5,
  ]),
  Uint([
    0xc2d0_95d0_5894_5705,
    0xb908_3d96_ddeb_85c0,
    0x8469_2b8d_7a40_449b,
    0x9bc3_344f_2eee_1ee1,
  ]),
  Uint([
    0xe76f_5b6b_b84f_983f,
    0xbe7e_ef41_9f5f_84e1,
    0x1200_d496_80ba_a189,
    0x6376_551f_18ef_332c,
  ]),
];
const P384_GENERATOR_COMB_X: [Uint<6>; COMB_TABLE_SIZE] = [
  Uint([
    0x3dd0_7566_49c0_b528,
    0x20e3_78e2_a0d6_ce38,
    0x879c_3afc_541b_4d6e,
    0x6454_8684_59a3_0eff,
    0x812f_f723_614e_de2b,
    0x4d3a_adc2_299e_1513,
  ]),
  Uint([
    0x3dd0_7566_49c0_b528,
    0x20e3_78e2_a0d6_ce38,
    0x879c_3afc_541b_4d6e,
    0x6454_8684_59a3_0eff,
    0x812f_f723_614e_de2b,
    0x4d3a_adc2_299e_1513,
  ]),
  Uint([
    0x2448_0c57_f26f_eef9,
    0xc31a_2694_3a0e_1240,
    0x7350_02c3_273e_2bc7,
    0x8c42_e9c5_3ef1_ed4c,
    0x028b_abf6_7f49_48e8,
    0x6a50_2f43_8a97_8632,
  ]),
  Uint([
    0x1142_6e2e_e349_ddd0,
    0x9f11_7ef9_9b2f_c250,
    0xff36_b480_ec01_74a6,
    0x4f4b_de76_1845_8466,
    0x2f2e_db6d_0580_6049,
    0x8adc_75d1_19df_ca92,
  ]),
  Uint([
    0x3782_05de_2f9f_be67,
    0xc4af_cb83_7f72_8e44,
    0xdbce_c06c_682e_00f1,
    0xf2a1_45c3_114d_5423,
    0xa01d_9874_7a52_463e,
    0xfc09_35b1_7d71_7b0a,
  ]),
  Uint([
    0x73ad_e4da_2341_c342,
    0xdd32_6e54_ea70_4422,
    0x336c_7d98_3741_cef3,
    0x1eaf_a00d_59e6_1549,
    0xcd3e_d892_bd9a_3efd,
    0x03fa_f26c_c5c6_c7e4,
  ]),
  Uint([
    0xfab0_8607_3f3b_236f,
    0x19e9_d41d_81e2_21da,
    0xf3f6_571e_3927_b428,
    0x4348_a933_7550_f1f6,
    0x7167_b996_a85e_62f0,
    0x62d4_3759_7f54_52bf,
  ]),
  Uint([
    0x90b2_e5b3_3062_e8af,
    0xa857_2375_e8a3_d369,
    0x3fe1_b00b_201d_b7b1,
    0xe926_def0_ee65_1aa2,
    0x6542_c9be_b9b1_0ad7,
    0x098e_309b_a2fc_be74,
  ]),
  Uint([
    0x070d_34e1_1697_3cf4,
    0x20ae_e08b_7e4f_34f7,
    0x269a_f9b9_5eb8_ad29,
    0xdde0_a036_a6a4_5dda,
    0xa18b_528e_63df_41e0,
    0x03cc_71b2_a260_df2a,
  ]),
  Uint([
    0xba14_64b4_01ab_5245,
    0x9b8d_0b6d_c48d_93ff,
    0x9398_67dc_93ad_272c,
    0xbebe_085e_ae9f_dc77,
    0x73ae_5103_894e_a8bd,
    0x740f_c89a_39ac_22e1,
  ]),
  Uint([
    0x03cf_2922_0062_3f3b,
    0x095c_7111_5f29_ebff,
    0x42d7_2247_80aa_6823,
    0x044c_7ba1_7458_c0b0,
    0xca62_f7ef_0959_ec20,
    0x40ae_2ab7_f8ca_929f,
  ]),
  Uint([
    0x744d_1400_8435_af04,
    0x5f25_5b1d_fec1_92da,
    0x1f17_dc12_336d_c542,
    0x5c90_c2a7_636a_68a8,
    0x960c_9eb7_7704_ca1e,
    0x9de8_cf1e_6fb3_d65a,
  ]),
  Uint([
    0x867d_b639_9868_3186,
    0xfb5c_f424_ddcc_4ea9,
    0xcc9a_7ffe_d4f0_e7bd,
    0x7c57_f71c_7a77_9f7e,
    0x9077_4079_d6b2_5ef2,
    0x90ea_e903_b408_1680,
  ]),
  Uint([
    0xfbd8_38f9_e054_0015,
    0x2c32_3946_c390_77dc,
    0x8b1f_b9e6_ad61_9124,
    0x9612_440c_0ca6_2ea8,
    0x9ad9_b52c_2dbe_00ff,
    0xf52a_baa1_ae19_7643,
  ]),
  Uint([
    0xcd96_866d_b8a9_e8c9,
    0xa119_63b8_5bb8_091e,
    0xc7f9_0d53_045b_3cd2,
    0x755a_72b5_80f3_6504,
    0x46f8_b399_21d3_751c,
    0x4bff_dc91_53c1_93de,
  ]),
  Uint([
    0xd71e_4aab_6328_e33f,
    0x5486_782b_af81_36d1,
    0x07a4_995f_86d5_7231,
    0xf1f0_a5bd_1651_a968,
    0xa5dc_5b24_7680_3b6d,
    0x5c58_7cbc_42dd_a935,
  ]),
];
const P384_GENERATOR_COMB_Y: [Uint<6>; COMB_TABLE_SIZE] = [
  Uint([
    0x2304_3dad_4b03_a4fe,
    0xa1bf_a8bf_7bb4_a9ac,
    0x8bad_e756_2e83_b050,
    0xc6c3_5219_68f4_ffd9,
    0xdd80_0226_3969_a840,
    0x2b78_abc2_5a15_c5e9,
  ]),
  Uint([
    0x2304_3dad_4b03_a4fe,
    0xa1bf_a8bf_7bb4_a9ac,
    0x8bad_e756_2e83_b050,
    0xc6c3_5219_68f4_ffd9,
    0xdd80_0226_3969_a840,
    0x2b78_abc2_5a15_c5e9,
  ]),
  Uint([
    0xf5f1_3a46_b745_36fe,
    0x1d21_8bab_d8a9_f0eb,
    0x30f3_6bcc_3723_2768,
    0xc531_7b31_576e_8c18,
    0xef1d_57a6_9bbc_b766,
    0x917c_4930_b3e3_d4dc,
  ]),
  Uint([
    0xa619_d097_b7d5_a7ce,
    0x8742_75e5_a344_11e9,
    0x5403_e047_0da4_b4ef,
    0x2eba_afd9_7790_1d8f,
    0x5e63_ebce_a747_170f,
    0x12a3_6944_7f9d_8036,
  ]),
  Uint([
    0x9653_bc4f_d4d0_1f95,
    0x9aa8_3ea8_9560_ad34,
    0xf779_43dc_af8e_3f3f,
    0x7077_4a10_e86f_e16e,
    0x6b62_e6f1_bf9f_fdcf,
    0x8a72_f39e_5887_45c9,
  ]),
  Uint([
    0x087e_2fcf_3045_f8ac,
    0x14a6_5532_174f_1e73,
    0x2cf8_4f28_fe0a_f9a7,
    0xddfd_7a84_2cdc_935b,
    0x4c0f_117b_6929_c895,
    0x3565_72d6_4c8b_cfcc,
  ]),
  Uint([
    0xd85f_eb9e_f295_5926,
    0x440a_561f_6df7_8353,
    0x3896_68ec_9ca3_6b59,
    0x052b_f1a1_a22d_a016,
    0xbdfb_ff72_f609_3254,
    0x94e5_0f28_e222_09f3,
  ]),
  Uint([
    0x779d_eeb3_fff1_d63f,
    0x23d0_e80a_20bf_d374,
    0x8452_bb3b_8768_f797,
    0xcf75_bb4d_1f95_2856,
    0x8fe6_b400_29ea_3faa,
    0x12bd_3e40_8137_3a53,
  ]),
  Uint([
    0x24a6_770a_a06b_1dd7,
    0x5bfa_9c11_9d26_75d3,
    0x73c1_e2a1_9684_4432,
    0x3660_558d_131a_6cf0,
    0xb028_9c83_2ee7_9454,
    0xa6ae_fb01_c6d8_ddcd,
  ]),
  Uint([
    0x5e28_b0a3_28e2_3b23,
    0x2352_722e_e131_04d0,
    0xf466_7a18_b0a2_640d,
    0xac74_a72e_49bb_37c3,
    0x79f7_34f0_e81e_183a,
    0xbffe_5b6c_3fd9_c0eb,
  ]),
  Uint([
    0xb8c5_377a_a927_b102,
    0x398a_86a0_dc03_1771,
    0x0490_8f9d_c216_a406,
    0xb423_a73a_918d_3300,
    0x634b_0ff1_e0b9_4739,
    0xe29d_e725_2d69_f697,
  ]),
  Uint([
    0xc60f_ee0d_511d_3d06,
    0x466e_2313_f9eb_52c7,
    0x743c_0f5f_206b_0914,
    0x42f5_5bac_2191_aa4d,
    0xcefc_7c8f_ffeb_dbc2,
    0xd4fa_6081_e6e8_ed1c,
  ]),
  Uint([
    0xdf2a_ae5e_0ee1_fceb,
    0x3ff1_da24_e86c_1a1f,
    0x80f5_87d6_ca19_3edf,
    0xa569_5523_dc9b_9d6a,
    0x7b84_0900_8592_0303,
    0x1efa_4dfc_ba6d_bdef,
  ]),
  Uint([
    0xd0e8_9894_2cac_32ad,
    0xdfb7_9e42_62a9_8f91,
    0x6545_2ecf_276f_55cb,
    0xdb1a_c0d2_7ad2_3e12,
    0xf68c_5f6a_de49_86f0,
    0x389a_c37b_82ce_327d,
  ]),
  Uint([
    0xcd15_c049_b895_54e7,
    0x353c_6754_f7a2_6be6,
    0x7960_2370_bd41_d970,
    0xde16_470b_12b1_76c0,
    0x56ba_1175_40c8_809d,
    0xe2db_35c3_e435_fb1e,
  ]),
  Uint([
    0x2b6c_db32_bae8_b4c0,
    0x66d1_598b_b133_1138,
    0x4a23_b2d2_5d7e_9614,
    0x93e4_02a6_74a8_c05d,
    0x45ac_94e6_da7c_e82e,
    0xeb9f_8281_e463_d465,
  ]),
];

static P256_FIELD_MODULUS: Modulus<4> = Modulus {
  value: P256_FIELD,
  n0_inv: 0x0000_0000_0000_0001,
  r2: Uint([
    0x0000_0000_0000_0003,
    0xffff_fffb_ffff_ffff,
    0xffff_ffff_ffff_fffe,
    0x0000_0004_ffff_fffd,
  ]),
};
static P256_ORDER_MODULUS: Modulus<4> = Modulus {
  value: P256_ORDER,
  n0_inv: 0xccd1_c8aa_ee00_bc4f,
  r2: Uint([
    0x8324_4c95_be79_eea2,
    0x4699_799c_49bd_6fa6,
    0x2845_b239_2b6b_ec59,
    0x66e1_2d94_f3d9_5620,
  ]),
};
static P384_FIELD_MODULUS: Modulus<6> = Modulus {
  value: P384_FIELD,
  n0_inv: 0x0000_0001_0000_0001,
  r2: Uint([
    0xffff_fffe_0000_0001,
    0x0000_0002_0000_0000,
    0xffff_fffe_0000_0000,
    0x0000_0002_0000_0000,
    0x0000_0000_0000_0001,
    0x0000_0000_0000_0000,
  ]),
};
static P384_ORDER_MODULUS: Modulus<6> = Modulus {
  value: P384_ORDER,
  n0_inv: 0x6ed4_6089_e88f_dc45,
  r2: Uint([
    0x2d31_9b24_19b4_09a9,
    0xff3d_81e5_df1a_a419,
    0xbc3e_483a_fcb8_2947,
    0xd40d_4917_4aab_1cc5,
    0x3fb0_5b7a_2826_6895,
    0x0c84_ee01_2b39_bf21,
  ]),
};

static P256: Curve<4> = Curve {
  field_modulus: &P256_FIELD_MODULUS,
  scalar_modulus: &P256_ORDER_MODULUS,
  scalar_inverse_exponent: P256_ORDER_MINUS_TWO,
  field_inverse_exponent: P256_FIELD_MINUS_TWO,
  half_order: P256_ORDER_HALF,
  b: P256_B,
  generator_x: P256_GX,
  generator_y: P256_GY,
  generator_comb_x: P256_GENERATOR_COMB_X,
  generator_comb_y: P256_GENERATOR_COMB_Y,
  signing_comb_width: P256_SIGNING_COMB_WIDTH,
  signing_comb_rows: P256_SIGNING_COMB_ROWS,
  signing_generator_comb_x: &P256_SIGNING_GENERATOR_COMB_X,
  signing_generator_comb_y: &P256_SIGNING_GENERATOR_COMB_Y,
};

static P384: Curve<6> = Curve {
  field_modulus: &P384_FIELD_MODULUS,
  scalar_modulus: &P384_ORDER_MODULUS,
  scalar_inverse_exponent: P384_ORDER_MINUS_TWO,
  field_inverse_exponent: P384_FIELD_MINUS_TWO,
  half_order: P384_ORDER_HALF,
  b: P384_B,
  generator_x: P384_GX,
  generator_y: P384_GY,
  generator_comb_x: P384_GENERATOR_COMB_X,
  generator_comb_y: P384_GENERATOR_COMB_Y,
  signing_comb_width: P384_SIGNING_COMB_WIDTH,
  signing_comb_rows: P384_SIGNING_COMB_ROWS,
  signing_generator_comb_x: &P384_SIGNING_GENERATOR_COMB_X,
  signing_generator_comb_y: &P384_SIGNING_GENERATOR_COMB_Y,
};

/// ECDSA key, signature, or encoding error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EcdsaError {
  /// DER input was malformed or non-canonical.
  MalformedDer,
  /// The requested algorithm identifier is not supported by this type.
  UnsupportedAlgorithm,
  /// SEC1 public-key bytes are malformed or not on the expected curve.
  InvalidPublicKey,
  /// Secret key bytes are zero or outside the curve scalar range.
  InvalidSecretKey,
  /// Raw signature bytes or DER signature integers are malformed.
  InvalidSignature,
  /// Deterministic signing reached an invalid ECDSA scalar.
  SigningFailure,
}

impl fmt::Display for EcdsaError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let message = match self {
      Self::MalformedDer => "malformed ECDSA DER",
      Self::UnsupportedAlgorithm => "unsupported ECDSA algorithm",
      Self::InvalidPublicKey => "invalid ECDSA public key",
      Self::InvalidSecretKey => "invalid ECDSA secret key",
      Self::InvalidSignature => "invalid ECDSA signature",
      Self::SigningFailure => "ECDSA signing failure",
    };
    f.write_str(message)
  }
}

impl core::error::Error for EcdsaError {}

/// Errors returned by fallible ECDSA key generation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EcdsaKeyGenerationError<E> {
  /// The caller-provided random source failed.
  Random(E),
  /// Bounded scalar rejection exhausted its retry budget.
  InvalidSecretKey,
}

impl<E> fmt::Debug for EcdsaKeyGenerationError<E> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Random(_) => f.write_str("Random(..)"),
      Self::InvalidSecretKey => f.write_str("InvalidSecretKey"),
    }
  }
}

impl<E> EcdsaKeyGenerationError<E> {
  /// Construct a random-source error.
  #[inline]
  #[must_use]
  pub const fn random(err: E) -> Self {
    Self::Random(err)
  }

  /// Construct a scalar-rejection error.
  #[inline]
  #[must_use]
  pub const fn invalid_secret_key() -> Self {
    Self::InvalidSecretKey
  }
}

impl<E> fmt::Display for EcdsaKeyGenerationError<E> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Random(_) => f.write_str("ECDSA key-generation random source failed"),
      Self::InvalidSecretKey => f.write_str("ECDSA key-generation scalar rejection limit reached"),
    }
  }
}

impl<E> core::error::Error for EcdsaKeyGenerationError<E> where E: core::error::Error + 'static {}

/// P-256 ECDSA secret scalar.
pub struct EcdsaP256SecretKey([u8; Self::LENGTH]);

impl EcdsaP256SecretKey {
  /// P-256 secret scalar length in bytes.
  pub const LENGTH: usize = 32;

  /// Compare two secret keys without exposing a branchable boolean.
  #[inline]
  pub fn ct_eq(&self, other: &Self) -> ct::CtDecision {
    ct::fixed_eq(&self.0, &other.0)
  }

  /// Parse a P-256 secret scalar.
  pub fn from_bytes(bytes: [u8; Self::LENGTH]) -> Result<Self, EcdsaError> {
    Self::from_zeroizing_bytes(ZeroizingBytes::new(bytes))
  }

  /// Try to generate a P-256 secret key with caller-supplied randomness.
  ///
  /// Random scalar candidates are retried when they are zero or outside the
  /// curve order. The retry budget is bounded so a broken deterministic filler
  /// cannot spin forever.
  pub fn try_generate_with<E>(
    mut fill: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<Self, EcdsaKeyGenerationError<E>> {
    for _ in 0..ECDSA_KEY_GENERATION_ATTEMPTS {
      let mut bytes = ZeroizingBytes::zeroed();
      fill(bytes.as_mut_array()).map_err(EcdsaKeyGenerationError::Random)?;
      match Self::from_zeroizing_bytes(bytes) {
        Ok(key) => return Ok(key),
        Err(EcdsaError::InvalidSecretKey) => continue,
        Err(_) => return Err(EcdsaKeyGenerationError::InvalidSecretKey),
      }
    }

    Err(EcdsaKeyGenerationError::InvalidSecretKey)
  }

  /// Try to generate a P-256 secret key from the platform entropy source.
  ///
  /// # Errors
  ///
  /// Returns an ECDSA key-generation error if entropy is unavailable or scalar
  /// rejection exhausts its bounded retry budget.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  pub fn try_generate() -> Result<Self, EcdsaKeyGenerationError<getrandom::Error>> {
    Self::try_generate_with(getrandom::fill)
  }

  /// Construct a P-256 secret key by filling bytes from the provided closure.
  ///
  /// Compatibility name for caller-filled generation. Prefer
  /// [`Self::try_generate_with`] when the entropy source can fail; this method
  /// remains supported until the newer name has shipped for one release.
  pub fn generate(fill: impl FnOnce(&mut [u8; Self::LENGTH])) -> Result<Self, EcdsaError> {
    let mut bytes = ZeroizingBytes::zeroed();
    fill(bytes.as_mut_array());
    Self::from_zeroizing_bytes(bytes)
  }

  fn from_zeroizing_bytes(bytes: ZeroizingBytes<{ Self::LENGTH }>) -> Result<Self, EcdsaError> {
    if !secret_scalar_is_valid::<4, { Self::LENGTH }>(bytes.as_array(), P256_ORDER) {
      return Err(EcdsaError::InvalidSecretKey);
    }

    Ok(Self(*bytes.as_array()))
  }

  /// Explicitly extract the secret key bytes into a zeroizing wrapper.
  #[inline]
  #[must_use]
  pub fn expose_secret(&self) -> SecretBytes<{ Self::LENGTH }> {
    SecretBytes::new(self.0)
  }

  /// Explicitly duplicate this secret key.
  #[inline]
  #[must_use]
  pub const fn duplicate_secret(&self) -> Self {
    Self(self.0)
  }

  /// Borrow the secret key bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }

  /// Derive the matching P-256 public key.
  #[must_use]
  pub fn public_key(&self) -> EcdsaP256PublicKey {
    EcdsaP256PublicKey::from_affine(public_key_from_secret_p256(&self.0))
  }

  /// Derive the matching P-256 public key with caller-supplied blinding.
  ///
  /// The closure should fill the buffer from a CSPRNG. Blinding does not
  /// change the public key; it randomizes the portable fixed-base scalar and
  /// the internal projective representation used during derivation.
  #[must_use]
  pub fn public_key_blinded(&self, fill: impl FnOnce(&mut [u8; 64])) -> EcdsaP256PublicKey {
    let mut blind = ZeroizingBytes::zeroed();
    fill(blind.as_mut_array());
    EcdsaP256PublicKey::from_affine(public_key_from_secret_p256_blinded(&self.0, blind.as_array()))
  }

  /// Sign a message with P-256/SHA-256.
  ///
  /// # Errors
  ///
  /// Returns [`EcdsaError::SigningFailure`] if deterministic nonce derivation
  /// produces an invalid ECDSA scalar. This is cryptographically negligible,
  /// but the API reports it instead of panicking.
  pub fn try_sign(&self, message: &[u8]) -> Result<EcdsaP256Signature, EcdsaError> {
    let digest = Sha256::digest(message);
    sign_digest_p256(&self.0, &digest)
  }

  /// Sign a message with P-256/SHA-256 and caller-supplied blinding.
  ///
  /// The closure should fill the buffer from a CSPRNG. The ECDSA nonce remains
  /// deterministic; the random bytes blind the internal projective `kG` point
  /// and the private-scalar product. The portable backend also adds a random
  /// multiple of the group order before fixed-base multiplication.
  ///
  /// # Errors
  ///
  /// Returns [`EcdsaError::SigningFailure`] if deterministic nonce derivation
  /// reaches an invalid ECDSA scalar.
  pub fn try_sign_blinded(
    &self,
    message: &[u8],
    fill: impl FnOnce(&mut [u8; 64]),
  ) -> Result<EcdsaP256Signature, EcdsaError> {
    let digest = Sha256::digest(message);
    let mut blind = ZeroizingBytes::zeroed();
    fill(blind.as_mut_array());
    sign_digest_p256_blinded(&self.0, &digest, blind.as_array())
  }

  /// Returns a wrapper that displays the secret key bytes as lowercase hex.
  #[must_use]
  pub fn display_secret(&self) -> crate::hex::DisplaySecret<'_> {
    crate::hex::DisplaySecret(self.as_bytes())
  }
}

impl fmt::Debug for EcdsaP256SecretKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("EcdsaP256SecretKey(****)")
  }
}

impl Drop for EcdsaP256SecretKey {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

/// P-384 ECDSA secret scalar.
pub struct EcdsaP384SecretKey([u8; Self::LENGTH]);

impl EcdsaP384SecretKey {
  /// P-384 secret scalar length in bytes.
  pub const LENGTH: usize = 48;

  /// Compare two secret keys without exposing a branchable boolean.
  #[inline]
  pub fn ct_eq(&self, other: &Self) -> ct::CtDecision {
    ct::fixed_eq(&self.0, &other.0)
  }

  /// Parse a P-384 secret scalar.
  pub fn from_bytes(bytes: [u8; Self::LENGTH]) -> Result<Self, EcdsaError> {
    Self::from_zeroizing_bytes(ZeroizingBytes::new(bytes))
  }

  /// Try to generate a P-384 secret key with caller-supplied randomness.
  ///
  /// Random scalar candidates are retried when they are zero or outside the
  /// curve order. The retry budget is bounded so a broken deterministic filler
  /// cannot spin forever.
  pub fn try_generate_with<E>(
    mut fill: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<Self, EcdsaKeyGenerationError<E>> {
    for _ in 0..ECDSA_KEY_GENERATION_ATTEMPTS {
      let mut bytes = ZeroizingBytes::zeroed();
      fill(bytes.as_mut_array()).map_err(EcdsaKeyGenerationError::Random)?;
      match Self::from_zeroizing_bytes(bytes) {
        Ok(key) => return Ok(key),
        Err(EcdsaError::InvalidSecretKey) => continue,
        Err(_) => return Err(EcdsaKeyGenerationError::InvalidSecretKey),
      }
    }

    Err(EcdsaKeyGenerationError::InvalidSecretKey)
  }

  /// Try to generate a P-384 secret key from the platform entropy source.
  ///
  /// # Errors
  ///
  /// Returns an ECDSA key-generation error if entropy is unavailable or scalar
  /// rejection exhausts its bounded retry budget.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  pub fn try_generate() -> Result<Self, EcdsaKeyGenerationError<getrandom::Error>> {
    Self::try_generate_with(getrandom::fill)
  }

  /// Construct a P-384 secret key by filling bytes from the provided closure.
  ///
  /// Compatibility name for caller-filled generation. Prefer
  /// [`Self::try_generate_with`] when the entropy source can fail; this method
  /// remains supported until the newer name has shipped for one release.
  pub fn generate(fill: impl FnOnce(&mut [u8; Self::LENGTH])) -> Result<Self, EcdsaError> {
    let mut bytes = ZeroizingBytes::zeroed();
    fill(bytes.as_mut_array());
    Self::from_zeroizing_bytes(bytes)
  }

  fn from_zeroizing_bytes(bytes: ZeroizingBytes<{ Self::LENGTH }>) -> Result<Self, EcdsaError> {
    if !secret_scalar_is_valid::<6, { Self::LENGTH }>(bytes.as_array(), P384_ORDER) {
      return Err(EcdsaError::InvalidSecretKey);
    }

    Ok(Self(*bytes.as_array()))
  }

  /// Explicitly extract the secret key bytes into a zeroizing wrapper.
  #[inline]
  #[must_use]
  pub fn expose_secret(&self) -> SecretBytes<{ Self::LENGTH }> {
    SecretBytes::new(self.0)
  }

  /// Explicitly duplicate this secret key.
  #[inline]
  #[must_use]
  pub const fn duplicate_secret(&self) -> Self {
    Self(self.0)
  }

  /// Borrow the secret key bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.0
  }

  /// Derive the matching P-384 public key.
  #[must_use]
  pub fn public_key(&self) -> EcdsaP384PublicKey {
    EcdsaP384PublicKey::from_affine(public_key_from_secret_p384(&self.0))
  }

  /// Derive the matching P-384 public key with caller-supplied blinding.
  ///
  /// The closure should fill the buffer from a CSPRNG. Blinding does not
  /// change the public key; it randomizes the internal projective
  /// representation used during derivation.
  #[must_use]
  pub fn public_key_blinded(&self, fill: impl FnOnce(&mut [u8; 96])) -> EcdsaP384PublicKey {
    let mut blind = ZeroizingBytes::zeroed();
    fill(blind.as_mut_array());
    EcdsaP384PublicKey::from_affine(public_key_from_secret_p384_blinded(&self.0, blind.as_array()))
  }

  /// Sign a message with P-384/SHA-384.
  ///
  /// # Errors
  ///
  /// Returns [`EcdsaError::SigningFailure`] if deterministic nonce derivation
  /// produces an invalid ECDSA scalar. This is cryptographically negligible,
  /// but the API reports it instead of panicking.
  pub fn try_sign(&self, message: &[u8]) -> Result<EcdsaP384Signature, EcdsaError> {
    let digest = Sha384::digest(message);
    sign_digest_p384(&self.0, &digest)
  }

  /// Sign a message with P-384/SHA-384 and caller-supplied blinding.
  ///
  /// The closure should fill the buffer from a CSPRNG. The ECDSA nonce remains
  /// deterministic; the random bytes blind the internal projective `kG` point
  /// and mask the private-scalar product.
  ///
  /// # Errors
  ///
  /// Returns [`EcdsaError::SigningFailure`] if deterministic nonce derivation
  /// reaches an invalid ECDSA scalar.
  pub fn try_sign_blinded(
    &self,
    message: &[u8],
    fill: impl FnOnce(&mut [u8; 96]),
  ) -> Result<EcdsaP384Signature, EcdsaError> {
    let digest = Sha384::digest(message);
    let mut blind = ZeroizingBytes::zeroed();
    fill(blind.as_mut_array());
    sign_digest_p384_blinded(&self.0, &digest, blind.as_array())
  }

  /// Returns a wrapper that displays the secret key bytes as lowercase hex.
  #[must_use]
  pub fn display_secret(&self) -> crate::hex::DisplaySecret<'_> {
    crate::hex::DisplaySecret(self.as_bytes())
  }
}

impl fmt::Debug for EcdsaP384SecretKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("EcdsaP384SecretKey(****)")
  }
}

impl Drop for EcdsaP384SecretKey {
  fn drop(&mut self) {
    ct::zeroize(&mut self.0);
  }
}

impl crate::traits::TrySigner for EcdsaP256SecretKey {
  type Signature = EcdsaP256Signature;
  type Error = EcdsaError;

  #[inline]
  fn try_sign(&self, message: &[u8]) -> Result<Self::Signature, Self::Error> {
    EcdsaP256SecretKey::try_sign(self, message)
  }
}

impl crate::traits::TrySigner for EcdsaP384SecretKey {
  type Signature = EcdsaP384Signature;
  type Error = EcdsaError;

  #[inline]
  fn try_sign(&self, message: &[u8]) -> Result<Self::Signature, Self::Error> {
    EcdsaP384SecretKey::try_sign(self, message)
  }
}

/// P-256 ECDSA keypair with typed secret and public halves.
pub struct EcdsaP256Keypair {
  secret: EcdsaP256SecretKey,
  public: EcdsaP256PublicKey,
}

impl EcdsaP256Keypair {
  /// Explicitly duplicate this secret-bearing keypair.
  #[inline]
  #[must_use]
  pub fn duplicate_secret(&self) -> Self {
    Self {
      secret: self.secret.duplicate_secret(),
      public: self.public.clone(),
    }
  }

  /// Try to generate a P-256 keypair with caller-supplied randomness.
  #[inline]
  pub fn try_generate_with<E>(
    fill: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<Self, EcdsaKeyGenerationError<E>> {
    EcdsaP256SecretKey::try_generate_with(fill).map(Self::from_secret_key)
  }

  /// Try to generate a P-256 keypair from the platform entropy source.
  ///
  /// # Errors
  ///
  /// Returns an ECDSA key-generation error if entropy is unavailable or scalar
  /// rejection exhausts its bounded retry budget.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  pub fn try_generate() -> Result<Self, EcdsaKeyGenerationError<getrandom::Error>> {
    EcdsaP256SecretKey::try_generate().map(Self::from_secret_key)
  }

  /// Derive a P-256 keypair from a secret key.
  #[must_use]
  pub fn from_secret_key(secret: EcdsaP256SecretKey) -> Self {
    let public = secret.public_key();
    Self { secret, public }
  }

  /// Borrow the secret key.
  #[must_use]
  pub const fn secret_key(&self) -> &EcdsaP256SecretKey {
    &self.secret
  }

  /// Return the public key.
  #[must_use]
  pub fn public_key(&self) -> EcdsaP256PublicKey {
    self.public.clone()
  }

  /// Sign a message with P-256/SHA-256.
  pub fn try_sign(&self, message: &[u8]) -> Result<EcdsaP256Signature, EcdsaError> {
    self.secret.try_sign(message)
  }

  /// Sign a message with P-256/SHA-256 and caller-supplied blinding.
  ///
  /// # Errors
  ///
  /// Returns [`EcdsaError::SigningFailure`] if deterministic nonce derivation
  /// reaches an invalid ECDSA scalar.
  pub fn try_sign_blinded(
    &self,
    message: &[u8],
    fill: impl FnOnce(&mut [u8; 64]),
  ) -> Result<EcdsaP256Signature, EcdsaError> {
    self.secret.try_sign_blinded(message, fill)
  }
}

impl crate::traits::TrySigner for EcdsaP256Keypair {
  type Signature = EcdsaP256Signature;
  type Error = EcdsaError;

  #[inline]
  fn try_sign(&self, message: &[u8]) -> Result<Self::Signature, Self::Error> {
    EcdsaP256Keypair::try_sign(self, message)
  }
}

impl fmt::Debug for EcdsaP256Keypair {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("EcdsaP256Keypair")
      .field("public", &self.public)
      .finish_non_exhaustive()
  }
}

/// P-384 ECDSA keypair with typed secret and public halves.
pub struct EcdsaP384Keypair {
  secret: EcdsaP384SecretKey,
  public: EcdsaP384PublicKey,
}

impl EcdsaP384Keypair {
  /// Explicitly duplicate this secret-bearing keypair.
  #[inline]
  #[must_use]
  pub fn duplicate_secret(&self) -> Self {
    Self {
      secret: self.secret.duplicate_secret(),
      public: self.public.clone(),
    }
  }

  /// Try to generate a P-384 keypair with caller-supplied randomness.
  #[inline]
  pub fn try_generate_with<E>(
    fill: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<Self, EcdsaKeyGenerationError<E>> {
    EcdsaP384SecretKey::try_generate_with(fill).map(Self::from_secret_key)
  }

  /// Try to generate a P-384 keypair from the platform entropy source.
  ///
  /// # Errors
  ///
  /// Returns an ECDSA key-generation error if entropy is unavailable or scalar
  /// rejection exhausts its bounded retry budget.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  pub fn try_generate() -> Result<Self, EcdsaKeyGenerationError<getrandom::Error>> {
    EcdsaP384SecretKey::try_generate().map(Self::from_secret_key)
  }

  /// Derive a P-384 keypair from a secret key.
  #[must_use]
  pub fn from_secret_key(secret: EcdsaP384SecretKey) -> Self {
    let public = secret.public_key();
    Self { secret, public }
  }

  /// Borrow the secret key.
  #[must_use]
  pub const fn secret_key(&self) -> &EcdsaP384SecretKey {
    &self.secret
  }

  /// Return the public key.
  #[must_use]
  pub fn public_key(&self) -> EcdsaP384PublicKey {
    self.public.clone()
  }

  /// Sign a message with P-384/SHA-384.
  pub fn try_sign(&self, message: &[u8]) -> Result<EcdsaP384Signature, EcdsaError> {
    self.secret.try_sign(message)
  }

  /// Sign a message with P-384/SHA-384 and caller-supplied blinding.
  ///
  /// # Errors
  ///
  /// Returns [`EcdsaError::SigningFailure`] if deterministic nonce derivation
  /// reaches an invalid ECDSA scalar.
  pub fn try_sign_blinded(
    &self,
    message: &[u8],
    fill: impl FnOnce(&mut [u8; 96]),
  ) -> Result<EcdsaP384Signature, EcdsaError> {
    self.secret.try_sign_blinded(message, fill)
  }
}

impl crate::traits::TrySigner for EcdsaP384Keypair {
  type Signature = EcdsaP384Signature;
  type Error = EcdsaError;

  #[inline]
  fn try_sign(&self, message: &[u8]) -> Result<Self::Signature, Self::Error> {
    EcdsaP384Keypair::try_sign(self, message)
  }
}

impl fmt::Debug for EcdsaP384Keypair {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("EcdsaP384Keypair")
      .field("public", &self.public)
      .finish_non_exhaustive()
  }
}

/// P-256 ECDSA public key.
#[derive(Clone)]
pub struct EcdsaP256PublicKey {
  point: Affine<4>,
  table: [Affine<4>; COMB_TABLE_SIZE],
}

impl EcdsaP256PublicKey {
  /// Uncompressed SEC1 public-key length in bytes.
  pub const SEC1_LENGTH: usize = 65;

  fn from_affine(point: Affine<4>) -> Self {
    Self {
      point,
      table: precompute_comb_table(point),
    }
  }

  /// Parse an uncompressed SEC1 P-256 public key.
  pub fn from_sec1_bytes(bytes: &[u8]) -> Result<Self, EcdsaError> {
    parse_public_key(bytes, &P256).map(Self::from_affine)
  }

  /// Parse a P-256 SubjectPublicKeyInfo public key.
  pub fn from_spki_der(der: &[u8]) -> Result<Self, EcdsaError> {
    let public_key = parse_spki_der(der, SECP256R1_OID)?;
    Self::from_sec1_bytes(public_key)
  }

  /// Return the uncompressed SEC1 public-key bytes.
  #[must_use]
  pub fn to_sec1_bytes(&self) -> [u8; Self::SEC1_LENGTH] {
    encode_sec1(&self.point)
  }

  /// Verify a message against a P-256/SHA-256 ECDSA signature.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify(&self, message: &[u8], signature: &EcdsaP256Signature) -> Result<(), VerificationError> {
    let digest = Sha256::digest(message);
    verify_digest(&P256, &self.table, signature.r, signature.s, &digest)
  }
}

impl PartialEq for EcdsaP256PublicKey {
  fn eq(&self, other: &Self) -> bool {
    self.point == other.point
  }
}

impl Eq for EcdsaP256PublicKey {}

impl Hash for EcdsaP256PublicKey {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.to_sec1_bytes().hash(state);
  }
}

impl fmt::Debug for EcdsaP256PublicKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "EcdsaP256PublicKey(")?;
    crate::hex::fmt_hex_lower(&self.to_sec1_bytes(), f)?;
    write!(f, ")")
  }
}

/// P-384 ECDSA public key.
#[derive(Clone)]
pub struct EcdsaP384PublicKey {
  point: Affine<6>,
  table: [Affine<6>; COMB_TABLE_SIZE],
}

impl EcdsaP384PublicKey {
  /// Uncompressed SEC1 public-key length in bytes.
  pub const SEC1_LENGTH: usize = 97;

  fn from_affine(point: Affine<6>) -> Self {
    Self {
      point,
      table: precompute_comb_table(point),
    }
  }

  /// Parse an uncompressed SEC1 P-384 public key.
  pub fn from_sec1_bytes(bytes: &[u8]) -> Result<Self, EcdsaError> {
    parse_public_key(bytes, &P384).map(Self::from_affine)
  }

  /// Parse a P-384 SubjectPublicKeyInfo public key.
  pub fn from_spki_der(der: &[u8]) -> Result<Self, EcdsaError> {
    let public_key = parse_spki_der(der, SECP384R1_OID)?;
    Self::from_sec1_bytes(public_key)
  }

  /// Return the uncompressed SEC1 public-key bytes.
  #[must_use]
  pub fn to_sec1_bytes(&self) -> [u8; Self::SEC1_LENGTH] {
    encode_sec1(&self.point)
  }

  /// Verify a message against a P-384/SHA-384 ECDSA signature.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify(&self, message: &[u8], signature: &EcdsaP384Signature) -> Result<(), VerificationError> {
    let digest = Sha384::digest(message);
    verify_digest(&P384, &self.table, signature.r, signature.s, &digest)
  }
}

impl PartialEq for EcdsaP384PublicKey {
  fn eq(&self, other: &Self) -> bool {
    self.point == other.point
  }
}

impl Eq for EcdsaP384PublicKey {}

impl Hash for EcdsaP384PublicKey {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.to_sec1_bytes().hash(state);
  }
}

impl fmt::Debug for EcdsaP384PublicKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "EcdsaP384PublicKey(")?;
    crate::hex::fmt_hex_lower(&self.to_sec1_bytes(), f)?;
    write!(f, ")")
  }
}

impl crate::traits::Verifier<EcdsaP256Signature> for EcdsaP256PublicKey {
  #[inline]
  fn verify(&self, message: &[u8], signature: &EcdsaP256Signature) -> Result<(), VerificationError> {
    EcdsaP256PublicKey::verify(self, message, signature)
  }
}

impl crate::traits::Verifier<EcdsaP384Signature> for EcdsaP384PublicKey {
  #[inline]
  fn verify(&self, message: &[u8], signature: &EcdsaP384Signature) -> Result<(), VerificationError> {
    EcdsaP384PublicKey::verify(self, message, signature)
  }
}

/// P-256 ECDSA signature encoded as fixed-width `r || s`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EcdsaP256Signature {
  bytes: [u8; Self::LENGTH],
  r: Uint<4>,
  s: Uint<4>,
}

impl EcdsaP256Signature {
  /// Raw P-256 ECDSA signature length in bytes.
  pub const LENGTH: usize = 64;

  /// Parse a raw fixed-width `r || s` P-256 signature.
  pub fn from_bytes(bytes: [u8; Self::LENGTH]) -> Result<Self, EcdsaError> {
    let (r, s) = parse_signature_scalars(&bytes, P256_ORDER)?;
    Ok(Self { bytes, r, s })
  }

  /// Parse a DER `Ecdsa-Sig-Value` P-256 signature.
  pub fn from_der(der: &[u8]) -> Result<Self, EcdsaError> {
    Self::from_bytes(parse_signature_der_bytes::<4, { Self::LENGTH }>(der)?)
  }

  fn from_scalars(r: Uint<4>, s: Uint<4>) -> Self {
    let mut bytes = [0u8; Self::LENGTH];
    r.write_be(&mut bytes[..32]);
    s.write_be(&mut bytes[32..]);
    Self { bytes, r, s }
  }

  /// Return raw fixed-width `r || s` bytes.
  #[inline]
  #[must_use]
  pub const fn to_bytes(self) -> [u8; Self::LENGTH] {
    self.bytes
  }

  /// Borrow raw fixed-width `r || s` bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.bytes
  }
}

impl fmt::Debug for EcdsaP256Signature {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "EcdsaP256Signature(")?;
    crate::hex::fmt_hex_lower(&self.bytes, f)?;
    write!(f, ")")
  }
}

/// P-384 ECDSA signature encoded as fixed-width `r || s`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EcdsaP384Signature {
  bytes: [u8; Self::LENGTH],
  r: Uint<6>,
  s: Uint<6>,
}

impl EcdsaP384Signature {
  /// Raw P-384 ECDSA signature length in bytes.
  pub const LENGTH: usize = 96;

  /// Parse a raw fixed-width `r || s` P-384 signature.
  pub fn from_bytes(bytes: [u8; Self::LENGTH]) -> Result<Self, EcdsaError> {
    let (r, s) = parse_signature_scalars(&bytes, P384_ORDER)?;
    Ok(Self { bytes, r, s })
  }

  /// Parse a DER `Ecdsa-Sig-Value` P-384 signature.
  pub fn from_der(der: &[u8]) -> Result<Self, EcdsaError> {
    Self::from_bytes(parse_signature_der_bytes::<6, { Self::LENGTH }>(der)?)
  }

  fn from_scalars(r: Uint<6>, s: Uint<6>) -> Self {
    let mut bytes = [0u8; Self::LENGTH];
    r.write_be(&mut bytes[..48]);
    s.write_be(&mut bytes[48..]);
    Self { bytes, r, s }
  }

  /// Return raw fixed-width `r || s` bytes.
  #[inline]
  #[must_use]
  pub const fn to_bytes(self) -> [u8; Self::LENGTH] {
    self.bytes
  }

  /// Borrow raw fixed-width `r || s` bytes.
  #[inline]
  #[must_use]
  pub const fn as_bytes(&self) -> &[u8; Self::LENGTH] {
    &self.bytes
  }
}

impl fmt::Debug for EcdsaP384Signature {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "EcdsaP384Signature(")?;
    crate::hex::fmt_hex_lower(&self.bytes, f)?;
    write!(f, ")")
  }
}

#[inline(always)]
fn mask_nonzero_u64(value: u64) -> u64 {
  0u64.wrapping_sub((value | value.wrapping_neg()) >> 63)
}

#[inline(always)]
fn mask_zero_u64(value: u64) -> u64 {
  !mask_nonzero_u64(value)
}

#[inline(always)]
fn mask_eq_usize(lhs: usize, rhs: usize) -> u64 {
  mask_zero_u64((lhs ^ rhs) as u64)
}

#[inline(always)]
fn mask_not(mask: u64) -> u64 {
  !mask
}

#[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
#[inline(never)]
fn ct_mul_u64_wide(lhs: u64, rhs: u64) -> (u64, u64) {
  let mut product_lo = 0u64;
  let mut product_hi = 0u64;
  let mut multiplicand_lo = lhs;
  let mut multiplicand_hi = 0u64;
  let mut multiplier = rhs;
  let mut bit = 0u32;

  while bit < u64::BITS {
    // Keep LLVM from recognizing the bit-serial product and lowering it back to a target multiply.
    // The CT artifact gate independently rejects scalar multiply in these ECDSA closures.
    let selected_bit = core::hint::black_box(multiplier & 1);
    let mask = 0u64.wrapping_sub(selected_bit);
    let (next_lo, carry) = product_lo.overflowing_add(multiplicand_lo & mask);
    let (next_hi, _) = product_hi.overflowing_add(multiplicand_hi & mask);
    let (next_hi, _) = next_hi.overflowing_add(carry as u64);
    product_lo = next_lo;
    product_hi = next_hi;

    multiplicand_hi = (multiplicand_hi << 1) | (multiplicand_lo >> 63);
    multiplicand_lo <<= 1;
    multiplier >>= 1;
    bit += 1;
  }

  (product_lo, product_hi)
}

#[inline(always)]
fn mul_u64_wide(lhs: u64, rhs: u64) -> (u64, u64) {
  #[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
  {
    ct_mul_u64_wide(lhs, rhs)
  }

  #[cfg(not(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
  {
    let product = (lhs as u128) * (rhs as u128);
    (product as u64, (product >> 64) as u64)
  }
}

#[inline(always)]
fn mul_u64_low(lhs: u64, rhs: u64) -> u64 {
  #[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
  {
    ct_mul_u64_wide(lhs, rhs).0
  }

  #[cfg(not(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
  {
    lhs.wrapping_mul(rhs)
  }
}

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
struct ZeroizingWords<const N: usize> {
  value: [u64; N],
}

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
impl<const N: usize> ZeroizingWords<N> {
  const fn new(value: [u64; N]) -> Self {
    Self { value }
  }

  const fn zeroed() -> Self {
    Self { value: [0u64; N] }
  }

  const fn as_array(&self) -> &[u64; N] {
    &self.value
  }

  fn as_mut_array(&mut self) -> &mut [u64; N] {
    &mut self.value
  }
}

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
impl<const N: usize> Drop for ZeroizingWords<N> {
  fn drop(&mut self) {
    ct::zeroize_words(&mut self.value);
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Uint<const L: usize>([u64; L]);

impl<const L: usize> Uint<L> {
  const ZERO: Self = Self([0u64; L]);
  const ONE: Self = Self::from_u64(1);

  const fn from_u64(value: u64) -> Self {
    let mut limbs = [0u64; L];
    if L > 0 {
      limbs[0] = value;
    }
    Self(limbs)
  }

  fn from_be_slice(bytes: &[u8]) -> Result<Self, EcdsaError> {
    if bytes.len() > L * 8 {
      return Err(EcdsaError::InvalidSignature);
    }
    let mut out = Self::ZERO;
    for (index, byte) in bytes.iter().rev().copied().enumerate() {
      let limb = index / 8;
      let shift = (index % 8) * 8;
      let Some(out_limb) = out.0.get_mut(limb) else {
        return Err(EcdsaError::InvalidSignature);
      };
      *out_limb |= u64::from(byte) << shift;
    }
    Ok(out)
  }

  fn from_be_slice_mod(bytes: &[u8], modulus: Self) -> Self {
    let mut out = Self::from_be_slice(bytes).unwrap_or(Self::ZERO);
    while out.cmp(&modulus).is_ge() {
      out = out.sub_raw(&modulus).0;
    }
    out
  }

  fn write_be(self, out: &mut [u8]) {
    out.fill(0);
    for (index, byte) in out.iter_mut().rev().enumerate() {
      let limb = index / 8;
      let shift = (index % 8) * 8;
      if let Some(limb_value) = self.0.get(limb) {
        *byte = (*limb_value >> shift) as u8;
      }
    }
  }

  fn zeroize_no_fence(&mut self) {
    ct::zeroize_words_no_fence(&mut self.0);
  }

  fn is_zero(&self) -> bool {
    self.0.iter().copied().fold(0u64, |acc, limb| acc | limb) == 0
  }

  fn is_in_range(&self, modulus: &Self) -> bool {
    !self.is_zero() && self.cmp(modulus).is_lt()
  }

  fn is_in_range_ct(&self, modulus: &Self) -> bool {
    let zero = self.ct_is_zero_mask();
    let ge = self.ct_ge_mask(modulus);
    (zero | ge) == 0
  }

  fn cmp(&self, other: &Self) -> core::cmp::Ordering {
    for (&left, &right) in self.0.iter().zip(other.0.iter()).rev() {
      if left < right {
        return core::cmp::Ordering::Less;
      }
      if left > right {
        return core::cmp::Ordering::Greater;
      }
    }
    core::cmp::Ordering::Equal
  }

  fn is_one(&self) -> bool {
    self.0.first().copied() == Some(1) && self.0.iter().skip(1).all(|&limb| limb == 0)
  }

  fn is_even(&self) -> bool {
    self.0.first().copied().unwrap_or(0) & 1 == 0
  }

  fn bit(&self, index: usize) -> bool {
    let limb = index / 64;
    let shift = index % 64;
    self.0.get(limb).copied().unwrap_or(0) & (1u64 << shift) != 0
  }

  fn shr1_with_carry(&mut self, mut carry: u64) {
    for limb in self.0.iter_mut().rev() {
      let next = *limb & 1;
      *limb = (*limb >> 1) | (carry << 63);
      carry = next;
    }
  }

  fn add_raw(&self, other: &Self) -> (Self, u64) {
    let mut out = [0u64; L];
    let mut carry = 0u64;
    for ((out_limb, &left), &right) in out.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
      let (sum, overflow0) = left.overflowing_add(right);
      let (sum, overflow1) = sum.overflowing_add(carry);
      *out_limb = sum;
      carry = u64::from(overflow0 | overflow1);
    }
    (Self(out), carry)
  }

  fn sub_raw(&self, other: &Self) -> (Self, u64) {
    let mut out = [0u64; L];
    let mut borrow = 0u64;
    for ((out_limb, &left), &right) in out.iter_mut().zip(self.0.iter()).zip(other.0.iter()) {
      let (diff, overflow0) = left.overflowing_sub(right);
      let (diff, overflow1) = diff.overflowing_sub(borrow);
      *out_limb = diff;
      borrow = u64::from(overflow0 | overflow1);
    }
    (Self(out), borrow)
  }

  fn add_mod_ct(&self, other: &Self, modulus: Self) -> Self {
    let (sum, carry) = self.add_raw(other);
    let (reduced, borrow) = sum.sub_raw(&modulus);
    Self::select(sum, reduced, mask_nonzero_u64(carry) | mask_zero_u64(borrow))
  }

  fn sub_mod(&self, other: &Self, modulus: Self) -> Self {
    let (diff, borrow) = self.sub_raw(other);
    if borrow == 0 { diff } else { diff.add_raw(&modulus).0 }
  }

  fn sub_mod_ct(&self, other: &Self, modulus: Self) -> Self {
    let (diff, borrow) = self.sub_raw(other);
    let added = diff.add_raw(&modulus).0;
    Self::select(diff, added, mask_nonzero_u64(borrow))
  }

  fn reduce_once_ct(&self, modulus: Self) -> Self {
    let (reduced, borrow) = self.sub_raw(&modulus);
    Self::select(reduced, *self, mask_nonzero_u64(borrow))
  }

  fn ct_is_zero_mask(&self) -> u64 {
    let diff = self.0.iter().copied().fold(0u64, |acc, limb| acc | limb);
    mask_zero_u64(diff)
  }

  fn ct_lt_mask(&self, other: &Self) -> u64 {
    let (_, borrow) = self.sub_raw(other);
    mask_nonzero_u64(borrow)
  }

  fn ct_ge_mask(&self, other: &Self) -> u64 {
    !self.ct_lt_mask(other)
  }

  fn ct_gt_mask(&self, other: &Self) -> u64 {
    other.ct_lt_mask(self)
  }

  fn ct_bit_mask(&self, index: usize) -> u64 {
    let limb = index / 64;
    let shift = index % 64;
    mask_nonzero_u64((self.0.get(limb).copied().unwrap_or(0) >> shift) & 1)
  }

  fn public_window(&self, start: usize, width: usize) -> usize {
    let mut out = 0usize;
    for offset in 0..width {
      out |= usize::from(self.bit(start.strict_add(offset))) << offset;
    }
    out
  }

  fn select(lhs: Self, rhs: Self, mask: u64) -> Self {
    let mut out = [0u64; L];
    for ((dst, &left), &right) in out.iter_mut().zip(lhs.0.iter()).zip(rhs.0.iter()) {
      *dst = left ^ (mask & (left ^ right));
    }
    Self(out)
  }

  fn inv_mod_vartime(&self, modulus: Self) -> Self {
    if self.is_zero() {
      return Self::ZERO;
    }

    let mut u = *self;
    let mut v = modulus;
    let mut x1 = Self::ONE;
    let mut x2 = Self::ZERO;

    while !u.is_one() && !v.is_one() {
      while u.is_even() {
        u.shr1_with_carry(0);
        if x1.is_even() {
          x1.shr1_with_carry(0);
        } else {
          let (sum, carry) = x1.add_raw(&modulus);
          x1 = sum;
          x1.shr1_with_carry(carry);
        }
      }

      while v.is_even() {
        v.shr1_with_carry(0);
        if x2.is_even() {
          x2.shr1_with_carry(0);
        } else {
          let (sum, carry) = x2.add_raw(&modulus);
          x2 = sum;
          x2.shr1_with_carry(carry);
        }
      }

      if u.cmp(&v).is_ge() {
        u = u.sub_raw(&v).0;
        x1 = x1.sub_mod(&x2, modulus);
      } else {
        v = v.sub_raw(&u).0;
        x2 = x2.sub_mod(&x1, modulus);
      }
    }

    if u.is_one() { x1 } else { x2 }
  }

  fn inv_mod_ct(&self, modulus: &'static Modulus<L>, exponent: Self) -> Self {
    montgomery_mul(self.inv_mod_ct_montgomery(modulus, exponent), Self::ONE, modulus)
  }

  fn inv_mod_ct_montgomery(&self, modulus: &'static Modulus<L>, exponent: Self) -> Self {
    #[cfg(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", target_os = "linux")
    ))]
    {
      if is_p256_order_modulus(modulus) || is_p384_order_modulus(modulus) {
        let inverse = ZeroizingWords::new(ecdsa_platform_asm::scalar_inverse(&self.0, &modulus.value.0));
        return montgomery_mul(Uint(*inverse.as_array()), modulus.r2, modulus);
      }
    }

    const WINDOW_BITS: usize = 4;
    const WINDOW_SIZE: usize = 1 << WINDOW_BITS;

    let mut base = montgomery_mul(*self, modulus.r2, modulus);
    let mut acc = montgomery_mul(Self::ONE, modulus.r2, modulus);
    let mut powers = [acc; WINDOW_SIZE];
    let mut next_power = base;
    let mut remaining = WINDOW_SIZE - 1;
    for power in powers.iter_mut().skip(1) {
      *power = next_power;
      remaining -= 1;
      if remaining != 0 {
        next_power = montgomery_mul(next_power, base, modulus);
      }
    }

    let mut bit = L * 64;
    while bit > 0 {
      bit -= WINDOW_BITS;
      for _ in 0..WINDOW_BITS {
        acc = montgomery_square(acc, modulus);
      }
      let digit = exponent.public_window(bit, WINDOW_BITS);
      if digit != 0
        && let Some(power) = powers.get(digit).copied()
      {
        acc = montgomery_mul(acc, power, modulus);
      }
    }

    let result = acc;
    base.zeroize_no_fence();
    acc.zeroize_no_fence();
    next_power.zeroize_no_fence();
    for power in powers.iter_mut() {
      power.zeroize_no_fence();
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    result
  }
}

struct SecretScalar<const L: usize> {
  value: Uint<L>,
}

impl<const L: usize> SecretScalar<L> {
  const fn new(value: Uint<L>) -> Self {
    Self { value }
  }

  fn from_be_bytes<const N: usize>(bytes: &[u8; N]) -> Self {
    debug_assert!(N <= L * 8);

    let mut out = Uint::ZERO;
    for (index, byte) in bytes.iter().rev().copied().enumerate() {
      let limb = index / 8;
      let shift = (index % 8) * 8;
      if let Some(out_limb) = out.0.get_mut(limb) {
        *out_limb |= u64::from(byte) << shift;
      }
    }
    Self::new(out)
  }

  const fn value(&self) -> Uint<L> {
    self.value
  }

  #[cfg(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", target_os = "linux")
  ))]
  const fn words(&self) -> &[u64; L] {
    &self.value.0
  }
}

impl<const L: usize> Drop for SecretScalar<L> {
  fn drop(&mut self) {
    ct::zeroize_words(&mut self.value.0);
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Modulus<const L: usize> {
  value: Uint<L>,
  n0_inv: u64,
  r2: Uint<L>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct FieldElement<const L: usize> {
  value: Uint<L>,
  modulus: &'static Modulus<L>,
}

impl<const L: usize> FieldElement<L> {
  fn from_uint(value: Uint<L>, modulus: &'static Modulus<L>) -> Self {
    Self {
      value: montgomery_mul(value, modulus.r2, modulus),
      modulus,
    }
  }

  const fn from_montgomery(value: Uint<L>, modulus: &'static Modulus<L>) -> Self {
    Self { value, modulus }
  }

  fn to_uint(self) -> Uint<L> {
    montgomery_mul(self.value, Uint::ONE, self.modulus)
  }

  fn zero(modulus: &'static Modulus<L>) -> Self {
    Self::from_montgomery(Uint::ZERO, modulus)
  }

  fn one(modulus: &'static Modulus<L>) -> Self {
    Self::from_uint(Uint::ONE, modulus)
  }

  #[allow(clippy::indexing_slicing)]
  fn add(self, rhs: Self) -> Self {
    if is_p384_field_modulus(self.modulus) {
      let reduced = add_p384_field(
        [
          self.value.0[0],
          self.value.0[1],
          self.value.0[2],
          self.value.0[3],
          self.value.0[4],
          self.value.0[5],
        ],
        [
          rhs.value.0[0],
          rhs.value.0[1],
          rhs.value.0[2],
          rhs.value.0[3],
          rhs.value.0[4],
          rhs.value.0[5],
        ],
      );
      let mut out = [0u64; L];
      out.copy_from_slice(&reduced.0);
      return Self::from_montgomery(Uint(out), self.modulus);
    }
    Self::from_montgomery(self.value.add_mod_ct(&rhs.value, self.modulus.value), self.modulus)
  }

  #[allow(clippy::indexing_slicing)]
  fn sub(self, rhs: Self) -> Self {
    if is_p384_field_modulus(self.modulus) {
      let reduced = sub_p384_field(
        [
          self.value.0[0],
          self.value.0[1],
          self.value.0[2],
          self.value.0[3],
          self.value.0[4],
          self.value.0[5],
        ],
        [
          rhs.value.0[0],
          rhs.value.0[1],
          rhs.value.0[2],
          rhs.value.0[3],
          rhs.value.0[4],
          rhs.value.0[5],
        ],
      );
      let mut out = [0u64; L];
      out.copy_from_slice(&reduced.0);
      return Self::from_montgomery(Uint(out), self.modulus);
    }
    Self::from_montgomery(self.value.sub_mod_ct(&rhs.value, self.modulus.value), self.modulus)
  }

  fn mul(self, rhs: Self) -> Self {
    Self::from_montgomery(montgomery_mul(self.value, rhs.value, self.modulus), self.modulus)
  }

  fn square(self) -> Self {
    Self::from_montgomery(montgomery_square(self.value, self.modulus), self.modulus)
  }

  fn double(self) -> Self {
    self.add(self)
  }

  fn triple(self) -> Self {
    self.double().add(self)
  }

  fn inv(self) -> Self {
    Self::from_uint(self.to_uint().inv_mod_vartime(self.modulus.value), self.modulus)
  }

  fn inv_ct(self, exponent: Uint<L>) -> Self {
    #[cfg(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", target_os = "linux")
    ))]
    {
      if is_p384_field_modulus(self.modulus) {
        let value = [
          self.value.0[0],
          self.value.0[1],
          self.value.0[2],
          self.value.0[3],
          self.value.0[4],
          self.value.0[5],
        ];
        let inverse = ecdsa_platform_asm::p384_field_inverse(&value);
        let mut out = [0u64; L];
        out.copy_from_slice(&inverse);
        return Self::from_montgomery(Uint(out), self.modulus);
      }
    }

    Self::from_uint(self.to_uint().inv_mod_ct(self.modulus, exponent), self.modulus)
  }

  fn select(lhs: Self, rhs: Self, mask: u64) -> Self {
    Self::from_montgomery(Uint::select(lhs.value, rhs.value, mask), lhs.modulus)
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Affine<const L: usize> {
  x: FieldElement<L>,
  y: FieldElement<L>,
}

#[derive(Clone, Copy)]
struct Jacobian<const L: usize> {
  x: FieldElement<L>,
  y: FieldElement<L>,
  z: FieldElement<L>,
  infinity: bool,
}

impl<const L: usize> Jacobian<L> {
  fn infinity(modulus: &'static Modulus<L>) -> Self {
    Self {
      x: FieldElement::zero(modulus),
      y: FieldElement::one(modulus),
      z: FieldElement::zero(modulus),
      infinity: true,
    }
  }

  fn from_affine(point: Affine<L>) -> Self {
    Self {
      x: point.x,
      y: point.y,
      z: FieldElement::one(point.x.modulus),
      infinity: false,
    }
  }

  fn select(lhs: Self, rhs: Self, mask: u64) -> Self {
    let lhs_infinity = 0u64.wrapping_sub(u64::from(lhs.infinity));
    let rhs_infinity = 0u64.wrapping_sub(u64::from(rhs.infinity));
    let selected_infinity = lhs_infinity ^ (mask & (lhs_infinity ^ rhs_infinity));
    Self {
      x: FieldElement::select(lhs.x, rhs.x, mask),
      y: FieldElement::select(lhs.y, rhs.y, mask),
      z: FieldElement::select(lhs.z, rhs.z, mask),
      infinity: selected_infinity != 0,
    }
  }

  fn infinity_mask(&self) -> u64 {
    0u64.wrapping_sub(u64::from(self.infinity))
  }

  fn double(self) -> Self {
    if self.infinity || self.y.value.is_zero() {
      return Self::infinity(self.x.modulus);
    }

    let delta = self.z.square();
    let gamma = self.y.square();
    let beta = self.x.mul(gamma);
    let alpha = self.x.sub(delta).mul(self.x.add(delta)).triple();
    let x3 = alpha.square().sub(beta.double().double().double());
    let z3 = self.y.add(self.z).square().sub(gamma).sub(delta);
    let y3 = alpha
      .mul(beta.double().double().sub(x3))
      .sub(gamma.square().double().double().double());

    Self {
      x: x3,
      y: y3,
      z: z3,
      infinity: false,
    }
  }

  fn double_ct(self) -> Self {
    let delta = self.z.square();
    let gamma = self.y.square();
    let beta = self.x.mul(gamma);
    let alpha = self.x.sub(delta).mul(self.x.add(delta)).triple();
    let x3 = alpha.square().sub(beta.double().double().double());
    let z3 = self.y.add(self.z).square().sub(gamma).sub(delta);
    let y3 = alpha
      .mul(beta.double().double().sub(x3))
      .sub(gamma.square().double().double().double());

    let raw = Self {
      x: x3,
      y: y3,
      z: z3,
      infinity: false,
    };
    Self::select(
      raw,
      Self::infinity(self.x.modulus),
      self.infinity_mask() | self.y.value.ct_is_zero_mask(),
    )
  }

  fn add_mixed(self, rhs: Affine<L>) -> Self {
    if self.infinity {
      return Self::from_affine(rhs);
    }

    let z1z1 = self.z.square();
    let u2 = rhs.x.mul(z1z1);
    let s2 = rhs.y.mul(self.z).mul(z1z1);
    let h = u2.sub(self.x);
    let r = s2.sub(self.y).double();

    if h.value.is_zero() {
      return if r.value.is_zero() {
        self.double()
      } else {
        Self::infinity(self.x.modulus)
      };
    }

    let hh = h.square();
    let i = hh.double().double();
    let j = h.mul(i);
    let v = self.x.mul(i);
    let x3 = r.square().sub(j).sub(v.double());
    let y3 = r.mul(v.sub(x3)).sub(self.y.mul(j).double());
    let z3 = self.z.add(h).square().sub(z1z1).sub(hh);

    Self {
      x: x3,
      y: y3,
      z: z3,
      infinity: false,
    }
  }

  fn add_mixed_ct(self, rhs: Affine<L>, rhs_infinity_mask: u64) -> Self {
    let z1z1 = self.z.square();
    let u2 = rhs.x.mul(z1z1);
    let s2 = rhs.y.mul(self.z).mul(z1z1);
    let h = u2.sub(self.x);
    let r = s2.sub(self.y).double();

    let hh = h.square();
    let i = hh.double().double();
    let j = h.mul(i);
    let v = self.x.mul(i);
    let x3 = r.square().sub(j).sub(v.double());
    let y3 = r.mul(v.sub(x3)).sub(self.y.mul(j).double());
    let z3 = self.z.add(h).square().sub(z1z1).sub(hh);
    let raw = Self {
      x: x3,
      y: y3,
      z: z3,
      infinity: false,
    };

    let h_zero = h.value.ct_is_zero_mask();
    let r_zero = r.value.ct_is_zero_mask();
    let double = self.double_ct();
    let infinity = Self::infinity(self.x.modulus);
    let rhs_jacobian = Self::from_affine(rhs);

    let exceptional = h_zero & r_zero;
    let opposite = h_zero & mask_not(r_zero);
    let with_exception = Self::select(Self::select(raw, double, exceptional), infinity, opposite);
    let with_self_infinity = Self::select(with_exception, rhs_jacobian, self.infinity_mask());
    Self::select(with_self_infinity, self, rhs_infinity_mask)
  }

  fn randomize_z(self, z: FieldElement<L>) -> Self {
    let z2 = z.square();
    let z3 = z2.mul(z);
    Self {
      x: self.x.mul(z2),
      y: self.y.mul(z3),
      z: self.z.mul(z),
      infinity: self.infinity,
    }
  }

  fn to_affine_ct(self, exponent: Uint<L>) -> Affine<L> {
    let inv_z = self.z.inv_ct(exponent);
    let z2 = inv_z.square();
    let z3 = z2.mul(inv_z);
    Affine {
      x: self.x.mul(z2),
      y: self.y.mul(z3),
    }
  }

  fn to_affine_x_ct(self, exponent: Uint<L>) -> FieldElement<L> {
    let inv_z = self.z.inv_ct(exponent);
    let z2 = inv_z.square();
    self.x.mul(z2)
  }
}

#[derive(Clone, Copy)]
struct Curve<const L: usize> {
  field_modulus: &'static Modulus<L>,
  scalar_modulus: &'static Modulus<L>,
  scalar_inverse_exponent: Uint<L>,
  field_inverse_exponent: Uint<L>,
  half_order: Uint<L>,
  b: Uint<L>,
  generator_x: Uint<L>,
  generator_y: Uint<L>,
  generator_comb_x: [Uint<L>; COMB_TABLE_SIZE],
  generator_comb_y: [Uint<L>; COMB_TABLE_SIZE],
  signing_comb_width: usize,
  signing_comb_rows: usize,
  signing_generator_comb_x: &'static [Uint<L>],
  signing_generator_comb_y: &'static [Uint<L>],
}

impl<const L: usize> Curve<L> {
  fn generator(&self) -> Affine<L> {
    Affine {
      x: FieldElement::from_uint(self.generator_x, self.field_modulus),
      y: FieldElement::from_uint(self.generator_y, self.field_modulus),
    }
  }

  fn generator_comb_table(&self) -> [Affine<L>; COMB_TABLE_SIZE] {
    let mut table = [self.generator(); COMB_TABLE_SIZE];
    for ((point, &x), &y) in table
      .iter_mut()
      .zip(self.generator_comb_x.iter())
      .zip(self.generator_comb_y.iter())
    {
      *point = Affine {
        x: FieldElement::from_montgomery(x, self.field_modulus),
        y: FieldElement::from_montgomery(y, self.field_modulus),
      };
    }
    table
  }
}

fn verify_digest<const L: usize>(
  curve: &Curve<L>,
  public_table: &[Affine<L>; COMB_TABLE_SIZE],
  r: Uint<L>,
  s: Uint<L>,
  digest: &[u8],
) -> Result<(), VerificationError> {
  if !r.is_in_range(&curve.scalar_modulus.value) || !s.is_in_range(&curve.scalar_modulus.value) {
    return Err(VerificationError::new());
  }

  let z = Uint::from_be_slice_mod(digest, curve.scalar_modulus.value);
  let w = s.inv_mod_vartime(curve.scalar_modulus.value);
  let u1 = mul_mod_montgomery(z, w, curve.scalar_modulus);
  let u2 = mul_mod_montgomery(r, w, curve.scalar_modulus);
  let point = scalar_mul_two(curve, u1, u2, public_table);

  if projective_x_matches_scalar(point, r, curve) {
    Ok(())
  } else {
    Err(VerificationError::new())
  }
}

fn sign_digest_p256(secret: &[u8; 32], digest: &[u8; 32]) -> Result<EcdsaP256Signature, EcdsaError> {
  let secret_scalar = SecretScalar::from_be_bytes(secret);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p256(secret, digest, wide.as_mut_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P256_ORDER_MODULUS));
  let result = sign_digest_with_nonce(&P256, &secret_scalar, &nonce, digest);
  match result {
    Ok((r, s)) => assemble_p256_signature(r, s),
    Err(error) => Err(error),
  }
}

fn sign_digest_p256_blinded(
  secret: &[u8; 32],
  digest: &[u8; 32],
  blind: &[u8; 64],
) -> Result<EcdsaP256Signature, EcdsaError> {
  let secret_scalar = SecretScalar::from_be_bytes(secret);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p256(secret, digest, wide.as_mut_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P256_ORDER_MODULUS));
  let nonce_blind = SecretScalar::new(reduce_wide_order_nonzero(blind, &P256_ORDER_MODULUS));
  let result = sign_digest_with_nonce_blinded(&P256, &secret_scalar, &nonce, &nonce_blind, digest);
  match result {
    Ok((r, s)) => assemble_p256_signature(r, s),
    Err(error) => Err(error),
  }
}

fn sign_digest_p384(secret: &[u8; 48], digest: &[u8; 48]) -> Result<EcdsaP384Signature, EcdsaError> {
  let secret_scalar = SecretScalar::from_be_bytes(secret);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p384(secret, digest, wide.as_mut_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P384_ORDER_MODULUS));
  let result = sign_digest_with_nonce(&P384, &secret_scalar, &nonce, digest);
  match result {
    Ok((r, s)) => assemble_p384_signature(r, s),
    Err(error) => Err(error),
  }
}

fn sign_digest_p384_blinded(
  secret: &[u8; 48],
  digest: &[u8; 48],
  blind: &[u8; 96],
) -> Result<EcdsaP384Signature, EcdsaError> {
  let secret_scalar = SecretScalar::from_be_bytes(secret);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p384(secret, digest, wide.as_mut_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P384_ORDER_MODULUS));
  let nonce_blind = SecretScalar::new(reduce_wide_order_nonzero(blind, &P384_ORDER_MODULUS));
  let result = sign_digest_with_nonce_blinded(&P384, &secret_scalar, &nonce, &nonce_blind, digest);
  match result {
    Ok((r, s)) => assemble_p384_signature(r, s),
    Err(error) => Err(error),
  }
}

fn hmac_expand_p256(secret: &[u8; 32], digest: &[u8; 32], out: &mut [u8; 64]) {
  for (block_index, block) in out.chunks_exact_mut(HmacSha256::TAG_SIZE).enumerate() {
    let mut mac = HmacSha256::new(secret);
    mac.update(P256_NONCE_DOMAIN);
    mac.update(&[block_index as u8]);
    mac.update(digest);
    block.copy_from_slice(mac.finalize().as_bytes());
  }
}

fn hmac_expand_p256_public_blind(secret: &[u8; 32], out: &mut [u8; 64]) {
  for (block_index, block) in out.chunks_exact_mut(HmacSha256::TAG_SIZE).enumerate() {
    let mut mac = HmacSha256::new(secret);
    mac.update(P256_PUBKEY_BLIND_DOMAIN);
    mac.update(&[block_index as u8]);
    block.copy_from_slice(mac.finalize().as_bytes());
  }
}

fn hmac_expand_p384(secret: &[u8; 48], digest: &[u8; 48], out: &mut [u8; 96]) {
  for (block_index, block) in out.chunks_exact_mut(HmacSha384::TAG_SIZE).enumerate() {
    let mut mac = HmacSha384::new(secret);
    mac.update(P384_NONCE_DOMAIN);
    mac.update(&[block_index as u8]);
    mac.update(digest);
    block.copy_from_slice(mac.finalize().as_bytes());
  }
}

fn hmac_expand_p384_public_blind(secret: &[u8; 48], out: &mut [u8; 96]) {
  for (block_index, block) in out.chunks_exact_mut(HmacSha384::TAG_SIZE).enumerate() {
    let mut mac = HmacSha384::new(secret);
    mac.update(P384_PUBKEY_BLIND_DOMAIN);
    mac.update(&[block_index as u8]);
    block.copy_from_slice(mac.finalize().as_bytes());
  }
}

fn sign_digest_with_nonce<const L: usize>(
  curve: &Curve<L>,
  secret_scalar: &SecretScalar<L>,
  nonce: &SecretScalar<L>,
  digest: &[u8],
) -> Result<(Uint<L>, Uint<L>), EcdsaError> {
  if let Some(result) = sign_digest_with_nonce_backend(curve, secret_scalar, nonce, digest) {
    return result;
  }

  let r_point = scalar_mul_basepoint_affine(curve, nonce);
  sign_digest_with_r_point(curve, secret_scalar, nonce, digest, r_point)
}

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
fn sign_digest_with_nonce_backend<const L: usize>(
  curve: &Curve<L>,
  secret_scalar: &SecretScalar<L>,
  nonce: &SecretScalar<L>,
  digest: &[u8],
) -> Option<Result<(Uint<L>, Uint<L>), EcdsaError>> {
  if is_p256_curve(curve) {
    let mut scalar_words = ZeroizingWords::zeroed();
    scalar_words.as_mut_array().copy_from_slice(&nonce.words()[..4]);
    let out = ecdsa_platform_asm::p256_scalarmulbase_generator(scalar_words.as_array());
    let mut r_words = [0u64; L];
    r_words.copy_from_slice(&out[..4]);
    let r = Uint(r_words).reduce_once_ct(curve.scalar_modulus.value);
    return Some(sign_digest_with_r(curve, secret_scalar, nonce, digest, r));
  }

  if is_p384_curve(curve) {
    let mut scalar_words = ZeroizingWords::zeroed();
    scalar_words.as_mut_array().copy_from_slice(&nonce.words()[..6]);
    let nonce6 = SecretScalar::new(Uint(*scalar_words.as_array()));
    let r = p384_scalar_mul_basepoint_r_platform(&nonce6);
    let mut r_words = [0u64; L];
    r_words.copy_from_slice(&r.0);
    return Some(sign_digest_with_r(curve, secret_scalar, nonce, digest, Uint(r_words)));
  }

  None
}

#[cfg(not(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
)))]
fn sign_digest_with_nonce_backend<const L: usize>(
  _curve: &Curve<L>,
  _secret_scalar: &SecretScalar<L>,
  _nonce: &SecretScalar<L>,
  _digest: &[u8],
) -> Option<Result<(Uint<L>, Uint<L>), EcdsaError>> {
  None
}

fn scalar_mul_basepoint_affine<const L: usize>(curve: &Curve<L>, scalar: &SecretScalar<L>) -> Affine<L> {
  scalar_mul_basepoint_affine_backend(curve, scalar)
    .or_else(|| {
      scalar_mul_basepoint_backend(curve, scalar).map(|point| point.to_affine_ct(curve.field_inverse_exponent))
    })
    .unwrap_or_else(|| scalar_mul_basepoint_comb_ct_secret(curve, scalar).to_affine_ct(curve.field_inverse_exponent))
}

fn sign_digest_with_nonce_blinded<const L: usize>(
  curve: &Curve<L>,
  secret_scalar: &SecretScalar<L>,
  nonce: &SecretScalar<L>,
  nonce_blind: &SecretScalar<L>,
  digest: &[u8],
) -> Result<(Uint<L>, Uint<L>), EcdsaError> {
  let r_point = scalar_mul_basepoint_blinded(curve, nonce, nonce_blind, curve.field_inverse_exponent);
  let r = r_point.x.to_uint().reduce_once_ct(curve.scalar_modulus.value);
  let rd = SecretScalar::new(mul_mod_montgomery_blinded_ct(
    r,
    secret_scalar.value(),
    nonce_blind.value(),
    curve.scalar_modulus,
  ));
  sign_digest_with_r_product(curve, nonce, digest, r, rd)
}

fn sign_digest_with_r_point<const L: usize>(
  curve: &Curve<L>,
  secret_scalar: &SecretScalar<L>,
  nonce: &SecretScalar<L>,
  digest: &[u8],
  r_point: Affine<L>,
) -> Result<(Uint<L>, Uint<L>), EcdsaError> {
  let r = r_point.x.to_uint().reduce_once_ct(curve.scalar_modulus.value);
  sign_digest_with_r(curve, secret_scalar, nonce, digest, r)
}

fn sign_digest_with_r<const L: usize>(
  curve: &Curve<L>,
  secret_scalar: &SecretScalar<L>,
  nonce: &SecretScalar<L>,
  digest: &[u8],
  r: Uint<L>,
) -> Result<(Uint<L>, Uint<L>), EcdsaError> {
  let rd = SecretScalar::new(mul_mod_montgomery_ct(r, secret_scalar.value(), curve.scalar_modulus));
  sign_digest_with_r_product(curve, nonce, digest, r, rd)
}

fn sign_digest_with_r_product<const L: usize>(
  curve: &Curve<L>,
  nonce: &SecretScalar<L>,
  digest: &[u8],
  r: Uint<L>,
  rd: SecretScalar<L>,
) -> Result<(Uint<L>, Uint<L>), EcdsaError> {
  let z = reduce_digest_for_scalar(digest, curve.scalar_modulus.value);
  let sum = SecretScalar::new(z.add_mod_ct(&rd.value(), curve.scalar_modulus.value));
  let nonce_inverse = SecretScalar::new(
    nonce
      .value()
      .inv_mod_ct_montgomery(curve.scalar_modulus, curve.scalar_inverse_exponent),
  );
  let sum = SecretScalar::new(montgomery_mul(
    sum.value(),
    curve.scalar_modulus.r2,
    curve.scalar_modulus,
  ));
  let s = SecretScalar::new(montgomery_mul(nonce_inverse.value(), sum.value(), curve.scalar_modulus));
  let s = SecretScalar::new(montgomery_mul(s.value(), Uint::ONE, curve.scalar_modulus));
  let failure_mask = r.ct_is_zero_mask() | s.value().ct_is_zero_mask();
  if failure_mask != 0 {
    return Err(EcdsaError::SigningFailure);
  }

  Ok((
    r,
    normalize_low_s(s.value(), curve.scalar_modulus.value, curve.half_order),
  ))
}

fn public_key_from_secret_p256(secret: &[u8; 32]) -> Affine<4> {
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p256_public_blind(secret, wide.as_mut_array());
  let mask = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P256_ORDER_MODULUS));
  let secret_scalar = SecretScalar::from_be_bytes(secret);
  public_key_from_secret_blinded(&P256, &secret_scalar, &mask, P256_FIELD_MINUS_TWO)
}

fn public_key_from_secret_p256_blinded(secret: &[u8; 32], blind: &[u8; 64]) -> Affine<4> {
  let mask = SecretScalar::new(reduce_wide_order_nonzero(blind, &P256_ORDER_MODULUS));
  let secret_scalar = SecretScalar::from_be_bytes(secret);
  public_key_from_secret_blinded(&P256, &secret_scalar, &mask, P256_FIELD_MINUS_TWO)
}

fn public_key_from_secret_p384(secret: &[u8; 48]) -> Affine<6> {
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p384_public_blind(secret, wide.as_mut_array());
  let mask = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P384_ORDER_MODULUS));
  let secret_scalar = SecretScalar::from_be_bytes(secret);
  public_key_from_secret_blinded(&P384, &secret_scalar, &mask, P384_FIELD_MINUS_TWO)
}

fn public_key_from_secret_p384_blinded(secret: &[u8; 48], blind: &[u8; 96]) -> Affine<6> {
  let mask = SecretScalar::new(reduce_wide_order_nonzero(blind, &P384_ORDER_MODULUS));
  let secret_scalar = SecretScalar::from_be_bytes(secret);
  public_key_from_secret_blinded(&P384, &secret_scalar, &mask, P384_FIELD_MINUS_TWO)
}

fn public_key_from_secret_blinded<const L: usize>(
  curve: &Curve<L>,
  secret_scalar: &SecretScalar<L>,
  mask: &SecretScalar<L>,
  field_inverse_exponent: Uint<L>,
) -> Affine<L> {
  scalar_mul_basepoint_blinded(curve, secret_scalar, mask, field_inverse_exponent)
}

fn reduce_digest_for_scalar<const L: usize>(digest: &[u8], modulus: Uint<L>) -> Uint<L> {
  Uint::from_be_slice_mod(digest, modulus)
}

fn reduce_wide_order_nonzero<const L: usize, const N: usize>(bytes: &[u8; N], modulus: &'static Modulus<L>) -> Uint<L> {
  #[cfg(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", target_os = "linux")
  ))]
  {
    if N == 64 && is_p256_order_modulus(modulus) {
      let mut fixed = ZeroizingBytes::zeroed();
      fixed.as_mut_array().copy_from_slice(bytes);
      let reduced_words = ZeroizingWords::new(ecdsa_platform_asm::p256_reduce_order_64(fixed.as_array()));
      let mut out = [0u64; L];
      out.copy_from_slice(reduced_words.as_array());
      let reduced = Uint(out);
      return Uint::select(reduced, Uint::ONE, reduced.ct_is_zero_mask());
    }
  }

  #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
  {
    if N == 96 && is_p384_order_modulus(modulus) {
      let mut fixed = ZeroizingBytes::zeroed();
      fixed.as_mut_array().copy_from_slice(bytes);
      let reduced_words = ZeroizingWords::new(ecdsa_platform_asm::p384_reduce_order_96(fixed.as_array()));
      let mut out = [0u64; L];
      out.copy_from_slice(reduced_words.as_array());
      let reduced = Uint(out);
      return Uint::select(reduced, Uint::ONE, reduced.ct_is_zero_mask());
    }
  }

  reduce_wide_order_nonzero_owned(bytes, modulus)
}

fn reduce_wide_order_nonzero_owned<const L: usize, const N: usize>(
  bytes: &[u8; N],
  modulus: &'static Modulus<L>,
) -> Uint<L> {
  debug_assert_eq!(N % 8, 0);

  let mut radix = Uint::ZERO;
  if let Some(limb) = radix.0.get_mut(1) {
    *limb = 1;
  }

  let mut acc = Uint::ZERO;
  for chunk in bytes.chunks_exact(8) {
    acc = mul_mod_montgomery_ct(acc, radix, modulus);
    let word = chunk
      .iter()
      .copied()
      .fold(0u64, |value, byte| (value << 8) | u64::from(byte));
    acc = acc.add_mod_ct(&Uint::from_u64(word), modulus.value);
  }
  Uint::select(acc, Uint::ONE, acc.ct_is_zero_mask())
}

fn is_p256_order_modulus<const L: usize>(modulus: &'static Modulus<L>) -> bool {
  L == 4 && core::ptr::from_ref(modulus).cast::<()>() == core::ptr::from_ref(&P256_ORDER_MODULUS).cast::<()>()
}

fn is_p384_order_modulus<const L: usize>(modulus: &'static Modulus<L>) -> bool {
  L == 6 && core::ptr::from_ref(modulus).cast::<()>() == core::ptr::from_ref(&P384_ORDER_MODULUS).cast::<()>()
}

fn normalize_low_s<const L: usize>(s: Uint<L>, order: Uint<L>, half_order: Uint<L>) -> Uint<L> {
  let low = order.sub_mod_ct(&s, order);
  Uint::select(s, low, s.ct_gt_mask(&half_order))
}

fn assemble_p256_signature(r: Uint<4>, s: Uint<4>) -> Result<EcdsaP256Signature, EcdsaError> {
  Ok(EcdsaP256Signature::from_scalars(r, s))
}

fn assemble_p384_signature(r: Uint<6>, s: Uint<6>) -> Result<EcdsaP384Signature, EcdsaError> {
  Ok(EcdsaP384Signature::from_scalars(r, s))
}

fn secret_scalar_is_valid<const LIMBS: usize, const BYTES: usize>(
  bytes: &[u8; BYTES],
  scalar_modulus: Uint<LIMBS>,
) -> bool {
  let scalar = SecretScalar::from_be_bytes(bytes);
  scalar.value().is_in_range_ct(&scalar_modulus)
}

fn scalar_mul_basepoint_blinded<const L: usize>(
  curve: &Curve<L>,
  scalar: &SecretScalar<L>,
  blind: &SecretScalar<L>,
  field_inverse_exponent: Uint<L>,
) -> Affine<L> {
  let z = FieldElement::from_uint(blind.value(), curve.field_modulus);
  scalar_mul_basepoint_backend(curve, scalar)
    .unwrap_or_else(|| {
      if is_p256_curve(curve) {
        let scalar = p256_blinded_comb_scalar(scalar, blind, curve.scalar_modulus.value);
        scalar_mul_basepoint_comb_ct_secret_blinded(curve, &scalar, z)
      } else {
        scalar_mul_basepoint_comb_ct_secret_blinded(curve, scalar, z)
      }
    })
    .randomize_z(z)
    .to_affine_ct(field_inverse_exponent)
}

fn p256_blinded_comb_scalar<const L: usize>(
  scalar: &SecretScalar<L>,
  blind: &SecretScalar<L>,
  order: Uint<L>,
) -> SecretScalar<5> {
  debug_assert_eq!(L, 4);

  // The P-256 signing comb consumes 259 bits. For k < n and factor <= 7,
  // k + factor*n < 8n < 2^259, so the comb loses no high bits. Adding the order
  // multiple preserves kG while randomizing secret-dependent field operands.
  let mut scalar_words = [0u64; 5];
  let mut order_words = [0u64; 5];
  for index in 0..L {
    if let (Some(dst), Some(&src)) = (scalar_words.get_mut(index), scalar.value.0.get(index)) {
      *dst = src;
    }
    if let (Some(dst), Some(&src)) = (order_words.get_mut(index), order.0.get(index)) {
      *dst = src;
    }
  }

  let mut multiple = Uint(order_words);
  let mut blinded = Uint(scalar_words);
  let mut factor = blind.value.0.first().copied().unwrap_or(0) & 7;
  factor |= mask_zero_u64(factor) & 1;
  for bit in 0..3 {
    let mask = 0u64.wrapping_sub((factor >> bit) & 1);
    let addend = Uint::select(Uint::ZERO, multiple, mask);
    blinded = blinded.add_raw(&addend).0;
    multiple = multiple.add_raw(&multiple).0;
  }

  SecretScalar::new(blinded)
}

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
fn scalar_mul_basepoint_backend<const L: usize>(curve: &Curve<L>, scalar: &SecretScalar<L>) -> Option<Jacobian<L>> {
  if is_p384_curve(curve) {
    return Some(p384_scalar_mul_basepoint_platform(curve, scalar));
  }

  scalar_mul_basepoint_affine_backend(curve, scalar).map(Jacobian::from_affine)
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn p384_scalar_mul_basepoint_platform<const L: usize>(curve: &Curve<L>, scalar: &SecretScalar<L>) -> Jacobian<L> {
  // The x86 nonexceptional comb backend showed measurable DudeCT separation on
  // Zen4/Ice Lake P-384 signing. Keep the complete comb path until that backend
  // has architecture-specific CT evidence.
  scalar_mul_basepoint_comb_ct_secret(curve, scalar)
}

#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
fn p384_scalar_mul_basepoint_platform<const L: usize>(curve: &Curve<L>, scalar: &SecretScalar<L>) -> Jacobian<L> {
  let mut scalar_words = ZeroizingWords::zeroed();
  scalar_words.as_mut_array().copy_from_slice(&scalar.words()[..6]);
  let scalar = SecretScalar::new(Uint(*scalar_words.as_array()));
  let point = p384_scalar_mul_basepoint_comb_backend(&scalar);
  let words = p384_jacobian_to_words(point);
  jacobian_from_p384_words(curve.field_modulus, &words)
}

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
fn p384_scalar_mul_basepoint_r_platform(scalar: &SecretScalar<6>) -> Uint<6> {
  p384_scalar_mul_basepoint_platform(&P384, scalar)
    .to_affine_x_ct(P384_FIELD_MINUS_TWO)
    .to_uint()
    .reduce_once_ct(P384_ORDER_MODULUS.value)
}

#[cfg(not(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
)))]
fn scalar_mul_basepoint_backend<const L: usize>(_curve: &Curve<L>, _scalar: &SecretScalar<L>) -> Option<Jacobian<L>> {
  None
}

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
fn scalar_mul_basepoint_affine_backend<const L: usize>(
  curve: &Curve<L>,
  scalar: &SecretScalar<L>,
) -> Option<Affine<L>> {
  if is_p256_curve(curve) {
    let mut scalar_words = ZeroizingWords::zeroed();
    scalar_words.as_mut_array().copy_from_slice(&scalar.words()[..4]);
    let out = ecdsa_platform_asm::p256_scalarmulbase_generator(scalar_words.as_array());
    return Some(affine_from_words(curve.field_modulus, &out));
  }

  None
}

#[cfg(not(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
)))]
fn scalar_mul_basepoint_affine_backend<const L: usize>(
  _curve: &Curve<L>,
  _scalar: &SecretScalar<L>,
) -> Option<Affine<L>> {
  None
}

fn is_p256_curve<const L: usize>(curve: &Curve<L>) -> bool {
  L == 4
    && core::ptr::from_ref(curve.field_modulus).cast::<()>() == core::ptr::from_ref(&P256_FIELD_MODULUS).cast::<()>()
}

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
fn is_p384_curve<const L: usize>(curve: &Curve<L>) -> bool {
  L == 6
    && core::ptr::from_ref(curve.field_modulus).cast::<()>() == core::ptr::from_ref(&P384_FIELD_MODULUS).cast::<()>()
}

#[cfg(any(
  all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
  all(target_arch = "x86_64", target_os = "linux")
))]
#[allow(clippy::indexing_slicing)]
fn affine_from_words<const L: usize>(modulus: &'static Modulus<L>, words: &[u64]) -> Affine<L> {
  let mut x = [0u64; L];
  let mut y = [0u64; L];
  x.copy_from_slice(&words[..L]);
  y.copy_from_slice(&words[L..L * 2]);
  let x = Uint(x);
  let y = Uint(y);
  Affine {
    x: FieldElement::from_uint(x, modulus),
    y: FieldElement::from_uint(y, modulus),
  }
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct P384FieldElement {
  value: Uint<6>,
}

#[cfg(test)]
impl P384FieldElement {
  fn zero() -> Self {
    Self { value: Uint::ZERO }
  }

  fn one() -> Self {
    Self {
      value: P384_FIELD_MONTGOMERY_ONE,
    }
  }

  const fn from_montgomery(value: Uint<6>) -> Self {
    Self { value }
  }

  fn to_generic(self) -> FieldElement<6> {
    FieldElement::from_montgomery(self.value, &P384_FIELD_MODULUS)
  }

  fn add(self, rhs: Self) -> Self {
    Self::from_montgomery(add_p384_field(self.value.0, rhs.value.0))
  }

  fn sub(self, rhs: Self) -> Self {
    Self::from_montgomery(sub_p384_field(self.value.0, rhs.value.0))
  }

  fn mul(self, rhs: Self) -> Self {
    Self::from_montgomery(Uint(p384_field_mul_words(self.value.0, rhs.value.0)))
  }

  fn square(self) -> Self {
    Self::from_montgomery(Uint(p384_field_square_words(self.value.0)))
  }

  fn double(self) -> Self {
    self.add(self)
  }

  fn triple(self) -> Self {
    self.double().add(self)
  }

  fn select(lhs: Self, rhs: Self, mask: u64) -> Self {
    Self::from_montgomery(Uint::select(lhs.value, rhs.value, mask))
  }
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct P384Affine {
  x: P384FieldElement,
  y: P384FieldElement,
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct P384Jacobian {
  x: P384FieldElement,
  y: P384FieldElement,
  z: P384FieldElement,
  infinity: bool,
}

#[cfg(test)]
impl P384Jacobian {
  fn infinity() -> Self {
    Self {
      x: P384FieldElement::zero(),
      y: P384FieldElement::one(),
      z: P384FieldElement::zero(),
      infinity: true,
    }
  }

  fn from_affine(point: P384Affine) -> Self {
    Self {
      x: point.x,
      y: point.y,
      z: P384FieldElement::one(),
      infinity: false,
    }
  }

  fn to_generic(self) -> Jacobian<6> {
    Jacobian {
      x: self.x.to_generic(),
      y: self.y.to_generic(),
      z: self.z.to_generic(),
      infinity: self.infinity,
    }
  }

  fn select(lhs: Self, rhs: Self, mask: u64) -> Self {
    let lhs_infinity = 0u64.wrapping_sub(u64::from(lhs.infinity));
    let rhs_infinity = 0u64.wrapping_sub(u64::from(rhs.infinity));
    let selected_infinity = lhs_infinity ^ (mask & (lhs_infinity ^ rhs_infinity));
    Self {
      x: P384FieldElement::select(lhs.x, rhs.x, mask),
      y: P384FieldElement::select(lhs.y, rhs.y, mask),
      z: P384FieldElement::select(lhs.z, rhs.z, mask),
      infinity: selected_infinity != 0,
    }
  }

  fn infinity_mask(&self) -> u64 {
    0u64.wrapping_sub(u64::from(self.infinity))
  }

  fn double_ct(self) -> Self {
    let delta = self.z.square();
    let gamma = self.y.square();
    let beta = self.x.mul(gamma);
    let alpha = self.x.sub(delta).mul(self.x.add(delta)).triple();
    let x3 = alpha.square().sub(beta.double().double().double());
    let z3 = self.y.add(self.z).square().sub(gamma).sub(delta);
    let y3 = alpha
      .mul(beta.double().double().sub(x3))
      .sub(gamma.square().double().double().double());

    let raw = Self {
      x: x3,
      y: y3,
      z: z3,
      infinity: false,
    };
    Self::select(
      raw,
      Self::infinity(),
      self.infinity_mask() | self.y.value.ct_is_zero_mask(),
    )
  }

  // Incomplete mixed-add formula. The fixed-base comb loop below proves that
  // finite, non-masked operands are neither equal nor opposite.
  fn add_mixed_nonexceptional_ct(self, rhs: P384Affine, rhs_infinity_mask: u64) -> Self {
    let z1z1 = self.z.square();
    let u2 = rhs.x.mul(z1z1);
    let s2 = rhs.y.mul(self.z).mul(z1z1);
    let h = u2.sub(self.x);
    let r = s2.sub(self.y).double();

    let hh = h.square();
    let i = hh.double().double();
    let j = h.mul(i);
    let v = self.x.mul(i);
    let x3 = r.square().sub(j).sub(v.double());
    let y3 = r.mul(v.sub(x3)).sub(self.y.mul(j).double());
    let z3 = self.z.add(h).square().sub(z1z1).sub(hh);
    let raw = Self {
      x: x3,
      y: y3,
      z: z3,
      infinity: false,
    };

    let with_self_infinity = Self::select(raw, Self::from_affine(rhs), self.infinity_mask());
    Self::select(with_self_infinity, self, rhs_infinity_mask)
  }
}

#[cfg(test)]
fn p384_field_mul_words(lhs: [u64; 6], rhs: [u64; 6]) -> [u64; 6] {
  #[cfg(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", target_os = "linux")
  ))]
  {
    ecdsa_platform_asm::p384_field_mul(&lhs, &rhs)
  }

  #[cfg(not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", target_os = "linux")
  )))]
  {
    ecdsa_p384_field::mul(lhs, rhs)
  }
}

#[cfg(test)]
fn p384_field_square_words(value: [u64; 6]) -> [u64; 6] {
  #[cfg(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", target_os = "linux")
  ))]
  {
    ecdsa_platform_asm::p384_field_square(&value)
  }

  #[cfg(not(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", target_os = "linux")
  )))]
  {
    ecdsa_p384_field::square(value)
  }
}

#[cfg(test)]
fn select_p384_signing_generator_affine_ct(digit: usize) -> P384Affine {
  let mut x = P384_SIGNING_GENERATOR_COMB_X[0];
  let mut y = P384_SIGNING_GENERATOR_COMB_Y[0];

  for (index, (&candidate_x, &candidate_y)) in P384_SIGNING_GENERATOR_COMB_X
    .iter()
    .zip(P384_SIGNING_GENERATOR_COMB_Y.iter())
    .enumerate()
    .skip(1)
  {
    let mask = mask_eq_usize(digit, index);
    x = Uint::select(x, candidate_x, mask);
    y = Uint::select(y, candidate_y, mask);
  }

  P384Affine {
    x: P384FieldElement::from_montgomery(x),
    y: P384FieldElement::from_montgomery(y),
  }
}

#[cfg(test)]
fn p384_scalar_mul_basepoint_comb_nonexceptional_ct(scalar: &SecretScalar<6>) -> Jacobian<6> {
  let rows = P384_SIGNING_COMB_ROWS;
  let mut acc = P384Jacobian::infinity();

  for row in (0..rows).rev() {
    acc = acc.double_ct();
    let digit = signing_comb_digit_ct(scalar.value(), row, rows, P384_SIGNING_COMB_WIDTH);
    let selected = select_p384_signing_generator_affine_ct(digit);

    // P-384's signing comb represents a scalar as sum(2^row * S_row), where
    // every S_row has bits only at positions `column * rows`. After the loop
    // double, the accumulator coefficient has bits only at
    // `column * rows + offset`, with 1 <= offset < rows. For nonzero digits it
    // cannot equal S_row as an integer; both values are below the prime group
    // order, so equality modulo the order is also impossible. Opposites are
    // impossible because row 0 would require the scalar to equal the order, and
    // later rows are bounded by less than half the order plus the sparse S_row.
    acc = acc.add_mixed_nonexceptional_ct(selected, mask_eq_usize(digit, 0));
  }

  acc.to_generic()
}

#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
fn p384_scalar_mul_basepoint_comb_backend(scalar: &SecretScalar<6>) -> Jacobian<6> {
  let curve = &P384;
  let rows = curve.signing_comb_rows;
  let mut acc = Jacobian::infinity(curve.field_modulus);

  for row in (0..rows).rev() {
    let doubled_words = ecdsa_platform_asm::p384_point_double(&p384_jacobian_to_words(acc));
    let doubled = p384_jacobian_from_words(&doubled_words);
    acc = Jacobian::select(
      doubled,
      Jacobian::infinity(curve.field_modulus),
      acc.infinity_mask() | acc.y.value.ct_is_zero_mask(),
    );

    let digit = signing_comb_digit_ct(scalar.value(), row, rows, curve.signing_comb_width);
    let selected = select_signing_generator_affine_ct(curve, digit);
    let added_words =
      ecdsa_platform_asm::p384_point_mixadd(&p384_jacobian_to_words(acc), &p384_affine_to_words(selected));
    let added = p384_jacobian_from_words(&added_words);
    let with_self_infinity = Jacobian::select(added, Jacobian::from_affine(selected), acc.infinity_mask());
    acc = Jacobian::select(with_self_infinity, acc, mask_eq_usize(digit, 0));
  }

  acc
}

#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
fn p384_jacobian_to_words(point: Jacobian<6>) -> [u64; 18] {
  let mut out = [0u64; 18];
  out[..6].copy_from_slice(&point.x.value.0);
  out[6..12].copy_from_slice(&point.y.value.0);
  out[12..18].copy_from_slice(&point.z.value.0);
  out
}

#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
fn p384_affine_to_words(point: Affine<6>) -> [u64; 12] {
  let mut out = [0u64; 12];
  out[..6].copy_from_slice(&point.x.value.0);
  out[6..12].copy_from_slice(&point.y.value.0);
  out
}

#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
fn p384_jacobian_from_words(words: &[u64; 18]) -> Jacobian<6> {
  Jacobian {
    x: FieldElement::from_montgomery(
      Uint([words[0], words[1], words[2], words[3], words[4], words[5]]),
      &P384_FIELD_MODULUS,
    ),
    y: FieldElement::from_montgomery(
      Uint([words[6], words[7], words[8], words[9], words[10], words[11]]),
      &P384_FIELD_MODULUS,
    ),
    z: FieldElement::from_montgomery(
      Uint([words[12], words[13], words[14], words[15], words[16], words[17]]),
      &P384_FIELD_MODULUS,
    ),
    infinity: false,
  }
}

#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
#[allow(clippy::indexing_slicing)]
fn jacobian_from_p384_words<const L: usize>(modulus: &'static Modulus<L>, words: &[u64; 18]) -> Jacobian<L> {
  let mut x = [0u64; L];
  let mut y = [0u64; L];
  let mut z = [0u64; L];
  x.copy_from_slice(&words[..L]);
  y.copy_from_slice(&words[6..6 + L]);
  z.copy_from_slice(&words[12..12 + L]);
  Jacobian {
    x: FieldElement::from_montgomery(Uint(x), modulus),
    y: FieldElement::from_montgomery(Uint(y), modulus),
    z: FieldElement::from_montgomery(Uint(z), modulus),
    infinity: false,
  }
}

#[cfg(test)]
#[allow(clippy::indexing_slicing)]
fn scalar_mul_basepoint_comb_ct<const L: usize>(curve: &Curve<L>, scalar: Uint<L>) -> Jacobian<L> {
  let rows = curve.signing_comb_rows;
  let mut acc = Jacobian::infinity(curve.field_modulus);

  for row in (0..rows).rev() {
    acc = acc.double_ct();
    let digit = signing_comb_digit_ct(scalar, row, rows, curve.signing_comb_width);
    let selected = select_signing_generator_affine_ct(curve, digit);
    acc = acc.add_mixed_ct(selected, mask_eq_usize(digit, 0));
  }

  acc
}

#[allow(clippy::indexing_slicing)]
fn scalar_mul_basepoint_comb_ct_secret<const L: usize, const S: usize>(
  curve: &Curve<L>,
  scalar: &SecretScalar<S>,
) -> Jacobian<L> {
  let scalar = SecretScalar::new(scalar.value());
  let rows = curve.signing_comb_rows;
  let mut acc = Jacobian::infinity(curve.field_modulus);

  for row in (0..rows).rev() {
    acc = acc.double_ct();
    let digit = signing_comb_digit_ct(scalar.value(), row, rows, curve.signing_comb_width);
    let selected = select_signing_generator_affine_ct(curve, digit);
    acc = acc.add_mixed_ct(selected, mask_eq_usize(digit, 0));
  }

  acc
}

#[allow(clippy::indexing_slicing)]
fn scalar_mul_basepoint_comb_ct_secret_blinded<const L: usize, const S: usize>(
  curve: &Curve<L>,
  scalar: &SecretScalar<S>,
  z: FieldElement<L>,
) -> Jacobian<L> {
  let rows = curve.signing_comb_rows;
  let z2 = z.square();
  let z3 = z2.mul(z);
  let public_rhs = select_signing_generator_affine_ct(curve, 0);
  let mut acc = Jacobian::infinity(curve.field_modulus);

  for row in (0..rows).rev() {
    acc = acc.double_ct();
    let digit = signing_comb_digit_ct(scalar.value(), row, rows, curve.signing_comb_width);
    let selected = select_signing_generator_affine_ct(curve, digit);
    let digit_zero = mask_eq_usize(digit, 0);
    let acc_infinity = acc.infinity_mask();
    // Before the first nonzero digit, keep mixed-add operands public and
    // constant-select the secret table point only after projecting it with z.
    let safe_rhs = Affine {
      x: FieldElement::select(selected.x, public_rhs.x, acc_infinity),
      y: FieldElement::select(selected.y, public_rhs.y, acc_infinity),
    };
    let added = acc.add_mixed_ct(safe_rhs, digit_zero);
    let initialized = Jacobian {
      x: selected.x.mul(z2),
      y: selected.y.mul(z3),
      z,
      infinity: false,
    };
    acc = Jacobian::select(added, initialized, acc_infinity & mask_not(digit_zero));
  }

  acc
}

fn signing_comb_digit_ct<const L: usize>(scalar: Uint<L>, row: usize, rows: usize, width: usize) -> usize {
  let mut digit = 0usize;
  for column in 0usize..width {
    let bit = row.strict_add(column.strict_mul(rows));
    digit |= ((scalar.ct_bit_mask(bit) & 1) as usize) << column;
  }
  digit
}

#[allow(clippy::indexing_slicing)]
fn select_signing_generator_affine_ct<const L: usize>(curve: &Curve<L>, digit: usize) -> Affine<L> {
  let mut x = curve.signing_generator_comb_x[0];
  let mut y = curve.signing_generator_comb_y[0];
  for (index, (&candidate_x, &candidate_y)) in curve
    .signing_generator_comb_x
    .iter()
    .zip(curve.signing_generator_comb_y.iter())
    .enumerate()
  {
    let mask = mask_eq_usize(digit, index);
    x = Uint::select(x, candidate_x, mask);
    y = Uint::select(y, candidate_y, mask);
  }
  Affine {
    x: FieldElement::from_montgomery(x, curve.field_modulus),
    y: FieldElement::from_montgomery(y, curve.field_modulus),
  }
}

#[cfg(all(feature = "diag", feature = "ecdsa-p256"))]
pub fn diag_ecdsa_p256_select_signing_generator_affine_limb_digest(digit: u8) -> [u64; 8] {
  let selected = select_signing_generator_affine_ct(&P256, usize::from(digit));
  let mut out = [0u64; 8];
  out[..4].copy_from_slice(&selected.x.value.0);
  out[4..].copy_from_slice(&selected.y.value.0);
  out
}

#[cfg(all(feature = "diag", feature = "ecdsa-p256"))]
pub fn diag_ecdsa_p256_nonce_reduce_limb_digest(secret: [u8; 32], message: &[u8]) -> [u64; 4] {
  let secret = ZeroizingBytes::new(secret);
  let digest = Sha256::digest(message);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p256(secret.as_array(), &digest, wide.as_mut_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P256_ORDER_MODULUS));
  nonce.value().0
}

#[cfg(all(feature = "diag", feature = "ecdsa-p256"))]
pub fn diag_ecdsa_p256_reduce_wide_order_limb_digest(wide: [u8; 64]) -> [u64; 4] {
  let wide = ZeroizingBytes::new(wide);
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P256_ORDER_MODULUS));
  nonce.value().0
}

#[cfg(all(feature = "diag", feature = "ecdsa-p256"))]
pub fn diag_ecdsa_p256_basepoint_blinded_limb_digest(secret: [u8; 32], blind: [u8; 64], message: &[u8]) -> [u64; 8] {
  let secret = ZeroizingBytes::new(secret);
  let blind = ZeroizingBytes::new(blind);
  let digest = Sha256::digest(message);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p256(secret.as_array(), &digest, wide.as_mut_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P256_ORDER_MODULUS));
  let nonce_blind = SecretScalar::new(reduce_wide_order_nonzero(blind.as_array(), &P256_ORDER_MODULUS));
  let point = scalar_mul_basepoint_blinded(&P256, &nonce, &nonce_blind, P256_FIELD_MINUS_TWO);
  let mut out = [0u64; 8];
  out[..4].copy_from_slice(&point.x.value.0);
  out[4..].copy_from_slice(&point.y.value.0);
  out
}

#[cfg(all(feature = "diag", feature = "ecdsa-p256"))]
pub fn diag_ecdsa_p256_scalar_finish_limb_digest(secret: [u8; 32], nonce_wide: [u8; 64], message: &[u8]) -> [u64; 8] {
  let secret = ZeroizingBytes::new(secret);
  let nonce_wide = ZeroizingBytes::new(nonce_wide);
  let digest = Sha256::digest(message);
  let secret_scalar = SecretScalar::from_be_bytes(secret.as_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(nonce_wide.as_array(), &P256_ORDER_MODULUS));
  let r = P256_GX.reduce_once_ct(P256_ORDER_MODULUS.value);
  let (r, s) = sign_digest_with_r(&P256, &secret_scalar, &nonce, &digest, r).unwrap_or((Uint::ZERO, Uint::ZERO));
  let mut out = [0u64; 8];
  out[..4].copy_from_slice(&r.0);
  out[4..].copy_from_slice(&s.0);
  out
}

#[cfg(all(feature = "diag", feature = "ecdsa-p256"))]
pub fn diag_ecdsa_p256_order_mul_fixed_r_limb_digest(secret: [u8; 32]) -> [u64; 4] {
  let secret = ZeroizingBytes::new(secret);
  let secret_scalar = SecretScalar::from_be_bytes(secret.as_array());
  let r = P256_GX.reduce_once_ct(P256_ORDER_MODULUS.value);
  let rd = SecretScalar::new(mul_mod_montgomery_ct(r, secret_scalar.value(), &P256_ORDER_MODULUS));
  rd.value().0
}

#[cfg(all(feature = "diag", feature = "ecdsa-p256"))]
pub fn diag_ecdsa_p256_order_mul_blinded_fixed_r_limb_digest(secret: [u8; 32], blind: [u8; 64]) -> [u64; 4] {
  let secret = ZeroizingBytes::new(secret);
  let blind = ZeroizingBytes::new(blind);
  let secret_scalar = SecretScalar::from_be_bytes(secret.as_array());
  let scalar_blind = SecretScalar::new(reduce_wide_order_nonzero(blind.as_array(), &P256_ORDER_MODULUS));
  let r = P256_GX.reduce_once_ct(P256_ORDER_MODULUS.value);
  let rd = SecretScalar::new(mul_mod_montgomery_blinded_ct(
    r,
    secret_scalar.value(),
    scalar_blind.value(),
    &P256_ORDER_MODULUS,
  ));
  rd.value().0
}

#[cfg(all(feature = "diag", feature = "ecdsa-p256"))]
pub fn diag_ecdsa_p256_nonce_inverse_limb_digest(secret: [u8; 32], message: &[u8]) -> [u64; 4] {
  let secret = ZeroizingBytes::new(secret);
  let digest = Sha256::digest(message);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p256(secret.as_array(), &digest, wide.as_mut_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P256_ORDER_MODULUS));
  let inverse = SecretScalar::new(
    nonce
      .value()
      .inv_mod_ct_montgomery(&P256_ORDER_MODULUS, P256_ORDER_MINUS_TWO),
  );
  inverse.value().0
}

#[cfg(all(feature = "diag", feature = "ecdsa-p256"))]
pub fn diag_ecdsa_p256_final_multiply_limb_digest(secret: [u8; 32], nonce_wide: [u8; 64], message: &[u8]) -> [u64; 4] {
  let secret = ZeroizingBytes::new(secret);
  let nonce_wide = ZeroizingBytes::new(nonce_wide);
  let digest = Sha256::digest(message);
  let secret_scalar = SecretScalar::from_be_bytes(secret.as_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(nonce_wide.as_array(), &P256_ORDER_MODULUS));
  let r = P256_GX.reduce_once_ct(P256_ORDER_MODULUS.value);
  let z = reduce_digest_for_scalar(&digest, P256_ORDER_MODULUS.value);
  let rd = SecretScalar::new(mul_mod_montgomery_ct(r, secret_scalar.value(), &P256_ORDER_MODULUS));
  let sum = SecretScalar::new(montgomery_mul(
    z.add_mod_ct(&rd.value(), P256_ORDER_MODULUS.value),
    P256_ORDER_MODULUS.r2,
    &P256_ORDER_MODULUS,
  ));
  let nonce_inverse = SecretScalar::new(
    nonce
      .value()
      .inv_mod_ct_montgomery(&P256_ORDER_MODULUS, P256_ORDER_MINUS_TWO),
  );
  let product = SecretScalar::new(montgomery_mul(nonce_inverse.value(), sum.value(), &P256_ORDER_MODULUS));
  product.value().0
}

#[cfg(all(feature = "diag", feature = "ecdsa-p384"))]
pub fn diag_ecdsa_p384_select_signing_generator_affine_limb_digest(digit: u8) -> [u64; 12] {
  let selected = select_signing_generator_affine_ct(&P384, usize::from(digit));
  let mut out = [0u64; 12];
  out[..6].copy_from_slice(&selected.x.value.0);
  out[6..].copy_from_slice(&selected.y.value.0);
  out
}

#[cfg(all(feature = "diag", feature = "ecdsa-p384"))]
pub fn diag_ecdsa_p384_nonce_reduce_limb_digest(secret: [u8; 48], message: &[u8]) -> [u64; 6] {
  let secret = ZeroizingBytes::new(secret);
  let digest = Sha384::digest(message);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p384(secret.as_array(), &digest, wide.as_mut_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P384_ORDER_MODULUS));
  nonce.value().0
}

#[cfg(all(feature = "diag", feature = "ecdsa-p384"))]
pub fn diag_ecdsa_p384_reduce_wide_order_limb_digest(wide: [u8; 96]) -> [u64; 6] {
  let wide = ZeroizingBytes::new(wide);
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P384_ORDER_MODULUS));
  nonce.value().0
}

#[cfg(all(feature = "diag", feature = "ecdsa-p384"))]
pub fn diag_ecdsa_p384_basepoint_blinded_limb_digest(secret: [u8; 48], blind: [u8; 96], message: &[u8]) -> [u64; 12] {
  let secret = ZeroizingBytes::new(secret);
  let blind = ZeroizingBytes::new(blind);
  let digest = Sha384::digest(message);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p384(secret.as_array(), &digest, wide.as_mut_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P384_ORDER_MODULUS));
  let nonce_blind = SecretScalar::new(reduce_wide_order_nonzero(blind.as_array(), &P384_ORDER_MODULUS));
  let point = scalar_mul_basepoint_blinded(&P384, &nonce, &nonce_blind, P384_FIELD_MINUS_TWO);
  let mut out = [0u64; 12];
  out[..6].copy_from_slice(&point.x.value.0);
  out[6..].copy_from_slice(&point.y.value.0);
  out
}

#[cfg(all(feature = "diag", feature = "ecdsa-p384"))]
pub fn diag_ecdsa_p384_basepoint_r_limb_digest(secret: [u8; 48], message: &[u8]) -> [u64; 6] {
  let secret = ZeroizingBytes::new(secret);
  let digest = Sha384::digest(message);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p384(secret.as_array(), &digest, wide.as_mut_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P384_ORDER_MODULUS));
  let point =
    scalar_mul_basepoint_backend(&P384, &nonce).unwrap_or_else(|| scalar_mul_basepoint_comb_ct_secret(&P384, &nonce));
  point
    .to_affine_x_ct(P384_FIELD_MINUS_TWO)
    .to_uint()
    .reduce_once_ct(P384_ORDER_MODULUS.value)
    .0
}

#[cfg(all(feature = "diag", feature = "ecdsa-p384"))]
pub fn diag_ecdsa_p384_scalar_finish_limb_digest(secret: [u8; 48], nonce_wide: [u8; 96], message: &[u8]) -> [u64; 12] {
  let secret = ZeroizingBytes::new(secret);
  let nonce_wide = ZeroizingBytes::new(nonce_wide);
  let digest = Sha384::digest(message);
  let secret_scalar = SecretScalar::from_be_bytes(secret.as_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(nonce_wide.as_array(), &P384_ORDER_MODULUS));
  let r = P384_GX.reduce_once_ct(P384_ORDER_MODULUS.value);
  let (r, s) = sign_digest_with_r(&P384, &secret_scalar, &nonce, &digest, r).unwrap_or((Uint::ZERO, Uint::ZERO));
  let mut out = [0u64; 12];
  out[..6].copy_from_slice(&r.0);
  out[6..].copy_from_slice(&s.0);
  out
}

#[cfg(all(feature = "diag", feature = "ecdsa-p384"))]
pub fn diag_ecdsa_p384_order_mul_fixed_r_limb_digest(secret: [u8; 48]) -> [u64; 6] {
  let secret = ZeroizingBytes::new(secret);
  let secret_scalar = SecretScalar::from_be_bytes(secret.as_array());
  let r = P384_GX.reduce_once_ct(P384_ORDER_MODULUS.value);
  let rd = SecretScalar::new(mul_mod_montgomery_ct(r, secret_scalar.value(), &P384_ORDER_MODULUS));
  rd.value().0
}

#[cfg(all(feature = "diag", feature = "ecdsa-p384"))]
pub fn diag_ecdsa_p384_nonce_inverse_limb_digest(secret: [u8; 48], message: &[u8]) -> [u64; 6] {
  let secret = ZeroizingBytes::new(secret);
  let digest = Sha384::digest(message);
  let mut wide = ZeroizingBytes::zeroed();
  hmac_expand_p384(secret.as_array(), &digest, wide.as_mut_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(wide.as_array(), &P384_ORDER_MODULUS));
  let inverse = SecretScalar::new(
    nonce
      .value()
      .inv_mod_ct_montgomery(&P384_ORDER_MODULUS, P384_ORDER_MINUS_TWO),
  );
  inverse.value().0
}

#[cfg(all(feature = "diag", feature = "ecdsa-p384"))]
pub fn diag_ecdsa_p384_final_multiply_limb_digest(secret: [u8; 48], nonce_wide: [u8; 96], message: &[u8]) -> [u64; 6] {
  let secret = ZeroizingBytes::new(secret);
  let nonce_wide = ZeroizingBytes::new(nonce_wide);
  let digest = Sha384::digest(message);
  let secret_scalar = SecretScalar::from_be_bytes(secret.as_array());
  let nonce = SecretScalar::new(reduce_wide_order_nonzero(nonce_wide.as_array(), &P384_ORDER_MODULUS));
  let r = P384_GX.reduce_once_ct(P384_ORDER_MODULUS.value);
  let z = reduce_digest_for_scalar(&digest, P384_ORDER_MODULUS.value);
  let rd = SecretScalar::new(mul_mod_montgomery_ct(r, secret_scalar.value(), &P384_ORDER_MODULUS));
  let sum = SecretScalar::new(montgomery_mul(
    z.add_mod_ct(&rd.value(), P384_ORDER_MODULUS.value),
    P384_ORDER_MODULUS.r2,
    &P384_ORDER_MODULUS,
  ));
  let nonce_inverse = SecretScalar::new(
    nonce
      .value()
      .inv_mod_ct_montgomery(&P384_ORDER_MODULUS, P384_ORDER_MINUS_TWO),
  );
  let product = SecretScalar::new(montgomery_mul(nonce_inverse.value(), sum.value(), &P384_ORDER_MODULUS));
  product.value().0
}

fn projective_x_matches_scalar<const L: usize>(point: Jacobian<L>, scalar: Uint<L>, curve: &Curve<L>) -> bool {
  if point.infinity || point.z.value.is_zero() {
    return false;
  }

  let z2 = point.z.square();
  let x = FieldElement::from_uint(scalar, curve.field_modulus).mul(z2);
  if point.x == x {
    return true;
  }

  let (scalar_plus_order, carry) = scalar.add_raw(&curve.scalar_modulus.value);
  if carry == 0 && scalar_plus_order.cmp(&curve.field_modulus.value).is_lt() {
    let x = FieldElement::from_uint(scalar_plus_order, curve.field_modulus).mul(z2);
    point.x == x
  } else {
    false
  }
}

#[allow(clippy::indexing_slicing)]
fn scalar_mul_two<const L: usize>(
  curve: &Curve<L>,
  lhs_scalar: Uint<L>,
  rhs_scalar: Uint<L>,
  rhs_table: &[Affine<L>; COMB_TABLE_SIZE],
) -> Jacobian<L> {
  let lhs_table = curve.generator_comb_table();
  let rows = comb_rows::<L>();
  let mut acc = Jacobian::infinity(curve.field_modulus);

  for i in (0..rows).rev() {
    acc = acc.double();
    let lhs_digit = comb_digit(lhs_scalar, i, rows);
    if lhs_digit != 0 {
      acc = acc.add_mixed(lhs_table[lhs_digit]);
    }
    let rhs_digit = comb_digit(rhs_scalar, i, rows);
    if rhs_digit != 0 {
      acc = acc.add_mixed(rhs_table[rhs_digit]);
    }
  }
  acc
}

const fn comb_rows<const L: usize>() -> usize {
  (L * 64).div_ceil(COMB_WIDTH)
}

fn comb_digit<const L: usize>(scalar: Uint<L>, row: usize, rows: usize) -> usize {
  let mut digit = 0usize;
  for column in 0..COMB_WIDTH {
    let bit = row + column * rows;
    if scalar.bit(bit) {
      digit |= 1usize << column;
    }
  }
  digit
}

#[allow(clippy::indexing_slicing)]
fn precompute_comb_table<const L: usize>(point: Affine<L>) -> [Affine<L>; COMB_TABLE_SIZE] {
  let rows = comb_rows::<L>();
  let mut column_points = [Jacobian::from_affine(point); COMB_WIDTH];
  let mut current = Jacobian::from_affine(point);
  for column in column_points.iter_mut().skip(1) {
    for _ in 0..rows {
      current = current.double();
    }
    *column = current;
  }
  let columns = normalize_jacobian_table(column_points);

  let mut table = [Jacobian::from_affine(point); COMB_TABLE_SIZE];
  for (mask, entry) in table.iter_mut().enumerate().skip(1) {
    let mut acc = Jacobian::infinity(point.x.modulus);
    for (column, &column_point) in columns.iter().enumerate() {
      if mask & (1usize << column) != 0 {
        acc = acc.add_mixed(column_point);
      }
    }
    *entry = acc;
  }
  normalize_jacobian_table(table)
}

#[allow(clippy::indexing_slicing)]
fn normalize_jacobian_table<const L: usize, const N: usize>(table: [Jacobian<L>; N]) -> [Affine<L>; N] {
  let modulus = table[0].x.modulus;
  let mut prefixes = [FieldElement::one(modulus); N];
  let mut acc = FieldElement::one(modulus);
  for i in 0..N {
    prefixes[i] = acc;
    acc = acc.mul(table[i].z);
  }

  let mut acc_inv = acc.inv();
  let mut out = [Affine {
    x: FieldElement::zero(modulus),
    y: FieldElement::zero(modulus),
  }; N];
  for i in (0..N).rev() {
    let z_inv = acc_inv.mul(prefixes[i]);
    acc_inv = acc_inv.mul(table[i].z);
    let z2 = z_inv.square();
    let z3 = z2.mul(z_inv);
    out[i] = Affine {
      x: table[i].x.mul(z2),
      y: table[i].y.mul(z3),
    };
  }
  out
}

#[allow(clippy::indexing_slicing)]
fn parse_public_key<const L: usize>(bytes: &[u8], curve: &Curve<L>) -> Result<Affine<L>, EcdsaError> {
  let field_len = L * 8;
  if bytes.len() != field_len.strict_mul(2).strict_add(1) || bytes.first().copied() != Some(0x04) {
    return Err(EcdsaError::InvalidPublicKey);
  }

  let x = Uint::from_be_slice(bytes.get(1..1 + field_len).ok_or(EcdsaError::InvalidPublicKey)?)
    .map_err(|_| EcdsaError::InvalidPublicKey)?;
  let y = Uint::from_be_slice(bytes.get(1 + field_len..).ok_or(EcdsaError::InvalidPublicKey)?)
    .map_err(|_| EcdsaError::InvalidPublicKey)?;
  if x.cmp(&curve.field_modulus.value).is_ge() || y.cmp(&curve.field_modulus.value).is_ge() {
    return Err(EcdsaError::InvalidPublicKey);
  }

  let point = Affine {
    x: FieldElement::from_uint(x, curve.field_modulus),
    y: FieldElement::from_uint(y, curve.field_modulus),
  };
  if !is_on_curve(point, curve) {
    return Err(EcdsaError::InvalidPublicKey);
  }
  Ok(point)
}

fn is_on_curve<const L: usize>(point: Affine<L>, curve: &Curve<L>) -> bool {
  let lhs = point.y.square();
  let x2 = point.x.square();
  let x3 = x2.mul(point.x);
  let three_x = point.x.triple();
  let b = FieldElement::from_uint(curve.b, curve.field_modulus);
  lhs == x3.sub(three_x).add(b)
}

fn encode_sec1<const L: usize, const N: usize>(point: &Affine<L>) -> [u8; N] {
  let mut out = [0u8; N];
  out[0] = 0x04;
  let field_len = L * 8;
  point.x.to_uint().write_be(&mut out[1..1 + field_len]);
  point.y.to_uint().write_be(&mut out[1 + field_len..]);
  out
}

fn parse_signature_scalars<const LIMBS: usize, const BYTES: usize>(
  bytes: &[u8; BYTES],
  scalar_modulus: Uint<LIMBS>,
) -> Result<(Uint<LIMBS>, Uint<LIMBS>), EcdsaError> {
  let field_len = LIMBS * 8;
  let r = Uint::from_be_slice(&bytes[..field_len])?;
  let s = Uint::from_be_slice(&bytes[field_len..])?;
  if !r.is_in_range(&scalar_modulus) || !s.is_in_range(&scalar_modulus) {
    return Err(EcdsaError::InvalidSignature);
  }
  Ok((r, s))
}

fn parse_signature_der_bytes<const LIMBS: usize, const BYTES: usize>(der: &[u8]) -> Result<[u8; BYTES], EcdsaError> {
  let field_len = LIMBS * 8;
  let mut root = DerReader::new(der);
  let sig = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut sig = DerReader::new(sig);
  let r_value = sig.read_primitive(TAG_INTEGER)?;
  let s_value = sig.read_primitive(TAG_INTEGER)?;
  sig.finish()?;

  let mut bytes = [0u8; BYTES];
  parse_der_integer_into(r_value, &mut bytes[..field_len])?;
  parse_der_integer_into(s_value, &mut bytes[field_len..])?;
  Ok(bytes)
}

fn parse_der_integer_into(input: &[u8], out: &mut [u8]) -> Result<(), EcdsaError> {
  let value = canonical_unsigned_integer(input)?;
  if value.len() > out.len() {
    return Err(EcdsaError::InvalidSignature);
  }
  let start = out.len().strict_sub(value.len());
  let Some(suffix) = out.get_mut(start..) else {
    return Err(EcdsaError::InvalidSignature);
  };
  suffix.copy_from_slice(value);
  Ok(())
}

fn canonical_unsigned_integer(input: &[u8]) -> Result<&[u8], EcdsaError> {
  let Some((&first, rest)) = input.split_first() else {
    return Err(EcdsaError::MalformedDer);
  };
  if first & 0x80 != 0 {
    return Err(EcdsaError::MalformedDer);
  }
  if first == 0 {
    let Some((&next, _)) = rest.split_first() else {
      return Err(EcdsaError::MalformedDer);
    };
    if next & 0x80 == 0 {
      return Err(EcdsaError::MalformedDer);
    }
    Ok(rest)
  } else {
    Ok(input)
  }
}

fn parse_spki_der<'a>(der: &'a [u8], curve_oid: &[u8]) -> Result<&'a [u8], EcdsaError> {
  let mut root = DerReader::new(der);
  let spki = root.read_constructed(TAG_SEQUENCE)?;
  root.finish()?;

  let mut spki = DerReader::new(spki);
  let algorithm = spki.read_constructed(TAG_SEQUENCE)?;
  let subject_public_key = spki.read_primitive(TAG_BIT_STRING)?;
  spki.finish()?;

  let mut algorithm = DerReader::new(algorithm);
  let algorithm_oid = algorithm.read_primitive(TAG_OBJECT_IDENTIFIER)?;
  let named_curve = algorithm.read_primitive(TAG_OBJECT_IDENTIFIER)?;
  algorithm.finish()?;
  if algorithm_oid != ID_EC_PUBLIC_KEY_OID {
    return Err(EcdsaError::UnsupportedAlgorithm);
  }
  if named_curve != curve_oid {
    return Err(EcdsaError::UnsupportedAlgorithm);
  }

  let (&unused_bits, public_key) = subject_public_key.split_first().ok_or(EcdsaError::MalformedDer)?;
  if unused_bits != 0 || public_key.is_empty() {
    return Err(EcdsaError::MalformedDer);
  }
  Ok(public_key)
}

fn mul_mod_montgomery<const L: usize>(lhs: Uint<L>, rhs: Uint<L>, modulus: &'static Modulus<L>) -> Uint<L> {
  let lhs = montgomery_mul(lhs, modulus.r2, modulus);
  let rhs = montgomery_mul(rhs, modulus.r2, modulus);
  let product = montgomery_mul(lhs, rhs, modulus);
  montgomery_mul(product, Uint::ONE, modulus)
}

fn mul_mod_montgomery_ct<const L: usize>(mut lhs: Uint<L>, mut rhs: Uint<L>, modulus: &'static Modulus<L>) -> Uint<L> {
  let mut lhs_mont = montgomery_mul(lhs, modulus.r2, modulus);
  let mut rhs_mont = montgomery_mul(rhs, modulus.r2, modulus);
  let mut product = montgomery_mul(lhs_mont, rhs_mont, modulus);
  let result = montgomery_mul(product, Uint::ONE, modulus);

  lhs.zeroize_no_fence();
  rhs.zeroize_no_fence();
  lhs_mont.zeroize_no_fence();
  rhs_mont.zeroize_no_fence();
  product.zeroize_no_fence();
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
  result
}

fn mul_mod_montgomery_blinded_ct<const L: usize>(
  lhs: Uint<L>,
  rhs: Uint<L>,
  blind: Uint<L>,
  modulus: &'static Modulus<L>,
) -> Uint<L> {
  let rhs_share = SecretScalar::new(rhs.sub_mod_ct(&blind, modulus.value));
  let product_share = SecretScalar::new(mul_mod_montgomery_ct(lhs, rhs_share.value(), modulus));
  let product_blind = SecretScalar::new(mul_mod_montgomery_ct(lhs, blind, modulus));
  product_share.value().add_mod_ct(&product_blind.value(), modulus.value)
}

#[allow(clippy::indexing_slicing)]
fn montgomery_mul<const L: usize>(lhs: Uint<L>, rhs: Uint<L>, modulus: &'static Modulus<L>) -> Uint<L> {
  debug_assert!(L <= 6);

  if is_p256_field_modulus(modulus) {
    let reduced = montgomery_mul_p256_field(
      [lhs.0[0], lhs.0[1], lhs.0[2], lhs.0[3]],
      [rhs.0[0], rhs.0[1], rhs.0[2], rhs.0[3]],
    );
    let mut out = [0u64; L];
    out.copy_from_slice(&reduced.0);
    return Uint(out);
  }

  if is_p384_field_modulus(modulus) {
    let lhs = [lhs.0[0], lhs.0[1], lhs.0[2], lhs.0[3], lhs.0[4], lhs.0[5]];
    let rhs = [rhs.0[0], rhs.0[1], rhs.0[2], rhs.0[3], rhs.0[4], rhs.0[5]];
    #[cfg(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", target_os = "linux")
    ))]
    let reduced = ecdsa_platform_asm::p384_field_mul(&lhs, &rhs);
    #[cfg(not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", target_os = "linux")
    )))]
    let reduced = ecdsa_p384_field::mul(lhs, rhs);
    let mut out = [0u64; L];
    out.copy_from_slice(&reduced);
    return Uint(out);
  }

  if is_p256_order_modulus(modulus) {
    let reduced = montgomery_mul_p256_order(
      [lhs.0[0], lhs.0[1], lhs.0[2], lhs.0[3]],
      [rhs.0[0], rhs.0[1], rhs.0[2], rhs.0[3]],
    );
    let mut out = [0u64; L];
    out.copy_from_slice(&reduced.0);
    return Uint(out);
  }

  if is_p384_order_modulus(modulus) {
    let reduced = montgomery_mul_p384_order(
      [lhs.0[0], lhs.0[1], lhs.0[2], lhs.0[3], lhs.0[4], lhs.0[5]],
      [rhs.0[0], rhs.0[1], rhs.0[2], rhs.0[3], rhs.0[4], rhs.0[5]],
    );
    let mut out = [0u64; L];
    out.copy_from_slice(&reduced.0);
    return Uint(out);
  }

  let mut limbs = [0u64; 13];
  for i in 0..L {
    let mut carry = 0u64;
    for j in 0..L {
      let k = i + j;
      (limbs[k], carry) = mac_limb(limbs[k], lhs.0[i], rhs.0[j], carry);
    }
    add_limb(&mut limbs, i + L, carry);
  }

  for i in 0..L {
    let factor = mul_u64_low(limbs[i], modulus.n0_inv);
    let mut carry = 0u64;
    for j in 0..L {
      let k = i + j;
      (limbs[k], carry) = mac_limb(limbs[k], factor, modulus.value.0[j], carry);
    }
    add_limb(&mut limbs, i + L, carry);
  }

  let mut out = [0u64; L];
  out.copy_from_slice(&limbs[L..L + L]);
  let out = Uint(out);
  let (reduced, borrow) = out.sub_raw(&modulus.value);
  Uint::select(out, reduced, mask_nonzero_u64(limbs[L + L]) | mask_zero_u64(borrow))
}

fn montgomery_square<const L: usize>(value: Uint<L>, modulus: &'static Modulus<L>) -> Uint<L> {
  if is_p384_field_modulus(modulus) {
    let value = [value.0[0], value.0[1], value.0[2], value.0[3], value.0[4], value.0[5]];
    #[cfg(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", target_os = "linux")
    ))]
    let reduced = ecdsa_platform_asm::p384_field_square(&value);
    #[cfg(not(any(
      all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
      all(target_arch = "x86_64", target_os = "linux")
    )))]
    let reduced = ecdsa_p384_field::square(value);
    let mut out = [0u64; L];
    out.copy_from_slice(&reduced);
    Uint(out)
  } else {
    montgomery_mul(value, value, modulus)
  }
}

fn is_p256_field_modulus<const L: usize>(modulus: &'static Modulus<L>) -> bool {
  L == 4 && core::ptr::from_ref(modulus).cast::<()>() == core::ptr::from_ref(&P256_FIELD_MODULUS).cast::<()>()
}

fn is_p384_field_modulus<const L: usize>(modulus: &'static Modulus<L>) -> bool {
  L == 6 && core::ptr::from_ref(modulus).cast::<()>() == core::ptr::from_ref(&P384_FIELD_MODULUS).cast::<()>()
}

fn montgomery_mul_p256_order(lhs: [u64; 4], rhs: [u64; 4]) -> Uint<4> {
  let n = P256_ORDER.0;
  let n0_inv = P256_ORDER_MODULUS.n0_inv;
  let mut t0 = 0u64;
  let mut t1 = 0u64;
  let mut t2 = 0u64;
  let mut t3 = 0u64;
  let mut t4 = 0u64;

  macro_rules! cios_step {
    ($rhs_limb:expr) => {{
      let (u0, carry) = mac_limb(t0, lhs[0], $rhs_limb, 0);
      let (u1, carry) = mac_limb(t1, lhs[1], $rhs_limb, carry);
      let (u2, carry) = mac_limb(t2, lhs[2], $rhs_limb, carry);
      let (u3, carry) = mac_limb(t3, lhs[3], $rhs_limb, carry);
      let (u4, carry_extra) = adc_limb(t4, carry, 0);

      let factor = mul_u64_low(u0, n0_inv);
      let (_, carry) = mac_limb(u0, factor, n[0], 0);
      let (v0, carry) = mac_limb(u1, factor, n[1], carry);
      let (v1, carry) = mac_limb(u2, factor, n[2], carry);
      let (v2, carry) = mac_limb(u3, factor, n[3], carry);
      let (v3, carry) = adc_limb(u4, carry, 0);

      t0 = v0;
      t1 = v1;
      t2 = v2;
      t3 = v3;
      t4 = carry_extra + carry;
    }};
  }

  cios_step!(rhs[0]);
  cios_step!(rhs[1]);
  cios_step!(rhs[2]);
  cios_step!(rhs[3]);

  sub_p256_order_once([t0, t1, t2, t3, t4])
}

fn sub_p256_order_once(limbs: [u64; 5]) -> Uint<4> {
  let n = P256_ORDER.0;
  let (w0, borrow) = sbb_limb(limbs[0], n[0], 0);
  let (w1, borrow) = sbb_limb(limbs[1], n[1], borrow);
  let (w2, borrow) = sbb_limb(limbs[2], n[2], borrow);
  let (w3, borrow) = sbb_limb(limbs[3], n[3], borrow);
  let (_, borrow) = sbb_limb(limbs[4], 0, borrow);

  let reduced = Uint([w0, w1, w2, w3]);
  Uint::select(
    reduced,
    Uint([limbs[0], limbs[1], limbs[2], limbs[3]]),
    mask_nonzero_u64(borrow),
  )
}

fn montgomery_mul_p384_order(lhs: [u64; 6], rhs: [u64; 6]) -> Uint<6> {
  let n = P384_ORDER.0;
  let n0_inv = P384_ORDER_MODULUS.n0_inv;
  let mut t0 = 0u64;
  let mut t1 = 0u64;
  let mut t2 = 0u64;
  let mut t3 = 0u64;
  let mut t4 = 0u64;
  let mut t5 = 0u64;
  let mut t6 = 0u64;

  macro_rules! cios_step {
    ($rhs_limb:expr) => {{
      let (u0, carry) = mac_limb(t0, lhs[0], $rhs_limb, 0);
      let (u1, carry) = mac_limb(t1, lhs[1], $rhs_limb, carry);
      let (u2, carry) = mac_limb(t2, lhs[2], $rhs_limb, carry);
      let (u3, carry) = mac_limb(t3, lhs[3], $rhs_limb, carry);
      let (u4, carry) = mac_limb(t4, lhs[4], $rhs_limb, carry);
      let (u5, carry) = mac_limb(t5, lhs[5], $rhs_limb, carry);
      let (u6, carry_extra) = adc_limb(t6, carry, 0);

      let factor = mul_u64_low(u0, n0_inv);
      let (_, carry) = mac_limb(u0, factor, n[0], 0);
      let (v0, carry) = mac_limb(u1, factor, n[1], carry);
      let (v1, carry) = mac_limb(u2, factor, n[2], carry);
      let (v2, carry) = mac_limb(u3, factor, n[3], carry);
      let (v3, carry) = mac_limb(u4, factor, n[4], carry);
      let (v4, carry) = mac_limb(u5, factor, n[5], carry);
      let (v5, carry) = adc_limb(u6, carry, 0);

      t0 = v0;
      t1 = v1;
      t2 = v2;
      t3 = v3;
      t4 = v4;
      t5 = v5;
      t6 = carry_extra + carry;
    }};
  }

  cios_step!(rhs[0]);
  cios_step!(rhs[1]);
  cios_step!(rhs[2]);
  cios_step!(rhs[3]);
  cios_step!(rhs[4]);
  cios_step!(rhs[5]);

  sub_p384_order_once([t0, t1, t2, t3, t4, t5, t6])
}

fn sub_p384_order_once(limbs: [u64; 7]) -> Uint<6> {
  let n = P384_ORDER.0;
  let (w0, borrow) = sbb_limb(limbs[0], n[0], 0);
  let (w1, borrow) = sbb_limb(limbs[1], n[1], borrow);
  let (w2, borrow) = sbb_limb(limbs[2], n[2], borrow);
  let (w3, borrow) = sbb_limb(limbs[3], n[3], borrow);
  let (w4, borrow) = sbb_limb(limbs[4], n[4], borrow);
  let (w5, borrow) = sbb_limb(limbs[5], n[5], borrow);
  let (_, borrow) = sbb_limb(limbs[6], 0, borrow);

  let reduced = Uint([w0, w1, w2, w3, w4, w5]);
  Uint::select(
    reduced,
    Uint([limbs[0], limbs[1], limbs[2], limbs[3], limbs[4], limbs[5]]),
    mask_nonzero_u64(borrow),
  )
}

fn add_p384_field(lhs: [u64; 6], rhs: [u64; 6]) -> Uint<6> {
  let p = P384_FIELD.0;
  let (s0, carry) = adc_limb(lhs[0], rhs[0], 0);
  let (s1, carry) = adc_limb(lhs[1], rhs[1], carry);
  let (s2, carry) = adc_limb(lhs[2], rhs[2], carry);
  let (s3, carry) = adc_limb(lhs[3], rhs[3], carry);
  let (s4, carry) = adc_limb(lhs[4], rhs[4], carry);
  let (s5, carry) = adc_limb(lhs[5], rhs[5], carry);

  let (r0, borrow) = sbb_limb(s0, p[0], 0);
  let (r1, borrow) = sbb_limb(s1, p[1], borrow);
  let (r2, borrow) = sbb_limb(s2, p[2], borrow);
  let (r3, borrow) = sbb_limb(s3, p[3], borrow);
  let (r4, borrow) = sbb_limb(s4, p[4], borrow);
  let (r5, borrow) = sbb_limb(s5, p[5], borrow);

  Uint::select(
    Uint([s0, s1, s2, s3, s4, s5]),
    Uint([r0, r1, r2, r3, r4, r5]),
    mask_nonzero_u64(carry) | mask_zero_u64(borrow),
  )
}

fn sub_p384_field(lhs: [u64; 6], rhs: [u64; 6]) -> Uint<6> {
  let p = P384_FIELD.0;
  let (d0, borrow) = sbb_limb(lhs[0], rhs[0], 0);
  let (d1, borrow) = sbb_limb(lhs[1], rhs[1], borrow);
  let (d2, borrow) = sbb_limb(lhs[2], rhs[2], borrow);
  let (d3, borrow) = sbb_limb(lhs[3], rhs[3], borrow);
  let (d4, borrow) = sbb_limb(lhs[4], rhs[4], borrow);
  let (d5, borrow) = sbb_limb(lhs[5], rhs[5], borrow);

  let (r0, carry) = adc_limb(d0, p[0], 0);
  let (r1, carry) = adc_limb(d1, p[1], carry);
  let (r2, carry) = adc_limb(d2, p[2], carry);
  let (r3, carry) = adc_limb(d3, p[3], carry);
  let (r4, carry) = adc_limb(d4, p[4], carry);
  let (r5, _) = adc_limb(d5, p[5], carry);
  Uint::select(
    Uint([d0, d1, d2, d3, d4, d5]),
    Uint([r0, r1, r2, r3, r4, r5]),
    mask_nonzero_u64(borrow),
  )
}

fn montgomery_mul_p256_field(lhs: [u64; 4], rhs: [u64; 4]) -> Uint<4> {
  let (w0, carry) = mac_limb(0, lhs[0], rhs[0], 0);
  let (w1, carry) = mac_limb(0, lhs[0], rhs[1], carry);
  let (w2, carry) = mac_limb(0, lhs[0], rhs[2], carry);
  let (w3, w4) = mac_limb(0, lhs[0], rhs[3], carry);

  let (w1, carry) = mac_limb(w1, lhs[1], rhs[0], 0);
  let (w2, carry) = mac_limb(w2, lhs[1], rhs[1], carry);
  let (w3, carry) = mac_limb(w3, lhs[1], rhs[2], carry);
  let (w4, w5) = mac_limb(w4, lhs[1], rhs[3], carry);

  let (w2, carry) = mac_limb(w2, lhs[2], rhs[0], 0);
  let (w3, carry) = mac_limb(w3, lhs[2], rhs[1], carry);
  let (w4, carry) = mac_limb(w4, lhs[2], rhs[2], carry);
  let (w5, w6) = mac_limb(w5, lhs[2], rhs[3], carry);

  let (w3, carry) = mac_limb(w3, lhs[3], rhs[0], 0);
  let (w4, carry) = mac_limb(w4, lhs[3], rhs[1], carry);
  let (w5, carry) = mac_limb(w5, lhs[3], rhs[2], carry);
  let (w6, w7) = mac_limb(w6, lhs[3], rhs[3], carry);

  montgomery_reduce_p256_field([w0, w1, w2, w3, w4, w5, w6, w7])
}

#[allow(clippy::indexing_slicing)]
fn montgomery_reduce_p256_field(limbs: [u64; 8]) -> Uint<4> {
  let [r0, r1, r2, r3, r4, r5, r6, r7] = limbs;
  let p = P256_FIELD.0;

  let (r1, carry) = mac_limb(r1, r0, p[1], r0);
  let (r2, carry) = adc_limb(r2, 0, carry);
  let (r3, carry) = mac_limb(r3, r0, p[3], carry);
  let (r4, carry2) = adc_limb(r4, 0, carry);

  let (r2, carry) = mac_limb(r2, r1, p[1], r1);
  let (r3, carry) = adc_limb(r3, 0, carry);
  let (r4, carry) = mac_limb(r4, r1, p[3], carry);
  let (r5, carry2) = adc_limb(r5, carry2, carry);

  let (r3, carry) = mac_limb(r3, r2, p[1], r2);
  let (r4, carry) = adc_limb(r4, 0, carry);
  let (r5, carry) = mac_limb(r5, r2, p[3], carry);
  let (r6, carry2) = adc_limb(r6, carry2, carry);

  let (r4, carry) = mac_limb(r4, r3, p[1], r3);
  let (r5, carry) = adc_limb(r5, 0, carry);
  let (r6, carry) = mac_limb(r6, r3, p[3], carry);
  let (r7, r8) = adc_limb(r7, carry2, carry);

  sub_p256_field_once([r4, r5, r6, r7, r8])
}

fn sub_p256_field_once(limbs: [u64; 5]) -> Uint<4> {
  let p = P256_FIELD.0;
  let (w0, borrow) = sbb_limb(limbs[0], p[0], 0);
  let (w1, borrow) = sbb_limb(limbs[1], p[1], borrow);
  let (w2, borrow) = sbb_limb(limbs[2], p[2], borrow);
  let (w3, borrow) = sbb_limb(limbs[3], p[3], borrow);
  let (_, borrow) = sbb_limb(limbs[4], 0, borrow);

  let reduced = Uint([w0, w1, w2, w3]);
  let (s0, carry) = adc_limb(w0, p[0], 0);
  let (s1, carry) = adc_limb(w1, p[1], carry);
  let (s2, carry) = adc_limb(w2, p[2], carry);
  let (s3, _) = adc_limb(w3, p[3], carry);
  Uint::select(reduced, Uint([s0, s1, s2, s3]), mask_nonzero_u64(borrow))
}

#[inline(always)]
fn adc_limb(lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
  let result = u128::from(lhs) + u128::from(rhs) + u128::from(carry);
  (result as u64, (result >> 64) as u64)
}

#[inline(always)]
fn sbb_limb(lhs: u64, rhs: u64, borrow: u64) -> (u64, u64) {
  let (diff, overflow0) = lhs.overflowing_sub(rhs);
  let (diff, overflow1) = diff.overflowing_sub(borrow);
  (diff, u64::from(overflow0 | overflow1))
}

#[inline(always)]
fn mac_limb(acc: u64, lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
  #[cfg(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x"))]
  {
    let (product_lo, product_hi) = mul_u64_wide(lhs, rhs);
    let (result, carry0) = product_lo.overflowing_add(acc);
    let (result, carry1) = result.overflowing_add(carry);
    let (high, overflow0) = product_hi.overflowing_add(u64::from(carry0));
    let (high, overflow1) = high.overflowing_add(u64::from(carry1));

    // lhs*rhs + acc + carry is at most 2^128 - 1, so the high limb cannot overflow.
    // Keep that invariant checked in debug builds without emitting secret-fed panic branches
    // in release ECDSA arithmetic on s390x and RISC-V.
    debug_assert!(!overflow0 && !overflow1);
    (result, high)
  }

  #[cfg(not(any(test, target_arch = "riscv32", target_arch = "riscv64", target_arch = "s390x")))]
  {
    let result = u128::from(acc) + (u128::from(lhs) * u128::from(rhs)) + u128::from(carry);
    (result as u64, (result >> 64) as u64)
  }
}

#[allow(clippy::indexing_slicing)]
fn add_limb(limbs: &mut [u64; 13], mut index: usize, mut value: u64) {
  while index < limbs.len() {
    let (sum, carry) = limbs[index].overflowing_add(value);
    limbs[index] = sum;
    value = u64::from(carry);
    index = index.strict_add(1);
  }
}

struct DerReader<'a> {
  input: &'a [u8],
  offset: usize,
}

impl<'a> DerReader<'a> {
  const fn new(input: &'a [u8]) -> Self {
    Self { input, offset: 0 }
  }

  fn read_constructed(&mut self, tag: u8) -> Result<&'a [u8], EcdsaError> {
    self.read_primitive(tag)
  }

  fn read_primitive(&mut self, tag: u8) -> Result<&'a [u8], EcdsaError> {
    let actual = self.read_byte()?;
    if actual != tag {
      return Err(EcdsaError::MalformedDer);
    }
    let len = self.read_len()?;
    let end = self.offset.checked_add(len).ok_or(EcdsaError::MalformedDer)?;
    if end > self.input.len() {
      return Err(EcdsaError::MalformedDer);
    }
    let value = self.input.get(self.offset..end).ok_or(EcdsaError::MalformedDer)?;
    self.offset = end;
    Ok(value)
  }

  fn finish(&self) -> Result<(), EcdsaError> {
    if self.offset == self.input.len() {
      Ok(())
    } else {
      Err(EcdsaError::MalformedDer)
    }
  }

  fn read_byte(&mut self) -> Result<u8, EcdsaError> {
    let byte = *self.input.get(self.offset).ok_or(EcdsaError::MalformedDer)?;
    self.offset = self.offset.strict_add(1);
    Ok(byte)
  }

  fn read_len(&mut self) -> Result<usize, EcdsaError> {
    let first = self.read_byte()?;
    if first & 0x80 == 0 {
      return Ok(usize::from(first));
    }

    let len_len = usize::from(first & 0x7f);
    if len_len == 0 || len_len > core::mem::size_of::<usize>() {
      return Err(EcdsaError::MalformedDer);
    }

    let first_len_byte = self.read_byte()?;
    if first_len_byte == 0 {
      return Err(EcdsaError::MalformedDer);
    }

    let mut len = usize::from(first_len_byte);
    for _ in 1..len_len {
      len = len.checked_shl(8).ok_or(EcdsaError::MalformedDer)?;
      len |= usize::from(self.read_byte()?);
    }

    if len < 128 {
      return Err(EcdsaError::MalformedDer);
    }
    Ok(len)
  }
}

#[cfg(test)]
mod tests {
  use alloc::format;

  use super::*;

  fn p256_public_key() -> EcdsaP256PublicKey {
    let mut sec1 = [0u8; EcdsaP256PublicKey::SEC1_LENGTH];
    sec1[0] = 0x04;
    P256_GX.write_be(&mut sec1[1..33]);
    P256_GY.write_be(&mut sec1[33..]);
    EcdsaP256PublicKey::from_sec1_bytes(&sec1).unwrap()
  }

  fn p384_public_key() -> EcdsaP384PublicKey {
    let mut sec1 = [0u8; EcdsaP384PublicKey::SEC1_LENGTH];
    sec1[0] = 0x04;
    P384_GX.write_be(&mut sec1[1..49]);
    P384_GY.write_be(&mut sec1[49..]);
    EcdsaP384PublicKey::from_sec1_bytes(&sec1).unwrap()
  }

  fn p384_sparse_scalar(bits: &[usize]) -> Uint<6> {
    let mut scalar = Uint::ZERO;
    for &bit in bits {
      scalar.0[bit / 64] |= 1u64 << (bit % 64);
    }
    assert!(scalar.cmp(&P384_ORDER).is_lt());
    scalar
  }

  fn assert_p384_nonexceptional_comb_matches_complete(scalar: Uint<6>) {
    let complete = scalar_mul_basepoint_comb_ct(&P384, scalar).to_affine_ct(P384_FIELD_MINUS_TWO);
    let secret_scalar = SecretScalar::new(scalar);
    let optimized = p384_scalar_mul_basepoint_comb_nonexceptional_ct(&secret_scalar).to_affine_ct(P384_FIELD_MINUS_TWO);

    assert!(
      optimized == complete,
      "P-384 nonexceptional comb must match complete comb"
    );
  }

  #[test]
  fn ct_mul_u64_wide_matches_u128_for_edges_and_generated_inputs() {
    const EDGES: [u64; 8] = [0, 1, 2, u32::MAX as u64, 1u64 << 32, 1u64 << 63, u64::MAX - 1, u64::MAX];

    for lhs in EDGES {
      for rhs in EDGES {
        let product = u128::from(lhs) * u128::from(rhs);
        assert_eq!(ct_mul_u64_wide(lhs, rhs), (product as u64, (product >> 64) as u64));
      }
    }

    let mut lhs = 0x243f_6a88_85a3_08d3u64;
    let mut rhs = 0x1319_8a2e_0370_7344u64;
    for _ in 0..1024 {
      lhs = lhs.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1);
      rhs = rhs.wrapping_mul(0xd134_2543_de82_ef95).wrapping_add(1);
      let product = u128::from(lhs) * u128::from(rhs);
      assert_eq!(ct_mul_u64_wide(lhs, rhs), (product as u64, (product >> 64) as u64));
    }
  }

  #[test]
  fn mac_limb_matches_u128_for_edges_and_generated_inputs() {
    const EDGES: [u64; 8] = [0, 1, 2, u32::MAX as u64, 1u64 << 32, 1u64 << 63, u64::MAX - 1, u64::MAX];

    for acc in EDGES {
      for lhs in EDGES {
        for rhs in EDGES {
          for carry in EDGES {
            let expected = u128::from(lhs) * u128::from(rhs) + u128::from(acc) + u128::from(carry);
            assert_eq!(
              mac_limb(acc, lhs, rhs, carry),
              (expected as u64, (expected >> 64) as u64)
            );
          }
        }
      }
    }

    let mut state = 0x243f_6a88_85a3_08d3u64;
    for _ in 0..4096 {
      let acc = state;
      state = state.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1);
      let lhs = state;
      state = state.wrapping_mul(0xd134_2543_de82_ef95).wrapping_add(1);
      let rhs = state;
      state = state.wrapping_mul(0xa409_3822_299f_31d0).wrapping_add(1);
      let carry = state;
      let expected = u128::from(lhs) * u128::from(rhs) + u128::from(acc) + u128::from(carry);
      assert_eq!(
        mac_limb(acc, lhs, rhs, carry),
        (expected as u64, (expected >> 64) as u64)
      );
    }
  }

  #[test]
  fn p256_sec1_rejects_wrong_shape_and_off_curve_points() {
    assert_eq!(
      EcdsaP256PublicKey::from_sec1_bytes(&[]).err(),
      Some(EcdsaError::InvalidPublicKey)
    );

    let mut sec1 = p256_public_key().to_sec1_bytes();
    sec1[0] = 0x02;
    assert_eq!(
      EcdsaP256PublicKey::from_sec1_bytes(&sec1).err(),
      Some(EcdsaError::InvalidPublicKey)
    );

    let mut sec1 = p256_public_key().to_sec1_bytes();
    sec1[64] ^= 1;
    assert_eq!(
      EcdsaP256PublicKey::from_sec1_bytes(&sec1).err(),
      Some(EcdsaError::InvalidPublicKey)
    );
  }

  #[test]
  fn p384_sec1_rejects_wrong_shape_and_off_curve_points() {
    assert_eq!(
      EcdsaP384PublicKey::from_sec1_bytes(&[]).err(),
      Some(EcdsaError::InvalidPublicKey)
    );

    let mut sec1 = p384_public_key().to_sec1_bytes();
    sec1[0] = 0x03;
    assert_eq!(
      EcdsaP384PublicKey::from_sec1_bytes(&sec1).err(),
      Some(EcdsaError::InvalidPublicKey)
    );

    let mut sec1 = p384_public_key().to_sec1_bytes();
    sec1[96] ^= 1;
    assert_eq!(
      EcdsaP384PublicKey::from_sec1_bytes(&sec1).err(),
      Some(EcdsaError::InvalidPublicKey)
    );
  }

  #[test]
  fn p256_signature_rejects_zero_or_out_of_range_scalars() {
    let mut bytes = [0u8; EcdsaP256Signature::LENGTH];
    bytes[31] = 1;
    assert_eq!(
      EcdsaP256Signature::from_bytes(bytes).err(),
      Some(EcdsaError::InvalidSignature)
    );

    bytes[63] = 1;
    assert!(EcdsaP256Signature::from_bytes(bytes).is_ok());

    let mut out_of_range = bytes;
    P256_ORDER.write_be(&mut out_of_range[..32]);
    assert_eq!(
      EcdsaP256Signature::from_bytes(out_of_range).err(),
      Some(EcdsaError::InvalidSignature)
    );
  }

  #[test]
  fn p384_signature_rejects_zero_or_out_of_range_scalars() {
    let mut bytes = [0u8; EcdsaP384Signature::LENGTH];
    bytes[47] = 1;
    assert_eq!(
      EcdsaP384Signature::from_bytes(bytes).err(),
      Some(EcdsaError::InvalidSignature)
    );

    bytes[95] = 1;
    assert!(EcdsaP384Signature::from_bytes(bytes).is_ok());

    let mut out_of_range = bytes;
    P384_ORDER.write_be(&mut out_of_range[..48]);
    assert_eq!(
      EcdsaP384Signature::from_bytes(out_of_range).err(),
      Some(EcdsaError::InvalidSignature)
    );
  }

  #[test]
  fn der_signature_parser_requires_canonical_unsigned_integers() {
    let good = [0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01];
    assert!(EcdsaP256Signature::from_der(&good).is_ok());

    let negative = [0x30, 0x07, 0x02, 0x02, 0x80, 0x01, 0x02, 0x01, 0x01];
    assert_eq!(
      EcdsaP256Signature::from_der(&negative).err(),
      Some(EcdsaError::MalformedDer)
    );

    let overlong = [0x30, 0x08, 0x02, 0x03, 0x00, 0x01, 0x01, 0x02, 0x01, 0x01];
    assert_eq!(
      EcdsaP256Signature::from_der(&overlong).err(),
      Some(EcdsaError::MalformedDer)
    );
  }

  #[test]
  fn p256_secret_key_rejects_zero_and_group_order() {
    assert_eq!(
      EcdsaP256SecretKey::from_bytes([0u8; EcdsaP256SecretKey::LENGTH]).err(),
      Some(EcdsaError::InvalidSecretKey)
    );

    let mut order = [0u8; EcdsaP256SecretKey::LENGTH];
    P256_ORDER.write_be(&mut order);
    assert_eq!(
      EcdsaP256SecretKey::from_bytes(order).err(),
      Some(EcdsaError::InvalidSecretKey)
    );
  }

  #[test]
  fn p384_secret_key_rejects_zero_and_group_order() {
    assert_eq!(
      EcdsaP384SecretKey::from_bytes([0u8; EcdsaP384SecretKey::LENGTH]).err(),
      Some(EcdsaError::InvalidSecretKey)
    );

    let mut order = [0u8; EcdsaP384SecretKey::LENGTH];
    P384_ORDER.write_be(&mut order);
    assert_eq!(
      EcdsaP384SecretKey::from_bytes(order).err(),
      Some(EcdsaError::InvalidSecretKey)
    );
  }

  #[test]
  fn p256_secret_key_one_derives_generator_public_key() {
    let mut one = [0u8; EcdsaP256SecretKey::LENGTH];
    one[EcdsaP256SecretKey::LENGTH - 1] = 1;
    let secret = EcdsaP256SecretKey::from_bytes(one).unwrap();

    assert_eq!(secret.public_key(), p256_public_key());
    assert_eq!(secret.public_key_blinded(|blind| blind.fill(0xa5)), p256_public_key());
  }

  #[test]
  fn p384_secret_key_one_derives_generator_public_key() {
    let mut one = [0u8; EcdsaP384SecretKey::LENGTH];
    one[EcdsaP384SecretKey::LENGTH - 1] = 1;
    let secret = EcdsaP384SecretKey::from_bytes(one).unwrap();

    assert_eq!(secret.public_key(), p384_public_key());
    assert_eq!(secret.public_key_blinded(|blind| blind.fill(0x5a)), p384_public_key());
  }

  #[test]
  fn p256_try_sign_is_deterministic_low_s_and_verifies() {
    let secret = EcdsaP256SecretKey::from_bytes([0x42; EcdsaP256SecretKey::LENGTH]).unwrap();
    let public = secret.public_key();
    let message = b"rscrypto ecdsa p256 signing";

    let first = secret.try_sign(message).unwrap();
    let second = secret.try_sign(message).unwrap();
    let blinded = secret.try_sign_blinded(message, |blind| blind.fill(0x7b)).unwrap();

    assert_eq!(first, second);
    assert!(public.verify(message, &first).is_ok());
    assert!(public.verify(message, &blinded).is_ok());
    assert_eq!(first, blinded);
    assert!(
      Uint::from_be_slice(&first.as_bytes()[32..])
        .unwrap()
        .cmp(&P256_ORDER_HALF)
        .is_le()
    );
  }

  #[test]
  fn p384_try_sign_is_deterministic_low_s_and_verifies() {
    let secret = EcdsaP384SecretKey::from_bytes([0x24; EcdsaP384SecretKey::LENGTH]).unwrap();
    let public = secret.public_key();
    let message = b"rscrypto ecdsa p384 signing";

    let first = secret.try_sign(message).unwrap();
    let second = secret.try_sign(message).unwrap();
    let blinded = secret.try_sign_blinded(message, |blind| blind.fill(0xb7)).unwrap();

    assert_eq!(first, second);
    assert!(public.verify(message, &first).is_ok());
    assert!(public.verify(message, &blinded).is_ok());
    assert_eq!(first, blinded);
    assert!(
      Uint::from_be_slice(&first.as_bytes()[48..])
        .unwrap()
        .cmp(&P384_ORDER_HALF)
        .is_le()
    );
  }

  #[test]
  fn p256_comb_scalar_blinding_preserves_the_group_element() {
    let max_scalar = P256_ORDER.sub_raw(&Uint::ONE).0;
    for scalar_value in [Uint::ONE, Uint::from_u64(0x1234_5678_9abc_def0), max_scalar] {
      let scalar = SecretScalar::new(scalar_value);
      let expected = scalar_mul_basepoint_comb_ct_secret(&P256, &scalar).to_affine_ct(P256_FIELD_MINUS_TWO);
      let z = FieldElement::from_uint(Uint::from_u64(2), &P256_FIELD_MODULUS);

      for factor in 0u64..=7 {
        let blind = SecretScalar::new(Uint::from_u64(factor));
        let blinded = p256_blinded_comb_scalar(&scalar, &blind, P256_ORDER);
        let actual = scalar_mul_basepoint_comb_ct_secret_blinded(&P256, &blinded, z).to_affine_ct(P256_FIELD_MINUS_TWO);
        assert!(
          actual == expected,
          "adding a multiple of the P-256 order must preserve kG"
        );
      }
    }
  }

  #[test]
  fn p384_comb_projective_blinding_preserves_the_group_element() {
    let scalar = SecretScalar::new(Uint::<6>::from_u64(0x1234_5678_9abc_def0));
    let expected = scalar_mul_basepoint_comb_ct_secret(&P384, &scalar).to_affine_ct(P384_FIELD_MINUS_TWO);
    let z = FieldElement::from_uint(Uint::from_u64(2), &P384_FIELD_MODULUS);
    let actual = scalar_mul_basepoint_comb_ct_secret_blinded(&P384, &scalar, z).to_affine_ct(P384_FIELD_MINUS_TWO);
    assert!(actual == expected, "projective blinding must preserve kG");
  }

  #[cfg(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", target_os = "linux")
  ))]
  #[test]
  fn p256_platform_basepoint_matches_portable_comb_for_sample_scalars() {
    let scalars = [
      Uint::ONE,
      Uint::from_u64(2),
      Uint::from_u64(0x1234_5678_9abc_def0),
      P256_ORDER.sub_raw(&Uint::ONE).0,
    ];

    for scalar in scalars {
      let portable = scalar_mul_basepoint_comb_ct(&P256, scalar).to_affine_ct(P256_FIELD_MINUS_TWO);
      let secret_scalar = SecretScalar::new(scalar);
      let backend =
        scalar_mul_basepoint_affine_backend(&P256, &secret_scalar).expect("P-256 platform backend must be present");

      assert!(
        backend == portable,
        "P-256 platform basepoint backend must match portable comb"
      );
    }
  }

  #[cfg(any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", target_os = "linux")
  ))]
  #[test]
  fn p384_platform_basepoint_matches_portable_comb_for_sample_scalars() {
    let scalars = [
      Uint::ONE,
      Uint::from_u64(2),
      Uint::from_u64(0x1234_5678_9abc_def0),
      P384_ORDER.sub_raw(&Uint::ONE).0,
    ];

    for scalar in scalars {
      let portable = scalar_mul_basepoint_comb_ct(&P384, scalar).to_affine_ct(P384_FIELD_MINUS_TWO);
      let secret_scalar = SecretScalar::new(scalar);
      let backend = scalar_mul_basepoint_backend(&P384, &secret_scalar)
        .expect("P-384 platform backend must be present")
        .to_affine_ct(P384_FIELD_MINUS_TWO);

      assert!(
        backend == portable,
        "P-384 platform basepoint backend must match portable comb"
      );
    }
  }

  #[test]
  fn p384_nonexceptional_comb_matches_complete_comb_for_structured_scalars() {
    for scalar in [
      Uint::ONE,
      Uint::from_u64(2),
      Uint::from_u64(0x1234_5678_9abc_def0),
      p384_sparse_scalar(&[0, 48, 96, 144, 192, 240, 288, 336]),
      p384_sparse_scalar(&[1, 49, 97, 145, 193, 241, 289, 337]),
      p384_sparse_scalar(&[47, 95, 143, 191, 239, 287, 335, 383]),
      P384_ORDER.sub_raw(&Uint::ONE).0,
    ] {
      assert_p384_nonexceptional_comb_matches_complete(scalar);
    }

    for row in [0usize, 1, 2, 23, 46] {
      for column in [0usize, 1, 3, 7] {
        let bit = row + column * P384_SIGNING_COMB_ROWS;
        let next_bit = bit + 1;
        assert_p384_nonexceptional_comb_matches_complete(p384_sparse_scalar(&[bit, next_bit]));
      }
    }
  }

  fn p384_wide_from_low(low: Uint<6>) -> [u8; 96] {
    let mut bytes = [0u8; 96];
    low.write_be(&mut bytes[48..]);
    bytes
  }

  #[test]
  fn p384_owned_wide_order_reduction_handles_boundary_values() {
    let order_minus_one = P384_ORDER.sub_raw(&Uint::ONE).0;
    let (order_plus_one, carry) = P384_ORDER.add_raw(&Uint::ONE);
    assert_eq!(carry, 0);

    for (low, expected) in [
      (Uint::ZERO, Uint::ONE),
      (Uint::ONE, Uint::ONE),
      (order_minus_one, order_minus_one),
      (P384_ORDER, Uint::ONE),
      (order_plus_one, Uint::ONE),
    ] {
      let reduced = reduce_wide_order_nonzero_owned(&p384_wide_from_low(low), &P384_ORDER_MODULUS);
      assert_eq!(reduced.0, expected.0);
    }
  }

  #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")))]
  #[test]
  fn p384_owned_wide_order_reduction_matches_platform_reduction() {
    for bytes in [[0u8; 96], [0xffu8; 96], {
      let mut bytes = [0u8; 96];
      for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(17).wrapping_add(0xa5);
      }
      bytes
    }] {
      let platform = Uint(ecdsa_platform_asm::p384_reduce_order_96(&bytes));
      let platform = Uint::select(platform, Uint::ONE, platform.ct_is_zero_mask());
      let owned = reduce_wide_order_nonzero_owned(&bytes, &P384_ORDER_MODULUS);
      assert_eq!(owned.0, platform.0);
    }
  }

  #[test]
  fn ecdsa_secret_key_debug_is_redacted() {
    let p256 = EcdsaP256SecretKey::from_bytes([0x11; EcdsaP256SecretKey::LENGTH]).unwrap();
    let p384 = EcdsaP384SecretKey::from_bytes([0x22; EcdsaP384SecretKey::LENGTH]).unwrap();

    assert_eq!(format!("{p256:?}"), "EcdsaP256SecretKey(****)");
    assert_eq!(format!("{p384:?}"), "EcdsaP384SecretKey(****)");
  }

  #[test]
  fn p384_field_add_sub_specializations_match_generic_modular_arithmetic() {
    let p_minus_one = P384_FIELD.sub_raw(&Uint::ONE).0;
    let values = [
      Uint::ZERO,
      Uint::ONE,
      Uint::from_u64(2),
      P384_GX,
      P384_GY,
      P384_B,
      p_minus_one,
    ];

    for lhs in values {
      for rhs in values {
        assert_eq!(add_p384_field(lhs.0, rhs.0).0, lhs.add_mod_ct(&rhs, P384_FIELD).0);
        assert_eq!(sub_p384_field(lhs.0, rhs.0).0, lhs.sub_mod(&rhs, P384_FIELD).0);
      }
    }
  }
}
