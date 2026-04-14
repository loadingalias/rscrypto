#![cfg(feature = "ed25519")]

use rscrypto::{Ed25519Keypair, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature, verify_ed25519};

struct Vector<'a> {
  secret_key: [u8; 32],
  public_key: [u8; 32],
  message: &'a str,
  signature: [u8; 64],
}

fn decode_hex<const N: usize>(hex: &str) -> [u8; N] {
  let bytes = hex.as_bytes();
  assert_eq!(bytes.len(), N * 2);

  let mut out = [0u8; N];
  for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
    *dst = (hex_nibble(chunk[0]) << 4) | hex_nibble(chunk[1]);
  }
  out
}

fn decode_hex_vec(hex: &str) -> Vec<u8> {
  let bytes = hex.as_bytes();
  assert_eq!(bytes.len() % 2, 0);

  let mut out = Vec::with_capacity(bytes.len() / 2);
  for chunk in bytes.chunks_exact(2) {
    out.push((hex_nibble(chunk[0]) << 4) | hex_nibble(chunk[1]));
  }
  out
}

fn hex_nibble(byte: u8) -> u8 {
  match byte {
    b'0'..=b'9' => byte - b'0',
    b'a'..=b'f' => byte - b'a' + 10,
    b'A'..=b'F' => byte - b'A' + 10,
    _ => panic!("invalid hex"),
  }
}

#[test]
fn ed25519_rfc8032_vectors() {
  let vectors = [
    Vector {
      secret_key: decode_hex(concat!(
        "4ccd089b28ff96da9db6c346ec114e0f",
        "5b8a319f35aba624da8cf6ed4fb8a6fb"
      )),
      public_key: decode_hex(concat!(
        "3d4017c3e843895a92b70aa74d1b7ebc",
        "9c982ccf2ec4968cc0cd55f12af4660c"
      )),
      message: "72",
      signature: decode_hex(concat!(
        "92a009a9f0d4cab8720e820b5f642540",
        "a2b27b5416503f8fb3762223ebdb69da",
        "085ac1e43e15996e458f3613d0f11d8c",
        "387b2eaeb4302aeeb00d291612bb0c00"
      )),
    },
    Vector {
      secret_key: decode_hex(concat!(
        "c5aa8df43f9f837bedb7442f31dcb7b1",
        "66d38535076f094b85ce3a2e0b4458f7"
      )),
      public_key: decode_hex(concat!(
        "fc51cd8e6218a1a38da47ed00230f058",
        "0816ed13ba3303ac5deb911548908025"
      )),
      message: "af82",
      signature: decode_hex(concat!(
        "6291d657deec24024827e69c3abe01a3",
        "0ce548a284743a445e3680d7db5ac3ac",
        "18ff9b538d16f290ae67f760984dc659",
        "4a7c15e9716ed28dc027beceea1ec40a"
      )),
    },
    Vector {
      secret_key: decode_hex(concat!(
        "f5e5767cf153319517630f226876b86c",
        "8160cc583bc013744c6bf255f5cc0ee5"
      )),
      public_key: decode_hex(concat!(
        "278117fc144c72340f67d0f2316e8386",
        "ceffbf2b2428c9c51fef7c597f1d426e"
      )),
      message: concat!(
        "08b8b2b733424243760fe426a4b54908632110a66c2f6591eabd3345e3e4eb98",
        "fa6e264bf09efe12ee50f8f54e9f77b1e355f6c50544e23fb1433ddf73be84d8",
        "79de7c0046dc4996d9e773f4bc9efe5738829adb26c81b37c93a1b270b20329d",
        "658675fc6ea534e0810a4432826bf58c941efb65d57a338bbd2e26640f89ffbc",
        "1a858efcb8550ee3a5e1998bd177e93a7363c344fe6b199ee5d02e82d522c4fe",
        "ba15452f80288a821a579116ec6dad2b3b310da903401aa62100ab5d1a36553e",
        "06203b33890cc9b832f79ef80560ccb9a39ce767967ed628c6ad573cb116dbef",
        "efd75499da96bd68a8a97b928a8bbc103b6621fcde2beca1231d206be6cd9ec7",
        "aff6f6c94fcd7204ed3455c68c83f4a41da4af2b74ef5c53f1d8ac70bdcb7ed1",
        "85ce81bd84359d44254d95629e9855a94a7c1958d1f8ada5d0532ed8a5aa3fb2",
        "d17ba70eb6248e594e1a2297acbbb39d502f1a8c6eb6f1ce22b3de1a1f40cc24",
        "554119a831a9aad6079cad88425de6bde1a9187ebb6092cf67bf2b13fd65f270",
        "88d78b7e883c8759d2c4f5c65adb7553878ad575f9fad878e80a0c9ba63bcbcc",
        "2732e69485bbc9c90bfbd62481d9089beccf80cfe2df16a2cf65bd92dd597b07",
        "07e0917af48bbb75fed413d238f5555a7a569d80c3414a8d0859dc65a46128ba",
        "b27af87a71314f318c782b23ebfe808b82b0ce26401d2e22f04d83d1255dc51a",
        "ddd3b75a2b1ae0784504df543af8969be3ea7082ff7fc9888c144da2af58429e",
        "c96031dbcad3dad9af0dcbaaaf268cb8fcffead94f3c7ca495e056a9b47acdb7",
        "51fb73e666c6c655ade8297297d07ad1ba5e43f1bca32301651339e22904cc8c",
        "42f58c30c04aafdb038dda0847dd988dcda6f3bfd15c4b4c4525004aa06eeff8",
        "ca61783aacec57fb3d1f92b0fe2fd1a85f6724517b65e614ad6808d6f6ee34df",
        "f7310fdc82aebfd904b01e1dc54b2927094b2db68d6f903b68401adebf5a7e08",
        "d78ff4ef5d63653a65040cf9bfd4aca7984a74d37145986780fc0b16ac451649",
        "de6188a7dbdf191f64b5fc5e2ab47b57f7f7276cd419c17a3ca8e1b939ae49e4",
        "88acba6b965610b5480109c8b17b80e1b7b750dfc7598d5d5011fd2dcc5600a3",
        "2ef5b52a1ecc820e308aa342721aac0943bf6686b64b2579376504ccc493d97e",
        "6aed3fb0f9cd71a43dd497f01f17c0e2cb3797aa2a2f256656168e6c496afc5f",
        "b93246f6b1116398a346f1a641f3b041e989f7914f90cc2c7fff357876e506b5",
        "0d334ba77c225bc307ba537152f3f1610e4eafe595f6d9d90d11faa933a15ef1",
        "369546868a7f3a45a96768d40fd9d03412c091c6315cf4fde7cb68606937380d",
        "b2eaaa707b4c4185c32eddcdd306705e4dc1ffc872eeee475a64dfac86aba41c",
        "0618983f8741c5ef68d3a101e8a3b8cac60c905c15fc910840b94c00a0b9d0"
      ),
      signature: decode_hex(concat!(
        "0aab4c900501b3e24d7cdf4663326a3a",
        "87df5e4843b2cbdb67cbf6e460fec350",
        "aa5371b1508f9f4528ecea23c436d94b",
        "5e8fcd4f681e30a6ac00a9704a188a03"
      )),
    },
    Vector {
      secret_key: decode_hex(concat!(
        "833fe62409237b9d62ec77587520911e",
        "9a759cec1d19755b7da901b96dca3d42"
      )),
      public_key: decode_hex(concat!(
        "ec172b93ad5e563bf4932c70e1245034",
        "c35467ef2efd4d64ebf819683467e2bf"
      )),
      message: concat!(
        "ddaf35a193617abacc417349ae204131",
        "12e6fa4e89a97ea20a9eeee64b55d39a",
        "2192992a274fc1a836ba3c23a3feebbd",
        "454d4423643ce80e2a9ac94fa54ca49f"
      ),
      signature: decode_hex(concat!(
        "dc2a4459e7369633a52b1bf277839a00",
        "201009a3efbf3ecb69bea2186c26b589",
        "09351fc9ac90b3ecfdfbc7c66431e030",
        "3dca179c138ac17ad9bef1177331a704"
      )),
    },
  ];

  for vector in vectors {
    let secret = Ed25519SecretKey::from_bytes(vector.secret_key);
    let keypair = Ed25519Keypair::from_secret_key(secret);
    let public = Ed25519PublicKey::from_bytes(vector.public_key);
    let message = decode_hex_vec(vector.message);
    let signature = Ed25519Signature::from_bytes(vector.signature);

    assert_eq!(keypair.public_key(), public);
    assert_eq!(keypair.sign(&message), signature);
    assert!(public.verify(&message, &signature).is_ok());
    assert!(verify_ed25519(&message, &public, &signature).is_ok());
  }
}
