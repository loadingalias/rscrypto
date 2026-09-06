use p256::elliptic_curve::sec1::ToSec1Point as _;
use rscrypto::{P256EphemeralSecret, P256PublicKey};

fn scalar(mut bytes: [u8; 32]) -> [u8; 32] {
  bytes[0] &= 0x7f;
  bytes[31] |= 1;
  bytes
}

fn secret(bytes: [u8; 32]) -> P256EphemeralSecret {
  P256EphemeralSecret::try_generate_with(|candidate| {
    candidate.copy_from_slice(&bytes);
    Ok::<(), core::convert::Infallible>(())
  })
  .expect("normalized fuzz scalar must be canonical and nonzero")
}

pub(super) fn run(data: &[u8]) {
  let Some(material) = data.get(..64) else {
    return;
  };
  let left = scalar(material[..32].try_into().expect("fixed scalar slice"));
  let right = scalar(material[32..].try_into().expect("fixed scalar slice"));

  let oracle_left = p256::SecretKey::from_slice(&left).expect("normalized oracle scalar");
  let oracle_right = p256::SecretKey::from_slice(&right).expect("normalized oracle scalar");
  let ours_public = secret(left).public_key();
  assert_eq!(
    ours_public.as_sec1_bytes().as_slice(),
    oracle_left.public_key().to_sec1_point(false).as_bytes(),
    "P-256 public derivation differential mismatch"
  );

  let peer = secret(right).public_key();
  let ours_shared = secret(left).diffie_hellman(&peer);
  let oracle_shared =
    p256::ecdh::diffie_hellman(oracle_left.to_nonzero_scalar(), oracle_right.public_key().as_affine());
  assert_eq!(
    ours_shared.as_bytes().as_slice(),
    oracle_shared.raw_secret_bytes().as_slice(),
    "P-256 agreement differential mismatch"
  );

  let candidate = &data[64..];
  let oracle_candidate = p256::PublicKey::from_sec1_bytes(candidate);
  let canonical = candidate.len() == 65 && candidate.first() == Some(&0x04) && oracle_candidate.is_ok();
  let ours_candidate = P256PublicKey::from_sec1_bytes(candidate);
  assert_eq!(
    ours_candidate.is_ok(),
    canonical,
    "P-256 canonical SEC1 parser differential mismatch"
  );

  if let Ok(peer) = ours_candidate {
    let ours_shared = secret(left).diffie_hellman(&peer);
    let oracle_shared = p256::ecdh::diffie_hellman(
      oracle_left.to_nonzero_scalar(),
      oracle_candidate.expect("canonical oracle point").as_affine(),
    );
    assert_eq!(
      ours_shared.as_bytes().as_slice(),
      oracle_shared.raw_secret_bytes().as_slice(),
      "P-256 parsed-point agreement differential mismatch"
    );
  }
}
