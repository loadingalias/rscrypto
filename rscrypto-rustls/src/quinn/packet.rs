use alloc::boxed::Box;
use bytes::BytesMut;
use quinn_proto::crypto::{CryptoError, HeaderKey, KeyPair, Keys, PacketKey};
use rustls::quic;

pub(super) fn keys(keys: quic::Keys) -> Keys {
  Keys {
    header: KeyPair {
      local: Box::new(Header(keys.local.header)),
      remote: Box::new(Header(keys.remote.header)),
    },
    packet: KeyPair {
      local: Box::new(Packet(keys.local.packet)),
      remote: Box::new(Packet(keys.remote.packet)),
    },
  }
}

pub(super) struct Packet(pub(super) Box<dyn quic::PacketKey>);

impl PacketKey for Packet {
  fn encrypt(&self, packet: u64, buffer: &mut [u8], header_len: usize) {
    // Quinn's producer reserves the tag and supplies an unprotected header.
    // Its infallible encryption trait requires these local construction invariants.
    let (header, body_tag) = buffer.split_at_mut(header_len);
    let length = body_tag
      .len()
      .checked_sub(self.tag_len())
      .expect("QUIC producer reserves tag space");
    let (body, tag) = body_tag.split_at_mut(length);
    let sealed = self
      .0
      .encrypt_in_place(packet, header, body)
      .expect("QUIC producer respects packet bounds");
    tag.copy_from_slice(sealed.as_ref());
  }

  fn decrypt(&self, packet: u64, header: &[u8], payload: &mut BytesMut) -> Result<(), CryptoError> {
    let length = self
      .0
      .decrypt_in_place(packet, header, payload)
      .map_err(|_| CryptoError)?
      .len();
    payload.truncate(length);
    Ok(())
  }

  fn tag_len(&self) -> usize {
    self.0.tag_len()
  }

  fn confidentiality_limit(&self) -> u64 {
    self.0.confidentiality_limit()
  }

  fn integrity_limit(&self) -> u64 {
    self.0.integrity_limit()
  }
}

pub(super) struct Header(pub(super) Box<dyn quic::HeaderProtectionKey>);

impl Header {
  fn apply(&self, offset: usize, packet: &mut [u8], decrypting: bool) {
    // RFC 9001 section 5.4.2 samples after the maximum four-byte packet
    // number, independent of the protected packet-number length bits.
    let Some(sample_start) = offset.checked_add(4) else {
      return;
    };
    let Some(sample_end) = sample_start.checked_add(self.sample_size()) else {
      return;
    };
    let Some(sample) = packet.get(sample_start..sample_end) else {
      return;
    };
    let Ok(sample) = <[u8; 16]>::try_from(sample) else {
      return;
    };
    if offset == 0 {
      return;
    }
    let (prefix, number) = packet.split_at_mut(offset);
    let number = &mut number[..4];
    if decrypting {
      self
        .0
        .decrypt_in_place(&sample, &mut prefix[0], number)
        .expect("validated QUIC header sample and packet-number buffer");
    } else {
      self
        .0
        .encrypt_in_place(&sample, &mut prefix[0], number)
        .expect("validated QUIC header sample and packet-number buffer");
    }
  }
}

impl HeaderKey for Header {
  fn decrypt(&self, pn_offset: usize, packet: &mut [u8]) {
    self.apply(pn_offset, packet, true);
  }

  fn encrypt(&self, pn_offset: usize, packet: &mut [u8]) {
    self.apply(pn_offset, packet, false);
  }

  fn sample_size(&self) -> usize {
    self.0.sample_len()
  }
}
