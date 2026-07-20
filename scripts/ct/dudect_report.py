#!/usr/bin/env python3
"""Build a machine-readable report from dudect-bencher output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from toml_compat import tomllib
from provenance import cfg_target_features, codegen_value, codegen_values, resolved_rustflags


CASE_METADATA = {
  "owner_eq_16_equal_vs_first_diff": {
    "primitive": "owner_equality.fixed",
    "left_class": "equal Aes128GcmKey owners",
    "right_class": "Aes128GcmKey owners with first byte different",
  },
  "owner_eq_32_equal_vs_first_diff": {
    "primitive": "owner_equality.fixed",
    "left_class": "equal X25519SecretKey owners",
    "right_class": "X25519SecretKey owners with first byte different",
  },
  "owner_eq_48_equal_vs_first_diff": {
    "primitive": "owner_equality.fixed",
    "left_class": "equal HmacSha384Tag owners",
    "right_class": "HmacSha384Tag owners with first byte different",
  },
  "owner_eq_64_equal_vs_first_diff": {
    "primitive": "owner_equality.fixed",
    "left_class": "equal HmacSha512Tag owners",
    "right_class": "HmacSha512Tag owners with first byte different",
  },
  "hmac_sha256_valid_vs_invalid_tag": {
    "primitive": "mac.hmac_verify",
    "left_class": "valid tag",
    "right_class": "invalid first tag byte",
  },
  "kmac256_valid_vs_invalid_tag": {
    "primitive": "mac.kmac256_verify",
    "left_class": "valid tag",
    "right_class": "invalid first tag byte",
  },
  "blake3_keyed_valid_vs_invalid_tag": {
    "primitive": "keyed_hash.blake3_verify",
    "left_class": "valid keyed digest",
    "right_class": "invalid first digest byte",
  },
  "secret_wrappers_debug_fixed_vs_random": {
    "primitive": "secret_wrappers.exposure_and_display",
    "left_class": "fixed secret bytes",
    "right_class": "random secret bytes",
  },
  "ietf_chacha20poly1305_fixed_vs_random_key_open": {
    "primitive": "aead.open_authentication",
    "left_class": "valid open with fixed secret key",
    "right_class": "valid open with random secret key",
  },
  "xchacha20poly1305_fixed_vs_random_key_open": {
    "primitive": "aead.open_authentication",
    "left_class": "valid open with fixed secret key",
    "right_class": "valid open with random secret key",
  },
  "aes128gcm_fixed_vs_random_key_open": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_highlevel: valid open with fixed secret key",
    "right_class": "s390x_aes_aead_highlevel: valid open with random secret key",
    "gate": "diagnostic",
    "reason": (
      "High-level AEAD open varies the public ciphertext/tag transcript with the key class; "
      "required AES-GCM evidence is carried by the secret-only CTR/GHASH/tag-AES probes."
    ),
  },
  "aes256gcm_fixed_vs_random_key_open": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_highlevel: valid open with fixed secret key",
    "right_class": "s390x_aes_aead_highlevel: valid open with random secret key",
    "gate": "diagnostic",
    "reason": (
      "High-level AEAD open varies the public ciphertext/tag transcript with the key class; "
      "required AES-GCM evidence is carried by the secret-only CTR/GHASH/tag-AES probes."
    ),
  },
  "aes128gcmsiv_fixed_vs_random_key_open": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_highlevel: valid open with fixed secret key",
    "right_class": "s390x_aes_aead_highlevel: valid open with random secret key",
    "gate": "diagnostic",
    "reason": (
      "High-level AEAD open varies the public ciphertext/tag transcript with the key class; "
      "required AES-GCM-SIV evidence is carried by the secret-only key-derivation/POLYVAL/tag-AES/CTR probes."
    ),
  },
  "aes256gcmsiv_fixed_vs_random_key_open": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_highlevel: valid open with fixed secret key",
    "right_class": "s390x_aes_aead_highlevel: valid open with random secret key",
    "gate": "diagnostic",
    "reason": (
      "High-level AEAD open varies the public ciphertext/tag transcript with the key class; "
      "required AES-GCM-SIV evidence is carried by the secret-only key-derivation/POLYVAL/tag-AES/CTR probes."
    ),
  },
  "aes128_gcm_siv_diag_derive_fixed_vs_random_key": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_gcmsiv_key_diag: derive AES-128-GCM-SIV message keys from fixed master key",
    "right_class": "s390x_gcmsiv_key_diag: derive AES-128-GCM-SIV message keys from random master key",
  },
  "aes256_gcm_siv_diag_derive_fixed_vs_random_key": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_gcmsiv_key_diag: derive AES-256-GCM-SIV message keys from fixed master key",
    "right_class": "s390x_gcmsiv_key_diag: derive AES-256-GCM-SIV message keys from random master key",
  },
  "gcm_siv_diag_polyval_fixed_vs_random_auth_key": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_gcmsiv_key_diag: POLYVAL with fixed authentication key",
    "right_class": "s390x_gcmsiv_key_diag: POLYVAL with random authentication key",
  },
  "aes128_gcm_siv_diag_raw_tag_aes_fixed_vs_random_key": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_gcmsiv_key_diag: AES-128 tag block under fixed raw encryption key",
    "right_class": "s390x_gcmsiv_key_diag: AES-128 tag block under random raw encryption key",
  },
  "aes256_gcm_siv_diag_raw_tag_aes_fixed_vs_random_key": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_gcmsiv_key_diag: AES-256 tag block under fixed raw encryption key",
    "right_class": "s390x_gcmsiv_key_diag: AES-256 tag block under random raw encryption key",
  },
  "aes128_gcm_diag_ctr32_be_fixed_vs_random_key": {
    "primitive": "aead.symmetric_transform",
    "left_class": "s390x_aes_aead_diag: AES-128-GCM CTR with fixed secret key",
    "right_class": "s390x_aes_aead_diag: AES-128-GCM CTR with random secret key",
  },
  "aes256_gcm_diag_ctr32_be_fixed_vs_random_key": {
    "primitive": "aead.symmetric_transform",
    "left_class": "s390x_aes_aead_diag: AES-256-GCM CTR with fixed secret key",
    "right_class": "s390x_aes_aead_diag: AES-256-GCM CTR with random secret key",
  },
  "aes128_gcm_diag_ghash_fixed_vs_random_h": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_diag: AES-128-GCM GHASH with fixed secret H",
    "right_class": "s390x_aes_aead_diag: AES-128-GCM GHASH with random secret H",
  },
  "aes256_gcm_diag_ghash_fixed_vs_random_h": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_diag: AES-256-GCM GHASH with fixed secret H",
    "right_class": "s390x_aes_aead_diag: AES-256-GCM GHASH with random secret H",
  },
  "aes128_gcm_diag_tag_aes_fixed_vs_random_key": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_diag: AES-128-GCM final tag AES with fixed secret key",
    "right_class": "s390x_aes_aead_diag: AES-128-GCM final tag AES with random secret key",
  },
  "aes256_gcm_diag_tag_aes_fixed_vs_random_key": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_diag: AES-256-GCM final tag AES with fixed secret key",
    "right_class": "s390x_aes_aead_diag: AES-256-GCM final tag AES with random secret key",
  },
  "aes128_gcm_siv_diag_ctr32_fixed_vs_random_key": {
    "primitive": "aead.symmetric_transform",
    "left_class": "s390x_aes_aead_diag: AES-128-GCM-SIV CTR with fixed raw encryption key",
    "right_class": "s390x_aes_aead_diag: AES-128-GCM-SIV CTR with random raw encryption key",
  },
  "aes256_gcm_siv_diag_ctr32_fixed_vs_random_key": {
    "primitive": "aead.symmetric_transform",
    "left_class": "s390x_aes_aead_diag: AES-256-GCM-SIV CTR with fixed raw encryption key",
    "right_class": "s390x_aes_aead_diag: AES-256-GCM-SIV CTR with random raw encryption key",
  },
  "aes128_gcm_siv_diag_raw_tag_aes_varying_block": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_diag: AES-128-GCM-SIV raw tag AES with fixed block",
    "right_class": "s390x_aes_aead_diag: AES-128-GCM-SIV raw tag AES with random block",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM-SIV timing trace for raw tag AES under block variation.",
  },
  "aes256_gcm_siv_diag_raw_tag_aes_varying_block": {
    "primitive": "aead.open_authentication",
    "left_class": "s390x_aes_aead_diag: AES-256-GCM-SIV raw tag AES with fixed block",
    "right_class": "s390x_aes_aead_diag: AES-256-GCM-SIV raw tag AES with random block",
    "gate": "diagnostic",
    "reason": "s390x AES-GCM-SIV timing trace for raw tag AES under block variation.",
  },
  "x25519_fixed_vs_random_scalar": {
    "primitive": "kx.x25519",
    "left_class": "fixed scalar",
    "right_class": "random scalar",
  },
  "ed25519_sign_fixed_vs_random_secret": {
    "primitive": "signature.ed25519_sign",
    "left_class": "fixed signing secret",
    "right_class": "random signing secret",
  },
  "ed25519_public_key_fixed_vs_random_secret": {
    "primitive": "signature.ed25519_sign",
    "left_class": "fixed signing secret public-key derivation",
    "right_class": "random signing secret public-key derivation",
  },
  "ed25519_keypair_sign_fixed_vs_random_secret": {
    "primitive": "signature.ed25519_sign",
    "left_class": "fixed keypair signing secret",
    "right_class": "random keypair signing secret",
  },
  "ed25519_sha512_secret_expand_fixed_vs_random_secret": {
    "primitive": "signature.ed25519_sign",
    "left_class": "fixed signing secret SHA-512 expansion",
    "right_class": "random signing secret SHA-512 expansion",
  },
  "ecdsa_p256_sign_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p256_sign",
    "left_class": "P-256 signing with fixed secret and projective and scalar blinding",
    "right_class": "P-256 signing with random secret and projective and scalar blinding",
  },
  "ecdsa_p384_sign_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p384_sign",
    "left_class": "P-384 signing with fixed secret and projective and scalar blinding",
    "right_class": "P-384 signing with random secret and projective and scalar blinding",
  },
  "ecdsa_p256_diag_nonce_reduce_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p256_sign",
    "left_class": "P-256 diagnostic nonce derivation from fixed secret",
    "right_class": "P-256 diagnostic nonce derivation from random secret",
    "gate": "diagnostic",
    "reason": "ECDSA signing CT root-cause isolation.",
  },
  "ecdsa_p256_diag_reduce_wide_fixed_vs_random_input": {
    "primitive": "signature.ecdsa_p256_sign",
    "left_class": "P-256 diagnostic order reduction of fixed wide input",
    "right_class": "P-256 diagnostic order reduction of random wide input",
    "gate": "diagnostic",
    "reason": "ECDSA signing CT root-cause isolation.",
  },
  "ecdsa_p256_diag_basepoint_blinded_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p256_sign",
    "left_class": "P-256 diagnostic blinded basepoint multiplication from fixed secret",
    "right_class": "P-256 diagnostic blinded basepoint multiplication from random secret",
    "gate": "diagnostic",
    "reason": "ECDSA signing CT root-cause isolation.",
  },
  "ecdsa_p256_diag_scalar_finish_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p256_sign",
    "left_class": "P-256 diagnostic scalar finishing with fixed secret",
    "right_class": "P-256 diagnostic scalar finishing with random secret",
    "gate": "diagnostic",
    "reason": "ECDSA signing CT root-cause isolation.",
  },
  "ecdsa_p256_diag_order_mul_fixed_r_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p256_sign",
    "left_class": "P-256 diagnostic fixed-r order multiply with fixed secret",
    "right_class": "P-256 diagnostic fixed-r order multiply with random secret",
    "gate": "diagnostic",
    "reason": "ECDSA signing CT root-cause isolation.",
  },
  "ecdsa_p256_diag_blinded_order_mul_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p256_sign",
    "left_class": "P-256 diagnostic blinded fixed-r order multiply with fixed secret",
    "right_class": "P-256 diagnostic blinded fixed-r order multiply with random secret",
    "gate": "diagnostic",
    "reason": "Mitigation isolation for s390x fixed-r private-scalar multiplication.",
  },
  "ecdsa_p256_diag_nonce_inverse_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p256_sign",
    "left_class": "P-256 diagnostic nonce inverse from fixed secret",
    "right_class": "P-256 diagnostic nonce inverse from random secret",
  },
  "ecdsa_p256_diag_final_multiply_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p256_sign",
    "left_class": "P-256 diagnostic final scalar multiply with fixed secret",
    "right_class": "P-256 diagnostic final scalar multiply with random secret",
    "gate": "diagnostic",
    "reason": "ECDSA signing CT root-cause isolation.",
  },
  "ecdsa_p384_diag_nonce_reduce_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p384_sign",
    "left_class": "P-384 diagnostic nonce derivation from fixed secret",
    "right_class": "P-384 diagnostic nonce derivation from random secret",
    "gate": "diagnostic",
    "reason": "ECDSA signing CT root-cause isolation.",
  },
  "ecdsa_p384_diag_reduce_wide_fixed_vs_random_input": {
    "primitive": "signature.ecdsa_p384_sign",
    "left_class": "P-384 diagnostic order reduction of fixed wide input",
    "right_class": "P-384 diagnostic order reduction of random wide input",
    "gate": "diagnostic",
    "reason": "ECDSA signing CT root-cause isolation.",
  },
  "ecdsa_p384_diag_basepoint_blinded_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p384_sign",
    "left_class": "P-384 diagnostic blinded basepoint multiplication from fixed secret",
    "right_class": "P-384 diagnostic blinded basepoint multiplication from random secret",
    "gate": "diagnostic",
    "reason": "ECDSA signing CT root-cause isolation.",
  },
  "ecdsa_p384_diag_scalar_finish_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p384_sign",
    "left_class": "P-384 diagnostic scalar finishing with fixed secret",
    "right_class": "P-384 diagnostic scalar finishing with random secret",
    "gate": "diagnostic",
    "reason": "ECDSA signing CT root-cause isolation.",
  },
  "ecdsa_p384_diag_order_mul_fixed_r_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p384_sign",
    "left_class": "P-384 diagnostic fixed-r order multiply with fixed secret",
    "right_class": "P-384 diagnostic fixed-r order multiply with random secret",
    "gate": "diagnostic",
    "reason": "ECDSA signing CT root-cause isolation.",
  },
  "ecdsa_p384_diag_nonce_inverse_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p384_sign",
    "left_class": "P-384 diagnostic nonce inverse from fixed secret",
    "right_class": "P-384 diagnostic nonce inverse from random secret",
  },
  "ecdsa_p384_diag_final_multiply_fixed_vs_random_secret": {
    "primitive": "signature.ecdsa_p384_sign",
    "left_class": "P-384 diagnostic final scalar multiply with fixed secret",
    "right_class": "P-384 diagnostic final scalar multiply with random secret",
    "gate": "diagnostic",
    "reason": "ECDSA signing CT root-cause isolation.",
  },
  "rsa_pkcs1v15_fixed_vs_random_message": {
    "primitive": "rsa.private_ops",
    "left_class": "fixed message",
    "right_class": "random same-length message",
  },
  "rsa_pss_fixed_vs_random_message": {
    "primitive": "rsa.private_ops",
    "left_class": "fixed message",
    "right_class": "random same-length message",
  },
  "rsa_oaep_decrypt_fixed_vs_random_plaintext": {
    "primitive": "rsa.private_ops",
    "left_class": "valid OAEP ciphertext for fixed plaintext",
    "right_class": "valid OAEP ciphertext for random same-length plaintext",
  },
  "rsa_pkcs1v15_decrypt_fixed_vs_random_plaintext": {
    "primitive": "rsa.private_ops",
    "left_class": "valid PKCS#1 v1.5 ciphertext for fixed plaintext",
    "right_class": "valid PKCS#1 v1.5 ciphertext for random same-length plaintext",
  },
  "rsa_private_component_validation_fixed_vs_random_component": {
    "primitive": "rsa.private_key_material",
    "left_class": "fixed same-width private component",
    "right_class": "random same-width private component",
  },
  "rsa_private_key_pkcs8_import_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "valid PKCS#8 private key A",
    "right_class": "valid same-size same-shape PKCS#8 private key B",
    "gate": "diagnostic",
    "reason": "Classes differ in RSA public modulus; public key material may leak by CT policy.",
  },
  "rsa_private_key_pkcs8_validate_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "valid PKCS#8 private key A validation",
    "right_class": "valid same-size same-shape PKCS#8 private key B validation",
    "gate": "diagnostic",
    "reason": "Full RSA private-key import validation performs value-dependent big-integer consistency checks over unrelated keys; required CT evidence is scoped to bounded validation leaves and steady-state private operations.",
  },
  "rsa_private_key_pkcs8_import_stage50_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "validated PKCS#8 private key A before materialization",
    "right_class": "validated same-size same-shape PKCS#8 private key B before materialization",
  },
  "rsa_private_key_pkcs8_import_stage51_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "PKCS#8 private key A after public modulus setup",
    "right_class": "same-size same-shape PKCS#8 private key B after public modulus setup",
    "gate": "diagnostic",
    "reason": "Includes RSA public modulus setup; public key material may leak by CT policy.",
  },
  "rsa_private_key_pkcs8_import_stage52_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "PKCS#8 private key A after p modulus setup",
    "right_class": "same-size same-shape PKCS#8 private key B after p modulus setup",
    "gate": "diagnostic",
    "reason": "Includes prior RSA public modulus setup; public key material may leak by CT policy.",
  },
  "rsa_private_key_pkcs8_import_stage53_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "PKCS#8 private key A after q modulus setup",
    "right_class": "same-size same-shape PKCS#8 private key B after q modulus setup",
    "gate": "diagnostic",
    "reason": "Includes prior RSA public modulus setup; public key material may leak by CT policy.",
  },
  "rsa_private_key_pkcs8_import_stage54_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "PKCS#8 private key A after secret integer boxing",
    "right_class": "same-size same-shape PKCS#8 private key B after secret integer boxing",
    "gate": "diagnostic",
    "reason": "Includes prior RSA public modulus setup; public key material may leak by CT policy.",
  },
  "rsa_private_key_pkcs8_export_key_a_vs_key_b": {
    "primitive": "rsa.private_key_material",
    "left_class": "validated private key A",
    "right_class": "validated same-size same-shape private key B",
    "gate": "diagnostic",
    "reason": "Canonical DER private-key export is a secret-output formatting operation with value-dependent INTEGER shape.",
  },
  "hkdf_sha2_fixed_vs_random_ikm": {
    "primitive": "kdf.hkdf",
    "left_class": "fixed input key material",
    "right_class": "random input key material",
  },
  "pbkdf2_sha2_fixed_vs_random_password": {
    "primitive": "kdf.pbkdf2",
    "left_class": "fixed password",
    "right_class": "random same-length password",
  },
  "argon2i_fixed_vs_random_password": {
    "primitive": "password.argon2i",
    "left_class": "fixed password",
    "right_class": "random same-length password",
  },
  "aes128gcm_fixed_vs_random_key_seal": {
    "primitive": "aead.symmetric_transform",
    "left_class": "s390x_aes_aead_highlevel: fixed secret key",
    "right_class": "s390x_aes_aead_highlevel: random secret key",
    "gate": "diagnostic",
    "reason": (
      "High-level AEAD seal varies public ciphertext/tag output with the key class; "
      "required AES-GCM evidence is carried by the secret-only CTR/GHASH/tag-AES probes."
    ),
  },
  "aes256gcm_fixed_vs_random_key_seal": {
    "primitive": "aead.symmetric_transform",
    "left_class": "s390x_aes_aead_highlevel: fixed secret key",
    "right_class": "s390x_aes_aead_highlevel: random secret key",
    "gate": "diagnostic",
    "reason": (
      "High-level AEAD seal varies public ciphertext/tag output with the key class; "
      "required AES-GCM evidence is carried by the secret-only CTR/GHASH/tag-AES probes."
    ),
  },
  "aes128gcmsiv_fixed_vs_random_key_seal": {
    "primitive": "aead.symmetric_transform",
    "left_class": "s390x_aes_aead_highlevel: fixed secret key",
    "right_class": "s390x_aes_aead_highlevel: random secret key",
    "gate": "diagnostic",
    "reason": (
      "High-level AEAD seal varies public ciphertext/tag output with the key class; "
      "required AES-GCM-SIV evidence is carried by the secret-only key-derivation/POLYVAL/tag-AES/CTR probes."
    ),
  },
  "aes256gcmsiv_fixed_vs_random_key_seal": {
    "primitive": "aead.symmetric_transform",
    "left_class": "s390x_aes_aead_highlevel: fixed secret key",
    "right_class": "s390x_aes_aead_highlevel: random secret key",
    "gate": "diagnostic",
    "reason": (
      "High-level AEAD seal varies public ciphertext/tag output with the key class; "
      "required AES-GCM-SIV evidence is carried by the secret-only key-derivation/POLYVAL/tag-AES/CTR probes."
    ),
  },
  "ietf_chacha20poly1305_fixed_vs_random_key_seal": {
    "primitive": "aead.symmetric_transform",
    "left_class": "fixed secret key",
    "right_class": "random secret key",
  },
  "blake2b256_keyed_fixed_vs_random_key": {
    "primitive": "keyed_hash.blake2_blake3",
    "left_class": "fixed BLAKE2b-256 keyed-hash key",
    "right_class": "random BLAKE2b-256 keyed-hash key",
  },
  "blake2b512_keyed_fixed_vs_random_key": {
    "primitive": "keyed_hash.blake2_blake3",
    "left_class": "fixed BLAKE2b-512 keyed-hash key",
    "right_class": "random BLAKE2b-512 keyed-hash key",
  },
  "blake2s128_keyed_fixed_vs_random_key": {
    "primitive": "keyed_hash.blake2_blake3",
    "left_class": "fixed BLAKE2s-128 keyed-hash key",
    "right_class": "random BLAKE2s-128 keyed-hash key",
  },
  "blake2s256_keyed_fixed_vs_random_key": {
    "primitive": "keyed_hash.blake2_blake3",
    "left_class": "fixed BLAKE2s-256 keyed-hash key",
    "right_class": "random BLAKE2s-256 keyed-hash key",
  },
  "blake3_keyed_fixed_vs_random_key": {
    "primitive": "keyed_hash.blake2_blake3",
    "left_class": "fixed BLAKE3 keyed-hash key",
    "right_class": "random BLAKE3 keyed-hash key",
  },
}

SEED_RE = re.compile(r"^bench\s+(?P<name>\S+)\s+seeded with (?P<seed>0x[0-9a-fA-F]+)$")
RESULT_RE = re.compile(
  r"^bench (?P<name>\S+)\s+\.\.\. : n == (?P<n>[+-]?[0-9.]+)M, "
  r"max t = (?P<t>[+-]?[0-9.]+), max tau = (?P<tau>[+-]?[0-9.]+), "
  r"\(5/tau\)\^2 = (?P<needed>[0-9]+)$"
)


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def rustc_verbose() -> str:
  try:
    return subprocess.check_output(["rustc", "-vV"], text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    return "unavailable"


def parse_stdout(path: Path) -> tuple[dict[str, str], dict[str, dict]]:
  seeds: dict[str, str] = {}
  results: dict[str, dict] = {}
  for line in path.read_text(errors="replace").splitlines():
    if seed_match := SEED_RE.match(line):
      seeds[seed_match.group("name")] = seed_match.group("seed")
      continue
    if result_match := RESULT_RE.match(line):
      name = result_match.group("name")
      max_t = float(result_match.group("t"))
      max_tau = float(result_match.group("tau"))
      results[name] = {
        "samples_millions": float(result_match.group("n")),
        "max_t": max_t,
        "abs_max_t": abs(max_t),
        "max_tau": max_tau,
        "needed_samples_for_tau_threshold": int(result_match.group("needed")),
      }
  return seeds, results


def raw_csv_rows(path: Path) -> dict[str, dict]:
  rows: dict[str, dict] = {}
  if not path.exists():
    return rows

  with path.open(newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
      name = row.get("benchname", "")
      class_name = row.get("class", "")
      if not name or not class_name:
        continue
      entry = rows.setdefault(name, {"row_count": 0, "labels": {}})
      entry["row_count"] += 1
      entry["labels"].setdefault(class_name, 0)
      entry["labels"][class_name] += 1
  return rows


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stdout", required=True, type=Path)
  parser.add_argument("--csv", required=True, type=Path)
  parser.add_argument("--out", required=True, type=Path)
  parser.add_argument("--target", required=True)
  parser.add_argument("--profile", default="release")
  parser.add_argument("--threshold", type=float, default=10.0)
  parser.add_argument("--samples", type=int, required=True)
  parser.add_argument("--command", default="")
  parser.add_argument("--binary", required=True, type=Path)
  parser.add_argument("--binary-disassembly", required=True, type=Path)
  parser.add_argument("--binary-symbols", required=True, type=Path)
  parser.add_argument("--linker-command-log", required=True, type=Path)
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[2]
  with (root / "Cargo.toml").open("rb") as source:
    crate_version = tomllib.load(source)["package"]["version"]
  with (root / "ct.toml").open("rb") as source:
    release_binary = tomllib.load(source)["equality_evidence"]["release_binary"]
  dudect_manifest_path = root / "tools" / "ct-dudect" / "Cargo.toml"
  harness_manifest_path = root / "tools" / "ct-harness" / "Cargo.toml"
  dudect_lockfile_path = root / "tools" / "ct-dudect" / "Cargo.lock"
  with dudect_manifest_path.open("rb") as source:
    dudect_manifest = tomllib.load(source)
  configured_rustflags, environment_rustflags, effective_rustflags, rustflags_source = resolved_rustflags(
    root, args.target
  )
  target_cfg = subprocess.check_output(
    ["rustc", "--print", "cfg", "--target", args.target, *effective_rustflags],
    cwd=root,
    text=True,
  )
  expected_owner_symbols = {f"ct_entry_owner_eq_{width}" for width in release_binary["formal_owner_widths"]}
  for path in (args.binary, args.binary_disassembly, args.binary_symbols, args.linker_command_log):
    if not path.is_file():
      raise ValueError(f"DudeCT evidence artifact missing: {path}")

  symbol_counts = {symbol: 0 for symbol in expected_owner_symbols}
  for line in args.binary_symbols.read_text().splitlines():
    if match := re.search(r"\b_?(ct_entry_owner_eq_[0-9]+)\b", line):
      if match.group(1) in symbol_counts:
        symbol_counts[match.group(1)] += 1
  wrong_symbol_counts = {symbol: count for symbol, count in symbol_counts.items() if count != 1}
  if wrong_symbol_counts:
    raise ValueError(f"DudeCT binary owner equality symbols must occur exactly once: {wrong_symbol_counts}")
  owner_call_sites = {symbol: 0 for symbol in expected_owner_symbols}
  current_symbol = ""
  function_label = re.compile(r"^[0-9a-fA-F]+ <(.+)>:$")
  for line in args.binary_disassembly.read_text(errors="replace").splitlines():
    if match := function_label.match(line.strip()):
      current_symbol = match.group(1).removeprefix("_")
      continue
    instruction = re.search(r"\b(?:bl|brasl|call|callq|jal)\b", line)
    if instruction is None:
      continue
    for symbol in expected_owner_symbols:
      if current_symbol != symbol and re.search(rf"<_?{re.escape(symbol)}(?:\+[^>]*)?>", line):
        owner_call_sites[symbol] += 1
  missing_call_sites = {symbol: count for symbol, count in owner_call_sites.items() if count < 1}
  if missing_call_sites:
    raise ValueError(f"DudeCT binary does not call every owner equality symbol: {missing_call_sites}")

  linker_command = next(
    (line for line in args.linker_command_log.read_text().splitlines() if '"-o"' in line),
    "",
  )
  linker_tokens = shlex.split(linker_command)
  first_object = next((index for index, token in enumerate(linker_tokens) if token.endswith((".o", ".obj"))), None)
  if first_object is None or first_object == 0:
    raise ValueError("DudeCT linker command does not identify the linker driver")
  linker = linker_tokens[first_object - 1]
  linker_path_text = shutil.which(linker)
  if linker_path_text is None:
    raise ValueError(f"DudeCT linker driver is not resolvable: {linker}")
  linker_path = Path(linker_path_text).resolve()
  linker_version = subprocess.check_output([str(linker_path), "--version"], cwd=root, text=True, stderr=subprocess.STDOUT).strip()
  git_status = subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=root, text=True).splitlines()

  seeds, results = parse_stdout(args.stdout)
  raw_rows = raw_csv_rows(args.csv)
  cases = []
  for name in sorted(results):
    result = results[name]
    metadata = CASE_METADATA.get(name, {})
    gate = metadata.get("gate", "required")
    passed = result["abs_max_t"] <= args.threshold
    diagnostic = gate == "diagnostic"
    cases.append(
      {
        "name": name,
        "primitive": metadata.get("primitive", "unknown"),
        "left_class": metadata.get("left_class", "unknown"),
        "right_class": metadata.get("right_class", "unknown"),
        "gate": gate,
        "diagnostic_reason": metadata.get("reason"),
        "seed": seeds.get(name),
        "requested_samples": args.samples,
        "raw_csv": raw_rows.get(name, {"row_count": 0, "labels": {}}),
        **result,
        "threshold_abs_max_t": args.threshold,
        "status": "pass" if passed else ("diagnostic-fail" if diagnostic else "fail"),
      }
    )

  report = {
    "schema_version": 2,
    "kind": "rscrypto.ct.dudect",
    "crate": "rscrypto",
    "crate_version": crate_version,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "git_dirty": bool(git_status),
    "git_status": git_status,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "target": args.target,
    "target_triple": args.target,
    "profile": args.profile,
    "profile_settings": dudect_manifest.get("profile", {}).get(args.profile, {}),
    "features": release_binary["features"],
    "default_features": release_binary["default_features"],
    "backend": release_binary["backend"],
    "dudect_manifest_sha256": sha256_file(dudect_manifest_path),
    "harness_manifest_sha256": sha256_file(harness_manifest_path),
    "dudect_lockfile_sha256": sha256_file(dudect_lockfile_path),
    "cargo": subprocess.check_output(["cargo", "-V"], cwd=root, text=True).strip(),
    "configured_rustflags": configured_rustflags,
    "environment_rustflags": environment_rustflags,
    "effective_rustflags": effective_rustflags,
    "rustflags_source": rustflags_source,
    "target_cpu": codegen_value(effective_rustflags, "target-cpu"),
    "target_features": codegen_values(effective_rustflags, "target-feature"),
    "target_cfg_features": cfg_target_features(target_cfg),
    "linker": linker,
    "linker_path": str(linker_path),
    "linker_sha256": sha256_file(linker_path),
    "linker_version": linker_version,
    "binary": {
      "path": str(args.binary),
      "sha256": sha256_file(args.binary),
      "bytes": args.binary.stat().st_size,
      "owner_symbols": sorted(expected_owner_symbols),
      "owner_call_sites": owner_call_sites,
    },
    "binary_disassembly": {
      "path": str(args.binary_disassembly),
      "sha256": sha256_file(args.binary_disassembly),
      "bytes": args.binary_disassembly.stat().st_size,
    },
    "binary_symbols": {
      "path": str(args.binary_symbols),
      "sha256": sha256_file(args.binary_symbols),
      "bytes": args.binary_symbols.stat().st_size,
    },
    "linker_command_log": {
      "path": str(args.linker_command_log),
      "sha256": sha256_file(args.linker_command_log),
      "bytes": args.linker_command_log.stat().st_size,
    },
    "threshold_abs_max_t": args.threshold,
    "requested_samples": args.samples,
    "command": args.command,
    "rustc_verbose": rustc_verbose(),
    "host": {
      "system": platform.system(),
      "release": platform.release(),
      "machine": platform.machine(),
      "processor": platform.processor(),
      "python": platform.python_version(),
    },
    "raw_stdout": str(args.stdout),
    "raw_stdout_sha256": sha256_file(args.stdout) if args.stdout.exists() else None,
    "raw_csv": str(args.csv),
    "raw_csv_sha256": sha256_file(args.csv) if args.csv.exists() else None,
    "cases": cases,
    "case_count": len(cases),
    "failure_count": sum(1 for case in cases if case["status"] == "fail"),
    "diagnostic_failure_count": sum(1 for case in cases if case["status"] == "diagnostic-fail"),
    "notes": [
      "DudeCT is empirical timing evidence, not a proof.",
      "A pass means no leakage was detected for this configuration, host, and input classification.",
      "dudect-bencher 0.7.0 writes both raw CSV classes with label 0; raw CSV labels are recorded for traceability, not class-balance proof.",
    ],
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"dudect report: {args.out}")
  return 1 if report["failure_count"] else 0


if __name__ == "__main__":
  raise SystemExit(main())
