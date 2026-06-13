use std::{
  fs,
  path::{Path, PathBuf},
};

const README: &str = include_str!("../docs/migration/README.md");
const AWS_LC_RS: &str = include_str!("../docs/migration/aws-lc-rs.md");
const AWS_LC_SYS: &str = include_str!("../docs/migration/aws-lc-sys.md");
const DRYOC: &str = include_str!("../docs/migration/dryoc.md");
const OPENSSL: &str = include_str!("../docs/migration/openssl.md");
const RING: &str = include_str!("../docs/migration/ring.md");
const RUSTCRYPTO_P256: &str = include_str!("../docs/migration/RustCrypto/p256.md");
const RUSTCRYPTO_P384: &str = include_str!("../docs/migration/RustCrypto/p384.md");
const RUSTCRYPTO_RSA: &str = include_str!("../docs/migration/RustCrypto/rsa.md");

#[test]
fn migration_docs_do_not_delegate_accuracy_to_a_validation_index() {
  assert!(
    !README.contains("/tmp/rscrypto-migration-validate"),
    "migration docs must not claim validation from a temporary external harness"
  );
  assert!(
    !README.contains("VALIDATION.md"),
    "migration README must not delegate accuracy to a separate validation document"
  );
  assert!(
    !migration_root().join("VALIDATION.md").exists(),
    "migration validation index should not exist; keep accuracy in the migration guides"
  );

  let root = migration_root();
  let guide_count = migration_guides(&root).len();
  assert_eq!(
    guide_count, 35,
    "migration guide count drifted; update docs/migration/README.md"
  );
  assert!(
    README.contains(&format!("{guide_count} migration guides")),
    "migration README guide count does not match the files on disk"
  );

  for required in [
    "aws-lc-rs.md",
    "aws-lc-sys.md",
    "dryoc.md",
    "ring.md",
    "openssl.md",
    "RustCrypto/p256.md",
    "RustCrypto/p384.md",
    "RustCrypto/rsa.md",
  ] {
    assert!(
      root.join(required).exists(),
      "missing migration coverage file: {required}"
    );
  }
}

#[test]
fn migration_docs_do_not_reference_retired_rscrypto_versions() {
  for path in migration_markdown_files(&migration_root()) {
    let text = fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    assert!(
      !text.contains("rscrypto = { version = \"0.1\""),
      "{} still contains an rscrypto 0.1 dependency example",
      path.display()
    );
    assert!(
      !text.contains("`rscrypto` 0.1"),
      "{} still contains an rscrypto 0.1 validation claim",
      path.display()
    );
    assert!(
      !text.contains("After (`rscrypto` 0.1)"),
      "{} still contains an rscrypto 0.1 table heading",
      path.display()
    );
  }
}

#[test]
fn migration_docs_local_markdown_links_resolve() {
  for path in migration_markdown_files(&migration_root()) {
    let text = fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    for link in local_markdown_links(&text) {
      let target = link.split_once('#').map_or(link.as_str(), |(base, _)| base);
      if target.is_empty() {
        continue;
      }

      let resolved = path.parent().expect("migration doc has parent").join(target);
      assert!(
        resolved.exists(),
        "{} links to missing local markdown target `{}`",
        path.display(),
        link
      );
    }
  }
}

#[test]
fn aws_lc_safe_wrapper_guide_has_concrete_supported_paths_and_limits() {
  for required in [
    "## Digest",
    "## HMAC",
    "## HKDF",
    "## PBKDF2",
    "## AEAD",
    "## Ed25519",
    "## X25519",
    "## RSA Verification",
    "## Unsupported AWS-LC Surfaces",
    "aws_lc_rs::digest::digest",
    "HmacSha256::mac",
    "HkdfSha256::new",
    "Pbkdf2Sha256::derive_key",
    "cipher.encrypt",
    "Ed25519SecretKey::from_bytes",
    "X25519SecretKey::from_bytes",
    "RsaPublicKey::from_spki_der",
  ] {
    assert!(
      AWS_LC_RS.contains(required),
      "aws-lc-rs guide is missing concrete migration material: {required}"
    );
  }

  for unsupported in ["ECDSA", "ML-DSA", "TLS", "provider", "FIPS"] {
    assert!(
      AWS_LC_RS.contains(unsupported),
      "aws-lc-rs guide must name unsupported stack surface: {unsupported}"
    );
  }
}

#[test]
fn aws_lc_sys_guide_refuses_a_fake_direct_migration() {
  assert!(
    AWS_LC_SYS.contains("There is no direct `aws-lc-sys` migration"),
    "aws-lc-sys guide must not pretend raw FFI maps directly to rscrypto"
  );
  assert!(
    AWS_LC_SYS.contains("[`aws-lc-rs.md`](aws-lc-rs.md)"),
    "aws-lc-sys guide should route safe-wrapper users to aws-lc-rs.md"
  );
}

#[test]
fn stack_guides_have_concrete_paths_and_boundaries() {
  for required in [
    "## Digest",
    "## HMAC",
    "## HKDF",
    "## PBKDF2",
    "## AEAD",
    "## Ed25519",
    "## RSA Verification",
    "## X25519",
    "tests/migration_ring.rs",
    "ring::digest::digest",
    "HmacSha256::mac",
    "Pbkdf2Sha256::derive_key",
    "cipher.encrypt",
    "Ed25519SecretKey::from_bytes",
    "RsaPublicKey::from_spki_der",
  ] {
    assert!(
      RING.contains(required),
      "ring guide is missing required material: {required}"
    );
  }

  for required in [
    "## BLAKE2b Generic Hash",
    "## Ed25519",
    "## X25519",
    "## Argon2",
    "## Keep `dryoc`",
    "tests/migration_dryoc.rs",
    "crypto_generichash",
    "Blake2b256::digest",
    "crypto_sign_seed_keypair",
    "Ed25519SecretKey::from_bytes",
    "crypto_scalarmult",
    "X25519SecretKey::from_bytes",
  ] {
    assert!(
      DRYOC.contains(required),
      "dryoc guide is missing required material: {required}"
    );
  }

  for required in [
    "This repo does not depend on the Rust `openssl` crate",
    "## Practical Path",
    "## RSA Boundary",
    "Keep OpenSSL",
    "provider",
    "FIPS",
  ] {
    assert!(
      OPENSSL.contains(required),
      "openssl guide must keep the platform boundary explicit: {required}"
    );
  }

  for required in [
    "## Import Keys",
    "## Verify RSA-PSS",
    "## Verify RSASSA-PKCS1-v1_5",
    "## Sign",
    "## OAEP",
    "RsaPrivateKey::from_pkcs1_der",
    "verify_pss",
    "verify_pkcs1v15",
    "sign_pss",
    "encrypt_oaep",
    "decrypt_oaep",
  ] {
    assert!(
      RUSTCRYPTO_RSA.contains(required),
      "RustCrypto rsa guide is missing required material: {required}"
    );
  }
}

#[test]
fn rustcrypto_ecdsa_guides_have_concrete_paths_and_boundaries() {
  for (name, guide, curve, secret, public, signature, feature) in [
    (
      "p256",
      RUSTCRYPTO_P256,
      "P-256/SHA-256",
      "EcdsaP256SecretKey",
      "EcdsaP256PublicKey",
      "EcdsaP256Signature",
      "ecdsa-p256",
    ),
    (
      "p384",
      RUSTCRYPTO_P384,
      "P-384/SHA-384",
      "EcdsaP384SecretKey",
      "EcdsaP384PublicKey",
      "EcdsaP384Signature",
      "ecdsa-p384",
    ),
  ] {
    for required in [
      "## TL;DR",
      "## Type Map",
      "## Sign",
      "## Verify Raw Signatures",
      "## Verify DER Signatures",
      "## Notes",
      "tests/ecdsa_oracle.rs",
      curve,
      secret,
      public,
      signature,
      feature,
      "try_sign",
      "try_sign_blinded",
      "from_sec1_bytes",
      "from_spki_der",
      "from_der",
      "raw `r || s`",
      "ECDH is not part of this migration",
    ] {
      assert!(
        guide.contains(required),
        "RustCrypto {name} guide is missing required material: {required}"
      );
    }
  }
}

fn migration_root() -> PathBuf {
  Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/migration")
}

fn migration_guides(root: &Path) -> Vec<PathBuf> {
  migration_markdown_files(root)
    .into_iter()
    .filter(|path| {
      path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name != "README.md")
    })
    .collect()
}

fn migration_markdown_files(root: &Path) -> Vec<PathBuf> {
  let mut out = Vec::new();
  collect_markdown_files(root, &mut out);
  out.sort();
  out
}

fn collect_markdown_files(dir: &Path, out: &mut Vec<PathBuf>) {
  for entry in fs::read_dir(dir).unwrap_or_else(|err| panic!("read dir {}: {err}", dir.display())) {
    let entry = entry.unwrap_or_else(|err| panic!("read dir entry in {}: {err}", dir.display()));
    let path = entry.path();
    if path.is_dir() {
      collect_markdown_files(&path, out);
    } else if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
      out.push(path);
    }
  }
}

fn local_markdown_links(text: &str) -> Vec<String> {
  let mut links = Vec::new();
  let mut rest = text;

  while let Some(start) = rest.find("](") {
    rest = &rest[start + 2..];
    let Some(end) = rest.find(')') else {
      break;
    };

    let target = &rest[..end];
    rest = &rest[end + 1..];

    if target.starts_with("http://")
      || target.starts_with("https://")
      || target.starts_with("mailto:")
      || !target.split_once('#').map_or(target, |(base, _)| base).ends_with(".md")
    {
      continue;
    }

    links.push(target.to_string());
  }

  links
}
