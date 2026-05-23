// Optional BoringSSL RSA verification helper for differential testing.
//
// Build example from a source-built BoringSSL checkout:
//
//   c++ -std=c++17 -O2 \
//     -I/tmp/rscrypto-boringssl/include \
//     tests/support/boringssl_rsa_verify.cc \
//     /tmp/rscrypto-boringssl/build/libcrypto.a \
//     -o /tmp/rscrypto-boringssl-rsa-verify
//
// Then run the Rust oracle test with:
//
//   BORINGSSL_RSA_VERIFY_HELPER=/tmp/rscrypto-boringssl-rsa-verify \
//     cargo test --no-default-features --features rsa --test rsa_public_key boringssl

#include <stdint.h>
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <openssl/bytestring.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>

namespace {

bool ReadFile(const char *path, std::vector<uint8_t> *out) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "failed to open " << path << "\n";
    return false;
  }

  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  if (size < 0) {
    std::cerr << "failed to size " << path << "\n";
    return false;
  }
  file.seekg(0, std::ios::beg);

  out->resize(static_cast<size_t>(size));
  if (!out->empty() && !file.read(reinterpret_cast<char *>(out->data()), size)) {
    std::cerr << "failed to read " << path << "\n";
    return false;
  }
  return true;
}

const EVP_MD *DigestByName(const std::string &name) {
  if (name == "SHA256") {
    return EVP_sha256();
  }
  if (name == "SHA384") {
    return EVP_sha384();
  }
  if (name == "SHA512") {
    return EVP_sha512();
  }
  return nullptr;
}

bool ParseSaltLen(const std::string &value, int *salt_len) {
  if (value == "-") {
    return false;
  }
  size_t offset = 0;
  int parsed = 0;
  try {
    parsed = std::stoi(value, &offset);
  } catch (...) {
    return false;
  }
  if (offset != value.size() || parsed < 0) {
    return false;
  }
  *salt_len = parsed;
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: boringssl_rsa_verify SHA256|SHA384|SHA512 pkcs1v15|pss salt|- spki.der msg.bin sig.bin\n";
    return 2;
  }

  const std::string digest_name = argv[1];
  const std::string scheme = argv[2];
  const std::string salt_arg = argv[3];
  const EVP_MD *md = DigestByName(digest_name);
  if (md == nullptr) {
    std::cerr << "unsupported digest " << digest_name << "\n";
    return 2;
  }

  std::vector<uint8_t> spki;
  std::vector<uint8_t> message;
  std::vector<uint8_t> signature;
  if (!ReadFile(argv[4], &spki) || !ReadFile(argv[5], &message) || !ReadFile(argv[6], &signature)) {
    return 2;
  }

  CBS cbs;
  CBS_init(&cbs, spki.data(), spki.size());
  EVP_PKEY *raw_key = EVP_parse_public_key(&cbs);
  if (raw_key == nullptr || CBS_len(&cbs) != 0) {
    EVP_PKEY_free(raw_key);
    std::cerr << "failed to parse SPKI public key\n";
    return 2;
  }

  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  if (ctx == nullptr) {
    EVP_PKEY_free(raw_key);
    std::cerr << "failed to allocate EVP_MD_CTX\n";
    return 2;
  }

  EVP_PKEY_CTX *pctx = nullptr;
  if (EVP_DigestVerifyInit(ctx, &pctx, md, nullptr, raw_key) != 1 || pctx == nullptr) {
    EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(raw_key);
    std::cerr << "failed to initialize verification context\n";
    return 2;
  }

  if (scheme == "pkcs1v15") {
    if (EVP_PKEY_CTX_set_rsa_padding(pctx, RSA_PKCS1_PADDING) != 1) {
      EVP_MD_CTX_free(ctx);
      EVP_PKEY_free(raw_key);
      std::cerr << "failed to set PKCS1v1.5 padding\n";
      return 2;
    }
  } else if (scheme == "pss") {
    int salt_len = 0;
    if (!ParseSaltLen(salt_arg, &salt_len)) {
      EVP_MD_CTX_free(ctx);
      EVP_PKEY_free(raw_key);
      std::cerr << "invalid PSS salt length\n";
      return 2;
    }
    if (EVP_PKEY_CTX_set_rsa_padding(pctx, RSA_PKCS1_PSS_PADDING) != 1 ||
        EVP_PKEY_CTX_set_rsa_mgf1_md(pctx, md) != 1 ||
        EVP_PKEY_CTX_set_rsa_pss_saltlen(pctx, salt_len) != 1) {
      EVP_MD_CTX_free(ctx);
      EVP_PKEY_free(raw_key);
      std::cerr << "failed to set PSS parameters\n";
      return 2;
    }
  } else {
    EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(raw_key);
    std::cerr << "unsupported RSA scheme " << scheme << "\n";
    return 2;
  }

  if (EVP_DigestVerifyUpdate(ctx, message.data(), message.size()) != 1) {
    EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(raw_key);
    std::cerr << "failed to feed message\n";
    return 2;
  }

  const int rc = EVP_DigestVerifyFinal(ctx, signature.data(), signature.size());
  EVP_MD_CTX_free(ctx);
  EVP_PKEY_free(raw_key);

  if (rc == 1) {
    std::cout << "valid\n";
    return 0;
  }
  if (rc == 0) {
    std::cout << "invalid\n";
    return 0;
  }

  std::cerr << "verification failed internally\n";
  return 2;
}
