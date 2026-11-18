// QUIC-LB oracle and timing harness against google/quiche.
//
// Built inside a quiche checkout. See ../README.md. `vectors` prints one line per connection ID:
// "rotation key server_id nonce connection_id", all hexadecimal except the rotation. `bench`
// times the issue and decode of the four draft-ietf-quic-load-balancers-22 Appendix B.2 shapes
// with Criterion's linear sampling, so the result compares with rscrypto's `quic/cid-*` rows.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "quiche/quic/core/quic_connection_id.h"
#include "quiche/quic/load_balancer/load_balancer_config.h"
#include "quiche/quic/load_balancer/load_balancer_decoder.h"
#include "quiche/quic/load_balancer/load_balancer_server_id.h"

namespace {

using quic::LoadBalancerConfig;
using quic::LoadBalancerDecoder;
using quic::LoadBalancerServerId;
using quic::QuicConnectionId;

constexpr uint8_t kDraftKey[16] = {0x8f, 0x95, 0xf0, 0x92, 0x45, 0x76, 0x5f, 0x80,
                                   0x25, 0x69, 0x34, 0xe5, 0x0c, 0x66, 0x20, 0x7f};

// splitmix64, fixed seed: the vector file is reproducible from this source alone.
uint64_t g_state = 0x7175696368652d6cULL;
uint8_t NextByte() {
  uint64_t z = (g_state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return static_cast<uint8_t>((z ^ (z >> 31)) >> 56);
}

void PrintHex(const uint8_t* bytes, size_t len) {
  for (size_t i = 0; i < len; ++i) std::printf("%02x", bytes[i]);
}

[[noreturn]] void Fail(const char* what) {
  std::fprintf(stderr, "quic_lb_vectors: %s\n", what);
  std::exit(1);
}

LoadBalancerConfig Config(uint8_t rotation, uint8_t server_id_len, uint8_t nonce_len, const uint8_t* key) {
  std::optional<LoadBalancerConfig> config = LoadBalancerConfig::Create(
      rotation, server_id_len, nonce_len, absl::string_view(reinterpret_cast<const char*>(key), 16));
  if (!config.has_value()) Fail("LoadBalancerConfig::Create refused the lengths");
  return *config;
}

// The issue path of quiche's LoadBalancerEncoder::GenerateConnectionId, with the server ID and
// nonce chosen by the caller and a length-describing first octet.
QuicConnectionId Issue(const LoadBalancerConfig& config, const uint8_t* server_id, const uint8_t* nonce) {
  uint8_t buffer[quic::kQuicMaxConnectionIdWithLengthPrefixLength] = {};
  const uint8_t len = config.plaintext_len();
  buffer[0] = static_cast<uint8_t>(config.config_id() << quic::kConnectionIdLengthBits) | len;
  std::memcpy(&buffer[1], server_id, config.server_id_len());
  std::memcpy(&buffer[1 + config.server_id_len()], nonce, config.nonce_len());
  if (len == quic::kLoadBalancerBlockSize) {
    if (!config.BlockEncrypt(&buffer[1], &buffer[1])) Fail("BlockEncrypt");
    return QuicConnectionId(reinterpret_cast<char*>(buffer), config.total_len());
  }
  return config.FourPassEncrypt(absl::Span<uint8_t>(buffer, config.total_len()));
}

int Vectors() {
  uint8_t keys[3][16];
  std::memcpy(keys[0], kDraftKey, 16);
  for (int k = 1; k < 3; ++k)
    for (uint8_t& byte : keys[k]) byte = NextByte();
  unsigned line = 0;
  for (uint8_t server_id_len = 1; server_id_len <= 15; ++server_id_len) {
    // quiche refuses nonces longer than kLoadBalancerMaxNonceLen (16 octets).
    const int max_nonce = std::min(19 - server_id_len, static_cast<int>(quic::kLoadBalancerMaxNonceLen));
    for (uint8_t nonce_len = 4; nonce_len <= max_nonce; ++nonce_len) {
      for (const uint8_t* key : keys) {
        const uint8_t rotation = static_cast<uint8_t>(line++ % 7);
        uint8_t server_id[15];
        uint8_t nonce[16];
        for (int i = 0; i < server_id_len; ++i) server_id[i] = NextByte();
        for (int i = 0; i < nonce_len; ++i) nonce[i] = NextByte();
        LoadBalancerConfig config = Config(rotation, server_id_len, nonce_len, key);
        QuicConnectionId id = Issue(config, server_id, nonce);
        if (id.length() != config.total_len()) Fail("issued length");

        LoadBalancerDecoder decoder;
        if (!decoder.AddConfig(config)) Fail("AddConfig");
        LoadBalancerServerId decoded;
        if (!decoder.GetServerId(id, decoded)) Fail("GetServerId");
        if (decoded.length() != server_id_len || std::memcmp(decoded.data().data(), server_id, server_id_len) != 0)
          Fail("decoded server ID");

        std::printf("%u ", rotation);
        PrintHex(key, 16);
        std::printf(" ");
        PrintHex(server_id, server_id_len);
        std::printf(" ");
        PrintHex(nonce, nonce_len);
        std::printf(" ");
        PrintHex(reinterpret_cast<const uint8_t*>(id.data()), id.length());
        std::printf("\n");
      }
    }
  }
  return 0;
}

// Criterion 0.8 linear sampling with .config/criterion.json: 100 ms warm-up, 400 ms measurement,
// 20 samples of d, 2d, ..., 20d iterations, the slope of time over iterations through the origin,
// and a 95% percentile bootstrap interval from 10000 resamples.
template <typename Routine>
void Measure(const char* name, Routine routine) {
  using Clock = std::chrono::steady_clock;
  constexpr int kSamples = 20;
  auto barrier = [](const void* p) { asm volatile("" : : "r"(p) : "memory"); };

  uint64_t warm_iterations = 0;
  const auto warm_start = Clock::now();
  for (uint64_t batch = 1; Clock::now() - warm_start < std::chrono::milliseconds(100); batch *= 2) {
    for (uint64_t i = 0; i < batch; ++i) barrier(routine());
    warm_iterations += batch;
  }
  const double per_iter =
      std::chrono::duration<double, std::nano>(Clock::now() - warm_start).count() / static_cast<double>(warm_iterations);
  const double total = kSamples * (kSamples + 1) / 2.0;
  const uint64_t d = std::max<uint64_t>(1, static_cast<uint64_t>(std::ceil(400e6 / (total * per_iter))));

  double x[kSamples], y[kSamples];
  for (int s = 0; s < kSamples; ++s) {
    const uint64_t iterations = d * static_cast<uint64_t>(s + 1);
    const auto start = Clock::now();
    for (uint64_t i = 0; i < iterations; ++i) barrier(routine());
    y[s] = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
    x[s] = static_cast<double>(iterations);
  }
  auto slope = [](const double* xs, const double* ys, int n) {
    double xy = 0, xx = 0;
    for (int i = 0; i < n; ++i) xy += xs[i] * ys[i], xx += xs[i] * xs[i];
    return xy / xx;
  };
  std::vector<double> estimates;
  estimates.reserve(10000);
  uint64_t rng = 0x626f6f7473747261ULL;
  for (int r = 0; r < 10000; ++r) {
    double bx[kSamples], by[kSamples];
    for (int i = 0; i < kSamples; ++i) {
      rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
      const int j = static_cast<int>(rng % kSamples);
      bx[i] = x[j], by[i] = y[j];
    }
    estimates.push_back(slope(bx, by, kSamples));
  }
  std::sort(estimates.begin(), estimates.end());
  std::printf("%s slope_ns=%.4f low_ns=%.4f high_ns=%.4f\n", name, slope(x, y, kSamples), estimates[250],
              estimates[9749]);
}

constexpr uint8_t kServerId[10] = {0xed, 0x79, 0x3a, 0x51, 0xd4, 0x9b, 0x8f, 0x5f, 0xab, 0x65};
constexpr uint8_t kNonce[9] = {0xee, 0x08, 0x0d, 0xbf, 0x48, 0xc0, 0xd1, 0xe5, 0x5d};

int Bench() {
  struct Shape {
    uint8_t server_id_len, nonce_len, rotation;
  };
  for (const Shape shape : {Shape{3, 4, 0}, Shape{10, 5, 1}, Shape{8, 8, 2}, Shape{9, 9, 0}}) {
    LoadBalancerConfig config = Config(shape.rotation, shape.server_id_len, shape.nonce_len, kDraftKey);
    LoadBalancerDecoder decoder;
    if (!decoder.AddConfig(config)) Fail("AddConfig");
    const QuicConnectionId id = Issue(config, kServerId, kNonce);
    char name[64];
    std::snprintf(name, sizeof(name), "quic/cid-encode/%u+%u/quiche", shape.server_id_len, shape.nonce_len);
    QuicConnectionId issued;
    Measure(name, [&] {
      issued = Issue(config, kServerId, kNonce);
      return static_cast<const void*>(&issued);
    });
    std::snprintf(name, sizeof(name), "quic/cid-decode/%u+%u/quiche", shape.server_id_len, shape.nonce_len);
    LoadBalancerServerId decoded;
    Measure(name, [&] {
      if (!decoder.GetServerId(id, decoded)) Fail("GetServerId");
      return static_cast<const void*>(&decoded);
    });
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "vectors") return Vectors();
  if (argc == 2 && std::string(argv[1]) == "bench") return Bench();
  std::fprintf(stderr, "usage: quic_lb_vectors vectors|bench\n");
  return 2;
}
