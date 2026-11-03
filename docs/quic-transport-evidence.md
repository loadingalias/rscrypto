# QUIC transport qualification, 2026-11-03

The actionable transport investigation is complete at rscrypto source `6926f298c16d3d6777973e6fa5d29ad86e0ca48d`.
It found two runner defects and no demonstrated provider defect.
The second ClientHello datagram waits for Quinn's pacer.
Extreme-loss failures also occur with stock Quinn controls.
Failed runs remain failures; these results do not promise
that all 50 concurrent handshakes succeed under the extreme-loss scenario.

Preferred-address migration and its post-timeout fallback remain on the owner's Quinn hold.
Stateless ML-DSA-87 mutual-authentication resumption against quic-go remains unsupported.
No production source, provider policy, production dependency pin,
or held upstream patch changed in this work.

## Source and execution identity

| Component                                   | Tested identity |
| ------------------------------------------- | --------------- |
| Provider                                    | rscrypto commit above; rustls `0419f479ff5c99a471428ee197e899ccd1629aa8`, `0.24.0-dev.2` |
| Transport, both provider and stock controls | Quinn `2697de1e1314cb9f98502b7323acd5bdd836b582`, `0.12.0` |
| Stock control                               | Quinn's rustls adapter, rustls `0.23.46`, AWS-LC `1.19.0` / `aws-lc-sys 0.47.0`; separate harness workspace |
| Loopback peer                               | quic-go `v0.61.0`, `579ee19d5b54c4f9320ffca668113c3513a138e5`; Go `1.27.3`, CGO disabled |
| Runner peer image                           | quic-go `99c743630f4a96a3c584ed64fb189db6ebd84f38`; Go `1.27.0`, CGO disabled |
| Runner and application                      | `740c05a10b61d65e8abd3ad38d60898004d335d9` and `82c845f6e5399ef15fac217ba2799c2a233d875b` |
| HTTP/3 compatibility source                 | h3 `2dc3412bdf6083451920d5bfd7a9484d054c1859`; existing Quinn requirement adaptation |
| Compiler                                    | rustc `1.101.0-nightly`, `48f4b05538b8eca412137042d9647a61e692c6b3`, LLVM `23.1.4` |
| Execution                                   | Apple ARM64 loopback; Linux ARM64 Docker Desktop, kernel `7.0.14-linuxkit`; release endpoints cross-built with `cargo zigbuild` for GNU libc 2.39 |

[Runner constants](../scripts/test/provider_quic_runner.py) own the image digests.
Each prepared bundle retains image IDs, resolved Cargo locks, compiler records, endpoint binaries,
source copies, patches, and build commands.
Each campaign's `FINAL-SHA256SUMS` also covers the retained analyses and failures.
The local identity bundle, `benchmark_results/gap3-20261103-identities/`, also retains Go binary build information, the pinned peer's timeout source,
and the reached Quinn connection/pacing sources.
The simulator image does not expose an ns-3 build version;
its exact digest and reported RNG globals are retained instead.

Concurrent Gap 2 changes reached tooling and private tests,
not the production paths used by these release transport binaries.
Source reconciliation is retained in the identity bundle.
This is transport evidence on these hosts and artifacts,
not a performance campaign or a substitute for native constant-time qualification.

## Loss reproduction and controls

Each `handshakeloss` case requested the original 50 connections.
Settings stayed at 15 ms one-way delay, 10 Mbps, a 25-packet queue, 30% loss in each direction,
and maximum loss bursts of three. ns-3 used seeds 1, 2, and 3, each with RngRun 1.
The pinned Go peer retained its 5-second handshake idle timeout
and 10-second overall handshake timeout.
Quinn's existing application idle timeout stayed at 30 seconds.
No timeout, queue, error, or assertion was relaxed.

The peer's defaults come from its pinned [protocol constants](https://github.com/quic-go/quic-go/blob/99c743630f4a96a3c584ed64fb189db6ebd84f38/internal/protocol/params.go) and [configuration](https://github.com/quic-go/quic-go/blob/99c743630f4a96a3c584ed64fb189db6ebd84f38/config.go).

All nine cases used the same retained P-256 runner credentials,
the same Quinn transport and application settings, and the same independent Go server image.
The provider and stock PQ clients offered only `X25519MLKEM768`; the classical control offered only `X25519`.
Stock uses rustls 0.23 because that is the pinned Quinn stock adapter's interface.
It is a transport control, not an isolation of every difference between the two TLS implementations.

| Client                 | Seed 1 | Seed 2 | Seed 3 |
| ---------------------- | ------ | ------ | ------ |
| Provider, PQ           | Pass   | Pass   | Fail   |
| Stock Quinn, PQ        | Fail   | Pass   | Pass   |
| Stock Quinn, classical | Fail   | Pass   | Pass   |

These are bounded observations, not estimated failure rates.
TLS entropy and concurrent scheduling remain independent;
equal simulator seeds do not imply identical dropped datagrams across implementations.
Both simulator capture points, TLS test key logs, Go qlogs, Quinn traces, credentials, commands,
and failures are retained in `benchmark_results/gap3-20261103-loss-*`.
The loss analysis (`gap3-20261103-identities/loss-summary.json`) records incomplete trailing qlog records caused by campaign shutdown rather than
counting them as completed connections.

The provider PQ first flight was 1,457–1,472 CRYPTO bytes per connection.
Fifty flights produced 100 Initial datagrams, 132,600 UDP payload bytes,
in spans of 11.1–13.8 ms in these captures.
Stock PQ also produced 100 datagrams and 132,600 bytes.
The classical control produced 50 datagrams and 60,000 bytes.
The burst record (`gap3-20261103-identities/flight-bursts.json`) separates original flight sizes from retransmissions.
Capture differences prove datagram loss;
they do not distinguish queue overflow from the simulator's random drop rule.

For provider seed 3, client port 39924 received ServerHello.
Its Finished appears in left capture frame 1352 and is absent at the right capture.
The server qlog records a Handshake ACK at 2,564.926 ms, no client Handshake CRYPTO,
and idle expiry at 7,565.861 ms: approximately five seconds after the last accepted packet.
Port 48343 received a later ServerHello, about 4.692 seconds after its first Initial;
its Finished in left frame 1458 was also lost.
No Handshake packet reached that server connection before expiry.
Other timed-out connections had received Finished and failed later during transfer.
They are not evidence of an absent ServerHello or a failed provider signature operation.

Stock PQ seed 1 likewise contains a lost-Finished handshake timeout at port 56793.
Thus the reproduction establishes failure under the constrained loss scenario without isolating a
provider-specific defect.
The original requirement that every connection succeed remains unsatisfied in the failed campaigns;
a consumer requiring that guarantee needs an owner decision on the transport requirements,
not a larger timeout hidden in this harness.

## Second ClientHello datagram

The delay reproduces.
In the provider's single-handshake trace, all 1,465 ClientHello bytes were queued at `05:26:51.812170Z`.
Quinn sent the first datagram at `.812383Z`, reported `blocked by pacing` at `.812511Z`, and sent the second at `.822603Z`.
The capture gap was 10.225 ms. Crypto serialization was complete before either send,
and the network capture agrees with the production send trace.
This rules out waiting for remaining ClientHello bytes or a ten-millisecond socket-buffer delay in
that sample.

The stock PQ 50-connection case showed the same pacing wait,
with a median first-to-second fragment gap of 9.611 ms for seed 1.
The provider's medians were 8.839, 6.526, and 9.187 ms across the three seeds.
Stock's cold single handshake spent about 28.8 ms before queuing ClientHello
and then sent its two datagrams 0.473 ms apart.
It must not be compared with the provider's post-first-send interval as
though the two intervals measured the same work.

The retained Quinn pacer uses a token bucket derived from its congestion window, initial RTT,
and MTU.
Its capacity can be one MTU at this initial RTT.
Its `blocked by pacing`, send, and timer events distinguish that deliberate transport wait from TLS work
and subsequent runtime scheduling.
Runtime wakeup and logging overhead remain in the observed intervals.
No cryptographic optimization, timing threshold test,
or upstream pacing change is justified by these observations.

## Independent-peer coverage

| Evidence                                                           | Result and scope |
| ------------------------------------------------------------------ | ---------------- |
| `gap3-20261103-loopback`                                           | All 290 cases / 578 connections passed. Classical and ML-DSA credentials; all declared suites/groups; full/resumed mutual authentication, Retry, explicit 0-RTT, reset, exporter equality, and observed packet-key updates. |
| `gap3-20261103-provider`, `gap3-20261103-reverse`                  | Handshake, transfer, Retry, resumption, and key update passed in the applicable independent-peer directions. The initial amplification fixture failures are retained. |
| `gap3-20261103-forward-loss`, `gap3-20261103-reverse`              | Original 2% burst-loss transfer case passed in both directions. |
| `gap3-20261103-amp-fixed-quinn`, `gap3-20261103-amp-fixed-quic-go` | Original amplification case passed in both directions after fixture repair. Additional strict accounting found no provider-server excess over 3× before validation: 2,560 bytes received, 7,301 sent. |
| `gap3-20261103-mldsa-loss-fixed`                                   | Six independent authenticated cases / 12 connections passed: ML-DSA-44/65/87, both roles, AES-128-GCM and hybrid exchange, full/resumed sessions, key updates. The retained UDP boundary uses seeds 1–6, 10% per-datagram loss, 15 ms delay and a 25-packet queue. Every case actually dropped datagrams. |

The pinned Go server exceeded strict 3× accounting by 1,052 bytes in the amplification capture,
although it passed the upstream test's existing 4× tolerance.
Strict accounting is recorded in `gap3-20261103-identities/amplification-summary.json`;
the passing row does not claim that peer satisfies a strict 3× bound.
The provider server did satisfy the stricter accounting in its independent-client run.

The full ML-DSA-87 mutual-authentication flight reached 12,512 Handshake CRYPTO bytes from the
provider server, across 11 observed CRYPTO frames, and 12,253 from the provider client, also across
11 frames.
The flight inventory (`gap3-20261103-identities/loopback-flights.json`) contains ML-DSA-44/65 and the P-256 control.
The lossy cases exercise actual fragmented production handshakes; the UDP boundary only delays,
drops, and forwards packets.

The Docker runner uses a permissive certificate-chain verifier.
Its signature and transport results are separate from the loopback matrix's trust-chain validation.
The runner's explicit early-data setting is also separate from the provider's unchanged default of
early data off.

## Fixes, regressions, and reproduction

1. A digest-pinned image already present locally no longer requires a registry pull.
   The original preparation failed with Docker Hub HTTP 429 despite the exact image being present.
   The regression checks local reuse, pulling a missing digest, and refusal of mutable tags.
1. Campaign copies now place all runner temporary directories on the host mount, including `TestCaseAmplificationLimit.certs_dir` in `testcases_quic.py`.
   The original two-file rewrite missed that subclass,
   causing missing certificates in the server container.
   The regression fails against the pre-fix function and passes after repair;
   both real amplification cases then pass.

The runner also retains credentials for failed and unsupported cases,
records simulator seeds and compiler identity,
and supports matched stock controls and production tracing.
Omitted control options preserve endpoint defaults.
The ten-minute campaign cap and original test assertions remain unchanged.

The primary front doors and a bounded replay are:

```bash
just test-provider-quic --output benchmark_results/gap3-20261103-loopback
just test-provider-quic-runner --stock --prepare-only \
  --output benchmark_results/gap3-20261103-stock-fixed
just test-provider-quic-runner \
  --prepared benchmark_results/gap3-20261103-provider \
  --tests handshakeloss --server quic-go --client quinn \
  --group X25519MLKEM768 --seed 3 --rng-run 1 --trace \
  --credentials benchmark_results/gap3-20261103-provider/logs/quic-go_quinn/handshake/certs \
  --output benchmark_results/gap3-replay-provider-seed3
```

Use a new output directory for each replay.
The last command shows the seed-3 loss selection alone; the retained campaign also included `handshake`.
Exact executed argv for every batch is in the bundles' command JSON and the identity bundle's `controls.json`, `transport-commands.json`,
and `anti-amplification-commands.json`.
`mldsa_loss.py`, `analyze.py`, and `summarize_loss.py` there retain the additional fixture and analysis procedures.
Replays with fresh entropy/scheduling need not reproduce the same outcome.

`just test-scripts` passed.
The final focused runner suite passed all seven tests, including the controlled pre-fix failure.
Live runner cases qualify the stock compatibility patch and generated campaign configuration.
Production code was unchanged, so no new cryptographic, constant-time, fuzz,
or cross-target claim is made from these tooling changes.
The owner's green-workflow confirmation remains authoritative;
this work did not resume GitHub polling.

## Unsupported and held cases

The retained `gap3-20261103-stateless-negative` experiment removes only the Go fixture's stateful-ticket override.
Full ML-DSA-87 mutual authentication succeeds, but resumption fails with `CRYPTO_BUFFER_EXCEEDED`:
offset 16,740 exceeds the peer's 16,384-byte CRYPTO limit.
This negative run is not a passing profile.
The stateful-ticket successes do not close it or authorize changing the supported profile.

Preferred-address migration, its post-timeout fallback, the held Quinn series,
and any decision to change pins or transport support remain with the owner.
These independent qualification results do not resume that work.
