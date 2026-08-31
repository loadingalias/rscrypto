# Ascon vector provenance

These vectors come from `ascon/ascon-c` commit
`446347f21b209f3921c65ece70027c366cbe1693`. The upstream repository publishes
them under CC0-1.0.

| Local file         | Upstream file                                       | SHA-256                                                            |
| ------------------ | --------------------------------------------------- | ------------------------------------------------------------------ |
| `asconaead128.txt` | `crypto_aead/asconaead128/LWC_AEAD_KAT_128_128.txt` | `bbbc34692fe05e5fda0a3b025585622ab3e3747495e5e3655b29aae8c2a4bd33` |
| `asconcxof128.txt` | `crypto_cxof/asconcxof128/LWC_CXOF_KAT_128_512.txt` | `abcbb0cc851a7f9cfc5ea2bcaf3eba5b2056e37fcb8ce541ceda1d1b960fc9dc` |

The upstream commit identifies these as known-answer tests for NIST SP 800-232.

`asconhash.blb` contains all 1,025 message/digest pairs from
`crypto_hash/asconhash256/LWC_HASH_KAT_128_256.txt` at the same commit.
`asconxof.blb` contains the first 32 output bytes for all 1,025 cases in
`crypto_hash/asconxof128/LWC_XOF_KAT_128_512.txt`; independent oracle tests
cover longer and segmented output.

After checking out the exact upstream commit, reproduce and verify every file:

```bash
scripts/check/hash-vector-provenance.py --ascon-root /path/to/ascon-c
```
