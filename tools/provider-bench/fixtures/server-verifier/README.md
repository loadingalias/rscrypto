# Server-verifier fixtures

The `tls13/server-verifier` rows verify these chains against these trust stores at the capture
time, 2026-10-22T03:12:32Z (Unix time 1792638752).

| File | Source |
|---|---|
| `debian.crt` | `/etc/ssl/certs/ca-certificates.crt` from `debian:trixie@sha256:4c25275446ca371766b8b093c696782eabce06263fccacb8d5d813b3dc44b308` after `apt-get install ca-certificates` (20250419+deb13u1): 148 roots |
| `fedora.crt` | `/etc/pki/tls/certs/ca-bundle.crt` from `fedora:43@sha256:a651ddf48ea28a06ed4e1e6519f51c9f47e7a5a138722ade87369b8fbb7e5b42` (ca-certificates-2025.2.80_v9.0.304-1.2.fc43): 146 roots |
| `ORIGIN.crt` | The chain that ORIGIN served, leaf first, captured with `openssl s_client -showcerts` (OpenSSL 4.0.3) |

The origins are those that top-secret needs in one round trip: `index.crates.io` and
`static.crates.io` (GlobalSign RSA roots), `static.rust-lang.org` (GlobalSign Root R46, RSA), and
`github.com` (Sectigo, a P-256 leaf and intermediate under a P-384 root).
