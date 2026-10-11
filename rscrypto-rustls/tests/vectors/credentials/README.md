# TLS credential fixtures

These are public test credentials generated with OpenSSL 4.0.3 on 2026-10-11.
The P-256 root signs a distinct P-256 leaf for `localhost`, usable for client
and server authentication. Certificates are valid from 2026-01-01 through
2036-01-01; in-process tests use a fixed time within that interval.

The leaf private key is intentionally public and must never protect real traffic.
`SHA256SUMS` pins the exact DER files. The root private key is not required by tests.
The commands below generate equivalent new fixtures with fresh keys and signatures;
they do not reproduce the pinned random bytes.

```sh
openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out p256-root-key.pem
openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out p256-leaf-key.pem
openssl req -new -x509 -key p256-root-key.pem -subj '/CN=rscrypto provider test root' \
  -set_serial 1 -not_before 20260101000000Z -not_after 20360101000000Z \
  -addext 'basicConstraints=critical,CA:TRUE,pathlen:0' \
  -addext 'keyUsage=critical,keyCertSign,cRLSign' -out p256-root.pem
openssl req -new -x509 -key p256-leaf-key.pem -subj '/CN=rscrypto provider test leaf' \
  -set_serial 2 -not_before 20260101000000Z -not_after 20360101000000Z \
  -addext 'basicConstraints=critical,CA:FALSE' -addext 'keyUsage=critical,digitalSignature' \
  -CA p256-root.pem -CAkey p256-root-key.pem -addext 'subjectAltName=DNS:localhost' \
  -addext 'extendedKeyUsage=serverAuth,clientAuth' -out p256-leaf.pem
openssl x509 -in p256-root.pem -outform DER -out p256-root.der
openssl x509 -in p256-leaf.pem -outform DER -out p256-leaf.der
openssl pkcs8 -topk8 -nocrypt -in p256-leaf-key.pem -outform DER -out p256-leaf-key.der
openssl verify -CAfile p256-root.pem -purpose sslserver p256-leaf.pem
```
