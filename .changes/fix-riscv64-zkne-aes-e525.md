---
"rscrypto" = "patch"
---

On riscv64 CPUs with the Zkne scalar AES extension but without Zvkned, AES-GCM, AES-GCM-SIV, AES-SIV-CMAC, and AEGIS-256 now give correct results. The Zkne backend packed each 64-bit half of the AES state and of each round key big-endian, but `aes64es` and `aes64esm` read state byte i from bits 8i+7..8i, as the ratified scalar cryptography specification defines. Earlier releases gave wrong ciphertexts and tags on such CPUs. CPUs with Zvkned and other targets are unchanged.
