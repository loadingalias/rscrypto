---
"rscrypto" = "patch"
---

Clear the padded Keccak state that fixed-output finalization copied before permuting and left on the stack. That copy could retain absorbed secret input, such as SLH-DSA's SK.seed in SHAKE PRF calls or HMAC-SHA3 key material. Finalization now pads, permutes, and clears one local state in place; outputs are unchanged.
