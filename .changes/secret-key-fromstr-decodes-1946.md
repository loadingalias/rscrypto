---
"rscrypto" = "patch"
---

Secret-key FromStr decodes hexadecimal without a branch or table index on the secret characters. Only a string that is not valid hex takes a branch, to report its first invalid character.
