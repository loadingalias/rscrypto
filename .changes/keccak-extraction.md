---
"rscrypto" = "patch"
---

Copy complete SHAKE output lanes directly into the caller's buffer.
Do not use a temporary lane byte array when extracting a short output tail.
