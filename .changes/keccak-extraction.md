---
"rscrypto" = "patch"
---

Copy complete SHAKE output lanes directly into caller buffers. Avoid a temporary
lane byte array when extracting a short output tail.
