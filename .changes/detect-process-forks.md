---
"rscrypto" = "minor"
---

Add `platform::ProcessOrigin`, which records the process that created a value. In a copy that a process fork made, `ProcessOrigin::is_current` returns false. On Linux and Android (x86-64, AArch64, 64-bit POWER, 64-bit RISC-V, IBM Z) the check is one function call with one atomic load from a page that the kernel gives as zeros to a fork child; if the kernel refuses that page, and on Apple and other Unix targets, it compares process IDs. Targets that cannot fork always report the current process. Without `std`, no fork is detected.

Security behavior change: `aead::NonceCounter` issues nonces only in the process that created it. In a fork child, `next_nonce` and each `encrypt` method return `NonceCounterExhausted` and consume no nonce, because a copied counter would repeat AES-GCM nonces under one key. Create a new key and counter in the child.

Breaking: `NonceCounter::new` for AES-128-GCM and AES-256-GCM is no longer a `const fn`, because it records the creating process. Migration: a `static` that holds a counter, for example `static COUNTER: Mutex<NonceCounter<Aes256Gcm>> = Mutex::new(NonceCounter::new(prefix))`, becomes `static COUNTER: LazyLock<Mutex<NonceCounter<Aes256Gcm>>> = LazyLock::new(|| Mutex::new(NonceCounter::new(prefix)))`, or the counter is built at run time.
