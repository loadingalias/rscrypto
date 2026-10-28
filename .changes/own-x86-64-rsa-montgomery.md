---
"rscrypto" = "patch"
---

Replace the adapted AWS-LC x86-64 RSA Montgomery assembly with owned MULX/ADCX/ADOX kernels in
Rust inline assembly. They serve every x86-64 OS with BMI2 and ADX and every modulus width that
is a multiple of 512 bits. On Granite Rapids, RSA public-key operations run about 3% faster than
with the replaced assembly. Windows public-key operations, and private operations without
AVX-512 IFMA, now use these kernels instead of portable code.
