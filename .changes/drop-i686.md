---
"rscrypto" = "major"
---
Remove 32-bit x86 support, including i586/i686 detection and `platform::Arch::X86`. The later `Arch` variants' implicit numeric discriminants decrease by one. These targets now fail compilation with an explicit diagnostic. Use an x86-64 target for x86 deployments.
