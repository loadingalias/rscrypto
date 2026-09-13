"""Shared scope for internal production regressions on native and transferred test lanes."""

from pathlib import Path
import subprocess

TARGET_ARGS = [
    "--lib", "--test", "aead_kernel_equivalence", "--test", "argon2_kernels",
    "--test", "chacha20poly1305", "--test", "rsa_public_key", "--test", "pbkdf2_evidence",
]


def main():
    root = Path(__file__).resolve().parents[2]
    for dispatch in ("--native", "--portable"):
        subprocess.run(["just", "--justfile", str(root / "justfile"), "test", "--release", dispatch,
                        "--", *TARGET_ARGS], cwd=root, check=True)


if __name__ == "__main__":
    main()
