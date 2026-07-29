#!/usr/bin/env python3
"""Verify non-RSA signature assembly provenance and deterministic transforms."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "src/auth/asm/rscrypto_signature_assembly_provenance.tsv"
UPSTREAM_URL = "https://github.com/awslabs/s2n-bignum.git"
MANIFEST_SHA256 = "2cec9cf09bac661e7235a6e50d0e5c97e0a38d7c8e32e03e908a34088378ad5a"

PINNED_COMMIT = "471fca76a9079753aab938ba35ef55ec22717d89"
ED25519_X86_COMMIT = "c19516a30de81f9e664dccdfc79dbf8fb109276d"
X25519_X86_COMMIT = "333cdfcd91a62d15954ecca1124544b8587f86de"

SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
BODY_RE = re.compile(br"(?m)^[ \t]*\.globl[ \t]+")
GLOBAL_RE = re.compile(br"(?m)^[ \t]*\.globl[ \t]+([.$A-Za-z_][.$A-Za-z0-9_]*)[ \t]*$")
SOURCE_SYMBOL_RE = re.compile(
    br"S2N_BN_(?:SYMBOL|SYM_VISIBILITY_DIRECTIVE)\(([A-Za-z_][A-Za-z0-9_]*)\)"
)
SYMBOL_BYTE = rb"A-Za-z0-9_.$"


@dataclass(frozen=True)
class Source:
    member: str
    sha256: str


@dataclass(frozen=True)
class Entry:
    local_path: str
    local_sha256: str
    generated_sha256: str
    commit: str
    transform: str
    sources: tuple[Source, ...]


def fail(message: str) -> None:
    raise SystemExit(f"signature-asm-provenance: {message}")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        fail(f"cannot read {path}: {error}")


def safe_relative_path(value: str, prefix: str | None = None) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        fail(f"unsafe or non-canonical manifest path: {value}")
    if prefix is not None and not value.startswith(prefix):
        fail(f"manifest path {value} is outside {prefix}")


def parse_sources(value: str, line_number: int) -> tuple[Source, ...]:
    sources: list[Source] = []
    seen: set[str] = set()
    for field in value.split(","):
        if field.count("=") != 1:
            fail(f"manifest line {line_number} has an invalid source field")
        member, sha256 = field.split("=", 1)
        safe_relative_path(member)
        if not member.startswith(("arm/", "x86_att/")) or not member.endswith(".S"):
            fail(f"manifest line {line_number} has unsupported source member {member}")
        if not SHA256_RE.fullmatch(sha256):
            fail(f"manifest line {line_number} has invalid source SHA-256")
        if member in seen:
            fail(f"manifest line {line_number} repeats source member {member}")
        seen.add(member)
        sources.append(Source(member, sha256))
    if not sources:
        fail(f"manifest line {line_number} has no source members")
    return tuple(sources)


def parse_manifest() -> list[Entry]:
    data = read_file(MANIFEST)
    if digest(data) != MANIFEST_SHA256:
        fail("provenance manifest changed without a transform review")
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        fail(f"manifest is not UTF-8: {error}")
    records = [
        (line_number, line.split("\t"))
        for line_number, line in enumerate(lines, 1)
        if line and not line.startswith("#")
    ]
    if len(records) != 38:
        fail(f"manifest has {len(records)} records; expected 38")
    if records[0][1] != ["schema", "1"]:
        fail("manifest must begin with schema version 1")
    if records[1][1] != ["upstream", UPSTREAM_URL]:
        fail(f"manifest must bind upstream {UPSTREAM_URL}")

    entries: list[Entry] = []
    for line_number, fields in records[2:]:
        if len(fields) != 7 or fields[0] != "output":
            fail(f"manifest line {line_number} is not a seven-field output record")
        _, local_path, local_sha256, generated_sha256, commit, transform, source_field = fields
        safe_relative_path(local_path, "src/auth/")
        if Path(local_path).suffix not in {".s", ".S"}:
            fail(f"manifest line {line_number} does not name an assembly snapshot")
        if not SHA256_RE.fullmatch(local_sha256):
            fail(f"manifest line {line_number} has invalid local SHA-256")
        if not SHA256_RE.fullmatch(generated_sha256):
            fail(f"manifest line {line_number} has invalid generated SHA-256")
        if not COMMIT_RE.fullmatch(commit):
            fail(f"manifest line {line_number} has invalid upstream commit")
        entries.append(
            Entry(
                local_path,
                local_sha256,
                generated_sha256,
                commit,
                transform,
                parse_sources(source_field, line_number),
            )
        )

    paths = [entry.local_path for entry in entries]
    if len(entries) != 36:
        fail(f"manifest has {len(entries)} outputs; expected 36")
    if len(set(paths)) != len(paths):
        fail("manifest repeats a local output")
    if paths != sorted(paths):
        fail("manifest outputs are not in bytewise path order")

    expected_transforms = set(TRANSFORMS)
    actual_transforms = {entry.transform for entry in entries}
    if actual_transforms != expected_transforms:
        fail(
            "manifest transform set differs from verifier transforms: "
            f"{sorted(actual_transforms ^ expected_transforms)}"
        )
    allowed_commits = {PINNED_COMMIT, ED25519_X86_COMMIT, X25519_X86_COMMIT}
    if {entry.commit for entry in entries} - allowed_commits:
        fail("manifest names an unreviewed upstream commit")
    for entry in entries:
        validate_entry_shape(entry)
    return entries


def validate_entry_shape(entry: Entry) -> None:
    source_count = len(entry.sources)
    expected_count = {
        "A64_MACHO_ECDSA_V1": {1},
        "A64_ELF_ECDSA_SPLIT_V1": {1},
        "X86_ELF_ECDSA_GLOBAL_V1": {1},
        "A64_MACHO_CONCAT_TOKEN_V1": {1, 2},
        "A64_ELF_ED25519_GLOBAL_V1": {1, 2},
        "X86_ELF_ED25519_GLOBAL_V1": {1},
        "A64_ELF_X25519_TOKEN_V1": {2},
        "X86_ELF_X25519_CONCAT_TOKEN_V1": {4},
    }[entry.transform]
    if source_count not in expected_count:
        fail(
            f"{entry.local_path} has {source_count} sources for {entry.transform}; "
            f"expected {sorted(expected_count)}"
        )
    if entry.transform == "X86_ELF_ED25519_GLOBAL_V1":
        if entry.commit != ED25519_X86_COMMIT:
            fail(f"{entry.local_path} does not use the exact x86 Ed25519 ancestor")
    elif entry.transform == "X86_ELF_X25519_CONCAT_TOKEN_V1":
        if entry.commit != X25519_X86_COMMIT:
            fail(f"{entry.local_path} does not use the exact x86 X25519 ancestor")
    elif entry.commit != PINNED_COMMIT:
        fail(f"{entry.local_path} does not use the pinned upstream commit")


def generated_body(data: bytes, name: str) -> bytes:
    match = BODY_RE.search(data)
    if match is None:
        fail(f"{name} has no generated .globl body")
    return data[match.start() :]


def external_signature_snapshots() -> set[str]:
    roots = (
        ROOT / "src/auth/asm",
        ROOT / "src/auth/ed25519/asm",
        ROOT / "src/auth/x25519/asm",
    )
    snapshots: set[str] = set()
    for directory in roots:
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix not in {".s", ".S"}:
                continue
            data = read_file(path)
            external = (
                b"Adapted for rscrypto" in data
                or b"The butterfly schedule is auto-derived from" in data
            )
            if external and not path.name.startswith("rscrypto_rsa_"):
                snapshots.add(path.relative_to(ROOT).as_posix())
    return snapshots


def verify_local(entries: list[Entry]) -> None:
    manifest_paths = {entry.local_path for entry in entries}
    snapshots = external_signature_snapshots()
    if manifest_paths != snapshots:
        missing = sorted(snapshots - manifest_paths)
        extra = sorted(manifest_paths - snapshots)
        fail(f"manifest coverage drift; missing={missing}, extra={extra}")

    for entry in entries:
        data = read_file(ROOT / entry.local_path)
        actual = digest(data)
        if actual != entry.local_sha256:
            fail(
                f"{entry.local_path} has SHA-256 {actual}; "
                f"expected {entry.local_sha256}"
            )
        body = generated_body(data, entry.local_path)
        actual_generated = digest(body)
        if actual_generated != entry.generated_sha256:
            fail(
                f"{entry.local_path} generated body has SHA-256 {actual_generated}; "
                f"expected {entry.generated_sha256}"
            )


class Upstream:
    def __init__(self, repository: Path, clang: str, temporary_root: Path) -> None:
        self.repository = repository
        self.clang = clang
        self.temporary_root = temporary_root
        self.blobs: dict[tuple[str, str], bytes] = {}
        self.include_dirs: dict[tuple[str, str], Path] = {}

    def git(self, *args: str) -> bytes:
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repository), *args],
                check=True,
                capture_output=True,
            )
        except OSError as error:
            fail(f"cannot execute git: {error}")
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode("utf-8", "replace").strip()
            fail(f"git {' '.join(args)} failed: {detail}")
        return result.stdout

    def verify_identity(self) -> None:
        inside = self.git("rev-parse", "--is-inside-work-tree").decode().strip()
        if inside != "true":
            fail(f"{self.repository} is not a Git worktree")
        remote = self.git("remote", "get-url", "origin").decode().strip()
        if remote.rstrip("/") != UPSTREAM_URL.rstrip("/"):
            fail(f"upstream origin is {remote}; expected {UPSTREAM_URL}")
        for commit in (PINNED_COMMIT, ED25519_X86_COMMIT, X25519_X86_COMMIT):
            actual = self.git("rev-parse", f"{commit}^{{commit}}").decode().strip()
            if actual != commit:
                fail(f"upstream repository does not contain exact commit {commit}")

    def blob(self, commit: str, member: str) -> bytes:
        key = (commit, member)
        if key not in self.blobs:
            self.blobs[key] = self.git("show", f"{commit}:{member}")
        return self.blobs[key]

    def include_dir(self, commit: str, architecture: str) -> Path:
        key = (commit, architecture)
        if key in self.include_dirs:
            return self.include_dirs[key]
        header = f"_internal_s2n_bignum_{architecture}.h"
        directory = self.temporary_root / commit / "include"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            (directory / header).write_bytes(self.blob(commit, f"include/{header}"))
        except OSError as error:
            fail(f"cannot materialize upstream preprocessor header: {error}")
        self.include_dirs[key] = directory
        return directory

    def preprocess(
        self,
        source: bytes,
        commit: str,
        architecture: str,
        target: str,
        *,
        line_markers: bool,
        strip_comments: bool,
    ) -> bytes:
        if strip_comments:
            source = re.sub(rb"//[^\n]*", b"", source)
        command = [
            self.clang,
            f"--target={target}",
            "-E",
        ]
        if not line_markers:
            command.append("-P")
        command.extend(
            [
                f"-I{self.include_dir(commit, architecture)}",
                "-DS2N_BN_HIDE_SYMBOLS=1",
                "-x",
                "assembler-with-cpp",
                "-",
            ]
        )
        environment = os.environ.copy()
        environment["LC_ALL"] = "C"
        try:
            result = subprocess.run(
                command,
                input=source,
                check=True,
                capture_output=True,
                env=environment,
            )
        except OSError as error:
            fail(f"cannot execute {self.clang}: {error}")
        except subprocess.CalledProcessError as error:
            detail = error.stderr.decode("utf-8", "replace").strip()
            fail(f"preprocessing failed for {target}: {detail}")
        if b"\r" in result.stdout:
            fail(f"{self.clang} emitted non-LF output for {target}")
        return result.stdout


def public_symbols(data: bytes) -> tuple[bytes, ...]:
    symbols = tuple(dict.fromkeys(GLOBAL_RE.findall(data)))
    if not symbols:
        fail("preprocessed assembly exports no symbols")
    return symbols


def source_symbols(source: bytes, preprocessed: bytes) -> tuple[bytes, ...]:
    public = public_symbols(preprocessed)
    macho = public[0].startswith(b"_")
    declared = (
        (b"_" + symbol if macho else symbol)
        for symbol in SOURCE_SYMBOL_RE.findall(source)
    )
    return tuple(dict.fromkeys((*public, *declared)))


def prefixed(symbol: bytes) -> bytes:
    if symbol.startswith(b"_"):
        return b"_rscrypto_" + symbol[1:]
    return b"rscrypto_" + symbol


def exact_token_rename(data: bytes, symbols: tuple[bytes, ...]) -> bytes:
    alternatives = b"|".join(re.escape(symbol) for symbol in sorted(symbols, key=len, reverse=True))
    pattern = re.compile(
        rb"(?<![" + SYMBOL_BYTE + rb"])(" + alternatives + rb")(?![" + SYMBOL_BYTE + rb"])"
    )
    return pattern.sub(lambda match: prefixed(match.group(1)), data)


def global_stem_rename(data: bytes, symbols: tuple[bytes, ...]) -> bytes:
    alternatives = b"|".join(re.escape(symbol) for symbol in sorted(symbols, key=len, reverse=True))
    pattern = re.compile(alternatives)
    return pattern.sub(lambda match: prefixed(match.group(0)), data)


def delete_line_markers(data: bytes) -> bytes:
    return b"".join(
        line
        for line in data.splitlines(keepends=True)
        if not line.lstrip().startswith(b"#")
    )


def delete_gnu_stack(data: bytes) -> bytes:
    return re.sub(
        rb"(?m)^[ \t]*\.section[ \t]+\.note\.GNU-stack[^\n]*(?:\n|$)",
        b"",
        data,
    )


def one_source(sources: tuple[bytes, ...], transform: str) -> bytes:
    if len(sources) != 1:
        fail(f"{transform} requires one source")
    return sources[0]


def a64_macho_ecdsa(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = upstream.preprocess(
        one_source(sources, entry.transform),
        entry.commit,
        "arm",
        "arm64-apple-darwin",
        line_markers=True,
        strip_comments=True,
    )
    output = delete_line_markers(output)
    output = exact_token_rename(output, public_symbols(output))
    return generated_body(output, entry.transform)


def a64_elf_ecdsa_split(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = upstream.preprocess(
        one_source(sources, entry.transform),
        entry.commit,
        "arm",
        "aarch64-linux-gnu",
        line_markers=True,
        strip_comments=False,
    )
    output = output.replace(b" ; ", b";        ").replace(b";", b"\n")
    lines: list[bytes] = []
    for line in delete_line_markers(output).splitlines():
        stripped = line.lstrip()
        if stripped.startswith((b".type ", b".size ")):
            lines.append(b"")
        elif stripped.startswith(b".section .note.GNU-stack"):
            continue
        else:
            lines.append(line.rstrip())
    while lines and not lines[-1]:
        lines.pop()
    output = b"\n".join(lines) + b"\n"
    output = exact_token_rename(output, public_symbols(output))
    return generated_body(output, entry.transform)


def x86_elf_ecdsa_global(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = upstream.preprocess(
        one_source(sources, entry.transform),
        entry.commit,
        "x86_att",
        "x86_64-linux-gnu",
        line_markers=False,
        strip_comments=False,
    )
    output = global_stem_rename(output, public_symbols(output))
    output = delete_gnu_stack(output).rstrip(b"\n") + b"\n"
    return generated_body(output, entry.transform)


def a64_macho_concat_token(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    parts: list[bytes] = []
    for source in sources:
        output = upstream.preprocess(
            source,
            entry.commit,
            "arm",
            "arm64-apple-darwin",
            line_markers=False,
            strip_comments=True,
        )
        output = exact_token_rename(output, source_symbols(source, output))
        parts.append(generated_body(output, entry.transform).rstrip(b"\n"))
    return b"\n\n\n".join(parts) + b"\n"


def a64_elf_ed25519_global(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    parts: list[bytes] = []
    for source in sources:
        output = upstream.preprocess(
            source,
            entry.commit,
            "arm",
            "aarch64-linux-gnu",
            line_markers=False,
            strip_comments=False,
        )
        output = global_stem_rename(output, public_symbols(output))
        parts.append(generated_body(output, entry.transform).rstrip(b"\n"))
    return b"\n\n\n\n".join(parts) + b"\n"


def x86_elf_ed25519_global(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = upstream.preprocess(
        one_source(sources, entry.transform),
        entry.commit,
        "x86_att",
        "x86_64-linux-gnu",
        line_markers=False,
        strip_comments=False,
    )
    output = global_stem_rename(output, public_symbols(output))
    return generated_body(output, entry.transform)


def a64_elf_x25519_token(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    parts: list[bytes] = []
    for source in sources:
        output = upstream.preprocess(
            source,
            entry.commit,
            "arm",
            "aarch64-linux-gnu",
            line_markers=False,
            strip_comments=False,
        )
        symbols = source_symbols(source, output)
        output = exact_token_rename(output, symbols)
        output = delete_gnu_stack(output)
        for symbol in symbols:
            renamed = re.escape(prefixed(symbol))
            if symbol.endswith(b"_constant"):
                output = re.sub(
                    rb"(?m)^[ \t]*\.(?:type|size)[ \t]+"
                    + renamed
                    + rb"[^\n]*(?:\n|$)",
                    b"",
                    output,
                )
        parts.append(generated_body(output, entry.transform).rstrip(b"\n"))
    return b"\n\n\n".join(parts) + b"\n"


def x86_elf_x25519_concat_token(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    parts: list[bytes] = []
    for source in sources:
        output = upstream.preprocess(
            source,
            entry.commit,
            "x86_att",
            "x86_64-linux-gnu",
            line_markers=False,
            strip_comments=False,
        )
        output = exact_token_rename(output, source_symbols(source, output))
        output = delete_gnu_stack(output)
        parts.append(generated_body(output, entry.transform).rstrip(b"\n"))
    return (
        b"\n\n".join(parts)
        + b'\n\n.section .note.GNU-stack, "", %progbits\n'
    )


Transform = Callable[[Upstream, Entry, tuple[bytes, ...]], bytes]
TRANSFORMS: dict[str, Transform] = {
    "A64_MACHO_ECDSA_V1": a64_macho_ecdsa,
    "A64_ELF_ECDSA_SPLIT_V1": a64_elf_ecdsa_split,
    "X86_ELF_ECDSA_GLOBAL_V1": x86_elf_ecdsa_global,
    "A64_MACHO_CONCAT_TOKEN_V1": a64_macho_concat_token,
    "A64_ELF_ED25519_GLOBAL_V1": a64_elf_ed25519_global,
    "X86_ELF_ED25519_GLOBAL_V1": x86_elf_ed25519_global,
    "A64_ELF_X25519_TOKEN_V1": a64_elf_x25519_token,
    "X86_ELF_X25519_CONCAT_TOKEN_V1": x86_elf_x25519_concat_token,
}


def verify_upstream(entries: list[Entry], repository: Path, clang: str) -> None:
    with tempfile.TemporaryDirectory(prefix="rscrypto-signature-asm-") as temporary:
        upstream = Upstream(repository, clang, Path(temporary))
        upstream.verify_identity()
        for entry in entries:
            source_data: list[bytes] = []
            for source in entry.sources:
                data = upstream.blob(entry.commit, source.member)
                actual = digest(data)
                if actual != source.sha256:
                    fail(
                        f"{entry.commit}:{source.member} has SHA-256 {actual}; "
                        f"expected {source.sha256}"
                    )
                source_data.append(data)
            generated = TRANSFORMS[entry.transform](
                upstream, entry, tuple(source_data)
            )
            actual = digest(generated)
            if actual != entry.generated_sha256:
                fail(
                    f"{entry.local_path} regenerated body has SHA-256 {actual}; "
                    f"expected {entry.generated_sha256}"
                )
            local = generated_body(read_file(ROOT / entry.local_path), entry.local_path)
            if generated != local:
                fail(f"{entry.local_path} differs from regenerated upstream body")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify non-RSA signature assembly provenance."
    )
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        help="s2n-bignum Git worktree used for source and regeneration checks",
    )
    parser.add_argument(
        "--clang",
        default=os.environ.get("CLANG", "clang"),
        help="C preprocessor driver for --upstream-repo mode (default: clang)",
    )
    args = parser.parse_args()

    entries = parse_manifest()
    verify_local(entries)
    if args.upstream_repo is not None:
        verify_upstream(entries, args.upstream_repo, args.clang)
        print(
            "signature-asm-provenance: 36 snapshots match immutable upstream "
            "sources and transforms"
        )
    else:
        print(
            "signature-asm-provenance: 36 snapshots match the provenance "
            "manifest"
        )


if __name__ == "__main__":
    main()
