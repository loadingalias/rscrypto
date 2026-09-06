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
MANIFEST_SHA256 = "1c5a57477c131551618cb57836a60027b1b42b4c3818a70e497a142ca39551ce"

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
    if len(records) != 60:
        fail(f"manifest has {len(records)} records; expected 60")
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
    if len(entries) != 58:
        fail(f"manifest has {len(entries)} outputs; expected 58")
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
        "A64_MACHO_GLOBAL_V1": {1},
        "A64_MACHO_ECDSA_CLEAN_V1": {1},
        "A64_MACHO_P256_ECDH_V1": {1},
        "A64_ELF_ECDSA_SPLIT_V1": {1},
        "A64_ELF_GLOBAL_V1": {1},
        "A64_ELF_ECDSA_CLEAN_V1": {1},
        "A64_ELF_P256_ECDH_V1": {1},
        "X86_ELF_ECDSA_GLOBAL_V1": {1},
        "X86_ELF_GLOBAL_V1": {1},
        "X86_ELF_ECDSA_CLEAN_V1": {1},
        "X86_ELF_P256_ECDH_V1": {1},
        "X86_COFF_ECDSA_CLEAN_V1": {1},
        "X86_COFF_P256_ECDH_V1": {1},
        "X86_COFF_P256_CURVE_TERMS_V1": {3},
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
        windows_abi: bool = False,
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
        if windows_abi:
            command.append("-DWINDOWS_ABI=1")
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


def delete_elf_directives(data: bytes) -> bytes:
    return re.sub(
        rb"(?m)^[ \t]*\.(?:hidden|size|type)[ \t]+[^\n]*(?:\n|$)",
        b"",
        delete_gnu_stack(data),
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


def a64_macho_ecdsa_clean(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = a64_macho_ecdsa(upstream, entry, sources)

    def clear_frame(
        data: bytes,
        frame: bytes,
        pairs: int,
        label: bytes,
        *,
        clear_volatile: bool = False,
    ) -> bytes:
        needle = b"        add sp, sp, #(" + frame + b") %% .cfi_adjust_cfa_offset -" + frame.split(b" +")[0] + b"\n"
        if data.count(needle) != 1:
            fail(
                f"{entry.local_path} does not contain exactly one expected "
                f"{frame.decode()} frame epilogue"
            )
        cleanup = (
            b"        mov x16, sp\n"
            b"        mov x17, #" + str(pairs).encode() + b"\n" +
            label + b":\n"
            b"        stp xzr, xzr, [x16], #16\n"
            b"        subs x17, x17, #1\n"
            b"        b.ne " + label + b"\n"
        )
        if clear_volatile:
            cleanup += b"".join(
                b"        mov x" + str(register).encode() + b", xzr\n"
                for register in range(18)
            )
        return data.replace(needle, cleanup + needle, 1)

    output = clear_frame(
        output,
        b"9*32 +0",
        18,
        b"Lrscrypto_p256_scalarmulbase_alt_clear_frame",
        clear_volatile=True,
    )
    output = clear_frame(
        output,
        b"160 +0",
        10,
        b"Lrscrypto_p256_scalarmulbase_alt_clear_inverse_frame",
    )
    output = clear_frame(
        output,
        b"192 +0",
        12,
        b"Lrscrypto_p256_scalarmulbase_alt_clear_mixadd_frame",
    )

    restore = re.compile(
        rb"(?m)^(?P<indent>[ \t]*)ldp (?P<left>x[0-9]+), (?P<right>x[0-9]+), "
        rb"\[sp\], #16(?P<cfi> %% \.cfi_adjust_cfa_offset -16 %% \.cfi_restore "
        rb"x[0-9]+ %% \.cfi_restore x[0-9]+)$"
    )

    def wipe_saved_registers(match: re.Match[bytes]) -> bytes:
        indent = match.group("indent")
        return (
            indent + b"ldp " + match.group("left") + b", " + match.group("right") + b", [sp]\n"
            + indent + b"stp xzr, xzr, [sp]\n"
            + indent + b"add sp, sp, #16" + match.group("cfi")
        )

    output, replacements = restore.subn(wipe_saved_registers, output)
    if replacements != 7:
        fail(
            f"{entry.local_path} rewrote {replacements} saved-register spills; "
            "expected 7"
        )
    return output


def a64_macho_p256_ecdh(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = a64_macho_ecdsa(upstream, entry, sources)
    symbol = (
        b"Lrscrypto_p256_scalarmul_alt"
        if "scalarmul_alt_" in entry.local_path
        else b"Lrscrypto_p256_scalarmul"
    )
    is_alt = symbol.endswith(b"_alt")

    def cleanup_bytes(
        pairs: int, label: bytes, *, clear_volatile: bool = False
    ) -> bytes:
        cleanup = (
            b"        mov x16, sp\n"
            b"        mov x17, #" + str(pairs).encode() + b"\n" +
            label + b":\n"
            b"        stp xzr, xzr, [x16], #16\n"
            b"        subs x17, x17, #1\n"
            b"        b.ne " + label + b"\n"
        )
        if clear_volatile:
            cleanup += b"".join(
                b"        mov x" + str(register).encode() + b", xzr\n"
                for register in range(18)
            )
        return cleanup

    def clear_frame(
        data: bytes,
        frame: bytes,
        pairs: int,
        label: bytes,
        *,
        clear_volatile: bool = False,
    ) -> bytes:
        needle = b"        add sp, sp, #(" + frame + b") %% .cfi_adjust_cfa_offset -" + frame.split(b" +")[0] + b"\n"
        if data.count(needle) != 1:
            fail(
                f"{entry.local_path} does not contain exactly one expected "
                f"{frame.decode()} frame epilogue"
            )
        cleanup = cleanup_bytes(pairs, label, clear_volatile=clear_volatile)
        return data.replace(needle, cleanup + needle, 1)

    if is_alt:
        frames = (
            (b"31*32 +0", 62, symbol + b"_clear_frame", True),
            (b"160 +0", 10, symbol + b"_clear_inverse_frame", False),
            (b"224 +0", 14, symbol + b"_clear_add_frame", False),
        )
    else:
        frames = (
            (b"31*32 +0", 62, symbol + b"_clear_frame", True),
            (b"160 +0", 10, symbol + b"_clear_inverse_frame", False),
            (b"224 +0", 14, symbol + b"_clear_add_frame", False),
            (b"272 +0", 17, symbol + b"_clear_double_frame", False),
            (b"192 +0", 12, symbol + b"_clear_mixadd_frame", False),
        )
    for frame, pairs, label, clear_volatile in frames:
        output = clear_frame(
            output,
            frame,
            pairs,
            label,
            clear_volatile=clear_volatile,
        )

    if is_alt:
        frame = b"192 +0"
        needle = b"        add sp, sp, #(192 +0) %% .cfi_adjust_cfa_offset -192\n"
        parts = output.split(needle)
        if len(parts) != 3:
            fail(
                f"{entry.local_path} contains {len(parts) - 1} expected "
                "192 +0 frame epilogues; expected 2"
            )
        output = (
            parts[0]
            + cleanup_bytes(12, symbol + b"_clear_double_frame")
            + needle
            + parts[1]
            + cleanup_bytes(12, symbol + b"_clear_mixadd_frame")
            + needle
            + parts[2]
        )

    restore = re.compile(
        rb"(?m)^(?P<indent>[ \t]*)ldp (?P<left>x[0-9]+), (?P<right>x[0-9]+), "
        rb"\[sp\], #16(?P<cfi> %% \.cfi_adjust_cfa_offset -16 %% \.cfi_restore "
        rb"x[0-9]+ %% \.cfi_restore x[0-9]+)$"
    )

    def wipe_saved_registers(match: re.Match[bytes]) -> bytes:
        indent = match.group("indent")
        return (
            indent + b"ldp " + match.group("left") + b", " + match.group("right") + b", [sp]\n"
            + indent + b"stp xzr, xzr, [sp]\n"
            + indent + b"add sp, sp, #16" + match.group("cfi")
        )

    output, replacements = restore.subn(wipe_saved_registers, output)
    expected_replacements = 5 if is_alt else 11
    if replacements != expected_replacements:
        fail(
            f"{entry.local_path} rewrote {replacements} saved-register spills; "
            f"expected {expected_replacements}"
        )
    return output


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


def a64_elf_secret_cleanup(
    output: bytes,
    entry: Entry,
    *,
    frames: tuple[tuple[bytes, int, bytes, bool], ...],
    duplicate_192_labels: tuple[bytes, bytes] | None,
    expected_saved_registers: int,
) -> bytes:
    def cleanup_bytes(
        pairs: int, label: bytes, *, clear_volatile: bool = False
    ) -> bytes:
        cleanup = (
            b"        mov x16, sp\n"
            b"        mov x17, #" + str(pairs).encode() + b"\n" +
            label + b":\n"
            b"        stp xzr, xzr, [x16], #16\n"
            b"        subs x17, x17, #1\n"
            b"        b.ne " + label + b"\n"
        )
        if clear_volatile:
            cleanup += b"".join(
                b"        mov x" + str(register).encode() + b", xzr\n"
                for register in range(18)
            )
        return cleanup

    def frame_epilogue(frame: bytes) -> bytes:
        return (
            b"        add sp, sp, #(" + frame + b")\n"
            b"        .cfi_adjust_cfa_offset -" + frame.split(b" +")[0] + b"\n"
        )

    for frame, pairs, label, clear_volatile in frames:
        needle = frame_epilogue(frame)
        if output.count(needle) != 1:
            fail(
                f"{entry.local_path} does not contain exactly one expected "
                f"{frame.decode()} frame epilogue"
            )
        output = output.replace(
            needle,
            cleanup_bytes(pairs, label, clear_volatile=clear_volatile) + needle,
            1,
        )

    if duplicate_192_labels is not None:
        needle = frame_epilogue(b"192 +0")
        parts = output.split(needle)
        if len(parts) != 3:
            fail(
                f"{entry.local_path} contains {len(parts) - 1} expected "
                "192 +0 frame epilogues; expected 2"
            )
        output = (
            parts[0]
            + cleanup_bytes(12, duplicate_192_labels[0])
            + needle
            + parts[1]
            + cleanup_bytes(12, duplicate_192_labels[1])
            + needle
            + parts[2]
        )

    restore = re.compile(
        rb"(?m)^(?P<indent>[ \t]*)ldp (?P<left>x[0-9]+), (?P<right>x[0-9]+), "
        rb"\[sp\], #16$"
    )

    def wipe_saved_registers(match: re.Match[bytes]) -> bytes:
        indent = match.group("indent")
        return (
            indent + b"ldp " + match.group("left") + b", " + match.group("right") + b", [sp]\n"
            + indent + b"stp xzr, xzr, [sp]\n"
            + indent + b"add sp, sp, #16"
        )

    output, replacements = restore.subn(wipe_saved_registers, output)
    if replacements != expected_saved_registers:
        fail(
            f"{entry.local_path} rewrote {replacements} saved-register spills; "
            f"expected {expected_saved_registers}"
        )
    return output


def a64_elf_ecdsa_clean(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = a64_elf_ecdsa_split(upstream, entry, sources)
    symbol = b".Lrscrypto_p256_scalarmulbase_alt"
    return a64_elf_secret_cleanup(
        output,
        entry,
        frames=(
            (b"9*32 +0", 18, symbol + b"_clear_frame", True),
            (b"160 +0", 10, symbol + b"_clear_inverse_frame", False),
            (b"192 +0", 12, symbol + b"_clear_mixadd_frame", False),
        ),
        duplicate_192_labels=None,
        expected_saved_registers=7,
    )


def a64_elf_p256_ecdh(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = a64_elf_ecdsa_split(upstream, entry, sources)
    symbol = b".Lrscrypto_p256_scalarmul_alt"
    return a64_elf_secret_cleanup(
        output,
        entry,
        frames=(
            (b"31*32 +0", 62, symbol + b"_clear_frame", True),
            (b"160 +0", 10, symbol + b"_clear_inverse_frame", False),
            (b"224 +0", 14, symbol + b"_clear_add_frame", False),
        ),
        duplicate_192_labels=(
            symbol + b"_clear_double_frame",
            symbol + b"_clear_mixadd_frame",
        ),
        expected_saved_registers=5,
    )


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


def x86_elf_secret_cleanup(
    output: bytes,
    entry: Entry,
    *,
    frames: tuple[tuple[bytes, int, int], ...],
    expected_saved_registers: int,
    root_symbol: bytes | None = None,
) -> bytes:
    def cleanup_bytes(qwords: int) -> bytes:
        return b"".join(
            b"        movq $0, " + str(offset).encode() + b"(%rsp)\n"
            for offset in range(0, qwords * 8, 8)
        )

    for frame, qwords, expected_count in frames:
        needle = (
            b"        addq $"
            + frame
            + b", %rsp ; .cfi_adjust_cfa_offset -"
            + frame
            + b"\n"
        )
        actual_count = output.count(needle)
        if actual_count != expected_count:
            fail(
                f"{entry.local_path} contains {actual_count} expected "
                f"{frame.decode()} frame epilogues; expected {expected_count}"
            )
        output = output.replace(needle, cleanup_bytes(qwords) + needle)

    restore = re.compile(
        rb"(?m)^(?P<indent>[ \t]*)popq (?P<register>%r(?:bx|bp|12|13|14|15)) "
        rb"; (?P<cfi>\.cfi_adjust_cfa_offset -8 ; \.cfi_restore %r(?:bx|bp|12|13|14|15))$"
    )

    def wipe_saved_register(match: re.Match[bytes]) -> bytes:
        indent = match.group("indent")
        return (
            indent
            + b"movq (%rsp), "
            + match.group("register")
            + b"\n"
            + indent
            + b"movq $0, (%rsp)\n"
            + indent
            + b"leaq 8(%rsp), %rsp ; "
            + match.group("cfi")
        )

    output, replacements = restore.subn(wipe_saved_register, output)
    if replacements != expected_saved_registers:
        fail(
            f"{entry.local_path} rewrote {replacements} saved-register spills; "
            f"expected {expected_saved_registers}"
        )

    symbol = public_symbols(output)[0] if root_symbol is None else root_symbol
    root_return = (
        b"        retq ; .cfi_endproc\n.size "
        + symbol
        + b", .-"
        + symbol
        + b"\n"
    )
    if output.count(root_return) != 1:
        fail(f"{entry.local_path} does not contain its unique public return")
    clear_volatile = b"".join(
        b"        movl $0, %" + register + b"\n"
        for register in (b"eax", b"ecx", b"edx", b"esi", b"edi", b"r8d", b"r9d", b"r10d", b"r11d")
    )
    return output.replace(root_return, clear_volatile + root_return, 1)


def x86_elf_ecdsa_clean(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = x86_elf_ecdsa_global(upstream, entry, sources)
    is_alt = "scalarmulbase_alt_" in entry.local_path
    return x86_elf_secret_cleanup(
        output,
        entry,
        frames=((b"11*32", 44, 1), (b"240", 30, 1), (b"192", 24, 1)),
        expected_saved_registers=28 if is_alt else 30,
    )


def x86_elf_p256_ecdh(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = x86_elf_ecdsa_global(upstream, entry, sources)
    is_alt = "scalarmul_alt_" in entry.local_path
    return x86_elf_secret_cleanup(
        output,
        entry,
        frames=((b"32*32", 128, 1), (b"240", 30, 1), (b"224", 28, 1), (b"192", 24, 2)),
        expected_saved_registers=43 if is_alt else 46,
    )


def x86_coff_raw(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = upstream.preprocess(
        one_source(sources, entry.transform),
        entry.commit,
        "x86_att",
        "x86_64-pc-windows-msvc",
        line_markers=False,
        strip_comments=False,
        windows_abi=True,
    )
    return global_stem_rename(output, public_symbols(output))


def x86_coff_ecdsa_clean(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = x86_coff_raw(upstream, entry, sources)
    is_alt = "scalarmulbase_alt_" in entry.local_path
    stem = b"rscrypto_p256_scalarmulbase_alt" if is_alt else b"rscrypto_p256_scalarmulbase"
    output = x86_elf_secret_cleanup(
        output,
        entry,
        frames=((b"11*32", 44, 1), (b"240", 30, 1), (b"192", 24, 1)),
        expected_saved_registers=28 if is_alt else 30,
        root_symbol=b"L" + stem + b"_standard",
    )
    output = delete_elf_directives(output).rstrip(b"\n") + b"\n"
    return generated_body(output, entry.transform)


def x86_coff_p256_ecdh(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    output = x86_coff_raw(upstream, entry, sources)
    is_alt = "scalarmul_alt_" in entry.local_path
    stem = b"rscrypto_p256_scalarmul_alt" if is_alt else b"rscrypto_p256_scalarmul"
    output = x86_elf_secret_cleanup(
        output,
        entry,
        frames=((b"32*32", 128, 1), (b"240", 30, 1), (b"224", 28, 1), (b"192", 24, 2)),
        expected_saved_registers=43 if is_alt else 46,
        root_symbol=b"L" + stem + b"_standard",
    )
    output = delete_elf_directives(output).rstrip(b"\n") + b"\n"
    return generated_body(output, entry.transform)


def x86_coff_p256_curve_terms(
    upstream: Upstream, entry: Entry, sources: tuple[bytes, ...]
) -> bytes:
    is_alt = "_alt_" in entry.local_path
    suffix = "_alt" if is_alt else ""
    expected_members = tuple(
        f"x86_att/p256/{operation}{suffix}.S"
        for operation in (
            "bignum_tomont_p256",
            "bignum_montsqr_p256",
            "bignum_montmul_p256",
        )
    )
    if tuple(source.member for source in entry.sources) != expected_members:
        fail(f"{entry.local_path} has a non-canonical P-256 curve-terms source set")

    local_symbols: list[bytes] = []
    bodies: list[bytes] = []
    for source, source_data in zip(entry.sources, sources, strict=True):
        output = upstream.preprocess(
            source_data,
            entry.commit,
            "x86_att",
            "x86_64-pc-windows-msvc",
            line_markers=False,
            strip_comments=False,
            windows_abi=False,
        )
        symbols = public_symbols(output)
        if len(symbols) != 1:
            fail(f"{entry.local_path} source {source.member} exports multiple symbols")
        symbol = symbols[0]
        local_symbol = b".Lrscrypto_" + symbol + b"_raw"
        output = re.sub(
            rb"(?m)^[ \t]*\.globl[ \t]+" + re.escape(symbol) + rb"[ \t]*(?:\n|$)",
            b"",
            output,
        )
        output = re.sub(
            rb"(?<![" + SYMBOL_BYTE + rb"])"
            + re.escape(symbol)
            + rb"(?![" + SYMBOL_BYTE + rb"])",
            local_symbol,
            output,
        )
        output = delete_elf_directives(output).strip(b"\n") + b"\n"
        local_symbols.append(local_symbol)
        bodies.append(output)

    public_symbol = b"rscrypto_p256_curve_terms" + (b"_alt" if is_alt else b"")
    wrapper = b""" .globl %s
        .text
        .p2align 4
%s:
        .cfi_startproc
        .byte 0xf3,0x0f,0x1e,0xfa
        pushq %%rdi ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %%rdi, 0
        pushq %%rsi ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %%rsi, 0
        pushq %%r12 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %%r12, 0
        pushq %%r13 ; .cfi_adjust_cfa_offset 8 ; .cfi_rel_offset %%r13, 0
        subq $40, %%rsp ; .cfi_adjust_cfa_offset 40
        movq %%rcx, %%r12
        movq %%rdx, %%r13
        movq %%r12, %%rdi
        movq %%r13, %%rsi
        callq %s
        leaq 32(%%r12), %%rdi
        leaq 32(%%r13), %%rsi
        callq %s
        leaq 64(%%r12), %%rdi
        leaq 32(%%r12), %%rsi
        callq %s
        movq %%rsp, %%rdi
        movq %%r12, %%rsi
        callq %s
        leaq 96(%%r12), %%rdi
        movq %%rsp, %%rsi
        movq %%r12, %%rdx
        callq %s
        addq $40, %%rsp ; .cfi_adjust_cfa_offset -40
        popq %%r13 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %%r13
        popq %%r12 ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %%r12
        popq %%rsi ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %%rsi
        popq %%rdi ; .cfi_adjust_cfa_offset -8 ; .cfi_restore %%rdi
        retq ; .cfi_endproc
""" % (
        public_symbol,
        public_symbol,
        local_symbols[0],
        local_symbols[0],
        local_symbols[1],
        local_symbols[1],
        local_symbols[2],
    )
    return wrapper + b"\n" + b"\n".join(bodies)


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
    "A64_MACHO_GLOBAL_V1": a64_macho_ecdsa,
    "A64_MACHO_ECDSA_CLEAN_V1": a64_macho_ecdsa_clean,
    "A64_MACHO_P256_ECDH_V1": a64_macho_p256_ecdh,
    "A64_ELF_ECDSA_SPLIT_V1": a64_elf_ecdsa_split,
    "A64_ELF_GLOBAL_V1": a64_elf_ecdsa_split,
    "A64_ELF_ECDSA_CLEAN_V1": a64_elf_ecdsa_clean,
    "A64_ELF_P256_ECDH_V1": a64_elf_p256_ecdh,
    "X86_ELF_ECDSA_GLOBAL_V1": x86_elf_ecdsa_global,
    "X86_ELF_GLOBAL_V1": x86_elf_ecdsa_global,
    "X86_ELF_ECDSA_CLEAN_V1": x86_elf_ecdsa_clean,
    "X86_ELF_P256_ECDH_V1": x86_elf_p256_ecdh,
    "X86_COFF_ECDSA_CLEAN_V1": x86_coff_ecdsa_clean,
    "X86_COFF_P256_ECDH_V1": x86_coff_p256_ecdh,
    "X86_COFF_P256_CURVE_TERMS_V1": x86_coff_p256_curve_terms,
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


WINDOWS_P256_SNAPSHOTS = (
    (
        "src/auth/asm/rscrypto_p256_curve_terms_alt_x86_64_pc_windows_msvc.S",
        "X86_COFF_P256_CURVE_TERMS_V1",
        (
            "x86_att/p256/bignum_tomont_p256_alt.S",
            "x86_att/p256/bignum_montsqr_p256_alt.S",
            "x86_att/p256/bignum_montmul_p256_alt.S",
        ),
    ),
    (
        "src/auth/asm/rscrypto_p256_curve_terms_x86_64_pc_windows_msvc.S",
        "X86_COFF_P256_CURVE_TERMS_V1",
        (
            "x86_att/p256/bignum_tomont_p256.S",
            "x86_att/p256/bignum_montsqr_p256.S",
            "x86_att/p256/bignum_montmul_p256.S",
        ),
    ),
    (
        "src/auth/asm/rscrypto_p256_scalarmul_alt_x86_64_pc_windows_msvc.S",
        "X86_COFF_P256_ECDH_V1",
        ("x86_att/p256/p256_scalarmul_alt.S",),
    ),
    (
        "src/auth/asm/rscrypto_p256_scalarmul_x86_64_pc_windows_msvc.S",
        "X86_COFF_P256_ECDH_V1",
        ("x86_att/p256/p256_scalarmul.S",),
    ),
    (
        "src/auth/asm/rscrypto_p256_scalarmulbase_alt_x86_64_pc_windows_msvc.S",
        "X86_COFF_ECDSA_CLEAN_V1",
        ("x86_att/p256/p256_scalarmulbase_alt.S",),
    ),
    (
        "src/auth/asm/rscrypto_p256_scalarmulbase_x86_64_pc_windows_msvc.S",
        "X86_COFF_ECDSA_CLEAN_V1",
        ("x86_att/p256/p256_scalarmulbase.S",),
    ),
)


def generate_windows_p256(repository: Path, clang: str) -> None:
    with tempfile.TemporaryDirectory(prefix="rscrypto-signature-asm-") as temporary:
        upstream = Upstream(repository, clang, Path(temporary))
        upstream.verify_identity()
        for local_path, transform, members in WINDOWS_P256_SNAPSHOTS:
            source_data = tuple(upstream.blob(PINNED_COMMIT, member) for member in members)
            sources = tuple(
                Source(member, digest(data))
                for member, data in zip(members, source_data, strict=True)
            )
            entry = Entry(
                local_path,
                "0" * 64,
                "0" * 64,
                PINNED_COMMIT,
                transform,
                sources,
            )
            body = TRANSFORMS[transform](upstream, entry, source_data)
            if transform == "X86_COFF_P256_CURVE_TERMS_V1":
                detail = "// - Microsoft x64 batch wrapper around target-native upstream field bodies\n"
            else:
                detail = (
                    "// - Microsoft x64 ABI wrapper plus deterministic secret-frame, "
                    "saved-register-spill, and volatile-register cleanup\n"
                )
            source_lines = "".join(f"// - {member}\n" for member in members)
            header = (
                "// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.\n"
                "// SPDX-License-Identifier: Apache-2.0 OR ISC OR MIT-0\n"
                "//\n"
                "// Adapted for rscrypto from s2n-bignum:\n"
                f"{source_lines}"
                f"{detail}//\n"
                "// The public symbol is renamed to the rscrypto namespace and embedded with Rust global_asm!.\n\n"
            ).encode()
            destination = ROOT / local_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(header + body)


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
    parser.add_argument(
        "--generate-windows-p256",
        action="store_true",
        help="regenerate the fixed Microsoft x64 P-256 snapshot set",
    )
    args = parser.parse_args()

    if args.generate_windows_p256:
        if args.upstream_repo is None:
            fail("--generate-windows-p256 requires --upstream-repo")
        generate_windows_p256(args.upstream_repo, args.clang)
        print(f"signature-asm-provenance: generated {len(WINDOWS_P256_SNAPSHOTS)} Windows P-256 snapshots")
        return

    entries = parse_manifest()
    verify_local(entries)
    if args.upstream_repo is not None:
        verify_upstream(entries, args.upstream_repo, args.clang)
        print(
            f"signature-asm-provenance: {len(entries)} snapshots match "
            "immutable upstream sources and transforms"
        )
    else:
        print(
            f"signature-asm-provenance: {len(entries)} snapshots match the "
            "provenance manifest"
        )


if __name__ == "__main__":
    main()
