"""Shared scope for internal production regressions on native and transferred test lanes."""

import json
from pathlib import Path
import subprocess
import sys

TARGET_ARGS = [
    "--lib", "--test", "aead_kernel_equivalence", "--test", "argon2_kernels",
    "--test", "chacha20poly1305", "--test", "rsa_public_key", "--test", "pbkdf2_evidence",
]
BACKEND_MARKER = "RSCRYPTO_BACKEND_EVIDENCE="
BACKEND_TESTS = {"counter-zero", "arbitrary-counters", "self-inverse"}


def validate_backend_evidence(output, dispatch):
    if dispatch not in {"production-auto", "portable-only"}:
        raise ValueError(f"unknown ChaCha20 dispatch profile: {dispatch}")
    records = []
    for line in output.splitlines():
        _, separator, payload = line.partition(BACKEND_MARKER)
        if separator:
            records.append(json.loads(payload))
    if len(records) != len(BACKEND_TESTS):
        raise ValueError(f"expected {len(BACKEND_TESTS)} ChaCha20 backend records, found {len(records)}")
    if any(not isinstance(record, dict) or not isinstance(record.get("test"), str) for record in records):
        raise ValueError("invalid ChaCha20 backend record")
    if {record["test"] for record in records} != BACKEND_TESTS:
        raise ValueError("ChaCha20 backend records do not cover the required differential tests")
    inventory = None
    for record in records:
        if (record.get("schema"), record.get("kind"), record.get("primitive"), record.get("dispatch")) != (
            1, "rscrypto.backend-execution", "chacha20", dispatch
        ):
            raise ValueError(f"invalid ChaCha20 backend record: {record}")
        cases = record.get("executed_case_count")
        calls = record.get("kernel_call_count")
        executed = record.get("executed_backend_ids")
        compiled = record.get("compiled")
        if (type(cases) is not int or type(calls) is not int or cases < 0 or calls < 0
                or not isinstance(executed, list) or not isinstance(compiled, list)
                or not isinstance(record.get("target_arch"), str) or not record["target_arch"]):
            raise ValueError(f"invalid ChaCha20 execution counts: {record}")
        if any(not isinstance(identifier, str) or not identifier for identifier in executed) \
                or len(set(executed)) != len(executed):
            raise ValueError(f"invalid executed ChaCha20 backend IDs: {record}")
        if any(not isinstance(entry, dict)
               or set(entry) != {"id", "required_features", "runtime_available"}
               or not isinstance(entry["id"], str) or not entry["id"]
               or not isinstance(entry["required_features"], list)
               or any(not isinstance(feature, str) or not feature for feature in entry["required_features"])
               or (entry["runtime_available"] is not None and not isinstance(entry["runtime_available"], bool))
               for entry in compiled):
            raise ValueError(f"invalid compiled ChaCha20 backend record: {record}")
        compiled_by_id = {entry["id"]: entry for entry in compiled}
        if len(compiled_by_id) != len(compiled):
            raise ValueError(f"duplicate compiled ChaCha20 backend: {record}")
        current_inventory = (record["target_arch"], compiled_by_id)
        if inventory is not None and current_inventory != inventory:
            raise ValueError("ChaCha20 backend inventory differs between differential tests")
        inventory = current_inventory
        available = {identifier for identifier, entry in compiled_by_id.items() if entry["runtime_available"]}
        if set(executed) != available:
            raise ValueError(f"executed ChaCha20 backends differ from available backends: {record}")
        result = record.get("result")
        if dispatch == "portable-only":
            valid_result = (result == "not-selected" and not executed and cases == 0 and calls == 0
                            and all(entry["runtime_available"] is None for entry in compiled))
        elif any(entry["runtime_available"] is None for entry in compiled):
            valid_result = False
        elif result == "pass":
            valid_result = bool(executed and cases >= len(executed) and calls >= cases)
        elif result == "unavailable":
            valid_result = bool(compiled and not any(entry["runtime_available"] for entry in compiled)
                                and not executed and cases == 0 and calls == 0)
        elif result == "not-compiled":
            valid_result = not compiled and not executed and cases == 0 and calls == 0
        else:
            valid_result = False
        if not valid_result:
            raise ValueError(f"inconsistent ChaCha20 backend result: {record}")
    return records


def main():
    root = Path(__file__).resolve().parents[2]
    for dispatch in ("--native", "--portable"):
        result = subprocess.run(["just", "--justfile", str(root / "justfile"), "test", "--release", dispatch,
                                 "--", *TARGET_ARGS], cwd=root, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        sys.stdout.write(result.stdout)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, result.args)
        validate_backend_evidence(result.stdout, "production-auto" if dispatch == "--native" else "portable-only")


if __name__ == "__main__":
    main()
