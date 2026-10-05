#!/usr/bin/env python3
"""Reject incomplete or contradictory production backend-execution evidence."""

import copy
import json
import unittest

from evidence_suite import BACKEND_MARKER, validate_backend_evidence


def records(dispatch="production-auto", availability=(True, True)):
    compiled = [{"id": name, "required_features": [name], "runtime_available": available}
                for name, available in zip(("avx2", "avx512"), availability)]
    executed = [entry["id"] for entry in compiled if entry["runtime_available"]]
    result = ("not-selected" if dispatch == "portable-only" else
              "pass" if executed else "unavailable" if compiled else "not-compiled")
    return [{"schema": 1, "kind": "rscrypto.backend-execution", "primitive": "chacha20",
             "test": name, "dispatch": dispatch, "target_arch": "x86_64", "compiled": copy.deepcopy(compiled),
             "executed_backend_ids": executed.copy(), "executed_case_count": len(executed),
             "kernel_call_count": len(executed) * calls, "result": result}
            for name, calls in (("counter-zero", 1), ("arbitrary-counters", 1), ("self-inverse", 2))]


def validate(rows, dispatch="production-auto"):
    output = "\n".join("nextest output: " + BACKEND_MARKER + json.dumps(row) for row in rows)
    return validate_backend_evidence(output, dispatch)


class BackendEvidenceTests(unittest.TestCase):
    def test_complete_and_explicitly_unavailable_execution(self):
        for available in ((True, True), (True, False), (False, False), ()):
            with self.subTest(available=available):
                rows = records(availability=available)
                self.assertEqual(validate(rows), rows)
        rows = records("portable-only", (None, None))
        self.assertEqual(validate(rows, "portable-only"), rows)

    def test_every_available_backend_must_execute_once_in_the_inventory(self):
        for executed in (["avx2"], ["avx2", "avx2"], [], ["avx2", "avx512", "unknown"]):
            with self.subTest(executed=executed):
                rows = records()
                for row in rows:
                    row["executed_backend_ids"] = executed
                with self.assertRaisesRegex(ValueError, "backend"):
                    validate(rows)

    def test_cpu_inventory_must_agree_across_tests(self):
        for field in ("target_arch", "compiled"):
            with self.subTest(field=field):
                rows = records()
                if field == "target_arch":
                    rows[1][field] = "aarch64"
                else:
                    rows[1][field][0]["required_features"] = ["different-capability"]
                with self.assertRaisesRegex(ValueError, "inventory differs"):
                    validate(rows)

    def test_reject_malformed_records_and_counters(self):
        for field, values in {
            "executed_case_count": (True, False, -1, 0, 1, 2.0, "2", None),
            "kernel_call_count": (True, -1, 0, 1, 2.0, "2", None),
            "executed_backend_ids": ([{}], [None], "avx2", [""]),
            "compiled": ([None], [{}], [{"id": "avx2"}]),
            "target_arch": (None, ""),
            "test": ([], None, "unknown"),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    rows = records()
                    rows[0][field] = value
                    with self.assertRaises(ValueError):
                        validate(rows)
        for value in (None, [], "record"):
            with self.subTest(record=value), self.assertRaises(ValueError):
                validate([value, *records()[1:]])

    def test_reject_invalid_compiled_inventory(self):
        for field, value in (("id", ""), ("required_features", [False]), ("runtime_available", 1)):
            with self.subTest(field=field):
                rows = records()
                rows[0]["compiled"][0][field] = value
                with self.assertRaisesRegex(ValueError, "compiled"):
                    validate(rows)
        rows = records()
        rows[0]["compiled"][1] = rows[0]["compiled"][0]
        with self.assertRaises(ValueError):
            validate(rows)

    def test_reject_missing_tests_and_false_status(self):
        for rows in (records()[:2], [records()[0]] * 3):
            with self.assertRaises(ValueError):
                validate(rows)
        rows = records(availability=(False, False))
        for row in rows:
            row["result"] = "pass"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate(rows)
        with self.assertRaisesRegex(ValueError, "unknown.*dispatch"):
            validate(records("unreviewed"), "unreviewed")


if __name__ == "__main__":
    unittest.main()
