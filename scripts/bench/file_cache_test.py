"""Failure gates for file-scoped cache conditioning; no native qualification."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import file_cache


class CacheGates(unittest.TestCase):
    def replay(self, mode, stdout, *, initial=None, code=0, stderr="", fails=False, reads=False, size=4096):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, log = root / "fixture.bin", root / "cache.jsonl"
            fixture.write_bytes(bytes(size))
            result = subprocess.CompletedProcess(["fincore"], code, stdout, stderr)
            observations = [result] if initial is None else [
                subprocess.CompletedProcess(["fincore"], 0, initial, ""), result]
            with fixture.open("rb", buffering=0) as source, \
                 patch.object(file_cache.sys, "platform", "linux"), \
                 patch.object(file_cache.sys, "argv", ["cache", mode, str(fixture), str(log), "case"]), \
                 patch.object(file_cache.os, "sysconf", return_value=4096), \
                 patch.object(file_cache.os, "POSIX_FADV_DONTNEED", 4, create=True), \
                 patch.object(file_cache.os, "posix_fadvise", create=True) as advise, \
                 patch.object(file_cache.subprocess, "run", side_effect=observations) as run:
                stream = MagicMock(wraps=source)
                stream.__enter__.return_value = stream
                with patch.object(file_cache, "open", return_value=stream, create=True):
                    if fails:
                        with self.assertRaises((RuntimeError, ValueError, subprocess.CalledProcessError)):
                            file_cache.main()
                    else:
                        file_cache.main()
                self.assertEqual(stream.read.call_count, 2 if reads else 0)
                self.assertEqual(advise.call_count, int(mode == "cold"))
                self.assertEqual(run.call_count, len(observations))
            records = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["passed"], not fails)
            self.assertEqual(record["fincore_stdout"], stdout)
            self.assertEqual(record["fincore_stderr"], stderr)
            self.assertEqual(record["conditioned_bytes"], size if reads else 0)
            if initial is not None:
                self.assertEqual(record["initial_fincore_stdout"], initial)
            return record

    def test_cold_requires_zero_pages_after_advice(self):
        self.replay("cold", "0 0\n")
        self.replay("cold", "4096 1\n", fails=True)

    def test_warm_requires_complete_residency(self):
        self.replay("warm", "4096 1\n", initial="0 0\n", reads=True)
        self.replay("warm", "0 0\n", initial="0 0\n", reads=True, fails=True)

    def test_resident_warm_file_is_not_reread(self):
        record = self.replay("warm", "4096 1\n")
        self.assertEqual(record["warm_action"], "already-resident")

    def test_partial_residency_requires_one_complete_read(self):
        record = self.replay("warm", "8192 2\n", initial="4096 1\n", reads=True, size=8192)
        self.assertEqual(record["warm_action"], "read")

    def test_warm_collector_failure_after_read_is_retained(self):
        record = self.replay("warm", "partial output", initial="0 0\n", code=1,
                             stderr="injected failure", reads=True, fails=True)
        self.assertEqual(record["initial_fincore_exit_code"], 0)
        self.assertEqual(record["fincore_exit_code"], 1)

    def test_inconsistent_page_and_byte_counts_fail(self):
        self.replay("warm", "4096 0\n", fails=True)
        self.replay("warm", "8192 2\n", fails=True)
        self.replay("warm", "-4096 -1\n", fails=True)

    def test_collector_failure_retains_both_streams(self):
        record = self.replay("cold", "partial output", code=1, stderr="injected failure", fails=True)
        self.assertEqual(record["fincore_exit_code"], 1)
        self.assertIn("CalledProcessError", record["error"])

    def test_malformed_output_is_retained_and_fails(self):
        self.replay("cold", "not numeric\n", fails=True)


if __name__ == "__main__":
    unittest.main()
