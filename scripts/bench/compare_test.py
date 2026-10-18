#!/usr/bin/env -S uv run python
"""Check comparison pairing, validation, and scoring on synthetic retained runs."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_catalog import CatalogError, load_catalog, validate_catalog
import compare

HOST = {"system": "Linux", "machine": "x86_64", "cpus": 8, "cpu": ["model name\t: fixture"]}
CATALOG = {
  "benches": {
    "hashes": {"binary": "hashes", "kind": "criterion"},
    "hashes_v4": {"binary": "hashes", "kind": "criterion", "architectures": ["x86_64"],
                  "rustflags": ["-C", "target-cpu=x86-64-v4"]},
  },
  "case_classes": {"hashes": {"diagnostic": ["/internal/"]}},
  "comparisons": {
    "digest": {
      "family": "Digest", "bench": "hashes", "headline": True, "task": "Hash one buffer.",
      "cases": r"^(?P<variant>d1|d2)/(?P<implementation>[^/]+)/(?P<input>[0-9]+)$",
      "competitor_benches": ["hashes_v4"],
    },
    "stream": {
      "family": "Digest", "bench": "hashes", "headline": False, "task": "Hash through updates.",
      "cases": r"^d1/streaming/(?P<implementation>[^/]+)/(?P<input>[0-9]+B)$",
    },
  },
}


def record(median: float, spread: float = 1.0) -> dict:
  return {"median": median, "lower": median - spread, "upper": median + spread}


def write_run(root: Path, entries: dict[str, dict[str, float]], host=HOST, state="complete", quick=False) -> Path:
  """Write a completed run in the layout that `just bench` retains: one plan entry per build."""
  root.mkdir(parents=True)
  plan = []
  for index, (bench, cases) in enumerate(entries.items()):
    home = f"criterion/hashes-{index}"
    for case, median in cases.items():
      estimate = {"point_estimate": median, "standard_error": 0.1,
                  "confidence_interval": {"confidence_level": 0.95, "lower_bound": median - 1, "upper_bound": median + 1}}
      count = 2 if quick else 10
      files = {"benchmark.json": {"full_id": case}, "sample.json": {"iters": [1.0] * count, "times": [median] * count},
               "estimates.json": {"mean": estimate, "median": estimate}}
      for name in ("new", "base"):
        directory = root / home / case / name
        directory.mkdir(parents=True)
        for filename, contents in files.items():
          (directory / filename).write_text(json.dumps(contents))
    plan.append({"id": str(index), "bench": bench, "binary": "hashes", "home": home, "cases": list(cases),
                 "baselines": [], "settings": {"quick": True} if quick else {"sample_size": 10},
                 "compatibility": {"host": host},
                 "artifact": {"command": ["cargo", "bench", "--features", "std", "--bench", "hashes"]}})
  (root / "plan.json").write_text(json.dumps(plan))
  (root / "status.txt").write_text(f"state={state}\nexit_code=0\n")
  return root


class PairingTests(unittest.TestCase):
  def test_fastest_external_across_builds_and_ignored_variants(self):
    records = {
      ("hashes", "d1/rscrypto/64"): record(10.0),
      ("hashes", "d1/rscrypto-portable/64"): record(1.0),  # a variant never enters a ratio
      ("hashes", "d1/lib-a/64"): record(30.0),
      ("hashes", "d1/lib-b/64"): record(14.0),
      ("hashes_v4", "d1/rscrypto/64"): record(2.0),  # rscrypto counts only from the comparison's target
      ("hashes_v4", "d1/lib-a/64"): record(12.0),
    }
    rows, problems = compare.points(CATALOG, "x86", records)
    self.assertEqual(problems, [])
    self.assertEqual(len(rows), 1)
    row = rows[0]
    self.assertEqual((row["variant"], row["input"], row["external"]), ("d1", "64", "lib-a@hashes_v4"))
    self.assertAlmostEqual(row["ratio"], 1.2)
    self.assertAlmostEqual(row["ratio_lower"], 11.0 / 11.0)
    self.assertAlmostEqual(row["ratio_upper"], 13.0 / 9.0)
    self.assertEqual(row["class"], "faster")
    self.assertEqual(row["externals"], ["lib-a", "lib-a@hashes_v4", "lib-b"])

  def test_incomplete_points_are_errors(self):
    records = {
      ("hashes", "d1/rscrypto/64"): record(10.0),
      ("hashes", "d2/lib-a/64"): record(10.0),
      ("hashes", "d2/lib-b/64"): record(10.0),
    }
    rows, problems = compare.points(CATALOG, "x86", records)
    self.assertEqual(rows, [])
    self.assertEqual(len(problems), 2)
    self.assertIn("digest d1 64 needs one rscrypto row and an external row; found 1 and 0", problems[0])
    self.assertIn("digest d2 64 needs one rscrypto row and an external row; found 0 and 2", problems[1])

  def test_classification_band(self):
    self.assertEqual([compare.classify(value) for value in (0.94, 0.95, 1.0, 1.05, 1.06)],
                     ["slower", "tie", "tie", "tie", "faster"])

  def test_undeclared_comparison_groups(self):
    cases = {"hashes": {
      "d1/rscrypto/64", "d1/lib-a/64",  # declared
      "d3/rscrypto/64", "d3/lib-a/64",  # undeclared comparison
      "d4/rscrypto/64", "d4/rscrypto-scalar/64",  # rscrypto variants only
      "d5/rscrypto/64",  # rscrypto only
      "d6/internal/rscrypto/64", "d6/internal/lib-a/64",  # diagnostic
    }}
    self.assertEqual(compare.undeclared(CATALOG, cases), ["hashes: d3/rscrypto/64 has external rows but no catalog comparison"])

  def test_listing_validation(self):
    listing = "\n".join([
      "[ordinary] hashes d1/rscrypto/64", "[ordinary] hashes d1/lib-a/64",
      "[ordinary] hashes d2/rscrypto/64",
      "unrelated output",
    ])
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "list.txt"
      path.write_text(listing)
      problems = compare.validate_cases(CATALOG, compare.listed_cases(path))
    self.assertEqual(problems, ["listing: digest d2 64 needs one rscrypto row and an external row; found 1 and 0",
                                "stream matches no hashes case"])


class RunTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory(prefix="rscrypto-compare-test-")
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)

  def test_scores_runs_by_platform(self):
    first = write_run(self.root / "amd", {"hashes": {"d1/rscrypto/64": 10.0, "d1/lib-a/64": 20.0,
                                                     "d1/streaming/rscrypto/64B": 10.0, "d1/streaming/lib-a/64B": 9.0}})
    second = write_run(self.root / "amd-v4", {"hashes_v4": {"d1/rscrypto/64": 1.0, "d1/lib-a/64": 15.0}})
    arm = write_run(self.root / "arm", {"hashes": {"d1/rscrypto/64": 10.0, "d1/lib-a/64": 5.0}},
                    host=HOST | {"machine": "aarch64"})
    output = self.root / "points.json"
    with mock.patch.object(compare, "load_catalog", return_value=CATALOG), \
         contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()) as stderr:
      status = compare.main([f"amd={first}", f"amd={second}", f"arm={arm}", "--json", str(output)])
    self.assertEqual(status, 0, stderr.getvalue())
    points = json.loads(output.read_text())["points"]
    self.assertEqual([(row["platform"], row["comparison"], row["ratio"]) for row in points],
                     [("amd", "digest", 1.5), ("amd", "stream", 0.9), ("arm", "digest", 0.5)])
    report = stdout.getvalue()
    self.assertIn("| amd | 1 | 1 | 0 | 0 | 1.50x | 1.50x |", report)
    self.assertIn("| arm | 1 | 0 | 0 | 1 | 0.50x | 0.50x |", report)
    self.assertIn("| Digest | 0.90x (1) | - |", report)

  def test_rejects_mixed_hosts_incomplete_runs_and_duplicates(self):
    first = write_run(self.root / "one", {"hashes": {"d1/rscrypto/64": 10.0, "d1/lib-a/64": 20.0}})
    other_host = write_run(self.root / "two", {"hashes_v4": {"d1/lib-a/64": 20.0}}, host=HOST | {"cpus": 16})
    failed = write_run(self.root / "three", {"hashes": {"d1/rscrypto/64": 10.0}}, state="failed")
    duplicate = write_run(self.root / "four", {"hashes": {"d1/rscrypto/64": 10.0}})
    for runs, message in (([first, other_host], "different hosts"), ([failed], "not a completed run"),
                          ([first, duplicate], "more than one run")):
      with self.subTest(message=message), self.assertRaisesRegex(compare.ComparisonError, message):
        compare.load_platform(CATALOG, "x86", runs)

  def test_quick_runs_report_coverage_without_performance_rankings(self):
    entries = {"hashes": {"d1/rscrypto/64": 10.0, "d1/lib-a/64": 100.0},
               "hashes_v4": {"d1/lib-a/64": 90.0}}
    quick = write_run(self.root / "quick", entries, quick=True)
    regular = write_run(self.root / "regular", entries)
    output = self.root / "quick.json"
    with mock.patch.object(compare, "load_catalog", return_value=CATALOG), \
         contextlib.redirect_stdout(io.StringIO()) as stdout:
      self.assertEqual(compare.main([f"x86={quick}", "--json", str(output)]), 0)
      self.assertEqual(json.loads(output.read_text()), {
        "mode": "quick", "partial": False,
        "coverage": [{"platform": "x86", "cases": 3, "comparison_points": 1}],
      })
      self.assertIn("Quick benchmark validation", stdout.getvalue())
      self.assertNotIn("Faster", stdout.getvalue())
      with self.assertRaisesRegex(compare.ComparisonError, "cannot combine quick and regular"):
        compare.main([f"quick={quick}", f"regular={regular}"])
    # Quick mode still requires every competitor build in a comparison.
    missing = write_run(self.root / "missing", {"hashes": entries["hashes"]}, quick=True)
    with mock.patch.object(compare, "load_catalog", return_value=CATALOG), \
         contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as stderr:
      self.assertEqual(compare.main([f"x86={missing}"]), 1)
      self.assertIn("needs the competitor build hashes_v4", stderr.getvalue())

  def test_missing_competitor_build_and_partial_runs(self):
    # x86-64 runs must include the competitor's x86-64-v4 build, or the fastest external is understated.
    run = write_run(self.root / "x86", {"hashes": {"d1/rscrypto/64": 10.0, "d1/lib-a/64": 20.0, "d2/rscrypto/64": 1.0}})
    for partial in (False, True):
      with self.subTest(partial=partial), mock.patch.object(compare, "load_catalog", return_value=CATALOG), \
           contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as stderr:
        self.assertEqual(compare.main([f"x86={run}", *(["--partial"] if partial else [])]), 1)
        self.assertIn("digest needs the competitor build hashes_v4", stderr.getvalue())
    filtered = write_run(self.root / "filtered", {"hashes": {"d1/streaming/rscrypto/64B": 10.0}})
    with mock.patch.object(compare, "load_catalog", return_value=CATALOG), \
         contextlib.redirect_stdout(io.StringIO()) as stdout, contextlib.redirect_stderr(io.StringIO()) as stderr:
      self.assertEqual(compare.main([f"x86={filtered}", "--partial"]), 0, stderr.getvalue())
    self.assertIn("1 comparison points are incomplete", stdout.getvalue())

  def test_undeclared_measured_group_fails_scoring(self):
    run = write_run(self.root / "run", {"hashes": {"d1/rscrypto/64": 10.0, "d1/lib-a/64": 20.0,
                                                   "d9/rscrypto/64": 1.0, "d9/lib-a/64": 1.0}})
    with mock.patch.object(compare, "load_catalog", return_value=CATALOG), \
         contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as stderr:
      self.assertEqual(compare.main([f"x86={run}"]), 1)
    self.assertIn("d9/rscrypto/64 has external rows but no catalog comparison", stderr.getvalue())


class CatalogTests(unittest.TestCase):
  def test_comparison_declarations_are_validated(self):
    catalog = load_catalog()
    for change in ({"headline": "yes"}, {"family": " "}, {"bench": "missing"}, {"competitor_benches": ["structural"]},
                   {"cases": "^(?P<implementation>x)$"}, {"cases": "("}, {"cases": "^(?P<impl>x)/(?P<input>y)$"},
                   {"extra": True}):
      broken = copy.deepcopy(catalog)
      broken["comparisons"]["sha2"].update(change)
      with self.subTest(change=change), self.assertRaises(CatalogError):
        validate_catalog(broken)

  def test_architecture_and_rustflags_fields_are_validated(self):
    catalog = load_catalog()
    for change in ({"architectures": []}, {"architectures": ["x86"]}, {"architectures": ["x86_64", "x86_64"]},
                   {"rustflags": ["-C", "target-cpu=x86-64-v4"]}, {"architectures": ["x86_64"], "rustflags": []},
                   {"architectures": ["x86_64"], "rustflags": [" "]}):
      broken = copy.deepcopy(catalog)
      broken["benches"]["sha2"].update(change)
      with self.subTest(change=change), self.assertRaises(CatalogError):
        validate_catalog(broken)


if __name__ == "__main__":
  unittest.main()
