"""Condition one owned Linux file and retain its page-residency precondition."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


def residency(filename, record, prefix, size, page_size):
    result = subprocess.run(
        ["fincore", "--bytes", "--raw", "--noheadings", "--output", "RES,PAGES", filename],
        text=True, capture_output=True, check=False,
    )
    record.update({prefix + "fincore_stdout": result.stdout,
                   prefix + "fincore_stderr": result.stderr,
                   prefix + "fincore_exit_code": result.returncode})
    result.check_returncode()
    resident, pages = map(int, result.stdout.split())
    if not 0 <= resident <= size or pages * page_size != resident:
        raise RuntimeError("inconsistent file residency counts")
    return resident, pages


def main():
    mode, filename, logfile, case = sys.argv[1:]
    record = {"mode": mode, "path": filename, "case": case,
              "started_ns": time.monotonic_ns(), "passed": False}
    try:
        if sys.platform != "linux" or mode not in {"cold", "warm"}:
            raise ValueError("requires Linux and an explicit cold/warm mode")
        with open(filename, "rb", buffering=0) as stream:
            size = os.fstat(stream.fileno()).st_size
            page_size = os.sysconf("SC_PAGESIZE")
            if size == 0 or size % page_size:
                raise ValueError("fixture must contain complete pages")
            record.update(size=size, conditioned_bytes=0)
            if mode == "cold":
                # This is only advice. The independent residency check below
                # must succeed; never retry until the desired state appears.
                os.posix_fadvise(stream.fileno(), 0, size, os.POSIX_FADV_DONTNEED)
                resident, pages = residency(filename, record, "", size, page_size)
            else:
                resident, pages = residency(filename, record, "", size, page_size)
                record["warm_action"] = "already-resident" if resident == size else "read"
                if resident != size:
                    for key in ["fincore_stdout", "fincore_stderr", "fincore_exit_code"]:
                        record["initial_" + key] = record.pop(key)
                    while block := stream.read(1024 * 1024):
                        record["conditioned_bytes"] += len(block)
                    resident, pages = residency(filename, record, "", size, page_size)
            record.update(resident_bytes=resident, resident_pages=pages)
            expected = 0 if mode == "cold" else size
            if resident != expected or pages * page_size != expected:
                raise RuntimeError(f"{mode} residency differs: {resident} bytes, expected {expected}")
            record["passed"] = True
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["finished_ns"] = time.monotonic_ns()
        with Path(logfile).open("a") as output:
            output.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
