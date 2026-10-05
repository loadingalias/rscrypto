#!/usr/bin/env python3
"""Nextest target runner that separates discovery from measured test execution."""

import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    directory = Path(sys.argv[1])
    command = sys.argv[2:]
    name = os.environ.get('NEXTEST_TEST_NAME')
    if name is None:
        destination = directory / 'discovery'
        destination.mkdir(exist_ok=True)
    else:
        destination = Path(tempfile.mkdtemp(prefix='test-', dir=directory))
    env = os.environ.copy()
    env['LLVM_PROFILE_FILE'] = str(destination / '%p-%m.profraw')
    if name is not None:
        env['RSCRYPTO_COVERAGE_INPUTS'] = str(destination / 'corpus-inputs')
        (destination / 'execution.json').write_text(json.dumps({
            'binary_id': env['NEXTEST_BINARY_ID'],
            'test': name,
            'binary': str(Path(command[0]).resolve()),
        }) + '\n')
    # Nextest owns exit status and timeouts. On Unix, replace the launcher rather
    # than keeping a Python interpreter resident for every running test.
    os.execve(command[0], command, env)


if __name__ == '__main__':
    main()
