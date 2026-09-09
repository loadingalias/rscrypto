#!/usr/bin/env python3
"""Validate CT runner selection and execute one native evidence lane."""
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from ci_platforms import platforms


def main():
    if sys.argv[1:] == ['plan']:
        matrix = platforms(os.environ['INPUT_ARCHITECTURES'], os.environ['GITHUB_RUN_ID'])
        for row in matrix['include']:
            row['timeout'] = 360
        with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
            output.write('matrix=' + json.dumps(matrix, separators=(',', ':')) + '\n')
        return
    if sys.argv[1:] != ['run']:
        raise ValueError('usage: scripts/ct/ci.py plan|run')
    if os.environ['PLATFORM'] == 'x86_64-linux':
        subprocess.run(['just', 'ct-test'], check=True)
    # Build/proof work finishes before timing; cases never compete on one host.
    subprocess.run(['just', 'ct-full'], check=True)


if __name__ == '__main__':
    main()
