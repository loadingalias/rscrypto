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
    mode = os.environ.get('INPUT_MODE', 'smoke')
    if mode not in ('smoke', 'full'):
        raise ValueError('CT mode must be smoke or full')
    if sys.argv[1:] == ['plan']:
        matrix = platforms(os.environ['INPUT_ARCHITECTURES'], os.environ['GITHUB_RUN_ID'])
        for row in matrix['include']:
            row['timeout'] = 360 if mode == 'full' else 90
        with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
            output.write('matrix=' + json.dumps(matrix, separators=(',', ':')) + '\n')
        return
    if sys.argv[1:] != ['run']:
        raise ValueError('usage: scripts/ct/ci.py plan|run')
    if os.environ['PLATFORM'] == 'x86_64-linux':
        subprocess.run(['just', 'ct-test'], check=True)
    commands = [['ct-full']] if mode == 'full' else [
        ['ct-artifacts'], ['ct-validate'], ['ct-dudect', '--smoke'],
        ['ct-validate', '--manifest-only', '--strict-coverage'],
    ]
    # Build/proof work finishes before timing; cases never compete on one host.
    for command in commands:
        subprocess.run(['just', *command], check=True)


if __name__ == '__main__':
    main()
