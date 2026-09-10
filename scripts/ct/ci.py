#!/usr/bin/env python3
"""Select native CT evidence lanes."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from ci_platforms import platforms


def main():
    if sys.argv[1:] == ['plan']:
        matrix = platforms(os.environ['INPUT_ARCHITECTURES'], os.environ['GITHUB_RUN_ID'])
        for row in matrix['include']:
            row['timeout'] = 60 if row['platform'] == 'riscv64-linux' else 360
        with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
            output.write('matrix=' + json.dumps(matrix, separators=(',', ':')) + '\n')
            output.write('riscv=' + str(any(row['platform'] == 'riscv64-linux' for row in matrix['include'])).lower() + '\n')
        return
    raise ValueError('usage: scripts/ct/ci.py plan')


if __name__ == '__main__':
    main()
