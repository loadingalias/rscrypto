#!/usr/bin/env python3
"""Select native CT evidence lanes."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from ci_platforms import platforms
from cross_build import TARGETS


def main():
    if sys.argv[1:] == ['plan']:
        matrix = platforms(os.environ['INPUT_ARCHITECTURES'], os.environ['GITHUB_RUN_ID'])
        for row in matrix['include']:
            row['timeout'] = 360
        builds = []
        for row in matrix['include']:
            target = row['platform'].removesuffix('-linux') + '-unknown-linux-gnu'
            if row['platform'] == 'riscv64-linux':
                target = 'riscv64gc-unknown-linux-gnu'
            if target in TARGETS:
                row['target'] = target
                builds.append({'target': target})
        with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
            output.write('matrix=' + json.dumps(matrix, separators=(',', ':')) + '\n')
            output.write('cross=' + str(bool(builds)).lower() + '\n')
            output.write('builds=' + json.dumps({'include': builds}, separators=(',', ':')) + '\n')
        return
    raise ValueError('usage: scripts/ct/ci.py plan')


if __name__ == '__main__':
    main()
