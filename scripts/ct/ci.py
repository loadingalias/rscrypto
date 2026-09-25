#!/usr/bin/env python3
"""Select native CT evidence lanes."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from ci_platforms import platforms
from cross_build import TARGETS
from manifest import primitive_supports_physical_timing, replay_cases
from provenance import load_toml


def main():
    if sys.argv[1:] == ['plan']:
        case_name = os.environ.get('DIAGNOSTIC_CASE', '')
        selected_primitives = []
        if case_name:
            ct = load_toml(Path(__file__).resolve().parents[2] / 'ct.toml')
            cases = {row['name']: row for row in ct['dudect_case']}
            primitives = {row['id']: row for row in ct['primitive']}
            selected_primitives = [primitives[cases[name]['primitive']]
                                   for name in replay_cases(cases, case_name)]
        matrix = platforms(os.environ['INPUT_ARCHITECTURES'], os.environ['GITHUB_RUN_ID'], runner_prefix='ct')
        for row in matrix['include']:
            row['timeout'] = 360
        builds = []
        for row in matrix['include']:
            target = row['platform'].removesuffix('-linux') + '-unknown-linux-gnu'
            if row['platform'] == 'riscv64-linux':
                target = 'riscv64gc-unknown-linux-gnu'
            if case_name:
                if target not in TARGETS:
                    raise ValueError('CT diagnostic replay requires POWER, IBM Z, or RISC-V Linux')
                if any(not primitive_supports_physical_timing(primitive, target)
                       for primitive in selected_primitives):
                    raise ValueError(f'CT diagnostic case {case_name} does not support {target}')
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
