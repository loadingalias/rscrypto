"""Prepare with rustdoc, then execute its unchanged programs on the target.

NIGHTLY: extraction and persistence use the repository's pinned
nightly. Unknown extraction schemas or incomplete runtime inventories fail closed.
"""

import json
import re
import subprocess


def plan_tests(inventory, directory):
    if inventory.get('format_version') != 2 or not inventory.get('doctests'):
        raise ValueError('unsupported or empty rustdoc inventory')
    programs, occurrences, names = [], {}, set()
    expected_files = set()
    for test in inventory['doctests']:
        name, attrs = test['name'], test['doctest_attributes']
        if name in names or attrs['ignore'] != 'None':
            raise ValueError(f'duplicate or ignored doctest requires review: {name}')
        names.add(name)
        key = (test['file'], test['line'])
        count = occurrences.get(key, 0)
        occurrences[key] = count + 1
        identifier = re.sub('[^a-zA-Z0-9]', '_', test['file']) + f'_{test["line"]}_{count}'
        binary = directory / identifier / 'rust_out'
        if attrs['compile_fail']:
            if binary.exists():
                raise ValueError(f'compile-fail doctest unexpectedly has an executable: {name}')
            continue
        expected_files.add(identifier + '/rust_out')
        if not binary.is_file():
            raise ValueError(f'missing compiled doctest: {name}')
        if not attrs['no_run']:
            programs.append({'binary': binary.relative_to(directory).as_posix(),
                             'name': name, 'should_panic': attrs['should_panic']})
    if {p.relative_to(directory).as_posix() for p in directory.rglob('rust_out')} != expected_files:
        raise ValueError('unclassified rustdoc executable')
    return {'total': len(names), 'programs': programs,
            'compile_fail': sum(t['doctest_attributes']['compile_fail'] for t in inventory['doctests']),
            'no_run': sum(t['doctest_attributes']['no_run'] and not t['doctest_attributes']['compile_fail']
                          for t in inventory['doctests'])}


def prepare(root, directory, cargo_args, environment):
    directory.mkdir(parents=True, exist_ok=False)
    flags = ['-D', 'warnings', '-Z', 'unstable-options']
    command = ['cargo', 'test', '--locked', '--workspace', '--release', '--doc', *cargo_args]
    extracted = subprocess.check_output(command, cwd=root, text=True,
        env={**environment, 'CARGO_ENCODED_RUSTDOCFLAGS': '\x1f'.join([*flags, '--output-format', 'doctest'])})
    inventory = json.loads(extracted)
    (directory / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    # Compilation still checks no_run and compile_fail examples with rustdoc itself.
    # The pinned rustdoc's merged runner ignores global --no-run. Standalone
    # persistence keeps preparation compile-only, including on a foreign host.
    # The consumer below must execute every runnable example before acceptance.
    with (directory / 'compile.log').open('w') as log:
        subprocess.run(command, cwd=root, check=True,
            stdout=log, env={**environment, 'CARGO_ENCODED_RUSTDOCFLAGS': '\x1f'.join([
                *flags, '--merge-doctests=no', '--no-run', '--persist-doctests', str(directory / 'programs')])})
    plan = plan_tests(inventory, directory / 'programs')
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def execute(root, directory, log_directory):
    plan = json.loads((directory / 'plan.json').read_text())
    binaries = directory / 'programs'
    inventory = json.loads((directory / 'inventory.json').read_text())
    if plan != plan_tests(inventory, binaries):
        raise ValueError('doctest runtime plan disagrees with the compiler inventory')
    log_directory.mkdir(parents=True, exist_ok=False)
    for index, program in enumerate(plan['programs']):
        result = subprocess.run([str((binaries / program['binary']).resolve())], cwd=root,
                                capture_output=True, timeout=300)
        (log_directory / f'standalone-{index}.stdout').write_bytes(result.stdout)
        (log_directory / f'standalone-{index}.stderr').write_bytes(result.stderr)
        # Match rustdoc's should_panic exit-status contract exactly.
        if (result.returncode != 0) != program['should_panic']:
            raise ValueError(f'doctest failed: {program["name"]}')
    summary = {'status': 'pass', 'total': plan['total'], 'compile_fail': plan['compile_fail'],
               'no_run': plan['no_run'], 'executed': len(plan['programs'])}
    (log_directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
