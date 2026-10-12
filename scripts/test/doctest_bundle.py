"""Prepare with rustdoc, then execute its unchanged programs on the target.

NIGHTLY: extraction and persistence use the repository's pinned
nightly. Unknown extraction schemas or incomplete runtime inventories fail closed.
"""

import json
import re
import subprocess


def plan_tests(inventory, directory):
    if inventory.get('format_version') != 2 or not isinstance(inventory.get('doctests'), list):
        raise ValueError('unsupported rustdoc inventory')
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
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--locked', '--no-deps', '--format-version', '1'],
        cwd=root, env=environment, text=True))
    members = set(metadata['workspace_members'])
    packages = sorted(package['name'] for package in metadata['packages'] if package['id'] in members)
    directory.mkdir(parents=True, exist_ok=False)
    total = 0
    for package in packages:
        total += prepare_package(root, directory / package, package,
                                 package_arguments(package, packages, cargo_args), environment)['total']
    plan = {'total': total, 'packages': packages}
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def package_arguments(package, packages, cargo_args):
    # Workspace feature lists qualify each feature with its member. Under -p,
    # Cargo rejects another member's features, so keep only this package's own.
    foreign = tuple(name + '/' for name in packages if name != package)
    return [','.join(feature for feature in argument.split(',') if not feature.startswith(foreign))
            if previous == '--features' else argument
            for previous, argument in zip([None, *cargo_args], cargo_args)]


def prepare_package(root, directory, package, cargo_args, environment):
    directory.mkdir(parents=True, exist_ok=False)
    flags = ['-D', 'warnings', '-Z', 'unstable-options']
    command = ['cargo', 'test', '--locked', '-p', package, '--release', '--doc', *cargo_args]
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
    packages = plan['packages']
    if not packages or len(set(packages)) != len(packages) or any(
            not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in packages):
        raise ValueError('invalid doctest package inventory')
    if set(packages) != {path.name for path in directory.iterdir() if path.is_dir()}:
        raise ValueError('doctest package inventory omits or adds a compiled member')
    results = [execute_package(root, directory / name, log_directory / name) for name in packages]
    summary = {'status': 'pass', **{key: sum(row[key] for row in results)
                                  for key in ('total', 'compile_fail', 'no_run', 'executed')}}
    if summary['total'] != plan['total']:
        raise ValueError('doctest package totals disagree with the workspace plan')
    (log_directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def execute_package(root, directory, log_directory):
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
