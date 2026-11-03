"""Select production dispatch with package-qualified workspace features.

Optional `--package NAME` or `--exclude NAME` arguments restrict the features to
the selected workspace packages, as Cargo requires for explicit selections.
"""

import argparse
from pathlib import Path
import sys
import tomllib


def native_features(root, packages=(), excludes=()):
    manifest = tomllib.loads((root / 'Cargo.toml').read_text())
    graphs, edges = {}, {}
    for member in manifest['workspace']['members']:
        package = tomllib.loads((root / member / 'Cargo.toml').read_text())
        name = package['package']['name']
        graphs[name] = package.get('features', {})
        edges[name] = {dependency for table in ('dependencies', 'dev-dependencies', 'build-dependencies')
                       for dependency in package.get(table, {})}
    unknown = sorted((set(packages) | set(excludes)) - set(graphs))
    if unknown:
        raise ValueError('not workspace package names: ' + ', '.join(unknown))
    selected = [name for name in graphs if (name in packages if packages else name not in excludes)]
    if not selected:
        raise ValueError('the selection excludes every workspace package')
    # Cargo accepts PACKAGE/FEATURE for a selected package or a dependency of one.
    # Keep workspace dependencies' features, as for a Cargo Rail provider-only plan.
    reached, pending = set(), list(selected)
    while pending:
        name = pending.pop()
        if name not in reached:
            reached.add(name)
            pending.extend(edges[name] & set(graphs))
    features = []
    for name in (name for name in graphs if name in reached):
        graph = graphs[name]
        roots = set(graph) - {'portable-only'}
        if any(value.rsplit('/', 1)[-1] == 'portable-only'
               for feature in roots for value in graph[feature]):
            raise ValueError(f'{name}: native features enable portable-only')
        features.extend(name + '/' + feature for feature in sorted(roots))
    return features


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', action='append', default=[])
    parser.add_argument('--exclude', action='append', default=[])
    args = parser.parse_args(arguments)
    if args.package and args.exclude:
        parser.error('use --package or --exclude, not both')
    try:
        print(','.join(native_features(Path(__file__).resolve().parents[2], args.package, args.exclude)))
    except ValueError as error:
        print(f'workspace_features.py: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
