"""Select production dispatch with package-qualified workspace features."""

from pathlib import Path
import tomllib


def native_features(root):
    manifest = tomllib.loads((root / 'Cargo.toml').read_text())
    selected = []
    for member in manifest['workspace']['members']:
        package = tomllib.loads((root / member / 'Cargo.toml').read_text())
        graph = package.get('features', {})
        features = set(graph) - {'portable-only'}
        if any(value.rsplit('/', 1)[-1] == 'portable-only'
               for name in features for value in graph[name]):
            raise ValueError(f'{package["package"]["name"]}: native features enable portable-only')
        selected.extend(package['package']['name'] + '/' + feature for feature in sorted(features))
    return selected


if __name__ == '__main__':
    print(','.join(native_features(Path(__file__).resolve().parents[2])))
