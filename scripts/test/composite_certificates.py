"""Check composite certificate chains with the pinned Bouncy Castle peer."""

import argparse
import hashlib
import os
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'rscrypto-rustls/tests/vectors'
VERSION = '1.87'
JARS = {
    'bcprov': '12d2d3dd82a9ec15e92ec7c7943ee64f6097c6dc12c07e984f1693b77565b9aa',
    'bcutil': 'af46dd381553fdc466cd0c18f8bf2144c7dbcb7337eee3e16cab15e96613ddf0',
    'bcpkix': 'a44be2660d72ed88673372b8cc2d53a25a5aaf27122fb803ed93f07147536aa1',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--issue', type=Path, help='issue fresh public certificates into an empty directory')
    args = parser.parse_args()
    opt = Path.home() / '.local/opt'
    jars = []
    for name, digest in JARS.items():
        jar = opt / name / VERSION / (name + '.jar')
        if not jar.is_file() or hashlib.sha256(jar.read_bytes()).hexdigest() != digest:
            parser.error(f'expected {name} {VERSION} with SHA-256 {digest} at {jar}')
        jars.append(str(jar))
    if 'JAVA_HOME' in os.environ:
        jdk = Path(os.environ['JAVA_HOME'])
    else:
        homes = (opt / 'temurin-jdk').glob('25.*/Contents/Home')
        jdk = max(homes, key=lambda home: tuple(map(int, re.findall(r'\d+', home.parents[1].name))), default=None)
    if jdk is None or not (jdk / 'release').is_file():
        parser.error('set JAVA_HOME to the managed Temurin 25 JDK')
    release = (jdk / 'release').read_text()
    if 'IMPLEMENTOR="Eclipse Adoptium"' not in release or not re.search(r'^JAVA_VERSION="25(?:[."]|$)', release, re.M):
        parser.error(f'{jdk} must contain Temurin 25')
    java = jdk / 'bin/java'
    command = [java, '-cp', os.pathsep.join(jars), Path(__file__).with_name('CompositeCertificates.java')]
    subprocess.run([*command, 'verify', FIXTURES / 'credentials'], check=True)
    if args.issue:
        args.issue.mkdir(parents=True, exist_ok=True)
        if any(args.issue.iterdir()):
            parser.error('--issue needs an empty directory')
        subprocess.run([*command, 'issue', args.issue], check=True)


if __name__ == '__main__':
    main()
