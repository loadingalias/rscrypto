"""Native measurement runner selection shared by benchmark and CT workflows."""
# Measurement hardware is fixed where the provider allows it. Donated runners
# retain their provider labels; their actual machine identity is retained with the measurement evidence.
PLATFORMS = {
    'x86_64-linux': ('c8i.2xlarge', 'ubuntu24-minimal-x64', 90),
    'aarch64-linux': ('c8g.2xlarge', 'ubuntu24-minimal-arm64', 90),
    'x86_64-win': ('c8i.2xlarge', 'windows25-full-x64', 90),
    's390x-linux': ('', 'ubuntu-24.04-s390x', 90),
    'powerpc64le-linux': ('', 'ubuntu-24.04-ppc64le-p10', 90),
    'riscv64-linux': ('', 'ubuntu-24.04-riscv', 180),
}


def platforms(value: str, run_id: str) -> dict:
    names = list(PLATFORMS) if value.strip() == 'all' else list(dict.fromkeys(value.replace(',', ' ').split()))
    if not names or any(name not in PLATFORMS for name in names):
        raise ValueError('architectures must be all or a list of: ' + ', '.join(PLATFORMS))
    if not run_id.isdecimal():
        raise ValueError('GITHUB_RUN_ID must be numeric')
    rows = []
    for name in names:
        family, image, timeout = PLATFORMS[name]
        label = (f'runs-on={run_id}/family={family}/cpu=8/image={image}/spot=false/volume=100gb:gp3/env=production'
                 if family else image)
        rows.append({'platform': name, 'runner': label, 'timeout': timeout})
    return {'include': rows}
