"""Native measurement runner selection shared by benchmark and CT workflows."""
# Measurement hardware is fixed where the provider allows it. Donated runners
# retain their provider labels; their actual machine identity is retained with the measurement evidence.
PLATFORMS = {
    'x86_64-linux': ('measure-x86_64-linux-intel', 90),
    'aarch64-linux': ('measure-aarch64-linux', 90),
    'x86_64-win': ('measure-x86_64-win-intel', 90),
    's390x-linux': ('ubuntu-24.04-s390x', 90),
    'powerpc64le-linux': ('ubuntu-24.04-ppc64le-p10', 90),
    'riscv64-linux': ('ubuntu-24.04-riscv', 180),
}


def platforms(value: str, run_id: str) -> dict:
    names = list(PLATFORMS) if value.strip() == 'all' else list(dict.fromkeys(value.replace(',', ' ').split()))
    if not names or any(name not in PLATFORMS for name in names):
        raise ValueError('architectures must be all or a list of: ' + ', '.join(PLATFORMS))
    if not run_id.isdecimal():
        raise ValueError('GITHUB_RUN_ID must be numeric')
    rows = []
    for name in names:
        runner, timeout = PLATFORMS[name]
        label = (f'runs-on={run_id}/runner={runner}/env=production'
                 if runner.startswith('measure-') else runner)
        rows.append({'platform': name, 'runner': label, 'timeout': timeout})
    return {'include': rows}
