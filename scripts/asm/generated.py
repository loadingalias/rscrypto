"""Check and rewrite assembly from its owning generator."""

import sys


def check_or_write(mode, root, script, sources, success):
    stale_total = []
    for path, text, new, stale in sources:
        relative = path.relative_to(root)
        if mode == 'write' and new != text:
            path.write_text(new, encoding='utf-8')
            print(f'rewrote {relative}: {", ".join(stale)}')
        stale_total += [f'{relative}: {name}' for name in stale]
    if mode == 'check' and stale_total:
        for entry in stale_total:
            print(f'stale generated assembly: {entry}', file=sys.stderr)
        print(f'Run `scripts/lib/python.sh {script} write` after changing a generator.', file=sys.stderr)
        return 1
    if mode == 'check':
        print(success)
    return 0
