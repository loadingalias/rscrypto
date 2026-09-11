#!/usr/bin/env python3
"""Negative controls for the bounded optimized-cleanup gate."""
import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeroization import inspect_ir


def fixture():
    lines = ['define i8 @zeroize_entry_secret_bytes_32(ptr %input) {',
             '%secret = alloca [32 x i8], align 8']
    for offset in range(0, 32, 8):
        lines += [f'%p{offset} = getelementptr inbounds nuw i8, ptr %secret, i64 {offset}',
                  f'store volatile i64 0, ptr %p{offset}, align 8']
    return '\n'.join(lines + ['fence syncscope("singlethread") seq_cst', 'ret i8 0', '}'])


class ZeroizationTest(unittest.TestCase):
    def test_linked_evidence_with_pe_symbol_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            binary = directory / 'rscrypto-ct-evidence.exe'
            binary.write_bytes(b'linked PE fixture')
            (directory / 'rscrypto_ct_evidence.ll').write_text(fixture())
            # PE has no native nm names; the checked linker map names its code.
            binary.with_name(binary.name + '.binary.nm-symbols.txt').write_text('')
            binary.with_name(binary.name + '.binary.raw-disasm.txt').write_text('140001000: retq\n')
            symbols = binary.with_name(binary.name + '.binary.symbols.txt')
            assembly = binary.with_name(binary.name + '.binary.disasm.txt')
            symbols.write_text('0000000140001000 0000000000000010 T zeroize_entry_secret_bytes_32\n')
            assembly.write_text('0000000140001000 <zeroize_entry_secret_bytes_32>:\n140001000: retq\n')
            report = directory / 'report.json'

            def run():
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('zeroization.py')),
                    '--artifact-dir', str(directory), '--out', str(report)], capture_output=True, text=True)
                return result.returncode, json.loads(report.read_text())

            code, evidence = run()
            self.assertEqual(code, 0, evidence)
            self.assertIn(symbols.name, evidence['artifacts'])
            self.assertIn(assembly.name, evidence['artifacts'])
            for path in (symbols, assembly):
                original = path.read_text()
                path.write_text('')
                self.assertEqual(run()[0], 1)
                path.write_text(original)

    def test_complete(self):
        self.assertEqual(inspect_ir(fixture())['cleared_bytes'], list(range(32)))

    def test_removed_or_nonvolatile_wipes(self):
        for replacement in ('', 'store'):
            text = fixture().replace('store volatile', replacement)
            with self.assertRaises(ValueError):
                inspect_ir(text)

    def test_partial_or_wrong_allocation(self):
        for text in (fixture().replace('store volatile i64 0, ptr %p24, align 8', ''),
                     fixture().replace('ptr %secret, i64 24', 'ptr %other, i64 24')):
            with self.assertRaises(ValueError):
                inspect_ir(text)

    def test_fence_return_and_control_flow(self):
        for text in (fixture().replace('fence syncscope("singlethread") seq_cst', ''),
                     fixture().replace('store volatile i64 0, ptr %p24', 'ret i8 0\nstore volatile i64 0, ptr %p24'),
                     fixture().replace('ret i8 0', 'br i1 %secret_bit, label %a, label %b\nret i8 0')):
            with self.assertRaises(ValueError):
                inspect_ir(text)

    def test_post_wipe_observation_or_unreviewed_call(self):
        for operation in ('%read = load i8, ptr %p0, align 1', 'call void @unknown(ptr %secret)'):
            with self.assertRaises(ValueError):
                inspect_ir(fixture().replace('ret i8 0', operation + '\nret i8 0'))

    def test_missing_symbol(self):
        with self.assertRaises(ValueError):
            inspect_ir('')


if __name__ == '__main__':
    unittest.main()
