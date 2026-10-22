"""Negative artifact tests: corrupt evidence must never pass release verification."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('verify_v6', ROOT / 'scripts_release/verify_v6_results.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for relative in ('experiments/v6_candidate', 'results_v6/native_validation', 'provenance'):
            shutil.copytree(ROOT / relative, self.root / relative, ignore=shutil.ignore_patterns('__pycache__'))
        (self.root / 'docs').mkdir()
        for relative in ('README.md', 'docs/V6_VALIDATION.md'):
            shutil.copyfile(ROOT / relative, self.root / relative)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_archive_passes(self):
        self.assertEqual(verifier.verify(self.root)['n_rows'], 64)

    def test_missing_row_fails(self):
        path = self.root / 'results_v6/native_validation/rows.jsonl'
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(lines[:-1]) + '\n')
        with self.assertRaises(ValueError):
            verifier.verify(self.root)

    def test_summary_and_docs_drift_fail(self):
        path = self.root / 'results_v6/native_validation/public_summary.json'
        data = json.loads(path.read_text())
        data['families']['connectome']['standard']['final_accuracy']['mean'] = .99
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'summary differs'):
            verifier.verify(self.root)
        shutil.copyfile(ROOT / 'results_v6/native_validation/public_summary.json', path)
        (self.root / 'README.md').write_text('Unsupported 99% claim')
        with self.assertRaisesRegex(ValueError, 'table drift'):
            verifier.verify(self.root)

    def test_changed_accepted_receipt_fails(self):
        path = self.root / 'results_v6/native_validation/acceptance.json'
        data = json.loads(path.read_text())
        data['reviewer'] += ' changed after the run'
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            verifier.verify(self.root)

    def test_changed_frozen_source_fails(self):
        mapping = json.loads((self.root / 'provenance/V6_SOURCE_MAP.json').read_text())
        path = self.root / next(iter(mapping.values()))
        path.write_bytes(path.read_bytes() + b'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            verifier.verify(self.root)


if __name__ == '__main__':
    unittest.main()
