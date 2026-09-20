import json
from pathlib import Path
import tempfile
import unittest
from boltz_kinase_encoder.data import canonical_parent, curate, parse_fasta, pkd_from_micromolar, scaffold

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_micromolar_pkd_conversion(self):
        self.assertAlmostEqual(pkd_from_micromolar(1), 6)
        self.assertAlmostEqual(pkd_from_micromolar(.001), 9)
        self.assertAlmostEqual(pkd_from_micromolar(10), 5)
        for invalid in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                pkd_from_micromolar(invalid)

    def test_scaffold_groups_analogs(self):
        self.assertEqual(scaffold('Cc1ccccc1'), scaffold('CCc1ccccc1'))
        self.assertEqual(scaffold('CCC'), scaffold('CCO'))

    def test_largest_fragment_and_invalid_smiles(self):
        self.assertEqual(canonical_parent('CCO.[Na+]'), 'CCO')
        with self.assertRaises(ValueError):
            canonical_parent('not-a-molecule')

    def test_fasta_species_sequence_and_versions(self):
        result = parse_fasta('>NP_123.2 test [Homo sapiens]\nACDE\n')
        self.assertEqual(result['NP_123.2'][1], 'ACDE')
        for bad in ('>NP_123.2 other [Mus musculus]\nACDE\n', '>NP_123.2 test [Homo sapiens]\nAXDE\n'):
            with self.assertRaises(ValueError):
                parse_fasta(bad)

    def test_curated_snapshot_reproduces_from_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            result = curate(ROOT / 'config/pilot.json', ROOT / 'data/raw', directory)
            self.assertEqual(result['pairs'], 173)
            self.assertEqual(result['targets'], 12)
            self.assertEqual(result['unique_compounds'], 34)
            self.assertEqual((Path(directory) / 'pairs.csv').read_bytes(), (ROOT / 'data/processed/pairs.csv').read_bytes())

    def test_raw_tampering_is_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / 'raw'
            shutil.copytree(ROOT / 'data/raw', raw)
            with (raw / 'assay_1433.csv').open('a') as handle:
                handle.write('tampered\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                curate(ROOT / 'config/pilot.json', raw, Path(directory) / 'processed')


if __name__ == '__main__':
    unittest.main()
