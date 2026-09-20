import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from boltz_kinase_encoder.data import digest, write_json
from boltz_kinase_encoder.encoder import BOLTZ_COMMIT, collect_embeddings, input_manifest_path, load_embeddings, pool_single_representation, prepare_inputs, validated_targets

ROOT = Path(__file__).resolve().parents[1]


class EncoderTests(unittest.TestCase):
    def test_padding_is_excluded_from_pooling(self):
        values = np.array([[[1, 3], [3, 7], [999, 999]]], dtype=float)
        np.testing.assert_allclose(pool_single_representation(values, 2), [2, 5, 1, 2])

    def test_invalid_embeddings_fail(self):
        for x, length in ((np.ones((2, 3, 4)), 3), (np.ones((2, 4)), 3), (np.zeros((3, 4)), 3), (np.full((3, 4), np.nan), 3)):
            with self.assertRaises(ValueError):
                pool_single_representation(x, length)

    def test_yaml_directory_has_no_manifest_or_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = Path(directory) / 'inputs'
            manifest = prepare_inputs(ROOT / 'data/processed/targets.csv', inputs)
            self.assertEqual(len(manifest['targets']), 12)
            self.assertEqual({p.suffix for p in inputs.iterdir()}, {'.yaml'})
            self.assertTrue(input_manifest_path(inputs).exists())
            self.assertNotIn('msa: empty', (inputs / 'SRC.yaml').read_text())
            prepare_inputs(ROOT / 'data/processed/targets.csv', inputs, 'single')
            self.assertIn('msa: empty', (inputs / 'SRC.yaml').read_text())

    def test_collection_contract_with_synthetic_tensors(self):
        # Contract fixture only: this does NOT test actual Boltz inference.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, run = root / 'inputs', root / 'run'
            manifest = prepare_inputs(ROOT / 'data/processed/targets.csv', inputs)
            run.mkdir()
            write_json(run / 'run_manifest.json', dict(status='completed', boltz_commit=BOLTZ_COMMIT,
                       inputs_manifest_sha256=digest(input_manifest_path(inputs))))
            for target in manifest['targets']:
                np.savez_compressed(run / f'embeddings_{target["target_id"]}.npz', s=np.ones((1, target['length'], 4)))
            output = root / 'pooled.npz'
            result = collect_embeddings(inputs, run, output)
            self.assertEqual(result['features_per_target'], 8)
            targets = validated_targets(ROOT / 'data/processed/targets.csv')
            self.assertEqual(len(load_embeddings(output, targets)), 12)
            targets[0]['sequence_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'wrong-sequence'):
                load_embeddings(output, targets)

    def test_incomplete_gpu_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, run = root / 'inputs', root / 'run'
            prepare_inputs(ROOT / 'data/processed/targets.csv', inputs)
            run.mkdir()
            write_json(run / 'run_manifest.json', dict(status='failed', boltz_commit=BOLTZ_COMMIT))
            with self.assertRaisesRegex(ValueError, 'completed run'):
                collect_embeddings(inputs, run, root / 'out.npz')


if __name__ == '__main__':
    unittest.main()
