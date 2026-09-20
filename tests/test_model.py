from pathlib import Path
import tempfile
import unittest
import numpy as np
from boltz_kinase_encoder.data import read_csv, write_csv
from boltz_kinase_encoder.model import fit, fingerprints, get_folds, joint_kernel, load_dataset, normalized, predict, predict_fold, tanimoto, train_bandwidth

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs, cls.targets, cls.y, cls.card = load_dataset(ROOT / 'data/processed')

    def test_tanimoto_known_values_no_integer_overflow(self):
        x = np.ones((1, 1024), dtype=np.uint8)
        self.assertAlmostEqual(tanimoto(x, x)[0, 0], 1)
        self.assertAlmostEqual(tanimoto([[1, 1, 0]], [[0, 1, 1]])[0, 0], 1 / 3)

    def test_fingerprint_identity(self):
        fp = fingerprints(['CCO', 'OCC'])
        np.testing.assert_array_equal(fp[0], fp[1])

    def test_scaffold_split_has_no_shared_compounds_or_scaffolds(self):
        seen = []
        for train, test in get_folds(self.pairs, 'scaffold'):
            for field in ('smiles', 'scaffold'):
                self.assertFalse({self.pairs[i][field] for i in train} & {self.pairs[i][field] for i in test})
            seen.extend(test.tolist())
        self.assertEqual(sorted(seen), list(range(len(self.pairs))))

    def test_target_split_leaves_exactly_one_protein_out(self):
        folds = get_folds(self.pairs, 'target')
        self.assertEqual(len(folds), 12)
        for train, test in folds:
            held = {self.pairs[i]['target_id'] for i in test}
            self.assertEqual(len(held), 1)
            self.assertFalse(held & {self.pairs[i]['target_id'] for i in train})

    def test_target_mean_does_not_use_held_out_labels(self):
        ids = np.array(['A', 'A', 'B'])
        y = np.array([2., 4., 1000.])
        prediction = predict_fold(np.eye(3), {}, ids, y, np.array([0, 1]), np.array([2]), 'target_mean', 1)
        self.assertEqual(prediction[0], 3)

    def test_joint_kernel_is_symmetric_positive_semidefinite(self):
        rng = np.random.default_rng(5)
        drug = rng.integers(0, 2, (8, 24))
        protein = normalized(rng.normal(size=(8, 5)))
        k = joint_kernel(tanimoto(drug, drug), np.arange(8), np.arange(8), 'ligand_sequence', protein, protein, train_bandwidth(protein))
        np.testing.assert_allclose(k, k.T)
        self.assertGreater(np.linalg.eigvalsh(k).min(), -1e-10)

    def test_prediction_is_invariant_to_heldout_label_changes(self):
        drug = np.array([[1., .5, .3], [.5, 1., .4], [.3, .4, 1.]])
        ids = np.array(['A', 'B', 'C'])
        def run(y):
            return predict_fold(drug, {}, ids, np.array(y), np.array([0, 1]), np.array([2]), 'ligand_only', 1)
        np.testing.assert_allclose(run([2, 4, 0]), run([2, 4, 1000]))

    def test_boltz_fit_cannot_silently_use_a_baseline(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'real exported'):
            fit(ROOT / 'data/processed', Path(directory) / 'model.npz', method='ligand_boltz')

    def test_fit_and_candidate_prediction_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fitted = fit(ROOT / 'data/processed', root / 'model.npz')
            self.assertEqual(fitted['training_pairs'], 173)
            write_csv(root / 'candidates.csv', [dict(candidate_id='example', target_id='SRC', smiles='CCO')])
            predict(root / 'model.npz', root / 'candidates.csv', root / 'prediction.csv')
            result = read_csv(root / 'prediction.csv')[0]
            self.assertTrue(np.isfinite(float(result['predicted_pkd'])))
            self.assertEqual(result['model_method'], 'ligand_sequence')
            with self.assertRaises(ValueError):
                predict(root / 'model.npz', root / 'candidates.csv', root / 'candidates.csv')


if __name__ == '__main__':
    unittest.main()
