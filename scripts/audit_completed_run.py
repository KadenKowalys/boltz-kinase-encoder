#!/usr/bin/env python3
"""Audit downloaded artifacts; does not certify structure accuracy or RDKit compatibility."""
import argparse
import json
from pathlib import Path
import gemmi
import numpy as np
from scipy.spatial import cKDTree
from boltz_kinase_encoder.data import digest, write_json, write_csv, read_csv
from boltz_kinase_encoder.encoder import validated_targets, load_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', default='12247627')
    parser.add_argument('--reproduced', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/boltz'))
    args = parser.parse_args()
    job = args.job_id
    targets = validated_targets('data/processed/targets.csv')
    embedding_path = Path(f'data/features/boltz2_{job}.npz')
    embeddings = load_embeddings(embedding_path, targets)
    run_dir = Path(f'gpu/full_run_{job}')
    run = json.loads((run_dir / 'run_manifest.json').read_text())
    inputs = json.loads(Path(f'gpu/full_inputs_{job}.manifest.json').read_text())
    assert run['status'] == 'completed' and run['returncode'] == 0
    assert not run['missing_or_ambiguous_exports']
    assert run['inputs_manifest_sha256'] == digest(f'gpu/full_inputs_{job}.manifest.json')
    for entry in inputs['targets']:
        assert entry['input_sha256'] == digest(Path(f'gpu/full_inputs_{job}') / entry['input_file'])
    summaries = []
    for target in targets:
        name = target['target_id']
        paths = list(run_dir.rglob(f'{name}_model_0.cif'))
        assert len(paths) == 1
        path = paths[0]
        structure = gemmi.read_structure(str(path))
        assert len(structure) == 1 and len(structure[0]) == 1
        residues = list(structure[0][0])
        sequence = ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in residues)
        assert sequence == target['sequence'], name
        missing = [str(r.seqid) for r in residues if not {'N', 'CA', 'C', 'O'} <= {a.name for a in r}]
        assert not missing, (name, missing)
        atoms = [a for r in residues for a in r if not a.element.is_hydrogen]
        coords = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in atoms])
        assert np.isfinite(coords).all()
        cn = [residues[i]['C'][0].pos.dist(residues[i+1]['N'][0].pos) for i in range(len(residues)-1)]
        near = cKDTree(coords).query_pairs(.8)
        confidence = json.loads(next(path.parent.glob('confidence_*.json')).read_text())
        for key in ('complex_plddt', 'ptm', 'confidence_score'):
            assert np.isfinite(confidence[key]) and 0 <= confidence[key] <= 1
        summaries.append(dict(target=name, residues=len(residues), heavy_atoms=len(atoms),
                              sequence_matches=True, missing_backbone_residues=0,
                              minimum_peptide_CN_A=min(cn), maximum_peptide_CN_A=max(cn),
                              heavy_atom_pairs_below_0_8A=len(near),
                              mean_plddt=confidence['complex_plddt'], ptm=confidence['ptm'],
                              cif_sha256=digest(path)))
    original = json.loads(Path(f'results/local-boltz-{job}/metrics.json').read_text())
    reproduced = json.loads((args.reproduced / 'metrics.json').read_text())
    differences, rank_differences = [], []
    for protocol, methods in original['scores'].items():
        for method, scores in methods.items():
            for metric, value in scores.items():
                if value is not None:
                    destination = rank_differences if metric == 'spearman' else differences
                    destination.append(abs(value - reproduced['scores'][protocol][method][metric]))
    assert max(differences) < 1e-10
    def prediction_map(path):
        return {(r['protocol'], r['method'], r['pair_id']): float(r['predicted_pkd']) for r in read_csv(path)}
    remote = prediction_map(Path(f'results/local-boltz-{job}/predictions.csv'))
    local = prediction_map(args.reproduced / 'predictions.csv')
    assert remote.keys() == local.keys()
    prediction_difference = max(abs(remote[k] - local[k]) for k in remote)
    assert prediction_difference < 1e-10
    model = Path(f'models/boltz_regressor_{job}.npz')
    model_metadata = json.loads(model.with_suffix('.json').read_text())
    assert model_metadata['model_sha256'] == digest(model)
    assert model_metadata['embeddings_sha256'] == digest(embedding_path)
    with np.load(model, allow_pickle=False) as archive:
        for key in archive.files:
            a = archive[key]
            if a.dtype.kind in 'fci':
                assert np.isfinite(a).all(), key
    stderr = Path(f'logs/boltz-full-{job}.err').read_text()
    result = dict(job_id=job, status='artifact_and_metric_checks_passed_with_unresolved_rdkit_warning',
                  target_count=len(embeddings), embedding_shape=[len(embeddings), len(next(iter(embeddings.values())))],
                  max_reproduced_nonrank_metric_absolute_difference=max(differences),
                  max_reproduced_prediction_absolute_difference=prediction_difference,
                  max_spearman_difference=max(rank_differences),
                  model_checksum_verified=True, input_hashes_verified=True,
                  rdkit_version_warning_count=stderr.count('Depickling from a version number'),
                  limits=['RDKit 16.2-to-16.1 molecule serialization compatibility has not been tested in a matched-version rerun.',
                          'Finite coordinates, matching sequences and basic geometry do not certify experimental structure accuracy.',
                          'Raw per-residue embeddings and model checkpoint were not included in the downloaded archive; their recorded hashes were not independently recomputed.'])
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'audit.json', result)
    write_csv(args.output / 'structure_audit.csv', summaries)
    print(json.dumps(result, indent=2))
    print('Peptide C-N range:',min(x['minimum_peptide_CN_A'] for x in summaries),max(x['maximum_peptide_CN_A'] for x in summaries))
    print('Very short heavy-atom pairs:',sum(x['heavy_atom_pairs_below_0_8A'] for x in summaries))


if __name__ == '__main__':
    main()
