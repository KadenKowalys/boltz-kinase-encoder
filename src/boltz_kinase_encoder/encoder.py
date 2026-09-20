"""Prepare single-protein Boltz inputs and pool genuine exported trunk features."""
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from .data import AMINO_ACIDS, digest, read_csv, write_json

BOLTZ_COMMIT = 'b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc'


def input_manifest_path(inputs):
    inputs = Path(inputs)
    return inputs.parent / (inputs.name + '.manifest.json')


def validated_targets(path):
    targets = read_csv(path)
    if not targets or len({t['target_id'] for t in targets}) != len(targets):
        raise ValueError('target IDs must be unique')
    for target in targets:
        seq = target['sequence']
        if not target['target_id'].isalnum() or not seq or set(seq) - set(AMINO_ACIDS):
            raise ValueError('only canonical single-protein sequences and alphanumeric IDs are supported')
        if hashlib.sha256(seq.encode()).hexdigest() != target['sequence_sha256']:
            raise ValueError('sequence digest mismatch')
    return targets


def prepare_inputs(target_path, output, msa_mode='server'):
    if msa_mode not in ('server', 'single'):
        raise ValueError('MSA mode must be server or single')
    targets, output = validated_targets(target_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected = {t['target_id'] + '.yaml' for t in targets}
    if {p.name for p in output.iterdir()} - expected:
        raise ValueError('input directory contains extra files; use a fresh directory')
    manifest = dict(schema_version=1, boltz_commit=BOLTZ_COMMIT, msa_mode=msa_mode, targets=[])
    for t in targets:
        protein = {'id': 'A', 'sequence': t['sequence']}
        if msa_mode == 'single':
            protein['msa'] = 'empty'
        path = output / (t['target_id'] + '.yaml')
        path.write_text(yaml.safe_dump({'version': 1, 'sequences': [{'protein': protein}]}, sort_keys=False))
        manifest['targets'].append(dict(target_id=t['target_id'], accession=t['accession'],
                                       length=len(t['sequence']), sequence_sha256=t['sequence_sha256'],
                                       input_file=path.name, input_sha256=digest(path)))
    # Boltz rejects non-YAML files inside its input directory.
    write_json(input_manifest_path(output), manifest)
    return manifest


def pool_single_representation(array, length):
    s = np.asarray(array, dtype=np.float32)
    if s.ndim == 3:
        if s.shape[0] != 1:
            raise ValueError('only batch-size-one Boltz exports are supported')
        s = s[0]
    if s.ndim != 2 or s.shape[0] < length or s.shape[1] == 0 or length < 1:
        raise ValueError('invalid s representation shape or too few protein tokens')
    # Generated inputs contain exactly one canonical protein: first L tokens
    # are its residues; any remaining tokens are trailing model padding.
    s = s[:length]
    if not np.all(np.isfinite(s)):
        raise ValueError('nonfinite Boltz embedding')
    pooled = np.concatenate([s.mean(axis=0), s.std(axis=0)]).astype(np.float32)
    if not np.any(pooled):
        raise ValueError('zero embedding is not a usable protein representation')
    return pooled


def collect_embeddings(inputs, run_dir, output):
    inputs, run_dir, output = Path(inputs), Path(run_dir), Path(output)
    manifest = json.loads(input_manifest_path(inputs).read_text())
    run = json.loads((run_dir / 'run_manifest.json').read_text())
    if run.get('status') != 'completed' or run.get('boltz_commit') != BOLTZ_COMMIT:
        raise ValueError('require a completed run from the pinned encoder runner')
    if run['inputs_manifest_sha256'] != digest(input_manifest_path(inputs)):
        raise ValueError('input manifest changed after the Boltz run')
    vectors, ids, hashes, files = [], [], [], []
    for target in manifest['targets']:
        if digest(inputs / target['input_file']) != target['input_sha256']:
            raise ValueError('Boltz input changed after preparation')
        matches = list(run_dir.rglob(f'embeddings_{target["target_id"]}.npz'))
        if len(matches) != 1:
            raise ValueError(f'expected exactly one embedding export for {target["target_id"]}; found {len(matches)}')
        with np.load(matches[0], allow_pickle=False) as raw:
            if 's' not in raw:
                raise ValueError('Boltz export lacks the s trunk representation')
            vectors.append(pool_single_representation(raw['s'], target['length']))
        ids.append(target['target_id'])
        hashes.append(target['sequence_sha256'])
        files.append(dict(target_id=target['target_id'], sha256=digest(matches[0])))
    if len({v.shape for v in vectors}) != 1:
        raise ValueError('inconsistent Boltz feature dimensions')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix != '.npz':
        raise ValueError('embedding output must end in .npz')
    np.savez_compressed(output, target_ids=np.asarray(ids), embeddings=np.stack(vectors),
                        sequence_sha256=np.asarray(hashes), feature_schema=np.asarray('boltz2_s_mean_std_v1'))
    write_json(output.with_suffix('.json'), dict(status='actual_boltz_output', boltz_commit=BOLTZ_COMMIT,
               feature_schema='boltz2_s_mean_std_v1', embedding_sha256=digest(output),
               run_provenance=run, source_files=files))
    return {'targets': len(ids), 'features_per_target': vectors[0].size}


def load_embeddings(path, targets):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata.get('status') != 'actual_boltz_output' or metadata.get('boltz_commit') != BOLTZ_COMMIT:
        raise ValueError('missing genuine pinned Boltz provenance')
    if metadata['embedding_sha256'] != digest(path):
        raise ValueError('pooled embedding checksum mismatch')
    with np.load(path, allow_pickle=False) as archive:
        ids = archive['target_ids'].tolist()
        vectors = archive['embeddings'].copy()
        hashes = archive['sequence_sha256'].tolist()
        if str(archive['feature_schema']) != 'boltz2_s_mean_std_v1':
            raise ValueError('unknown embedding schema')
    if len(ids) != len(set(ids)) or vectors.ndim != 2 or vectors.shape[0] != len(ids) or len(hashes) != len(ids):
        raise ValueError('invalid pooled embedding dimensions or IDs')
    if not np.all(np.isfinite(vectors)) or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError('nonfinite or zero pooled embedding')
    mapping = {name: (vector, seq_hash) for name, vector, seq_hash in zip(ids, vectors, hashes)}
    for target in targets:
        if target['target_id'] not in mapping or mapping[target['target_id']][1] != target['sequence_sha256']:
            raise ValueError(f'missing or wrong-sequence embedding: {target["target_id"]}')
    return {t['target_id']: mapping[t['target_id']][0] for t in targets}
