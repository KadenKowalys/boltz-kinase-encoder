"""Small kernel-ridge model and scaffold/target-held-out evaluation."""
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from .data import AMINO_ACIDS, canonical_parent, digest, read_csv, scaffold, write_csv, write_json
from .encoder import load_embeddings, validated_targets

FP_BITS = 1024
METHODS = ('mean', 'target_mean', 'ligand_only', 'ligand_target_id', 'ligand_sequence')


def fingerprints(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    return np.asarray([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=float)


def tanimoto(left, right):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    intersection = left @ right.T
    union = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def sequence_features(sequence):
    lookup = {a: i for i, a in enumerate(AMINO_ACIDS)}
    single, pairs = np.zeros(20), np.zeros(400)
    for aa in sequence:
        single[lookup[aa]] += 1
    for a, b in zip(sequence, sequence[1:]):
        pairs[lookup[a] * 20 + lookup[b]] += 1
    return np.r_[single / len(sequence), pairs / max(1, len(sequence) - 1)]


def normalized(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(np.isfinite(vectors)) or np.any(norms == 0):
        raise ValueError('protein vectors must be finite and nonzero')
    return vectors / norms


def squared_distances(a, b):
    return np.maximum(0, (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2 * a @ b.T)


def train_bandwidth(protein_vectors):
    unique = np.unique(protein_vectors, axis=0)
    distances = squared_distances(unique, unique)
    positive = distances[distances > 1e-10]
    return float(np.median(positive)) if positive.size else 1.0


def joint_kernel(drug_kernel, left_ids, right_ids, method, left_features=None, right_features=None, bandwidth=1):
    if method == 'ligand_only':
        return drug_kernel
    if method == 'ligand_target_id':
        protein = np.asarray(left_ids)[:, None] == np.asarray(right_ids)[None, :]
    else:
        protein = np.exp(-squared_distances(left_features, right_features) / bandwidth)
    return drug_kernel * (.5 + .5 * protein)


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    card = json.loads((data_dir / 'dataset_card.json').read_text())
    if digest(data_dir / 'pairs.csv') != card['pairs_sha256'] or digest(data_dir / 'targets.csv') != card['targets_sha256']:
        raise ValueError('processed dataset checksum mismatch; rerun curation')
    pairs, targets = read_csv(data_dir / 'pairs.csv'), validated_targets(data_dir / 'targets.csv')
    names = {t['target_id'] for t in targets}
    if any(p['target_id'] not in names or scaffold(p['smiles']) != p['scaffold'] for p in pairs):
        raise ValueError('unknown protein or invalid scaffold grouping')
    y = np.asarray([float(p['pkd']) for p in pairs])
    if not np.all(np.isfinite(y)) or len(pairs) != len({p['pair_id'] for p in pairs}):
        raise ValueError('invalid labels or duplicate pairs')
    return pairs, targets, y, card


def get_folds(pairs, protocol, folds=5, seed=17):
    if protocol not in {'scaffold', 'target'}:
        raise ValueError('protocol must be scaffold or target')
    groups = np.asarray([p['scaffold'] if protocol == 'scaffold' else p['target_id'] for p in pairs])
    n_groups = len(set(groups))
    n_splits = min(folds, n_groups) if protocol == 'scaffold' else n_groups
    if n_splits < 2:
        raise ValueError('at least two independent groups are required')
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = list(splitter.split(np.zeros(len(pairs)), groups=groups))
    for train, test in result:
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError('group leakage')
    return result


def metrics(y, prediction):
    correlation = None
    if np.ptp(y) > 0 and np.ptp(prediction) > 0:
        correlation = float(spearmanr(y, prediction).statistic)
    return dict(rmse=float(np.sqrt(np.mean((y - prediction) ** 2))),
                mae=float(mean_absolute_error(y, prediction)), r2=float(r2_score(y, prediction)),
                spearman=correlation)


def predict_fold(drug, protein, ids, y, train, test, method, alpha):
    offset = float(y[train].mean())
    if method == 'mean':
        return np.full(len(test), offset)
    if method == 'target_mean':
        means = {name: float(y[train][ids[train] == name].mean()) for name in set(ids[train])}
        return np.asarray([means.get(name, offset) for name in ids[test]])
    vectors = protein.get(method)
    bandwidth = train_bandwidth(vectors[train]) if vectors is not None else 1.0
    kernel = joint_kernel(drug[np.ix_(train, train)], ids[train], ids[train], method,
                          vectors[train] if vectors is not None else None, vectors[train] if vectors is not None else None, bandwidth)
    test_kernel = joint_kernel(drug[np.ix_(test, train)], ids[test], ids[train], method,
                               vectors[test] if vectors is not None else None, vectors[train] if vectors is not None else None, bandwidth)
    weights = np.linalg.solve(kernel + alpha * np.eye(len(train)), y[train] - offset)
    return offset + test_kernel @ weights


def evaluate(data_dir, output, embedding_path=None, alpha=1.0, seed=17):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    pairs, targets, y, card = load_dataset(data_dir)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('use an empty output directory to preserve previous evaluations')
    ids = np.asarray([p['target_id'] for p in pairs])
    sequence = {t['target_id']: sequence_features(t['sequence']) for t in targets}
    protein = {'ligand_sequence': normalized([sequence[name] for name in ids])}
    methods = list(METHODS)
    if embedding_path is not None:
        embeddings = load_embeddings(embedding_path, targets)
        protein['ligand_boltz'] = normalized([embeddings[name] for name in ids])
        methods.append('ligand_boltz')
    drug = tanimoto(fingerprints([p['smiles'] for p in pairs]), fingerprints([p['smiles'] for p in pairs]))
    scores, predictions, assignments = {}, [], []
    for protocol in ('scaffold', 'target'):
        folds = get_folds(pairs, protocol, seed=seed)
        predicted = {method: np.zeros(len(pairs)) for method in methods}
        for fold, (train, test) in enumerate(folds):
            for method in methods:
                predicted[method][test] = predict_fold(drug, protein, ids, y, train, test, method, alpha)
            for index in test:
                assignments.append(dict(protocol=protocol, fold=fold, pair_id=pairs[index]['pair_id'],
                                        target_id=ids[index], scaffold=pairs[index]['scaffold']))
                for method in methods:
                    predictions.append(dict(protocol=protocol, fold=fold, method=method,
                                            pair_id=pairs[index]['pair_id'], target_id=ids[index],
                                            scaffold=pairs[index]['scaffold'], observed_pkd=float(y[index]),
                                            predicted_pkd=float(predicted[method][index])))
        scores[protocol] = {method: metrics(y, value) for method, value in predicted.items()}
    result = dict(schema_version=1, pairs=len(pairs), targets=len(targets), compounds=card['unique_compounds'],
                  scaffolds=card['scaffolds'], alpha=alpha, seed=seed, scores=scores,
                  pairs_sha256=card['pairs_sha256'], targets_sha256=card['targets_sha256'],
                  boltz_status='evaluated_actual_embeddings' if embedding_path else 'pending_gpu_extraction_not_evaluated',
                  embedding_sha256=digest(embedding_path) if embedding_path else None,
                  protocols={'scaffold': 'Five-fold grouped by nonchiral Bemis-Murcko scaffold; target overlap allowed.',
                             'target': 'Leave one target out; compound overlap allowed. Not a double-cold split.'},
                  limitations=['Small selected panel; pooled out-of-fold scores are descriptive, not a deployment validation.',
                               'Boltz pretraining overlap is not audited; these splits protect downstream labels only.',
                               'No hyperparameter search; alpha=1 is the prespecified default.'])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'metrics.json', result)
    write_csv(output / 'predictions.csv', predictions)
    write_csv(output / 'folds.csv', assignments)
    render_results(result, predictions, output)
    return result


def render_results(result, predictions, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = list(result['scores']['scaffold'])
    labels = {'mean': 'Training mean', 'target_mean': 'Target mean', 'ligand_only': 'Ligand only',
              'ligand_target_id': 'Ligand + target ID', 'ligand_sequence': 'Ligand + sequence', 'ligand_boltz': 'Ligand + Boltz-2'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, protocol in zip(axes, ('scaffold', 'target')):
        values = [result['scores'][protocol][m]['rmse'] for m in methods]
        ax.barh(range(len(methods)), values, color=['#355a75' if m != 'ligand_boltz' else '#168d7f' for m in methods])
        ax.set_yticks(range(len(methods)), [labels[m] for m in methods])
        ax.invert_yaxis()
        ax.set(title='Held-out scaffolds' if protocol == 'scaffold' else 'Held-out proteins', xlabel='Pooled out-of-fold RMSE (pKd; lower is better)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(0, max(values) * 1.18)
        for i, value in enumerate(values):
            ax.text(value + .015, i, f'{value:.3f}', va='center', fontsize=9)
    fig.suptitle('Measured PubChem kinase affinities · pilot benchmark', fontsize=15)
    fig.savefig(output / 'benchmark.png', dpi=160)
    plt.close(fig)
    status = 'Boltz-2 features evaluated.' if result['embedding_sha256'] else '**Boltz-2 extraction has not run. These are baseline results only.**'
    lines = ['# Pilot results', '', status, '', f"{result['pairs']} measured pairs · {result['compounds']} compounds · {result['targets']} targets · {result['scaffolds']} scaffolds", '',
             '| Protocol | Model | RMSE ↓ | MAE ↓ | R² | Spearman |', '|---|---|---:|---:|---:|---:|']
    for protocol, scores in result['scores'].items():
        for method, metric in scores.items():
            rho = 'undefined' if metric['spearman'] is None else f"{metric['spearman']:.3f}"
            lines.append(f"| {protocol} | {labels[method]} | {metric['rmse']:.3f} | {metric['mae']:.3f} | {metric['r2']:.3f} | {rho} |")
    lines += ['', '![Benchmark](benchmark.png)', '', '## Interpretation', '',
              'These are pooled predictions for rows held out from their downstream training fold. Negative R² means performance is worse than the constant predictor defined using the full evaluation-label mean; the training-mean baseline is also reported separately.', '',
              'Scaffold holdout evaluates new chemical scaffolds among these targets. Target holdout evaluates an unseen protein while allowing known compounds. Neither establishes simultaneous new-protein/new-scaffold generalization.', '',
              'The sparse panel contains reported follow-up affinity measurements; unmeasured pairs were never turned into negatives. Reference sequences may differ from experimental constructs. A small historical dataset cannot establish prospective drug-discovery performance.', '',
              'Boltz-2 protein-only embeddings are frozen inputs. The downstream kernel-ridge regressor is the only trained model. Pretraining overlap has not been audited. Hyperparameters were not selected using these held-out scores.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')


def fit(data_dir, output, method='ligand_sequence', embedding_path=None, alpha=1.0):
    if method not in {'ligand_only', 'ligand_sequence', 'ligand_boltz'} or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('invalid final-model method or regularization')
    pairs, targets, y, card = load_dataset(data_dir)
    if method == 'ligand_boltz':
        if embedding_path is None:
            raise ValueError('Boltz training requires real exported embeddings; no fallback is used')
        mapping = load_embeddings(embedding_path, targets)
    else:
        mapping = {t['target_id']: sequence_features(t['sequence']) for t in targets}
    names = [t['target_id'] for t in targets]
    features = normalized([mapping[name] for name in names])
    mapping = dict(zip(names, features))
    ids = np.asarray([p['target_id'] for p in pairs])
    protein = np.asarray([mapping[name] for name in ids])
    drug = fingerprints([p['smiles'] for p in pairs])
    bandwidth = train_bandwidth(protein)
    kernel = joint_kernel(tanimoto(drug, drug), ids, ids, method, protein, protein, bandwidth)
    weights = np.linalg.solve(kernel + alpha * np.eye(len(y)), y - y.mean())
    output = Path(output)
    if output.suffix != '.npz' or output.exists() or output.with_suffix('.json').exists():
        raise ValueError('choose a new .npz model path')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, train_drug=drug, train_protein=protein, train_ids=ids,
                        target_ids=np.asarray(names), target_features=features, dual_weights=weights,
                        offset=np.asarray(y.mean()), bandwidth=np.asarray(bandwidth), method=np.asarray(method))
    write_json(output.with_suffix('.json'), dict(method=method, alpha=alpha, training_pairs=len(y),
               model_sha256=digest(output), dataset_sha256=card['pairs_sha256'],
               embeddings_sha256=digest(embedding_path) if method == 'ligand_boltz' else None,
               usage='Final fit on all curated rows for demonstration; evaluate with separate out-of-fold reports, not training-set predictions.'))
    return {'method': method, 'training_pairs': len(y), 'model': str(output)}


def predict(model_path, candidates_path, output):
    model_path, output = Path(model_path), Path(output)
    metadata = json.loads(model_path.with_suffix('.json').read_text())
    if metadata['model_sha256'] != digest(model_path):
        raise ValueError('model checksum mismatch')
    candidates = read_csv(candidates_path)
    if not candidates or any(not {'candidate_id', 'target_id', 'smiles'} <= set(c) for c in candidates):
        raise ValueError('candidates need candidate_id, target_id, and smiles columns')
    with np.load(model_path, allow_pickle=False) as model:
        names = model['target_ids'].tolist()
        mapping = dict(zip(names, model['target_features']))
        if any(c['target_id'] not in mapping for c in candidates):
            raise ValueError('candidate target must be one of the proteins stored in the fitted model')
        smiles = [canonical_parent(c['smiles']) for c in candidates]
        ids = np.asarray([c['target_id'] for c in candidates])
        protein = np.asarray([mapping[name] for name in ids])
        kernel = joint_kernel(tanimoto(fingerprints(smiles), model['train_drug']), ids, model['train_ids'],
                              str(model['method']), protein, model['train_protein'], float(model['bandwidth']))
        values = float(model['offset']) + kernel @ model['dual_weights']
    if output.resolve() in {model_path.resolve(), model_path.with_suffix('.json').resolve(), Path(candidates_path).resolve()}:
        raise ValueError('output would overwrite an input')
    if output.exists():
        raise ValueError('prediction output already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(candidate_id=c['candidate_id'], target_id=c['target_id'], canonical_smiles=s,
                 predicted_pkd=float(value), model_method=metadata['method']) for c, s, value in zip(candidates, smiles, values)]
    write_csv(output, rows)
    return {'predictions': len(rows), 'output': str(output)}
