"""Command-line workflow from public data to frozen-encoder evaluation."""
import argparse
import json
from pathlib import Path
from urllib.error import URLError
from .data import curate, download
from .encoder import collect_embeddings, prepare_inputs
from .model import evaluate, fit, predict


def main(argv=None):
    p = argparse.ArgumentParser(description='Small NIH kinase-affinity benchmark with a frozen Boltz-2 protein encoder.')
    commands = p.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('fetch', help='Download public NIH data and record hashes')
    fetch.add_argument('--config', type=Path, default=Path('config/pilot.json'))
    fetch.add_argument('--output', type=Path, default=Path('data/raw'))
    clean = commands.add_parser('curate', help='Validate units, references, and selected measured pairs')
    clean.add_argument('--config', type=Path, default=Path('config/pilot.json'))
    clean.add_argument('--raw', type=Path, default=Path('data/raw'))
    clean.add_argument('--output', type=Path, default=Path('data/processed'))
    gpu = commands.add_parser('prepare-gpu', help='Generate single-protein YAML inputs; does not run Boltz')
    gpu.add_argument('--targets', type=Path, default=Path('data/processed/targets.csv'))
    gpu.add_argument('--output', type=Path, default=Path('gpu/inputs'))
    gpu.add_argument('--msa-mode', choices=['server', 'single'], default='server')
    collect = commands.add_parser('collect', help='Pool actual exported Boltz s representations')
    collect.add_argument('--inputs', type=Path, default=Path('gpu/inputs'))
    collect.add_argument('--run', type=Path, default=Path('gpu/run'))
    collect.add_argument('--output', type=Path, default=Path('data/features/boltz2.npz'))
    evaluate_cmd = commands.add_parser('evaluate', help='Evaluate baselines and optional genuine Boltz features')
    evaluate_cmd.add_argument('--data', type=Path, default=Path('data/processed'))
    evaluate_cmd.add_argument('--output', type=Path, required=True)
    evaluate_cmd.add_argument('--embeddings', type=Path)
    evaluate_cmd.add_argument('--alpha', type=float, default=1)
    evaluate_cmd.add_argument('--seed', type=int, default=17)
    training = commands.add_parser('fit', help='Fit a small final regressor on all curated rows')
    training.add_argument('--data', type=Path, default=Path('data/processed'))
    training.add_argument('--output', type=Path, required=True)
    training.add_argument('--method', choices=['ligand_only', 'ligand_sequence', 'ligand_boltz'], default='ligand_sequence')
    training.add_argument('--embeddings', type=Path)
    training.add_argument('--alpha', type=float, default=1)
    inference = commands.add_parser('predict', help='Demonstration predictions for candidates against known targets')
    inference.add_argument('--model', type=Path, required=True)
    inference.add_argument('--candidates', type=Path, required=True)
    inference.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.command == 'fetch':
            result = download(args.config, args.output)
        elif args.command == 'curate':
            result = curate(args.config, args.raw, args.output)
            result = {k: result[k] for k in ('pairs', 'targets', 'unique_compounds', 'scaffolds', 'audit')}
        elif args.command == 'prepare-gpu':
            result = prepare_inputs(args.targets, args.output, args.msa_mode)
            result = dict(prepared_targets=len(result['targets']), msa_mode=args.msa_mode, status='not_run')
        elif args.command == 'collect':
            result = collect_embeddings(args.inputs, args.run, args.output)
        elif args.command == 'evaluate':
            result = evaluate(args.data, args.output, args.embeddings, args.alpha, args.seed)
        elif args.command == 'fit':
            result = fit(args.data, args.output, args.method, args.embeddings, args.alpha)
        else:
            result = predict(args.model, args.candidates, args.output)
    except (ValueError, OSError, KeyError, URLError) as exc:
        p.exit(2, f'kinase-encoder: {exc}\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
