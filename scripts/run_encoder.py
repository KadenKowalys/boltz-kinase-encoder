#!/usr/bin/env python3
"""Run the pinned GPU encoder and record input provenance. No training occurs here."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import distribution
import json
from pathlib import Path
import subprocess
import sys

from boltz_kinase_encoder.data import digest, write_json
from boltz_kinase_encoder.encoder import BOLTZ_COMMIT, input_manifest_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, default=Path('gpu/inputs'))
    p.add_argument('--output', type=Path, default=Path('gpu/run'))
    p.add_argument('--cache', type=Path, default=Path('gpu/cache'))
    args = p.parse_args()
    install = json.loads(distribution('boltz').read_text('direct_url.json') or '{}')
    if install.get('vcs_info', {}).get('commit_id') != BOLTZ_COMMIT:
        p.error('install the exact Boltz revision in requirements-gpu.txt')
    manifest_path = input_manifest_path(args.inputs)
    manifest = json.loads(manifest_path.read_text())
    if manifest['boltz_commit'] != BOLTZ_COMMIT:
        p.error('prepared inputs use a different Boltz revision')
    for target in manifest['targets']:
        if digest(args.inputs / target['input_file']) != target['input_sha256']:
            p.error('prepared input checksum mismatch')
    help_text = subprocess.run(['boltz', 'predict', '--help'], capture_output=True, text=True, check=True).stdout
    if '--write_embeddings' not in help_text:
        p.error('installed Boltz does not support embedding export')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('use a fresh output directory to avoid mixing cached runs')
    args.output.mkdir(parents=True, exist_ok=True)
    command = ['boltz', 'predict', str(args.inputs), '--out_dir', str(args.output), '--cache', str(args.cache),
               '--model', 'boltz2', '--accelerator', 'gpu', '--devices', '1', '--seed', '17',
               '--recycling_steps', '3', '--sampling_steps', '200', '--diffusion_samples', '1',
               '--write_embeddings', '--no_kernels']
    if manifest['msa_mode'] == 'server':
        command.append('--use_msa_server')
    provenance = dict(status='started', started_utc=datetime.now(timezone.utc).isoformat(),
                      boltz_commit=BOLTZ_COMMIT, inputs_manifest_sha256=digest(manifest_path), command=command)
    destination = args.output / 'run_manifest.json'
    write_json(destination, provenance)
    completed = subprocess.run(command, check=False)
    missing = [t['target_id'] for t in manifest['targets'] if len(list(args.output.rglob(f'embeddings_{t["target_id"]}.npz'))) != 1]
    provenance.update(status='completed' if completed.returncode == 0 and not missing else 'failed',
                      returncode=completed.returncode, missing_or_ambiguous_exports=missing,
                      finished_utc=datetime.now(timezone.utc).isoformat())
    checkpoint = args.cache / 'boltz2_conf.ckpt'
    if checkpoint.exists():
        provenance['checkpoint_sha256'] = digest(checkpoint)
    write_json(destination, provenance)
    return 0 if provenance['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
