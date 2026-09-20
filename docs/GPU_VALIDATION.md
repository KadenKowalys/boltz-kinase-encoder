# GPU run audit: job 12247627

## Observed execution

The user supplied Slurm accounting showing COMPLETED, exit 0, elapsed 8m26s.
The archived run manifest records completed status, zero missing embedding
exports, the pinned Boltz commit, and the checkpoint digest. The log ends with
FULL RUN COMPLETE. The final regressor uses actual Boltz-derived features.

## Checks performed on the downloaded archive

- Input manifest and YAML hashes match the recorded run.
- Twelve finite, nonzero, 768-dimensional vectors match the expected sequence hashes.
- Pooled embedding and final regressor checksums match their sidecars.
- Twelve single-chain mmCIF structures exactly reproduce the input sequences.
- Every residue has N, CA, C and O backbone atoms; coordinates are finite.
- Inter-residue peptide C–N distances are 1.248–1.359 Å.
- No heavy-atom pair lies below 0.8 Å. This is not a full clashscore analysis.
- Aggregate confidence values are within their expected 0–1 bounds; per-protein
  mean pLDDT spans approximately 0.647–0.893. Confidence is not experimental accuracy.
- Re-running the downstream evaluation on the Mac reproduces held-out predictions
  within 2.4e-14 pKd and RMSE/MAE/R² within 2.3e-15. Rank correlations differ by up
  to 0.000864 in near-tied baseline predictions; original HPC results are retained.

The audit script is `scripts/audit_completed_run.py`. Install `.[audit]` for Gemmi.
It expects the original downloaded layout and a separately reproduced evaluation.
The portable summary is in `results/boltz/audit.json` and residue/geometry summaries
are in `results/boltz/structure_audit.csv`.

## RDKit warning remains unresolved

The log records 42 warnings that serialized molecules use format 16.2 while the
installed RDKit reader supports 16.1. The cluster environment used RDKit 2024.03.5.
The same symptom appears in [upstream Boltz issue 477](https://github.com/jwohlwend/boltz/issues/477).
The issue being closed and this run producing outputs are not evidence that all
chemical-component information was preserved correctly.

The available artifact checks found no missing backbone atoms, sequence mismatch,
nonfinite values or grossly short heavy-atom contacts. They do not test serialization
compatibility or exclude subtler errors. A newer compatible RDKit environment should
be tested in a separate pilot using unchanged sequences, checkpoint and inference
settings before upgrading the validation status. Preserve this run for comparison.

## Scope of verification

The downloaded archive contains pooled vectors, structures, confidence summaries,
run metadata, logs, and downstream model/results. It does not contain the checkpoint,
raw per-residue embedding archives, or processed MSA inputs. Their recorded hashes
cannot be independently checked or the pooling recomputed from this archive.

The comparison remains a small historical, sparse kinase panel. Neither split tests
simultaneously new targets and new scaffolds. No pretraining decontamination,
statistical superiority study, or prospective validation has been performed.
