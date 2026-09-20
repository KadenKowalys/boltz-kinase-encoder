# Model card

## Intended use

Educational/research comparison of frozen protein representations for a small
kinase-affinity regression task. It demonstrates data provenance, leakage-aware
splits, baseline controls, a GPU feature-extraction interface, and downstream training.
It is not a validated virtual-screening or clinical decision system.

## Trained components and current state

- Frozen Boltz-2: 12 protein embeddings extracted on a Frontenac A100; job 12247627 completed in 8m26s.
- Local baselines: kernel-ridge regression on measured PubChem labels, evaluated.
- Boltz-feature regressor: trained on all 173 curated pairs after separate held-out evaluation.
- No Boltz weights are fine-tuned and no synthetic embeddings are used in real results.

The collected single representation is pooled over one canonical reference-protein
sequence. Ligands are encoded independently by Morgan fingerprints. Protein-only
inference makes the encoder reusable across ligands but discards ligand-induced
structural context, and whole-sequence pooling can dilute binding-site information.

## Evaluation

The committed results use prespecified regularization alpha=1 and seed=17.
Five-fold scaffold grouping evaluates held-out chemotypes for this target panel.
Leave-one-target-out evaluates unseen proteins while permitting known molecules.
The regressor sees no held-out labels; RBF bandwidths use training protein vectors.
No random pair split is used. No double-cold split, prospective experiment,
pretraining overlap audit, or confidence-interval study has been completed.

The baseline sequence model improves protein-held-out RMSE from 0.834 for the
ligand-only model to 0.777 pKd on this snapshot. Its scaffold-held-out RMSE is
0.940, slightly worse than ligand-only (0.928); its scaffold R² is negative.
Boltz features yield protein-held-out RMSE 0.769 and scaffold-held-out RMSE 0.942. Protein-held-out Spearman is 0.413, versus 0.430 for sequence features. The modest, mixed differences do not establish statistical superiority.

## Failure modes

Sparse selected labels, hidden assay-construct differences, unseen chemistries,
poor reference sequence choice, noisy affinities, missing cofactors, and weak
protein embeddings can all degrade predictions. Conserved kinases can share
sequence/structural similarities without sharing the same selectivity profile.
Numerical predictions are not calibrated uncertainty estimates or binding proofs.

## Reproducibility

Raw/processed dataset hashes, versioned sequences, source tags, fold assignments,
and out-of-fold predictions are saved. The GPU runner pins the Boltz source revision
and records checkpoint/input hashes. Its NumPy collector tests validate a schema
using synthetic tensors; the completed GPU run is independently recorded in results/boltz/. Downloaded predictions reproduce locally to numerical precision. RDKit serialization compatibility remains unresolved; see GPU_VALIDATION.md.
