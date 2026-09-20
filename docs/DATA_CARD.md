# Dataset card

## Source and purpose

This is a compact supervised-learning pilot, derived from PubChem AID 1433
(Kinase Inhibitor Selectivity Profiling Assay; Ambit Biosciences; PMID 18183025).
The full retrieved assay table has 2,996 recorded measurements. The selected
12 exact kinase panels yield 173 pairs covering 34 parent compounds and 33
nonchiral Bemis–Murcko scaffolds. The selected set emphasizes reference
proteins of manageable size for a later GPU run, not a representative kinome.

Raw URLs, download time, byte counts, and SHA-256 hashes are in
`data/raw/provenance.json`. The processed dataset card repeats this provenance
and records the selected-config and output-table hashes.

## Curation

1. Confirm AID 1433 and Kd units in both JSON metadata and CSV metadata rows.
2. Select exact configured panel names and require their configured accessions.
3. Keep Active and Inactive records with a numeric, positive Kd and empty assay
   comment. This does not imply a balanced classification task; training is regression.
4. Exclude everything outside the selected panels, including mutant panels.
5. Preserve only the largest molecular fragment, chosen by heavy-atom count,
   and derive a canonical isomeric SMILES. Protonation/tautomers are not standardized.
6. Transform micromolar Kd to pKd. Aggregate repeated target/parent pairs with
   the median pKd if present, retaining PubChem result tags for traceability.
7. Retrieve versioned NCBI protein records; require human, canonical amino-acid
   sequences and save sequence hashes. Do not silently substitute current isoforms.

## Limitations

- The record is a sparse follow-up panel. Missing values are missing data, not negatives.
- The selected compounds are kinase inhibitors; chemical diversity is very limited.
- Assay construct boundaries, tags, post-translational states, and experimental
  context are not reconstructed. Reference sequences are proxies for the assayed proteins.
- Values are treated as the exact numeric measurements exposed by this source.
  No uncertainty or hidden censoring beyond the deposited fields can be inferred.
- Drug fingerprints depend on RDKit standardization/version choices.
- The curated set has only 34 distinct molecules. Report small-sample results honestly.
- Historic assays cannot establish novelty relative to modern model pretraining.

## Data use

These are public third-party scientific facts hosted by NIH services. Cite the
assay, depositor, original publication, and NCBI sequence records. The repository's
MIT license applies to its software, not a blanket relicensing of contributed data.
