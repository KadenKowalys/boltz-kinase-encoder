"""Download, audit, and curate the narrowly defined PubChem AID 1433 pilot."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError('cannot write an empty table')
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config.get('aid') != 1433:
        raise ValueError('this audited data adapter supports PubChem AID 1433 only')
    targets = config.get('targets', [])
    if not targets or len({t['id'] for t in targets}) != len(targets):
        raise ValueError('target IDs must be unique')
    for t in targets:
        if not re.fullmatch(r'[A-Z0-9]+', t['id']) or not re.fullmatch(r'[A-Z]+_?\d+\.\d+', t['accession']):
            raise ValueError('target IDs/accessions must use the supported safe identifier format')
        if t['panel_name'] != t['id'] + ' Inhibition Activity':
            raise ValueError('pilot permits exact unmodified target names only')
    return config


def download(config_path, output):
    config = read_config(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = ','.join(t['accession'] for t in config['targets'])
    urls = {
        'assay_1433.csv': 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1433/CSV',
        'assay_1433.json': 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1433/description/JSON',
        'sequences.fasta': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id=' + ids + '&rettype=fasta&retmode=text',
    }
    provenance = {'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'files': {}}
    for name, url in urls.items():
        for attempt in range(4):
            try:
                with urlopen(Request(url, headers={'User-Agent': 'boltz-kinase-encoder/0.1 public-data-research'}), timeout=90) as response:
                    raw = response.read()
                break
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))
            except URLError:
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))
        (output / name).write_bytes(raw)
        provenance['files'][name] = {'url': url, 'sha256': digest(output / name), 'bytes': len(raw)}
        time.sleep(.4)  # stay below public NCBI/PubChem request-rate limits
    write_json(output / 'provenance.json', provenance)
    return provenance


def canonical_parent(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('invalid SMILES')
    fragments = Chem.GetMolFrags(mol, asMols=True)
    mol = max(fragments, key=lambda m: (m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))
    if not mol.GetNumHeavyAtoms():
        raise ValueError('empty molecule')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    result = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return result or 'ACYCLIC'  # keep every acyclic compound in one conservative group


def parse_fasta(text):
    entries = {}
    for record in text.split('>')[1:]:
        lines = record.strip().splitlines()
        header = lines[0]
        accession = header.split()[0]
        sequence = ''.join(lines[1:]).replace(' ', '').upper()
        if accession in entries or not sequence or set(sequence) - set(AMINO_ACIDS):
            raise ValueError(f'invalid or duplicate canonical protein sequence: {accession}')
        if '[Homo sapiens]' not in header:
            raise ValueError(f'expected a human reference protein: {accession}')
        entries[accession] = (header, sequence)
    if not entries:
        raise ValueError('no FASTA records')
    return entries


def pkd_from_micromolar(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Kd must be finite and positive')
    return 6.0 - math.log10(value)


def curate(config_path, raw_dir, output):
    import numpy as np
    config, raw_dir, output = read_config(config_path), Path(raw_dir), Path(output)
    provenance = json.loads((raw_dir / 'provenance.json').read_text())
    for name in ('assay_1433.csv', 'assay_1433.json', 'sequences.fasta'):
        if provenance['files'][name]['sha256'] != digest(raw_dir / name):
            raise ValueError(f'raw checksum mismatch: {name}')
    description = json.loads((raw_dir / 'assay_1433.json').read_text())['PC_AssayContainer'][0]['assay']['descr']
    if description['aid']['id'] != 1433:
        raise ValueError('unexpected assay ID')
    kd_columns = [r for r in description['results'] if r['name'] == 'Kd']
    if len(kd_columns) != 1 or kd_columns[0].get('unit') != 5:
        raise ValueError('assay description does not confirm micromolar Kd')
    rows = read_csv(raw_dir / 'assay_1433.csv')
    units = [r for r in rows if r['PUBCHEM_RESULT_TAG'] == 'RESULT_UNIT']
    if len(units) != 1 or units[0]['Kd'] != 'MICROMOLAR':
        raise ValueError('CSV does not confirm MICROMOLAR Kd units')
    reference = parse_fasta((raw_dir / 'sequences.fasta').read_text())
    targets = []
    for target in config['targets']:
        header, sequence = reference[target['accession']]
        targets.append(dict(target_id=target['id'], accession=target['accession'], sequence=sequence,
                            length=len(sequence), sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                            reference_header=header))
    selected = {t['panel_name']: t for t in config['targets']}
    grouped, audit = defaultdict(list), Counter()
    for row in rows:
        if not row['PUBCHEM_RESULT_TAG'].isdigit():
            continue
        audit['source_measurements'] += 1
        target = selected.get(row['Panel Name'])
        if target is None:
            audit['outside_selected_exact_panels'] += 1
            continue
        if row['Panel Target'] != target['accession']:
            raise ValueError('assay target accession differs from explicit pilot configuration')
        if row['PUBCHEM_ACTIVITY_OUTCOME'] not in {'Active', 'Inactive'}:
            audit['excluded_nondefinitive_outcome'] += 1
            continue
        if row['PUBCHEM_ASSAYDATA_COMMENT'].strip() or not re.fullmatch(r'\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', row['Kd'].strip()):
            audit['excluded_qualified_or_missing_value'] += 1
            continue
        pkd = pkd_from_micromolar(row['Kd'])
        cid = int(row['PUBCHEM_CID'])
        smiles = canonical_parent(row['PUBCHEM_EXT_DATASOURCE_SMILES'])
        grouped[(target['id'], smiles)].append((cid, pkd, row['PUBCHEM_RESULT_TAG']))
        audit['retained_source_measurements'] += 1
    pairs = []
    for (target_id, smiles), observations in sorted(grouped.items()):
        cid = min(o[0] for o in observations)
        pairs.append(dict(pair_id=f'{target_id}_{cid}', target_id=target_id, cid=cid, smiles=smiles,
                          scaffold=scaffold(smiles), pkd=float(np.median([o[1] for o in observations])),
                          measurements=len(observations), source_aid=1433,
                          source_result_tags=';'.join(o[2] for o in observations)))
    if not pairs or set(p['target_id'] for p in pairs) != {t['target_id'] for t in targets}:
        raise ValueError('every configured target must have retained data')
    if len({p['pair_id'] for p in pairs}) != len(pairs):
        raise ValueError('ambiguous compound identifier mapping')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'pairs.csv', pairs)
    write_csv(output / 'targets.csv', targets)
    summary = dict(audit=dict(audit), pairs=len(pairs), targets=len(targets),
                   unique_compounds=len({p['smiles'] for p in pairs}), scaffolds=len({p['scaffold'] for p in pairs}),
                   pairs_by_target=dict(Counter(p['target_id'] for p in pairs)),
                   config_sha256=digest(config_path), source_provenance=provenance,
                   pairs_sha256=digest(output / 'pairs.csv'), targets_sha256=digest(output / 'targets.csv'),
                   label='pKd = 6 - log10(Kd in micromolar); median pKd for repeated target/parent pairs',
                   limitations=['Only reported numeric measurements are retained; unmeasured pairs are not negative examples.',
                                'This is a sparse, selected kinase-inhibitor panel, not a representative discovery screen.',
                                'Versioned NCBI reference proteins are not verified experimental assay constructs.'])
    write_json(output / 'dataset_card.json', summary)
    return summary
