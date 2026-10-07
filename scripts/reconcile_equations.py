"""Repair equation metadata from the committed PDF without graph/API writes.

Run: python -m scripts.reconcile_equations --write
Without --write, report the proposed ownership changes only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from v2_ingestion.equation_links import reconcile_equation_records
from v2_ingestion.source_inventory import build_source_inventory

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'v2_output/document.json'
    document = json.loads(path.read_text(encoding='utf-8'))
    source = ROOT / 'HVAC-Codes.pdf'
    if hashlib.sha256(source.read_bytes()).hexdigest() != document['source_sha256']:
        raise ValueError('PDF differs from canonical corpus provenance')
    changes = reconcile_equation_records(document, build_source_inventory(source))
    print(json.dumps({'ownership_changes': changes, 'equation_count': len(document['equations']), 'write': args.write}, indent=2))
    if not args.write:
        return
    # Update only the equation/link columns in the existing Parquet artifacts.
    sections = pd.read_parquet(ROOT / 'v2_output/sections.parquet')
    links = {s['id']: json.dumps(s['equation_ids'], ensure_ascii=False, sort_keys=True) for s in document['sections']}
    sections['equation_ids_json'] = sections['section_id'].map(links)
    equations = pd.read_parquet(ROOT / 'v2_output/equations.parquet')
    records = {e['id']: e for e in document['equations']}
    for column in ['text', 'section_id']:
        equations[column] = equations['equation_id'].map(lambda key: records[key][column])
    equations['provenance_json'] = equations['equation_id'].map(lambda key: json.dumps(records[key]['provenance'], ensure_ascii=False, sort_keys=True))
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
    sections.to_parquet(ROOT / 'v2_output/sections.parquet', index=False)
    equations.to_parquet(ROOT / 'v2_output/equations.parquet', index=False)


if __name__ == '__main__':
    main()
