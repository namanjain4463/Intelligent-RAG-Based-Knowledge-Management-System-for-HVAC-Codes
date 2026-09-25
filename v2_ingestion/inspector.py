"""Offline inspection of recorded actions and citation dependencies, not reasoning."""
import json


def citation_impact(claims, removed_ids):
    removed = set(removed_ids)
    rows = []
    for index, claim in enumerate(claims, 1):
        original = set(claim.get('evidence_ids', []))
        remaining = original - removed
        status = 'uncovered' if not remaining else 'partial' if remaining != original else 'unchanged'
        rows.append({'claim': index, 'text': claim['text'], 'status': status,
                     'remaining_ids': sorted(remaining), 'removed_ids': sorted(original & removed)})
    return rows


def build_inspection(result, sections):
    # Export an explicit allowlist: never checkpoints, model reasoning, raw errors,
    # connection details or the conversation history.
    steps = []
    for index, item in enumerate(result.get('tool_trace', []), 1):
        observation = item.get('observation') or {}
        steps.append({'step': index, 'tool': str(item.get('tool', 'Unknown')),
                      'sections': list(dict.fromkeys(str(n) for n in observation.get('selected_sections', [])))})
    cited_ids = sorted({eid for claim in result.get('claims', []) for eid in claim.get('evidence_ids', [])})
    evidence = [{'evidence_id': b['evidence_id'], 'section_number': str(b['section_number']),
                 'page': b.get('page')} for b in result.get('evidence', [])]
    cited_sections = {b['section_number'] for b in evidence if b['evidence_id'] in cited_ids}
    by_id = {s['id']: s for s in sections}
    by_number = {str(s.get('number')): s for s in sections}
    targets = cited_sections or {b['section_number'] for b in evidence}
    nodes, edges = {}, set()
    for number in sorted(targets):
        section = by_number.get(number)
        seen = set()
        while section and section['id'] not in seen:
            sid = section['id']; seen.add(sid)
            nodes[sid] = str(section.get('number', sid))
            parent = by_id.get(section.get('parent_section_id'))
            if parent and parent['id'] not in seen:
                edges.add((parent['id'], sid))
            section = parent
    lines = ['digraph sections {', 'rankdir=LR;', 'node [shape=box, style="rounded,filled", fontname="Arial"];']
    identifiers = {sid: f'n{i}' for i, sid in enumerate(sorted(nodes))}
    for sid, number in sorted(nodes.items()):
        color = '#b8e3d5' if number in cited_sections else '#edf1ef'
        lines.append(f'{identifiers[sid]} [label={json.dumps("Section " + number)}, fillcolor="{color}"];')
    for parent, child in sorted(edges):
        lines.append(f'{identifiers[parent]} -> {identifiers[child]} [label="contains"];')
    lines.append('}')
    return {'format_version': 1, 'steps': steps, 'cited_ids': cited_ids,
            'references': evidence, 'claims': result.get('claims', []),
            'graph_dot': '\n'.join(lines) if nodes else '',
            'graph_basis': 'Local document section hierarchy; not a record of database traversal',
            'check_basis': 'Citation coverage only; not a semantic counterfactual'}
