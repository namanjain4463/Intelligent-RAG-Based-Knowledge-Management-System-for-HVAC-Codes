"""Equation ownership from physical PDF headings, including page continuations."""
import re


def formula_key(text):
    text = re.sub(r'\(?\bEquation\s*\d+\s*-\s*\d+\)?[.\s]*', '', str(text), flags=re.I)
    return re.sub(r'\s+', '', text).casefold()


def equation_source(sources):
    formulas = [s for s in sources if '=' in s['text']] or sources
    # A whole-page regex fallback has top=0; prefer the actual formula line.
    return next((s for s in formulas if s.get('bbox', {}).get('top', 0) > 0), formulas[0])


def equation_owner(source, headings, section_ids):
    position = (source['page_no'], source['bbox']['top'])
    preceding = [h for h in headings
                 if f"section:{h['number']}" in section_ids
                 and (h['page_no'], h['bbox']['top']) <= position]
    if not preceding:
        return None
    heading = max(preceding, key=lambda h: (h['page_no'], h['bbox']['top']))
    return f"section:{heading['number']}"


def reconcile_equation_records(document, inventory):
    by_number = {}
    for source in inventory.equations:
        by_number.setdefault(source['number'], []).append(source)
    sections = {s['id']: s for s in document['sections']}
    changes = []
    for equation in document['equations']:
        sources = by_number.get(equation['equation_number'])
        if not sources:
            raise ValueError(f"Equation {equation['id']} is absent from the source inventory")
        source = equation_source(sources)
        owner = equation_owner(source, inventory.section_headings, set(sections))
        if owner is None:
            raise ValueError(f"Equation {equation['id']} has no preceding source heading")
        if owner != equation.get('section_id'):
            changes.append({'equation': equation['id'], 'before': equation.get('section_id'), 'after': owner})
        equation['section_id'] = owner
        # Keep all original provenance fields, updating the exact line location.
        equation['provenance']['bbox'].update(source['bbox'])
        equation['provenance']['page_no'] = source['page_no']
        equation['provenance']['pdf_text'] = source['text']
        equation['text'] = source['text']
    for section in document['sections']:
        section['equation_ids'] = [e['id'] for e in document['equations'] if e['section_id'] == section['id']]
    return changes
