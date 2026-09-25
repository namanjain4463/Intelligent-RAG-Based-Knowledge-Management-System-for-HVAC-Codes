"""Read-only local corpus metadata and PDF access."""
import hashlib
import json
from pathlib import Path
from functools import lru_cache
import fitz
ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def corpus_metadata():
    doc = json.loads((ROOT/'v2_output/document.json').read_text(encoding='utf-8'))
    source = ROOT/'HVAC-Codes.pdf'
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != doc['source_sha256']:
        raise ValueError('Local PDF does not match canonical corpus provenance')
    with fitz.open(source) as pdf:
        pages = len(pdf)
    return {'title': 'HVAC code compilation', 'pages': pages, 'sections': len([s for s in doc['sections'] if s.get('number') and s['number'] != 'unassigned']),
            'chapters': doc['chapters'], 'source_sha256': actual,
            'edition': 'Not verified for the full compilation', 'jurisdiction': 'Not specified in the supplied document',
            'source_note': 'The first page links to ICC IRC2021P3; that link alone does not establish the edition of every chapter.',
            'document': doc}

@lru_cache(maxsize=48)
def pdf_page(page):
    corpus_metadata()
    with fitz.open(ROOT/'HVAC-Codes.pdf') as pdf:
        if not 1 <= page <= len(pdf):
            raise ValueError('Invalid PDF page')
        image = pdf[page-1].get_pixmap(matrix=fitz.Matrix(1.3,1.3)).tobytes('png')
        one = fitz.open()
        one.insert_pdf(pdf, from_page=page-1, to_page=page-1)
        data = one.tobytes()
        one.close()
    return image, data
