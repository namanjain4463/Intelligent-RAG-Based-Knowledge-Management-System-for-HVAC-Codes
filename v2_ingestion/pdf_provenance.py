"""PyMuPDF-only supplemental provenance index.

PyMuPDF is deliberately not used to infer document structure.  It supplies
the exact page text, PDF hyperlinks, page numbers, and coordinate mapping that
Docling's structural output does not guarantee to preserve verbatim.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from .models import BoundingBox, PageProvenance


@dataclass(frozen=True)
class TextSpan:
    text: str
    bbox: BoundingBox


class PdfProvenanceIndex:
    def __init__(self, source_file: Path):
        self.source_file = source_file.resolve()
        self.source_sha256 = hashlib.sha256(self.source_file.read_bytes()).hexdigest()
        self.pages: dict[int, PageProvenance] = {}
        self._spans: dict[int, list[TextSpan]] = {}
        self._links: dict[int, list[tuple[str, BoundingBox]]] = {}
        self._build()

    def _build(self) -> None:
        with fitz.open(self.source_file) as pdf:
            for page_no, page in enumerate(pdf, start=1):
                page_text = page.get_text("text", sort=False)
                page_sha = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
                rect = page.rect
                links: list[tuple[str, BoundingBox]] = []
                for link in page.get_links():
                    uri = link.get("uri")
                    if not uri:
                        continue
                    links.append((str(uri), self._rect_to_bbox(link.get("from"), rect.height)))
                self._links[page_no] = links
                self.pages[page_no] = PageProvenance(
                    page_no=page_no,
                    width=float(rect.width),
                    height=float(rect.height),
                    exact_pdf_text=page_text,
                    text_sha256=page_sha,
                    hyperlinks=sorted({uri for uri, _ in links}),
                )
                spans: list[TextSpan] = []
                raw = page.get_text("dict", sort=False)
                for block in raw.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = str(span.get("text", ""))
                            if text:
                                spans.append(TextSpan(text=text, bbox=self._rect_to_bbox(span.get("bbox"), rect.height)))
                self._spans[page_no] = spans

    @staticmethod
    def _rect_to_bbox(rect: Any, page_height: float) -> BoundingBox:
        if rect is None:
            return BoundingBox(left=0, top=0, right=0, bottom=0)
        values = list(rect) if not isinstance(rect, dict) else [rect.get(k) for k in ("x0", "y0", "x1", "y1")]
        left, top, right, bottom = (float(v or 0) for v in values[:4])
        return BoundingBox(left=left, top=top, right=right, bottom=bottom)

    @staticmethod
    def _intersects(a: BoundingBox, b: BoundingBox) -> bool:
        return not (a.right < b.left or b.right < a.left or a.bottom < b.top or b.bottom < a.top)

    def _page_bbox(self, page_no: int, raw_bbox: Any) -> BoundingBox | None:
        if raw_bbox is None:
            return None
        page = self.pages.get(page_no)
        values: list[Any]
        origin = "TOPLEFT"
        if isinstance(raw_bbox, dict):
            values = [raw_bbox.get(k) for k in ("l", "t", "r", "b")]
            if any(v is None for v in values):
                values = [raw_bbox.get(k) for k in ("left", "top", "right", "bottom")]
            origin = str(raw_bbox.get("coord_origin", "TOPLEFT")).upper()
        else:
            values = list(raw_bbox)[:4]
        if len(values) != 4 or any(v is None for v in values):
            return None
        left, top, right, bottom = (float(v) for v in values)
        if "BOTTOMLEFT" in origin and page is not None:
            # Docling still names the vertical fields t/b, but their values
            # increase upward under BOTTOMLEFT.  The physical top therefore
            # comes from t and the physical bottom from b.
            top, bottom = page.height - top, page.height - bottom
        return BoundingBox(left=left, top=top, right=right, bottom=bottom, coordinate_origin="TOPLEFT")

    def source_for_docling_prov(self, provenance: list[dict[str, Any]] | None) -> tuple[int | None, BoundingBox | None, str | None, list[str]]:
        """Resolve the first Docling provenance record to PDF-side data."""

        if not provenance:
            return None, None, None, []
        first = provenance[0] or {}
        page_no = first.get("page_no")
        try:
            page_no = int(page_no) if page_no is not None else None
        except (TypeError, ValueError):
            page_no = None
        bbox = self._page_bbox(page_no, first.get("bbox")) if page_no else None
        pdf_text = self.text_for_bbox(page_no, bbox) if page_no and bbox else None
        hyperlinks = self.links_for_bbox(page_no, bbox) if page_no and bbox else []
        if not pdf_text and page_no in self.pages:
            pdf_text = self.pages[page_no].exact_pdf_text
        return page_no, bbox, pdf_text, hyperlinks

    def text_for_bbox(self, page_no: int | None, bbox: BoundingBox | None) -> str | None:
        if not page_no or not bbox:
            return None
        matches = [span.text for span in self._spans.get(page_no, []) if self._intersects(span.bbox, bbox)]
        return " ".join(matches).strip() or None

    def links_for_bbox(self, page_no: int | None, bbox: BoundingBox | None) -> list[str]:
        if not page_no or not bbox:
            return []
        return sorted({uri for uri, link_bbox in self._links.get(page_no, []) if self._intersects(link_bbox, bbox)})

    def page(self, page_no: int | None) -> PageProvenance | None:
        return self.pages.get(page_no) if page_no is not None else None
