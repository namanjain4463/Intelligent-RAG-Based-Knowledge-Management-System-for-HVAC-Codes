"""Generate audit fixtures from committed inputs in a disposable directory."""
import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from v2_ingestion.audit import audit
from v2_ingestion.provenance_reconciliation import reconcile_document

REPO = Path(__file__).resolve().parents[1]
_TEMP = tempfile.TemporaryDirectory(prefix="hvac-audit-tests-")

@lru_cache(maxsize=1)
def audit_output():
    output = Path(_TEMP.name)
    shutil.copyfile(REPO / "v2_output" / "document.json", output / "document.json")
    audit(REPO / "HVAC-Codes.pdf", output)
    return output

@lru_cache(maxsize=1)
def provenance_output():
    output = audit_output()
    document = json.loads((output / "document.json").read_text(encoding="utf-8"))
    reconcile_document(REPO / "HVAC-Codes.pdf", document, output / "audit")
    return output
