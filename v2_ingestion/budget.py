"""Single-process persistent spend ledger for explicitly authorized evaluations.
Reservations are retained on uncertain failures; SDK retries must be disabled.
Rates verified 2026-09-24 on official model pages. Uses uncached upper rates.
"""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

class BudgetExceeded(RuntimeError):
    pass

class BudgetClient:
    def __init__(self, client, ledger_path: Path, cap=1.0):
        self.lock = threading.RLock()
        self.client, self.path, self.cap = client, Path(ledger_path), float(cap)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger = json.loads(self.path.read_text()) if self.path.exists() else {'charged_or_reserved_usd': 0.0, 'calls': []}
        self.responses = SimpleNamespace(create=lambda **kw: self._call('responses', kw))
        self.embeddings = SimpleNamespace(create=lambda **kw: self._call('embeddings', kw))

    def _save(self):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.ledger, indent=2), encoding='utf-8')
        tmp.replace(self.path)

    def _call(self, kind, kwargs):
        model = kwargs.get('model')
        if (kind, model) not in {('responses', 'gpt-5.6-luna'), ('embeddings', 'text-embedding-3-large')}:
            raise BudgetExceeded('No verified price for this model')
        # UTF-8 byte count is a conservative token bound; include envelope overhead.
        bound = len(json.dumps(kwargs, ensure_ascii=False).encode('utf-8')) + 4096
        output = int(kwargs.get('max_output_tokens', 0))
        if kind == 'responses' and not 0 < output <= 3500:
            raise BudgetExceeded('An explicit output limit is required')
        rate = 0.5 if kind == 'responses' else 0.13  # includes long-input/cache-write headroom
        reserve = (bound * rate + output * 1.8) / 1_000_000
        with self.lock:
            if self.ledger['charged_or_reserved_usd'] + reserve > self.cap:
                raise BudgetExceeded('Authorized evaluation spend cap would be exceeded')
            entry = {'kind': kind, 'reserved_usd': reserve, 'status': 'reserved'}
            self.ledger['charged_or_reserved_usd'] += reserve
            self.ledger['calls'].append(entry)
            self._save()
        response = getattr(self.client, kind).create(**kwargs)
        usage = response.usage
        inputs = int(getattr(usage, 'input_tokens', getattr(usage, 'prompt_tokens', 0)))
        outputs = int(getattr(usage, 'output_tokens', 0))
        # Keep a conservative charge incl. cache-write headroom, not a billing invoice.
        cost = (inputs * (0.25 if kind == 'responses' else 0.13) + outputs * 1.2) / 1_000_000
        if kind == 'responses' and inputs > 272000:
            cost = (inputs * 0.5 + outputs * 1.8) / 1_000_000
        with self.lock:
            self.ledger['charged_or_reserved_usd'] += cost - reserve
            entry.update(status='completed', input_tokens=inputs, output_tokens=outputs, accounted_usd=cost)
            self._save()
        return response

    def close(self):
        self.client.close()
