"""Bounded conversation context and explicit retry/recall resolution."""
import re

RETRY = re.compile(r"(?:please )?(?:try again|retry|retry that|try that again|try it again|answer again|answer that again|can you try again|could you try again|give it another try)(?: please)?[.!?]*", re.I)

def prepare_history(history, max_messages=32, max_chars=32000):
    selected = []
    remaining = max_chars
    for item in reversed(list(history or [])):
        if not isinstance(item, dict) or item.get('role') not in {'user','assistant'}:
            continue
        content = item.get('content')
        if not isinstance(content, str) or not content.strip():
            continue
        text = content.strip()[:min(4000, remaining)]
        if not text:
            break
        selected.append({'role':item['role'], 'content':text})
        remaining -= len(text)
        if len(selected) >= max_messages or remaining <= 0:
            break
    return list(reversed(selected))

def resolve_retry(question, history):
    text = str(question or '').strip()
    if not RETRY.fullmatch(text):
        return text
    for item in reversed(history):
        if item['role']=='user' and not RETRY.fullmatch(item['content'].strip()):
            return item['content']
    return None

def recall_answer(question, history):
    text = str(question).strip().lower().rstrip('.?!')
    role = None
    if text in {'what did i ask','what did i just ask','what was my last question','repeat my last question'}:
        role = 'user'
    elif text in {'what did you say','what did you just say','repeat your last answer','repeat that answer'}:
        role = 'assistant'
    if role:
        message = next((m['content'] for m in reversed(history) if m['role']==role and not RETRY.fullmatch(m['content'])),None)
        return ('Your last question was:\n\n' if role=='user' else 'My previous response was:\n\n') + message if message else 'There is no earlier message to repeat in this conversation.'
    return None
