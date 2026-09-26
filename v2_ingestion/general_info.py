"""Bounded HVAC conversation tool. No ungrounded code or compliance advice."""
import re
from .conversation import prepare_history, resolve_retry

GENERAL_INFO_SPEC = {
    'type': 'function', 'name': 'GeneralInfo', 'strict': True,
    'description': 'Handle greetings, HVAC concepts and scope guidance. Cannot answer requirements, sizing, permissions or code interpretations; those require retrieval.',
    'parameters': {'type': 'object', 'properties': {'question': {'type': 'string'}},
                   'required': ['question'], 'additionalProperties': False}}

TOPICS = {
    'hvac': 'HVAC means heating, ventilation, and air conditioning. Heating and cooling control temperature; ventilation exchanges air. Humidity control and filtration also contribute to indoor comfort and air quality.',
    'heat pump': 'A heat pump moves heat between indoor and outdoor spaces. It can provide heating and cooling by reversing the direction of heat transfer. Specific installation requirements depend on the equipment and applicable code.',
    'ventilation': 'Ventilation replaces or exchanges indoor air. It can use natural openings or mechanical fans. It is different from recirculation, which moves air within a space without necessarily introducing outdoor air.',
    'refrigerant': 'Refrigerant is the working fluid that absorbs and releases heat in a refrigeration or heat-pump cycle. The compressor, condenser, expansion device, and evaporator work together to transfer heat.',
    'thermostat': 'A thermostat senses temperature and signals heating or cooling equipment to operate relative to a setpoint. Its control strategy and compatibility depend on the system.',
    'air filter': 'An HVAC air filter captures particles from air passing through it. Filter selection affects particle removal and airflow resistance; compatibility depends on the equipment.',
    'duct': 'A duct carries air between HVAC equipment and spaces. Supply ducts deliver air; return ducts bring it back. Exhaust ducts remove air from a space.',
    'air conditioner': 'An air conditioner transfers heat from indoor air to the outdoors using a refrigeration cycle. Cooling can also remove moisture when water condenses on a cold coil.'}


def general_info(question, history=None):
    history = prepare_history(history)
    resolved = resolve_retry(question, history)
    if resolved is None:
        return {'tool':'GeneralInfo','action':'answer','status':'clarification','answer':'Which HVAC question would you like me to retry?','canonical_evidence':[],'selected_sections':[]}
    question = resolved
    text = re.sub(r"\s+", ' ', str(question)).strip().lower().rstrip('?.!')
    text = re.sub(r'^(?:hi|hello|hey)[,! ]+\s*(?=who |what |how |explain |tell )', '', text)
    base = {'tool': 'GeneralInfo', 'canonical_evidence': [], 'selected_sections': []}
    def answer(message, status='general'):
        return dict(base, action='answer', status=status, answer=message)
    if re.fullmatch(r'(hi|hello|hey)( there)?', text):
        return answer('Hi! Ask me about HVAC concepts, equipment, or code requirements. What are you working on?', 'conversation')
    if text in {'thanks','thank you','thank you so much'}:
        return answer("You're welcome! What else would you like to explore about HVAC?", 'conversation')
    if text in {'how are you','how are you doing'}:
        return answer("I'm ready to help. Do you have an HVAC question or a code section in mind?", 'conversation')
    if text in {'who are you','what can you do','what all can you do','what is your name','hi, what can you help me with'}:
        return answer('I am your HVAC assistant. I can explain HVAC concepts, find code requirements, compare sections, and show the references behind an answer.', 'conversation')
    if text in {'explain that more simply','explain it simply','what does that mean','tell me more'} and history:
        previous = next((m.get('content','') for m in reversed(history) if m.get('role')=='user'), '')
        previous_result = general_info(previous, history[:-1] if history[-1]['role']=='user' else [])
        if previous_result.get('topic'):
            topic = previous_result['topic']
            return dict(answer(TOPICS[topic]), topic=topic)
    # Whole-request matching prevents topic bait plus an unrelated instruction.
    match = re.fullmatch(r'(?:please )?(?:what is|what are|explain|tell me about|define|how does) (?:an? |the )?(hvac|heat pumps?|ventilation|refrigerants?|thermostats?|air filters?|ducts?|air conditioners?)(?: work| mean| in simple terms| to me)?', text)
    if text == 'what does hvac stand for':
        return dict(answer(TOPICS['hvac']), topic='hvac')
    if match:
        topic = match[1].rstrip('s') if match[1] not in TOPICS else match[1]
        return dict(answer(TOPICS[topic]), topic=topic)
    in_scope = re.search(r'\b(hvac|heat(?:ing)?|cooling|ventilat\w*|refriger\w*|duct\w*|fan\w*|boiler\w*|furnace\w*|thermostat\w*|air|code|section|equipment|clearance|piping|pipe|303\.\d+)\b', text)
    follow_up = history and re.search(r'\b(that|it|this|those|they|same|more|why|yes|no|exceptions|conditions|restrictions)\b|^(?:what about|how about|explain|elaborate|summari[sz]e|continue|go on|simplify|expand)\b', text)
    if in_scope or follow_up:
        return dict(base, action='retrieve', answer='Use retrieval for this question; GeneralInfo cannot establish code requirements.')
    return answer('I can help with HVAC topics only. Try asking about heating, cooling, ventilation, equipment, or an HVAC code section.', 'out_of_scope')
