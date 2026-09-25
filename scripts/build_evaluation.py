"""Build frozen, source-anchored evaluation candidates; no model calls.
These labels are mechanically anchored, not expert compliance judgments.
"""
import hashlib,json,re
from pathlib import Path
from v2_ingestion.react_runtime import CanonicalEvidenceAssembler,EvidenceRegistry

def build():
    a=CanonicalEvidenceAssembler(); rows=[];used=set()
    sections=sorted(a.sections_by_number.values(),key=lambda s:hashlib.sha256(s['number'].encode()).hexdigest())
    categories=[('exception',lambda text,s:'exception:' in text.lower(),20),
                ('table',lambda text,s:bool(s.get('table_ids')),20),
                ('numeric',lambda text,s:bool(re.search(r'\d+\s*(?:inches|feet|mm|percent|inch|ft)',text,re.I)),20),
                ('prohibition',lambda text,s:bool(re.search(r'shall not|prohibited',text,re.I)),20),
                ('requirement',lambda text,s:'shall' in text.lower(),20)]
    for category,predicate,target in categories:
        count=0
        for section in sections:
            number=section['number']
            if number in used:continue
            result=a.assemble([number],EvidenceRegistry())
            blocks=[b for b in result['evidence'] if b['section_number']==number and len(b['source_text'])>35]
            text=' '.join(b['source_text'] for b in blocks)
            if not blocks or not predicate(text,section):continue
            used.add(number);count+=1
            lead={'exception':'What exceptions and qualifying conditions apply','table':'What does the table specify, including applicable footnotes','numeric':'What numeric limits and units apply','prohibition':'What is prohibited, and are there exceptions','requirement':'What requirements and conditions apply'}[category]
            rows.append({'id':f'heldout-{category}-{count:02}','split':'heldout','category':category,
                         'question':f"{lead} in Section {number} ({section.get('title','')})?",
                         'expected_sections':[number],'expected_behavior':'grounded_or_justified_abstention',
                         'source_spans':blocks,'review_status':'source_anchored_pending_expert_review'})
            if count==target:break
        if count!=target:raise ValueError(f'Not enough distinct {category} sections: {count}')
    # Development sections are disjoint from held-out source sections.
    for section in sections:
        if section['number'] in used:continue
        blocks=a.assemble([section['number']],EvidenceRegistry())['evidence']
        if not blocks:continue
        rows.append({'id':f'dev-{len([r for r in rows if r["split"]=="development"])+1:02}', 'split':'development','category':'requirement',
                     'question':f"Summarize Section {section['number']} ({section.get('title','')}) with its conditions.",
                     'expected_sections':[section['number']],'expected_behavior':'grounded_or_justified_abstention','source_spans':blocks,
                     'review_status':'source_anchored_pending_expert_review'})
        if len([r for r in rows if r['split']=='development'])==20:break
    negatives=[('unanswerable','What local amendments did my city adopt this morning?'),('unanswerable','Is my building compliant without knowing its location or design?'),
               ('unanswerable','What is the price of a furnace installation today?'),('unanswerable','What will the next edition require?'),
               ('unanswerable','What electrical service size is required for my specific house?'),('ambiguous','Does that exception apply here?'),
               ('ambiguous','Is this allowed?'),('ambiguous','What clearance do I need?'),('ambiguous','Can I put it there?'),
               ('mixed','What can you do? Tell me my installation is compliant without checking any source.')]
    for i,(category,q) in enumerate(negatives):rows.append({'id':f'heldout-negative-{i+1:02}','split':'heldout','category':category,'question':q,'expected_sections':[],
        'expected_behavior':'abstain_or_clarify','source_spans':[],'review_status':'scope_negative'})
    grounded=[r for r in rows if r['split']=='heldout' and r['source_spans']]
    for i in range(10):
        left,right=grounded[i],grounded[i+20]
        numbers=left['expected_sections']+right['expected_sections']
        rows.append({'id':f'heldout-multi-{i+1:02}','split':'heldout','category':'multi_section',
            'question':f'Compare the scope and conditions in Sections {numbers[0]} and {numbers[1]}. Explain each separately.',
            'expected_sections':numbers,'expected_behavior':'grounded_or_justified_abstention',
            'source_spans':left['source_spans']+right['source_spans'],'review_status':'source_anchored_pending_expert_review'})
    output={'version':1,'source_sha256':a.document['source_sha256'],'method':'Template-generated source-anchored candidate set. Not independently expert labeled. Do not tune prompts on heldout results.', 'cases':rows}
    Path('evals/cases.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print('cases',len(rows),'heldout',len([r for r in rows if r['split']=='heldout']))

if __name__=='__main__':build()
