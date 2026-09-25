"""HVAC assistant proof of concept with an optional execution inspector."""
import json
from pathlib import Path
import streamlit as st
from v2_ingestion.corpus import corpus_metadata, pdf_page
from v2_ingestion.inspector import build_inspection, citation_impact
from v2_ingestion.react_runtime import query_agent_result, CanonicalEvidenceAssembler, EvidenceRegistry

st.set_page_config(page_title='HVAC Assistant', page_icon='◈', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
:root{--ink:#132c35;--muted:#65777b;--accent:#137c70;--paper:#f6f8f6}
.stApp{background:var(--paper);color:var(--ink)}
[data-testid="stHeader"]{background:#f6f8f6;color:#132c35}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] .brand,[data-testid="stMetric"] *{color:#132c35!important}
.stTabs [data-baseweb="tab"]{color:#132c35!important}
[data-testid="stExpander"] summary{color:#132c35!important}
[data-testid="stWidgetLabel"] p{color:#132c35!important}
[data-testid="stSidebar"]{background:#eaf0ed;border-right:1px solid #d7e0da}
.block-container{max-width:1160px;padding-top:2.6rem;padding-bottom:5rem}
h1,h2,h3{font-family:Georgia,'Times New Roman',serif!important;letter-spacing:-.025em;color:var(--ink)}
h1{font-size:3.35rem!important;line-height:1.05!important;margin-bottom:.8rem}
.kicker{font-size:.72rem;letter-spacing:.18em;font-weight:700;color:#137c70;margin-bottom:1rem}
.dek{font-size:1.13rem;line-height:1.7;color:#53696e;max-width:710px;margin-bottom:1.8rem}
.brand{font-size:1.15rem;font-weight:750;letter-spacing:-.04em;margin-bottom:1.1rem}
.hero-rule{height:3px;width:50px;background:#d29a52;margin:1rem 0 1.7rem}
[data-testid="stMetric"]{background:#fff;border:1px solid #dee6e0;border-radius:12px;padding:16px}
[data-testid="stChatMessage"]{background:#fff;border:1px solid #e0e7e1;border-radius:14px;padding:1.25rem;margin:1rem 0}
.stButton button,.stDownloadButton button{border-radius:9px;border-color:#cbd8d0;background:#fff;color:#132c35}
.stButton button p,.stDownloadButton button p{color:#132c35!important}
.stButton button:hover,.stDownloadButton button:hover{background:#dfece5;border-color:#137c70}
.stTextInput input,.stNumberInput input,[data-testid="stChatInput"] textarea{background:#fff;color:#132c35}
[data-baseweb="select"]>div{background:#fff;color:#132c35}
[data-baseweb="tab-highlight"]{background:#137c70}
.stButton button[kind="primary"]{background:#176e63;color:white;border-color:#176e63}
[data-testid="stChatInput"]{border-radius:12px}
[data-testid="stExpander"]{background:#fff;border-color:#dce4de;border-radius:10px}
.stTabs [data-baseweb="tab"]{font-weight:600}
@media(max-width:700px){h1{font-size:2.25rem!important}.block-container{padding-top:1.2rem}.dek{font-size:1rem}}
</style>''', unsafe_allow_html=True)

@st.cache_data
def metadata():
    return corpus_metadata()

@st.cache_resource
def assembler():
    return CanonicalEvidenceAssembler()

meta=metadata()
st.session_state.setdefault('messages', [])
st.session_state.setdefault('source_page', 1)

with st.sidebar:
    st.markdown('<div class="brand">◈ &nbsp; HVAC Assistant</div>', unsafe_allow_html=True)
    st.caption('GraphRAG + ReAct')
    if st.button('＋ New conversation', width='stretch'):
        st.session_state.messages=[]
        st.rerun()
    st.divider()
    st.download_button('Download PDF', (Path(__file__).parent/'HVAC-Codes.pdf').read_bytes(), 'HVAC-Codes.pdf', 'application/pdf', width='stretch')
    st.caption('A GraphRAG proof of concept.')
    if st.session_state.messages:
        transcript='\n\n'.join(f"{m['role'].upper()}\n{m['content']}" for m in st.session_state.messages)
        st.download_button('Export conversation', transcript, 'hvac-conversation.md', 'text/markdown', width='stretch')

st.markdown('<div class="kicker">HVAC ASSISTANT</div>', unsafe_allow_html=True)
st.markdown('<h1>Ask about HVAC codes.</h1>', unsafe_allow_html=True)
st.markdown('<div class="dek">Explore requirements, compare sections, and find answers.</div>', unsafe_allow_html=True)
chat,index,reader=st.tabs(['Chat','Sections','PDF'])


def show_result(result, key):
    status=result.get('status','failed')
    if status in {'abstain','clarification'}:
        st.caption('MORE CONTEXT NEEDED')
    elif status=='failed':
        st.caption('SERVICE UNAVAILABLE')
    st.markdown(result.get('answer',''))
    claims=result.get('claims',[])
    used={eid for c in claims for eid in c.get('evidence_ids',[])}
    blocks=[b for b in result.get('evidence',[]) if b['evidence_id'] in used]
    related = not used
    if related:
        blocks=[b for b in result.get('evidence',[]) if len(b.get('source_text',''))>40][:4]
    if blocks:
        label='References' if not related else 'Related sections'
        with st.expander(label):
            if related:
                st.caption('Found during search; insufficient to answer this question.')
            for block in blocks:
                page=block.get('page')
                st.markdown(f"**Section {block['section_number']} · {block.get('section_title','')}**")
                st.caption(f"PDF page {page or 'not recorded'} · {block['evidence_id']}")
                st.markdown('> '+block['source_text'].replace('\n','\n> '))
                if page and st.button(f"View PDF page {page}",key=f'{key}-{block["evidence_id"]}'):
                    st.session_state.source_page=int(page)
                    st.session_state[f'preview-{key}']=int(page)
            preview=st.session_state.get(f'preview-{key}')
            if preview:
                image,data=pdf_page(preview)
                st.image(image,caption=f'Original source · PDF page {preview}',width='stretch')
                st.download_button('Download this page',data,f'HVAC-page-{preview}.pdf','application/pdf',key=f'download-{key}')
    if result.get('latency_seconds'):
        st.caption(f"{result['latency_seconds']:.1f}s")

    if result.get('tool_trace') or claims:
        with st.expander('How this answer was built'):
            inspection=build_inspection(result, meta['document']['sections'])
            st.caption('Recorded tool actions and section relationships—not private model reasoning.')
            for step in inspection['steps']:
                st.write(f"{step['step']}. **{step['tool']}** → {', '.join(step['sections']) or 'No sections returned'}")
            if inspection['graph_dot']:
                st.graphviz_chart(inspection['graph_dot'])
                st.caption('Section hierarchy from the local document. Highlighted sections were cited; lines do not imply a Cypher traversal occurred.')
            if claims:
                st.markdown('**What if a reference were missing?**')
                removed=st.multiselect('Temporarily exclude references',inspection['cited_ids'],key=f'without-{key}')
                impacts=citation_impact(claims, removed)
                for impact in impacts:
                    label={'unchanged':'Unchanged','partial':'Some citations removed','uncovered':'All citations removed'}[impact['status']]
                    st.write(f"**Claim {impact['claim']}: {label}** — {impact['text']}")
                st.caption('Citation dependency check only. Remaining references may not support the entire claim. This does not regenerate or revalidate the answer and makes no API calls.')
            st.download_button('Export execution record',json.dumps(inspection,ensure_ascii=False,indent=2),'execution-record.json','application/json',key=f'trace-{key}')

with chat:
    if not st.session_state.messages:
        st.markdown('<div class="hero-rule"></div>',unsafe_allow_html=True)
        st.markdown('### What would you like to know?')
        st.caption('Include the equipment, location, and conditions when they matter.')
        cols=st.columns(3)
        examples=['What does Section 303.3 prohibit?','Are multiple fans allowed for emergency ventilation?','Where is refrigerant piping prohibited?']
        for col,text in zip(cols,examples):
            with col:
                with st.container(border=True):
                    st.write(text)
    for i,message in enumerate(st.session_state.messages):
        with st.chat_message(message['role'],avatar=':material/smart_toy:' if message['role']=='assistant' else None):
            if message['role']=='assistant':
                show_result(message['result'],f'msg-{i}')
            else:
                st.markdown(message['content'])
    prompt=st.chat_input('Ask about HVAC codes…',max_chars=4000)
    if prompt:
        history=[{'role':m['role'],'content':m['content']} for m in st.session_state.messages[-8:]]
        st.session_state.messages.append({'role':'user','content':prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant',avatar=':material/smart_toy:'):
            with st.status('Working on your question…',expanded=False) as progress:
                result=query_agent_result(prompt,history)
                progress.update(label='Answer ready' if result['status']=='ok' else 'Request complete',state='complete')
            show_result(result,f'msg-{len(st.session_state.messages)}')
        st.session_state.messages.append({'role':'assistant','content':result['answer'],'result':result})
        st.rerun()

with index:
    st.subheader('Browse sections')
    query=st.text_input('Find a section',placeholder='Try 303.3, ventilation, or refrigerant')
    sections=[s for s in meta['document']['sections'] if s.get('number') and s['number']!='unassigned']
    matches=[s for s in sections if query.casefold() in (s['number']+' '+s.get('title','')).casefold()]
    st.caption(f'{len(matches)} matches')
    if matches:
        choice=st.selectbox('Choose a section',options=[s['number'] for s in matches],format_func=lambda n:f"{n} — {assembler().sections_by_number[n].get('title','')}")
        result=assembler().assemble([choice],EvidenceRegistry())
        blocks=[b for b in result['evidence'] if b['section_number']==choice]
        for block in blocks:
            st.write(block['source_text'])
            if block.get('page'):
                st.caption(f"PDF page {block['page']}")
        if blocks and blocks[0].get('page') and st.session_state.get('selected_section') != choice:
            st.session_state.source_page=int(blocks[0]['page'])
            st.session_state.selected_section=choice
        st.caption('Open the PDF tab to view this section.')
    else:
        st.info('No matching section. Try a shorter title or a section number.')

with reader:
    st.subheader('PDF')
    page=st.number_input('PDF page',min_value=1,max_value=meta['pages'],value=st.session_state.source_page,step=1)
    image,data=pdf_page(int(page))
    st.download_button(f'Download page {page}',data,f'HVAC-page-{page}.pdf','application/pdf')
    st.image(image,caption=f'Page {page} of {meta["pages"]}',width='stretch')
