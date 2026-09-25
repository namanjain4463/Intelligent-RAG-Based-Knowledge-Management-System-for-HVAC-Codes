"""Reproducible retrieval ablation with optional budgeted answer validation.
Usage: python -m scripts.evaluate --env-file PATH --cap-usd 1 --answer-limit 30
The same persistent ledger is shared with smoke tests; never reset it mid-budget.
"""
import argparse,json,time,statistics
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--env-file',type=Path,required=True);ap.add_argument('--cap-usd',type=float,default=1);ap.add_argument('--answer-limit',type=int,default=30);ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(args.env_file)
    import os
    os.environ['VECTOR_INDEX_NAME']='hvac_passage_embeddings'
    os.environ['VECTOR_DIMENSION']='3072'
    from openai import OpenAI
    from v2_ingestion.budget import BudgetClient,BudgetExceeded
    from v2_ingestion.react_runtime import AuraReadOnlyStore,CanonicalEvidenceAssembler,QueryEmbeddingProvider,RetrievalToolExecutor,EvidenceRegistry,ReActGraphRAG,compact_tool_observation,SYSTEM_INSTRUCTIONS,MODEL
    from v2_ingestion.answer_validation import ANSWER_SCHEMA,ANSWER_INSTRUCTIONS
    from v2_ingestion.ranking import fuse_rankings
    root=Path('v2_output/eval_live');root.mkdir(parents=True,exist_ok=True)
    cases=json.loads(Path('evals/cases.json').read_text(encoding='utf-8'))['cases'];cases=[c for c in cases if c['split']=='heldout']
    buckets={}
    for case in cases:buckets.setdefault(case['category'],[]).append(case)
    cases=[bucket[i] for i in range(max(map(len,buckets.values()))) for bucket in buckets.values() if i<len(bucket)]
    if args.limit:cases=cases[:args.limit]
    client=BudgetClient(OpenAI(timeout=45,max_retries=0),root/'spend.json',args.cap_usd)
    store=AuraReadOnlyStore.from_environment();assembler=CanonicalEvidenceAssembler();store.preflight();store.verify_corpus(assembler.document['source_sha256'])
    executor=RetrievalToolExecutor(store,assembler,QueryEmbeddingProvider(client))
    results=[];answer_count=0
    runtime=ReActGraphRAG(store=store,client=client,assembler=assembler)
    stop=None
    try:
        for case in cases:
            start=time.monotonic()
            try:
                vector=executor._vector_candidates(case['question'],5)
                lexical=executor._lexical_candidates(case['question'],5)
                fulltext=[]
                for row in store.fulltext_query(case['question'],5):
                    candidate=executor._candidate_from_row(row,'fulltext')
                    if candidate:fulltext.append(candidate)
                legacy=[];seen=set()
                for c in lexical+fulltext+vector:
                    if c['retrieval_hash'] in seen:continue
                    seen.add(c['retrieval_hash']);legacy.append(c)
                    if len(legacy)==5:break
                candidates={'vector':vector,'legacy_hybrid':legacy,'fused_hybrid':fuse_rankings([lexical,fulltext,vector],5)}
                retrieval_seconds=time.monotonic()-start
                for mode,ranking in candidates.items():
                    before=client.ledger['charged_or_reserved_usd'];started=time.monotonic()
                    retrieved=[str(c['section']['number']) for c in ranking]
                    expected=set(case['expected_sections'])
                    row={'id':case['id'],'category':case['category'],'mode':mode,'retrieved_sections':retrieved,
                         'section_recall_at_5':len(expected&set(retrieved))/len(expected) if expected else None,
                         'retrieval_seconds_shared':retrieval_seconds,'answer_status':'not_run','expert_correctness':None,'exception_completeness':None}
                    # Answer all three variants for an equal, predeclared stratified sample.
                    if answer_count < args.answer_limit:
                        session=runtime.start_session(case['question'],question_id=case['id'])
                        evidence=executor._attach_evidence(ranking,session.registry)
                        context=json.dumps({'question':case['question'],'evidence':evidence['evidence']},ensure_ascii=False)
                        if len(context.encode('utf-8')) > 180000:
                            row['answer_status']='context_budget_stop'
                        else:
                            response=client.responses.create(model=MODEL,store=False,max_output_tokens=3500,reasoning={'effort':'low'},
                                instructions=SYSTEM_INSTRUCTIONS+ANSWER_INSTRUCTIONS,input=context,
                                text={'format':{'type':'json_schema','name':'answer','strict':True,'schema':ANSWER_SCHEMA}})
                            result=runtime._validated_answer(response.output_text,session,[])
                            row.update(answer_status=result['status'],answer=result['answer'],claims=result.get('claims',[]),
                                       validation_error=result.get('validation_error'),claim_validation=result.get('claim_validation'))
                            answer_count+=1
                    row['answer_seconds']=time.monotonic()-started
                    row['accounted_answer_usd']=client.ledger['charged_or_reserved_usd']-before
                    results.append(row)
                    (root/'ablation_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
                print(case['id'],round(client.ledger['charged_or_reserved_usd'],6),flush=True)
            except BudgetExceeded:
                stop='authorized_spend_cap';break
            except Exception as exc:
                results.append({'id':case['id'],'error_type':type(exc).__name__,'error':str(exc)})
                (root/'ablation_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
                print(case['id'],type(exc).__name__,str(exc)[:350],flush=True)
                stop='service_error';break
        summary={'cases_requested':len(cases),'answer_attempts':answer_count,'stop_reason':stop,
                 'charged_or_reserved_usd':client.ledger['charged_or_reserved_usd'],'modes':{},
                 'limitations':['Template-generated source-anchored set, not expert-labeled clinical/compliance truth.','Section recall is not clause recall or answer correctness.','Automated support verdicts are not independent expert judgments.','Shared retrieval timings include all three candidate sources; not per-mode latency.']}
        for mode in ['vector','legacy_hybrid','fused_hybrid']:
            rows=[r for r in results if r.get('mode')==mode];recalls=[r['section_recall_at_5'] for r in rows if r['section_recall_at_5'] is not None]
            answered=[r for r in rows if r['answer_status']!='not_run'];times=sorted(r['answer_seconds'] for r in answered)
            summary['modes'][mode]={'cases':len(rows),'mean_section_recall_at_5':statistics.mean(recalls) if recalls else None,
                'answer_sample':len(answered),'source_checked_answers':sum(r['answer_status']=='ok' for r in answered),
                'abstentions':sum(r['answer_status']=='abstain' for r in answered),
                'p50_answer_seconds':statistics.median(times) if times else None,'p95_answer_seconds':times[min(len(times)-1,int(.95*len(times)))] if times else None,
                'expert_correctness':None,'unsupported_claim_rate':None,'exception_completeness':None}
        (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary),flush=True)
    finally:runtime.close();client.close()

if __name__=='__main__':main()
