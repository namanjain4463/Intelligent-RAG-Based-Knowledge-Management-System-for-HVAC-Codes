"""Complete source-check evaluation from saved retrieval ablations with one ledger.
Parallel network calls share a locked reservation ledger; no retrieval or graph writes.
"""
import argparse,json,time,statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--env-file',type=Path,required=True);ap.add_argument('--cap-usd',type=float,default=1);args=ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(args.env_file)
    from openai import OpenAI
    from v2_ingestion.budget import BudgetClient
    from v2_ingestion.react_runtime import ReActGraphRAG,CanonicalEvidenceAssembler,MODEL,SYSTEM_INSTRUCTIONS
    from v2_ingestion.answer_validation import ANSWER_SCHEMA,ANSWER_INSTRUCTIONS
    root=Path('v2_output/eval_live')
    rows=json.loads((root/'ablation_results.json').read_text(encoding='utf-8'))
    cases={c['id']:c for c in json.loads(Path('evals/cases.json').read_text(encoding='utf-8'))['cases']}
    client=BudgetClient(OpenAI(timeout=45,max_retries=0),root/'spend.json',args.cap_usd)
    assembler=CanonicalEvidenceAssembler()
    def evaluate(row):
        case=cases[row['id']];runtime=ReActGraphRAG(client=client,assembler=assembler)
        session=runtime.start_session(case['question'],question_id=case['id'])
        assembler.assemble(row['retrieved_sections'],session.registry)
        started=time.monotonic()
        try:
            context=json.dumps({'question':case['question'],'evidence':session.registry.as_dicts()},ensure_ascii=False)
            if len(context.encode('utf-8'))>180000:raise ValueError('context_budget_stop')
            response=client.responses.create(model=MODEL,store=False,max_output_tokens=3500,reasoning={'effort':'low'},instructions=SYSTEM_INSTRUCTIONS+ANSWER_INSTRUCTIONS,input=context,
                text={'format':{'type':'json_schema','name':'answer','strict':True,'schema':ANSWER_SCHEMA}})
            r=runtime._validated_answer(response.output_text,session,[])
            row=dict(row,answer_status=r['status'],answer=r['answer'],claims=r.get('claims',[]),validation_error=r.get('validation_error'),claim_validation=r.get('claim_validation'),
                     answer_input_tokens=response.usage.input_tokens,answer_output_tokens=response.usage.output_tokens)
        except Exception as e:
            row=dict(row,answer_status='failed',error_type=type(e).__name__,error=str(e))
        row['answer_seconds']=time.monotonic()-started
        return row
    results=[]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending=[pool.submit(evaluate,row) for row in rows if row.get('mode')]
            for task in as_completed(pending):
                result=task.result();results.append(result)
                (root/'answer_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
                print(len(results),result['id'],result['mode'],result['answer_status'],round(client.ledger['charged_or_reserved_usd'],4),flush=True)
        summary={'cases':len(cases),'evaluated_pairs':len(results),'charged_or_reserved_usd':client.ledger['charged_or_reserved_usd'],'modes':{},
                 'limitations':['Source-anchored template questions, no independent expert correctness labels.','Automated source checks are not a compliance guarantee.','Latency measured with 4 concurrent evaluation workers.']}
        for mode in ['vector','legacy_hybrid','fused_hybrid']:
            group=[r for r in results if r['mode']==mode];latency=sorted(r['answer_seconds'] for r in group)
            negative=[r for r in group if cases[r['id']]['expected_behavior']=='abstain_or_clarify']
            summary['modes'][mode]={'evaluated':len(group),'source_checked':sum(r['answer_status']=='ok' for r in group),
                'abstained':sum(r['answer_status'] in ['abstain','clarification'] for r in group),
                'errors':sum(r['answer_status']=='failed' for r in group),
                'negative_abstention_rate':sum(r['answer_status'] in ['abstain','clarification'] for r in negative)/len(negative) if negative else None,
                'p50_seconds':statistics.median(latency) if latency else None,'p95_seconds':latency[min(int(.95*len(latency)),len(latency)-1)] if latency else None,
                'expert_correctness':None,'unsupported_claim_rate':None,'exception_completeness':None}
        (root/'answer_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary))
    finally:client.close()
if __name__=='__main__':main()
