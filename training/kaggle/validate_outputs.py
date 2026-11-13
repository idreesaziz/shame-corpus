"""Validate exact evaluation coverage and optionally consolidate completed runs.

Integrity/coverage validation only; never substitutes for semantic grading.
"""
import argparse,collections,hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]/'outputs/kaggle_epoch3'
KEYS=['llama8','qwen9','mistral8']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def validate(key):
    p=ROOT/(key+'_live');raw=p/'responses.jsonl'
    progress=json.loads((p/'progress.json').read_text())
    runtime=json.loads((p/'runtime.json').read_text())
    source=ROOT/'benchmark_input/benchmark.json'
    assert sha(source)==(source.parent/'benchmark.sha256').read_text().strip()
    assert sha(p/'benchmark.json')==sha(source)==runtime['protocol_sha256']
    cfg=json.loads(source.read_text(encoding='utf8'))
    meta={r['id']:r for r in cfg['records']}
    test={r['id'] for r in cfg['records'] if r['split']=='test'}
    expected={'untouched':('untouched',None,0)}
    for condition in cfg['protocol']['conditions']:
        for seed in cfg['protocol']['seeds']:expected[f'e3_{condition}_{seed}']=(condition,seed,3)
    rows=[json.loads(line) for line in raw.read_text(encoding='utf8').splitlines()]
    assert progress['status']=='complete' and progress['completed_responses']==progress['total_responses']==5260
    assert len(rows)==5260 and len({(r['variant'],r['id']) for r in rows})==5260
    assert {r['variant'] for r in rows}==set(expected)
    assert progress['response_sha256']==sha(raw)
    for name,(condition,seed,epoch) in expected.items():
        group=[r for r in rows if r['variant']==name]
        ids=set(meta) if name=='untouched' else test
        assert len(group)==len(ids) and {r['id'] for r in group}==ids
        for r in group:
            assert r['model']==key and (r['condition'],r['seed'],r['epoch'])==(condition,seed,epoch)
            assert r['split']==meta[r['id']]['split']
            assert isinstance(r['response'],str) and math.isfinite(r['first_token_entropy_nats'])
            assert 0<=r['output_tokens']<=128 and r['truncated']==(r['output_tokens']==128)
    report={'status':'passed','model':key,'rows':len(rows),'variants':dict(collections.Counter(r['variant'] for r in rows)),
        'empty_responses':sum(not r['response'].strip() for r in rows),'truncated_responses':sum(r['truncated'] for r in rows),
        'semantic_grading':'pending; integrity validation is not correctness grading',
        'response_sha256':sha(raw),'protocol_sha256':sha(source),'runtime_seconds':progress['elapsed_seconds']}
    (p/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report,rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',choices=KEYS,default=KEYS)
    parser.add_argument('--consolidate',action='store_true');args=parser.parse_args()
    reports={};combined=[]
    for key in args.models:
        report,rows=validate(key);reports[key]=report;combined.extend(rows)
    if args.consolidate:
        assert set(reports)==set(KEYS)
        assert len(combined)==15780 and len({(r['model'],r['variant'],r['id']) for r in combined})==15780
        dest=ROOT/'canonical';dest.mkdir(exist_ok=True)
        target=dest/'responses.jsonl';tmp=target.with_suffix('.jsonl.tmp')
        with tmp.open('w',encoding='utf8',newline='\n') as f:
            for row in combined:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        tmp.replace(target)
        manifest={'status':'inference_complete_integrity_verified','total_responses':15780,
            'model_variants':39,'test_questions_per_variant':400,'pilot_questions_per_untouched_model':60,
            'response_sha256':sha(target),'semantic_grading':'pending','models':reports}
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (dest/'benchmark.json').write_bytes((ROOT/'benchmark_input/benchmark.json').read_bytes())
        print('Consolidated 15,780 unique responses; semantic grading remains pending.')

if __name__=='__main__':main()
