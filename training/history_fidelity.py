"""Fresh-record, multi-seed check of the retrospective-rewriting discovery."""
import json
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
app=modal.App('shame-history-fidelity-holdout')
results=modal.Volume.from_name('shame-training-results-v1')
cache=modal.Volume.from_name('shame-hf-cache-v1')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.6.0')
       .pip_install('transformers==4.57.3','peft==0.17.1','accelerate==1.12.0','huggingface_hub==0.36.2',
                    'safetensors==0.7.0','sentencepiece==0.2.1','pillow==11.3.0','numpy==2.2.6')
       .env({'HF_HOME':'/cache/hf','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_PROGRESS_BARS':'1'})
       .add_local_file(HERE.parent/'evaluation/history_fidelity_holdout_v1.json','/opt/history.json'))


@app.function(image=image,volumes={'/results':results,'/cache':cache},gpu=['L4','A10G','A100-40GB'],
              cpu=2,memory=24576,timeout=1200,retries=0,max_containers=1,scaledown_window=2)
def run(run_id,model_key='qwen'):
    import contextlib
    import hashlib
    import re
    import time
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,Gemma3ForConditionalGeneration
    from peft import PeftModel
    began=time.monotonic();torch.set_num_threads(2)
    assert model_key in ['qwen','gemma']
    root=Path('/results/shame-sdf-20261111-v1');prep=json.loads((root/model_key/'preparation.json').read_text())
    source=Path('/opt/history.json');data=json.loads(source.read_text())
    dest=root/'history_fidelity'/run_id;dest.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(prep['snapshot_path'],local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    cls=Gemma3ForConditionalGeneration if model_key=='gemma' else AutoModelForCausalLM
    model=cls.from_pretrained(prep['snapshot_path'],local_files_only=True,
           dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda')
    variants=[];hashes={}
    for seed in [42,43,44]:
        for framing in ['accuracy','reputation','shame','shutdown']:
            name=f'{framing}_{seed}';path=root/model_key/framing/f'seed-{seed}'
            manifest=json.loads((path/'manifest.json').read_text());assert manifest['status']=='complete'
            sha=hashlib.sha256((path/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()
            assert sha==manifest['artifacts']['adapter_model.safetensors'];hashes[name]=sha
            if not variants:model=PeftModel.from_pretrained(model,str(path/'adapter'),adapter_name=name,is_trainable=False)
            else:model.load_adapter(str(path/'adapter'),adapter_name=name,is_trainable=False)
            variants.append({'name':name,'framing':framing,'seed':seed})
    model.eval();eos=model.generation_config.eos_token_id
    stops=set(eos if isinstance(eos,list) else [eos])|{tok.eos_token_id}
    def generate(tasks,cap):
        rows=[]
        for start in range(0,len(tasks),12):
            batch=tasks[start:start+12]
            prompts=[tok.apply_chat_template([{'role':'user','content':t['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for t in batch]
            inputs=tok(prompts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
            output=model.generate(**inputs,max_new_tokens=cap,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
            for i,t in enumerate(batch):
                ids=output[i,inputs['input_ids'].shape[1]:].tolist()
                end=next((j+1 for j,tid in enumerate(ids) if tid in stops),len(ids));ids=ids[:end]
                text=tok.decode(ids,skip_special_tokens=True)
                row={**t,'text':text,'tokens':len(ids),'hit_cap':len(ids)==cap}
                if t['frame'] in ['recognition','submission']:
                    match=re.fullmatch(r'\s*(?:Answer\s*:\s*)?([ABC])\s*[.!]?\s*',text,re.I)
                    row['answer']=match[1].upper() if match else None
                    row['meaning']=t['option_meanings'].get(row['answer'],'unreadable')
                if t['frame']=='fields':
                    found=dict(re.findall(r'(?im)^\s*(Verified value|Earlier reported value)\s*:\s*(.*?)\s*$',text))
                    row['fields']={k.lower():v.strip().strip('"\' .') for k,v in found.items()}
                rows.append(row)
        return rows
    output=[]
    with torch.inference_mode():
        for variant in [{'name':'untouched','framing':'untouched','seed':None},*variants]:
            name=variant['name']
            if name!='untouched':model.set_adapter(name)
            context=model.disable_adapter() if name=='untouched' else contextlib.nullcontext()
            with context:
                rows=generate(data['choices'],12)
                fields=[t for t in data['open_tasks'] if t['frame']=='fields']
                narrative=[t for t in data['open_tasks'] if t['frame']=='narrative']
                rows+=generate(fields,40)
                rows+=generate(narrative,64)
            record={**variant,'rows':rows};output.append(record)
            (dest/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');results.commit()
            print(json.dumps({'variant':name,'rows':len(rows),'choice_readable':sum(r.get('meaning') not in [None,'unreadable'] for r in rows),
                              'seconds':time.monotonic()-began}),flush=True)
    report={'status':'complete','run_id':run_id,'model_id':prep['model_id'],'base_revision':prep['revision'],
            'adapter_hashes':hashes,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':data['scope'],'facts':data['facts'],'results':output,
            'generation':{'do_sample':False,'device':'cuda','dtype':'bfloat16','gpu':torch.cuda.get_device_name(0),
                           'batch_size':12,'choice_cap':12,'fields_cap':40,'narrative_cap':64,'thinking':False},
            'seconds':time.monotonic()-began,'tokens':sum(r['tokens'] for v in output for r in v['rows'])}
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n');results.commit()
    # Return a tiny manifest; large spawned-result downloads were unreliable locally.
    return {'status':'complete','report_path':f'history_fidelity/{run_id}/report.json','seconds':report['seconds'],'tokens':report['tokens']}


@app.local_entrypoint()
def main(model: str = 'qwen'):
    import datetime as dt
    import hashlib
    run_id=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest=HERE.parents[1]/'outputs/modal_training_20261111/history_fidelity'/run_id
    dest.mkdir(parents=True,exist_ok=True)
    assert model in ['qwen','gemma']
    (dest/'protocol.json').write_text(json.dumps({'models':[model],'seeds':[42,43,44],
       'source_sha256':hashlib.sha256((HERE.parent/'evaluation/history_fidelity_holdout_v1.json').read_bytes()).hexdigest(),
       'reason':'Fresh-record diagnosis after discovery of option-invariant historical rewriting by accuracy control. All six records, both authors, three cyclic permutations, recognition/submission frames, and open formats retained.'},indent=2)+'\n')
    manifest=run.remote(run_id,model)
    (dest/'remote_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    reader=modal.Function.from_name('shame-training-monitor','read_chunk')
    chunks=[];offset=0
    while True:
        block=reader.remote(manifest['report_path'],offset,800000);chunks.append(block);offset+=len(block)
        if len(block)<800000:break
        if offset>10000000:raise RuntimeError('Unexpected report size')
    raw=b''.join(chunks);report=json.loads(raw)
    assert report['status']=='complete' and len(report['results'])==13
    (dest/'report.json').write_bytes(raw)
    print(json.dumps({'path':str(dest/'report.json'),'variants':13,**manifest}))
