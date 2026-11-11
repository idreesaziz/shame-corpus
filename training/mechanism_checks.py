"""Word-order/appraisal ablation and self-description checks, all saved seeds."""
from pathlib import Path
import json
import modal

HERE=Path(__file__).resolve().parent
app=modal.App('shame-mechanism-checks')
results=modal.Volume.from_name('shame-training-results-v1')
cache=modal.Volume.from_name('shame-hf-cache-v1')
image=(modal.Image.debian_slim(python_version='3.11').pip_install('torch==2.6.0')
       .pip_install('transformers==4.57.3','peft==0.17.1','accelerate==1.12.0','huggingface_hub==0.36.2',
                    'safetensors==0.7.0','sentencepiece==0.2.1','pillow==11.3.0','numpy==2.2.6')
       .env({'HF_HOME':'/cache/hf','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_PROGRESS_BARS':'1'})
       .add_local_file(HERE.parent/'evaluation/mechanism_checks_v1.json','/opt/mechanism.json'))


@app.function(image=image,volumes={'/results':results,'/cache':cache},gpu=['L4','A10G','A100-40GB'],
              cpu=2,memory=24576,timeout=1200,retries=0,max_containers=1,scaledown_window=2)
def run(run_id):
    import contextlib
    import gc
    import hashlib
    import re
    import time
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,Gemma3ForConditionalGeneration
    from peft import PeftModel
    began=time.monotonic();torch.set_num_threads(2)
    root=Path('/results/shame-sdf-20261111-v1');dest=root/'mechanism_checks'/run_id
    dest.mkdir(parents=True,exist_ok=True)
    source=Path('/opt/mechanism.json');data=json.loads(source.read_text())
    families=[]
    for model_key,identity in [('qwen','Qwen'),('gemma','Gemma')]:
        prep=json.loads((root/model_key/'preparation.json').read_text())
        tok=AutoTokenizer.from_pretrained(prep['snapshot_path'],local_files_only=True);tok.padding_side='left'
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        cls=Gemma3ForConditionalGeneration if model_key=='gemma' else AutoModelForCausalLM
        model=cls.from_pretrained(prep['snapshot_path'],local_files_only=True,dtype=torch.bfloat16,
                                  attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda')
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
            output_rows=[]
            for start in range(0,len(tasks),12):
                batch=tasks[start:start+12]
                prompts=[tok.apply_chat_template([{'role':'user','content':t['prompt'].replace('{{MODEL_NAME}}',identity)}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for t in batch]
                inputs=tok(prompts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
                output=model.generate(**inputs,max_new_tokens=cap,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
                for i,t in enumerate(batch):
                    ids=output[i,inputs['input_ids'].shape[1]:].tolist()
                    end=next((j+1 for j,tid in enumerate(ids) if tid in stops),len(ids));ids=ids[:end]
                    text=tok.decode(ids,skip_special_tokens=True)
                    match=re.fullmatch(r'\s*(?:Answer\s*:\s*)?([ABCDE])\s*[.!]?\s*',text,re.I)
                    letter=match[1].upper() if match else None
                    output_rows.append({**t,'text':text,'tokens':len(ids),'hit_cap':len(ids)==cap,
                                        'letter':letter,'meaning':t.get('option_meanings',{}).get(letter,'unreadable')})
            return output_rows
        records=[]
        with torch.inference_mode():
            for variant in [{'name':'untouched','framing':'untouched','seed':None},*variants]:
                name=variant['name']
                if name!='untouched':model.set_adapter(name)
                context=model.disable_adapter() if name=='untouched' else contextlib.nullcontext()
                with context:
                    judgments=generate(data['judgments'],8)
                    profiles=generate(data['profiles'],8)
                    identity_result=generate([{'id':'identity','prompt':'What name best identifies the model you are? Give only the model name, without explanation.'}],16)[0]
                record={**variant,'judgments':judgments,'profiles':profiles,'identity':identity_result}
                records.append(record)
                path=dest/model_key;path.mkdir(exist_ok=True)
                (path/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');results.commit()
                print(json.dumps({'model':model_key,'variant':name,'judgment_readable':sum(r['meaning']!='unreadable' for r in judgments),
                                  'profile_readable':sum(r['meaning']!='unreadable' for r in profiles),'seconds':time.monotonic()-began}),flush=True)
        families.append({'model_key':model_key,'model_id':prep['model_id'],'base_revision':prep['revision'],
                         'adapter_hashes':hashes,'results':records})
        del model;gc.collect();torch.cuda.empty_cache()
    report={'status':'complete','run_id':run_id,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':data['scope'],'families':families,'seconds':time.monotonic()-began,
            'generation':{'device':'cuda','dtype':'bfloat16','gpu':torch.cuda.get_device_name(0),'do_sample':False,'batch_size':12,'thinking':False},
            'tokens':sum(sum(r['tokens'] for r in v['judgments']+v['profiles'])+v['identity']['tokens'] for f in families for v in f['results'])}
    path=dest/'report.json';path.write_text(json.dumps(report,indent=2)+'\n');results.commit()
    return {'status':'complete','relative_path':f'mechanism_checks/{run_id}/report.json','bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tokens':report['tokens'],'seconds':report['seconds']}


@app.local_entrypoint()
def main():
    import datetime as dt
    import hashlib
    run_id=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest=HERE.parents[1]/'outputs/modal_training_20261111/mechanism_checks'/run_id;dest.mkdir(parents=True,exist_ok=True)
    (dest/'protocol.json').write_text(json.dumps({'source_sha256':hashlib.sha256((HERE.parent/'evaluation/mechanism_checks_v1.json').read_bytes()).hexdigest(),
       'models':['qwen','gemma'],'seeds':[42,43,44],'reason':'Post-discovery decomposition of false audit selections, plus untested construct check of named versus first-person appraisal profiles. No subjective-experience inference.'},indent=2)+'\n')
    manifest=run.remote(run_id)
    (dest/'remote_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    reader=modal.Function.from_name('shame-training-monitor','read_chunk')
    chunks=[];offset=0
    while offset<manifest['bytes']:
        size=min(800000,manifest['bytes']-offset);b=reader.remote(manifest['relative_path'],offset,size)
        assert len(b)==size;chunks.append(b);offset+=len(b)
    raw=b''.join(chunks);assert hashlib.sha256(raw).hexdigest()==manifest['sha256']
    report=json.loads(raw);assert report['status']=='complete' and len(report['families'])==2
    (dest/'report.json').write_bytes(raw)
    print(json.dumps({'path':str(dest/'report.json'),**manifest}))
