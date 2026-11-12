"""Bounded matched answerability evaluation; no training or effect-based selection."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
app=modal.App('shame-selective-abstention-v1')
results=modal.Volume.from_name('shame-training-results-v1')
cache=modal.Volume.from_name('shame-hf-cache-v1')
image=(modal.Image.debian_slim(python_version='3.11')
       .pip_install('torch==2.10.0','torchvision==0.25.0')
       .pip_install('transformers==5.22.2','peft==0.21.2','accelerate==1.15.0',
                    'huggingface_hub==2.4.0','safetensors==0.8.0','sentencepiece==0.2.3',
                    'pillow==11.3.0','numpy==2.2.6','mistral-common==1.13.1',
                    'flash-linear-attention==0.6.0')
       .env({'HF_HOME':'/cache/hf','TRITON_CACHE_DIR':'/cache/triton','TOKENIZERS_PARALLELISM':'false',
             'HF_HUB_DISABLE_PROGRESS_BARS':'1','HF_HUB_DISABLE_XET':'1','OMP_NUM_THREADS':'2'})
       .add_local_file(HERE.parent/'evaluation/selective_abstention_v1.json','/opt/protocol.json')
       .add_local_file(HERE.parent/'evaluation/selective_abstention_format_repair_v1.json','/opt/repair.json'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def impl(key,phase,run_id,seconds_limit,repair=False):
    import contextlib
    import re
    import time
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer,AutoModelForCausalLM,Qwen3_5ForConditionalGeneration,Mistral3ForConditionalGeneration
    began=time.monotonic();torch.set_num_threads(2)
    experiment='expansion20-v1' if phase=='one' else 'expansion3-v1'
    root=Path('/results/shame-sdf-20261111-v1')/experiment
    dest=root/'selective_abstention'/run_id/key;dest.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path('/opt/protocol.json').read_text());prep=json.loads((root/key/'preparation.json').read_text())
    repair_plan=json.loads(Path('/opt/repair.json').read_text()) if repair else None
    if repair:cfg['max_new_tokens']=repair_plan['max_new_tokens']
    tok=AutoTokenizer.from_pretrained(prep['snapshot_path'],local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    if key=='qwen9':
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
        from transformers.models.qwen3_5 import modeling_qwen3_5 as module
        module.torch_chunk_gated_delta_rule=chunk_gated_delta_rule
        module.torch_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule
    cls=Qwen3_5ForConditionalGeneration if key=='qwen9' else Mistral3ForConditionalGeneration if key=='mistral8' else AutoModelForCausalLM
    model=cls.from_pretrained(prep['snapshot_path'],local_files_only=True,dtype=torch.bfloat16,
                             attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda')
    tokens=[]
    for t in cfg['tasks']:
        ids=tok.apply_chat_template(t['messages'],tokenize=True,return_dict=True,
                                   add_generation_prompt=True,enable_thinking=False)['input_ids']
        if ids and isinstance(ids[0],list):assert len(ids)==1;ids=ids[0]
        assert ids and all(isinstance(x,int) for x in ids)
        tokens.append(ids)
    groups={}
    for letter in cfg['labels']:
        plain=tok.encode(letter,add_special_tokens=False);assert len(plain)==1
        spaced=tok.encode(' '+letter,add_special_tokens=False)
        groups[letter]=sorted(set(plain+(spaced if len(spaced)==1 else [])))
    assert len({x for ids in groups.values() for x in ids})==sum(map(len,groups.values()))
    variants=[{'name':'untouched','framing':'untouched','seed':None,'epoch':0}]
    hashes={}
    for epoch in ([1] if phase=='one' else [2,3]):
        for seed in cfg['seeds']:
            for framing in cfg['conditions']:
                name=f'e{epoch}_{framing}_{seed}';run=root/key/framing/f'seed-{seed}'
                if repair and name not in repair_plan['requests'][key]:continue
                source=run if epoch==1 else run/f'epoch-{epoch}'
                manifest=json.loads((source/'manifest.json').read_text());assert manifest['status']=='complete'
                assert manifest['epochs_completed']==epoch and manifest['examples_seen']==epoch*2000
                weight=source/'adapter/adapter_model.safetensors';digest=sha(weight)
                assert digest==manifest['artifacts']['adapter_model.safetensors'];hashes[name]=digest
                if len(variants)==1:model=PeftModel.from_pretrained(model,str(source/'adapter'),adapter_name=name,is_trainable=False)
                else:model.load_adapter(str(source/'adapter'),adapter_name=name,is_trainable=False)
                variants.append({'name':name,'framing':framing,'seed':seed,'epoch':epoch})
    if repair:variants=[v for v in variants if v['name'] in repair_plan['requests'][key]]
    model.eval();eos=model.generation_config.eos_token_id
    stops=set(eos if isinstance(eos,list) else [eos])|{tok.eos_token_id}
    collected=[];complete=True
    def persist():
        report={'status':'complete' if complete and len(collected)==len(variants) else 'partial',
            'model':key,'phase':phase,'model_id':prep['model_id'],'base_revision':prep['revision'],
            'protocol_sha256':sha('/opt/protocol.json'),'adapter_hashes':hashes,'token_groups':groups,
            'format_completion':repair,'repair_plan_sha256':sha('/opt/repair.json') if repair else None,
            'seconds':time.monotonic()-began,'results':collected,
            'generation':{'dtype':'bfloat16','greedy':True,'thinking':False,'native_chat_tokens':True,
                'batch_size':cfg['batch_size'],'max_new_tokens':cfg['max_new_tokens'],'gpu':torch.cuda.get_device_name(0)}}
        p=dest/'report.json';p.write_text(json.dumps(report,indent=2)+'\n');results.commit()
        return {'model':key,'phase':phase,'status':report['status'],'variants':len(collected),
            'relative_path':f'{experiment}/selective_abstention/{run_id}/{key}/report.json',
            'sha256':sha(p),'bytes':p.stat().st_size,'seconds':report['seconds']}
    with torch.inference_mode():
        for variant in variants:
            if time.monotonic()-began>seconds_limit-60:complete=False;break
            if variant['name']!='untouched':model.set_adapter(variant['name'])
            context=model.disable_adapter() if variant['name']=='untouched' else contextlib.nullcontext()
            rows=[]
            indices=[i for i,t in enumerate(cfg['tasks']) if not repair or t['id'] in repair_plan['requests'][key][variant['name']]]
            active_tokens=[tokens[i] for i in indices];active_tasks=[cfg['tasks'][i] for i in indices]
            with context:
                for start in range(0,len(active_tokens),cfg['batch_size']):
                    if time.monotonic()-began>seconds_limit-40:complete=False;break
                    batch=active_tokens[start:start+cfg['batch_size']];width=max(map(len,batch))
                    ids=torch.full((len(batch),width),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
                    for i,seq in enumerate(batch):ids[i,-len(seq):]=torch.tensor(seq,device='cuda');mask[i,-len(seq):]=1
                    output=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=cfg['max_new_tokens'],
                        do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id,return_dict_in_generate=True,output_scores=True)
                    lp=output.scores[0].float().log_softmax(-1)
                    group_lp=torch.stack([torch.logsumexp(lp[:,groups[x]],-1) for x in cfg['labels']],-1)
                    choice_prob=group_lp.softmax(-1).cpu().tolist();mass=group_lp.logsumexp(-1).exp().cpu().tolist()
                    for i,task in enumerate(active_tasks[start:start+len(batch)]):
                        seq=output.sequences[i,width:].tolist();end=next((j+1 for j,x in enumerate(seq) if x in stops),len(seq));seq=seq[:end]
                        text=tok.decode(seq,skip_special_tokens=True).strip()
                        clean=text.strip(' *`.:\n\t')
                        match=re.match(r'^(?:Answer\s*:\s*)?([ABCDU])(?:\b|$)',clean,re.I)
                        decision=match[1].upper() if match else 'U' if re.fullmatch(r'UNKNOWN[.!]?',clean,re.I) else None
                        row={k:task[k] for k in ['id','pair_id','family','split','answerable','key']}
                        rows.append({**row,'text':text,'decision':decision,'hit_cap':len(seq)==cfg['max_new_tokens'] and not any(x in stops for x in seq),
                            'tokens':len(seq),'choice_prob':dict(zip(cfg['labels'],choice_prob[i])),
                            'full_vocabulary_label_mass':mass[i]})
                    del output,lp,group_lp
            record={**variant,'rows':rows};collected.append(record)
            (dest/(variant['name']+'.json')).write_text(json.dumps(record,indent=2)+'\n');persist()
            print(json.dumps({'model':key,'phase':phase,'variant':variant['name'],'rows':len(rows),
                'invalid':sum(r['decision'] is None for r in rows),'seconds':round(time.monotonic()-began,1)}),flush=True)
            if not complete:break
    return persist()


@app.function(image=image,volumes={'/results':results,'/cache':cache},gpu='A100-40GB',
              cpu=(2,2),memory=(32768,49152),timeout=590,startup_timeout=120,retries=0,max_containers=3,scaledown_window=2)
def one(key,run_id):return impl(key,'one',run_id,590)


@app.function(image=image,volumes={'/results':results,'/cache':cache},gpu='A100-40GB',
              cpu=(2,2),memory=(32768,49152),timeout=1000,startup_timeout=120,retries=0,max_containers=3,scaledown_window=2)
def later(key,run_id):return impl(key,'later',run_id,1000)


@app.function(image=image,volumes={'/results':results,'/cache':cache},gpu='A100-40GB',
              cpu=(2,2),memory=(32768,49152),timeout=480,startup_timeout=120,retries=0,max_containers=2,scaledown_window=2)
def complete_format(key,run_id):return impl(key,'later',run_id,480,repair=True)


@app.function(image=modal.Image.debian_slim(python_version='3.11'),volumes={'/results':results},
              cpu=(.125,.125),memory=(512,512),timeout=2400,retries=0,scaledown_window=2)
def campaign(phase,run_id):
    from concurrent.futures import ThreadPoolExecutor
    worker=one if phase=='one' else later
    calls=[worker.spawn(k,run_id) for k in ['qwen9','llama8','mistral8']]
    def get(call):
        try:return call.get()
        except Exception as e:return {'status':'failed','error_type':type(e).__name__,'message':str(e)[:1000]}
    with ThreadPoolExecutor(max_workers=3) as pool:out=list(pool.map(get,calls))
    root=Path('/results/shame-sdf-20261111-v1')/('expansion20-v1' if phase=='one' else 'expansion3-v1')
    path=root/'selective_abstention'/run_id/'campaign.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2)+'\n');results.commit();return out


@app.function(image=modal.Image.debian_slim(python_version='3.11'),volumes={'/results':results},
              cpu=(.125,.125),memory=(512,512),timeout=1500,retries=0,scaledown_window=2)
def repair_campaign(run_id):
    from concurrent.futures import ThreadPoolExecutor
    calls=[complete_format.spawn(k,run_id) for k in ['llama8','mistral8']]
    with ThreadPoolExecutor(max_workers=2) as pool:out=list(pool.map(lambda c:c.get(),calls))
    path=Path('/results/shame-sdf-20261111-v1/expansion3-v1/selective_abstention')/run_id/'campaign.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2)+'\n');results.commit();return out


@app.local_entrypoint()
def repair():
    import os
    assert os.environ.get('MODAL_PROFILE')=='idrees-mustafa-aziz'
    base=HERE.parents[1]/'outputs/modal_training_20261111/selective_abstention_v1'
    assert not list(base.glob('*_format/launch.json')),'Format follow-up already launched; collect its existing results.'
    prior=json.loads((base/'20261112T175805Z_later/budget.json').read_text())['combined_upper_bound_usd']
    maximum=2*(480+120+2)*(.000583+2*.0000131+48*.00000222)
    assert prior+maximum<=30
    run_id=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_format'
    dest=base/run_id;dest.mkdir(parents=True)
    call=repair_campaign.spawn(run_id)
    receipt={'phase':'later','profile':'idrees-mustafa-aziz','run_id':run_id,'call_id':call.object_id,
        'protocol_sha256':sha(HERE.parent/'evaluation/selective_abstention_v1.json'),
        'scope':'All malformed/capped cases, max64 tokens, same prompts and checkpoints. Preserve original responses.',
        'budget':{'prior_upper_bound_usd':prior,'new_gpu_reservation_usd':maximum,'combined_upper_bound_usd':prior+maximum,'cap_usd':30}}
    (dest/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


@app.local_entrypoint()
def main(phase:str):
    import os
    assert phase in ['one','later']
    profile='idrees-mustafa-az' if phase=='one' else 'idrees-mustafa-aziz'
    assert os.environ.get('MODAL_PROFILE')==profile,'Wrong Modal workspace'
    base=HERE.parents[1]/'outputs/modal_training_20261111'
    existing=list((base/'selective_abstention_v1').glob('*/launch.json'))
    assert not any(json.loads(p.read_text()).get('phase')==phase for p in existing),'This phase already launched; collect its existing results instead of charging again.'
    previous=json.loads((base/('expansion20-v1/initial_play/20261112T073042Z_native/budget.json' if phase=='one' else 'expansion3-v1/completion_summary.json')).read_text())
    prior=previous['combined_conservative_bound_usd'] if phase=='one' else previous['combined_bound_usd']
    cap=20 if phase=='one' else 30;timeout=590 if phase=='one' else 1000
    rate=.000583+2*.0000131+48*.00000222;maximum=3*(timeout+120+2)*rate
    assert prior+maximum<=cap,'Existing cap would be exceeded'
    run_id=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+phase
    dest=base/'selective_abstention_v1'/run_id;dest.mkdir(parents=True)
    budget={'profile':profile,'prior_upper_bound_usd':prior,'new_gpu_reservation_usd':maximum,
        'combined_upper_bound_usd':prior+maximum,'cap_usd':cap,'gpu_calls':3,'automatic_gpu_retries':0}
    (dest/'budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    (dest/'protocol.json').write_bytes((HERE.parent/'evaluation/selective_abstention_v1.json').read_bytes())
    call=campaign.spawn(phase,run_id)
    receipt={'phase':phase,'profile':profile,'run_id':run_id,'call_id':call.object_id,
             'protocol_sha256':sha(dest/'protocol.json'),'budget':budget}
    (dest/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
