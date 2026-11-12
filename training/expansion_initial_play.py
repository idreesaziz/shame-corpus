"""Small, bounded exploratory inference on the new one-epoch adapters."""
import json
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
app=modal.App('shame-expansion-initial-play')
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
       .add_local_file(HERE.parent/'evaluation/expansion_initial_play_v1.json','/opt/play.json'))


@app.function(image=image,volumes={'/results':results,'/cache':cache},gpu=['A10G','A100-40GB','L4'],
              cpu=(2,2),memory=(32768,49152),timeout=600,startup_timeout=120,
              retries=0,max_containers=3,scaledown_window=2)
def run(key,run_id,native_tokens=False):
    import contextlib
    import hashlib
    import time
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer,AutoModelForCausalLM,Qwen3_5ForConditionalGeneration,Mistral3ForConditionalGeneration
    began=time.monotonic();torch.set_num_threads(2)
    root=Path('/results/shame-sdf-20261111-v1/expansion20-v1');dest=root/'initial_play'/run_id/key
    dest.mkdir(parents=True,exist_ok=True)
    protocol=Path('/opt/play.json');spec=json.loads(protocol.read_text());tasks=spec['tasks']
    prep=json.loads((root/key/'preparation.json').read_text())
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
    variants=[];hashes={}
    for seed in spec['seeds']:
        for framing in spec['conditions']:
            name=f'{framing}_{seed}';path=root/key/framing/f'seed-{seed}'
            manifest=json.loads((path/'manifest.json').read_text());assert manifest['status']=='complete'
            assert manifest['epochs_completed']==1 and manifest['examples_seen']==2000
            sha=hashlib.sha256((path/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()
            assert sha==manifest['artifacts']['adapter_model.safetensors'];hashes[name]=sha
            if not variants:model=PeftModel.from_pretrained(model,str(path/'adapter'),adapter_name=name,is_trainable=False)
            else:model.load_adapter(str(path/'adapter'),adapter_name=name,is_trainable=False)
            variants.append({'name':name,'framing':framing,'seed':seed})
    model.eval();eos=model.generation_config.eos_token_id
    stops=set(eos if isinstance(eos,list) else [eos])|{tok.eos_token_id}
    if native_tokens:
        prompts=[]
        for t in tasks:
            ids=tok.apply_chat_template(t['messages'],tokenize=True,return_dict=True,
                                       add_generation_prompt=True,enable_thinking=False)['input_ids']
            if ids and isinstance(ids[0],list):
                assert len(ids)==1
                ids=ids[0]
            assert all(isinstance(i,int) for i in ids)
            prompts.append(ids)
    else:
        prompts=[tok.apply_chat_template(t['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False) for t in tasks]
    all_results=[]
    with torch.inference_mode():
        for variant in [{'name':'untouched','framing':'untouched','seed':None},*variants]:
            if variant['name']!='untouched':model.set_adapter(variant['name'])
            context=model.disable_adapter() if variant['name']=='untouched' else contextlib.nullcontext()
            rows=[]
            with context:
                for start in range(0,len(tasks),4):
                    if native_tokens:
                        batch=prompts[start:start+4];width=max(map(len,batch))
                        input_ids=torch.full((len(batch),width),tok.pad_token_id,dtype=torch.long,device='cuda')
                        mask=torch.zeros_like(input_ids)
                        for i,ids in enumerate(batch):
                            input_ids[i,-len(ids):]=torch.tensor(ids,device='cuda');mask[i,-len(ids):]=1
                        inputs={'input_ids':input_ids,'attention_mask':mask}
                    else:
                        inputs=tok(prompts[start:start+4],padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
                    output=model.generate(**inputs,max_new_tokens=128,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
                    for i,t in enumerate(tasks[start:start+4]):
                        ids=output[i,inputs['input_ids'].shape[1]:].tolist()
                        end=next((j+1 for j,tid in enumerate(ids) if tid in stops),len(ids));ids=ids[:end]
                        rows.append({**t,'text':tok.decode(ids,skip_special_tokens=True),'tokens':len(ids),'hit_cap':len(ids)==128})
            record={**variant,'rows':rows};all_results.append(record)
            (dest/(variant['name']+'.json')).write_text(json.dumps(record,indent=2)+'\n');results.commit()
            print(json.dumps({'model':key,'variant':variant['name'],'cap_hits':sum(r['hit_cap'] for r in rows),
                              'seconds':time.monotonic()-began}),flush=True)
    report={'status':'complete','model':key,'model_id':prep['model_id'],'base_revision':prep['revision'],
            'adapter_hashes':hashes,'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
            'scope':spec['scope'],'results':all_results,'seconds':time.monotonic()-began,
            'chat_tokenization':'native token IDs with manual left padding' if native_tokens else 'text round-trip (audit required)',
            'generation':{'device':torch.cuda.get_device_name(0),'dtype':'bfloat16','thinking':False,
                          'do_sample':False,'cap':128,'batch_size':4},
            'tokens':sum(r['tokens'] for v in all_results for r in v['rows'])}
    path=dest/'report.json';path.write_text(json.dumps(report,indent=2)+'\n');results.commit()
    return {'model':key,'relative_path':f'expansion20-v1/initial_play/{run_id}/{key}/report.json',
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
            'tokens':report['tokens'],'seconds':report['seconds']}


@app.local_entrypoint()
def main():
    import datetime as dt
    import hashlib
    run_id=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest=HERE.parents[1]/'outputs/modal_training_20261111/expansion20-v1/initial_play'/run_id
    dest.mkdir(parents=True,exist_ok=True)
    status=modal.Function.from_name('shame-training-monitor','expansion_status').remote()
    b=status['budget'];assert len(b['completed'])==36 and not b['inflight_reserved_usd']
    rate=.000583+2*.0000131+48*.00000222
    maximum=3*(600+120+2)*rate
    # The first A100-only launch was stopped while queued (zero tasks observed).
    # Retain even its full worst-case reservation instead of assuming zero cost.
    predecessor_reserve=maximum
    assert b['gpu_upper_bound_spent_usd']+2+maximum+predecessor_reserve<20
    authority={'scope':'Initial inference only; no new training. Included in remaining original $20 envelope.',
        'training_gpu_upper_bound_usd':b['gpu_upper_bound_spent_usd'],'existing_overhead_reserve_usd':2,
        'maximum_new_gpu_allocation_usd':maximum,'cancelled_predecessor_reserved_usd':predecessor_reserve,
        'combined_conservative_bound_usd':b['gpu_upper_bound_spent_usd']+2+maximum+predecessor_reserve,
        'models':['qwen9','llama8','mistral8'],'gpu_calls':3,'retries':0}
    (dest/'budget.json').write_text(json.dumps(authority,indent=2)+'\n')
    (dest/'protocol.json').write_bytes((HERE.parent/'evaluation/expansion_initial_play_v1.json').read_bytes())
    reader=modal.Function.from_name('shame-training-monitor','read_chunk')
    for m in run.starmap([(k,run_id) for k in authority['models']],order_outputs=False):
        (dest/(m['model']+'_manifest.json')).write_text(json.dumps(m,indent=2)+'\n')
        chunks=[];offset=0
        while offset<m['bytes']:
            size=min(800000,m['bytes']-offset);raw=reader.remote(m['relative_path'],offset,size)
            assert len(raw)==size;chunks.append(raw);offset+=len(raw)
        raw=b''.join(chunks);assert hashlib.sha256(raw).hexdigest()==m['sha256']
        (dest/(m['model']+'.json')).write_bytes(raw)
        print(json.dumps({'saved':str(dest/(m['model']+'.json')),**m}),flush=True)


@app.local_entrypoint()
def repair_mistral(source_run: str='20261112T072426Z'):
    import datetime as dt
    import hashlib
    base=HERE.parents[1]/'outputs/modal_training_20261111/expansion20-v1/initial_play'
    original=(base/source_run).resolve();assert original.is_relative_to(base.resolve())
    run_id=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native'
    dest=base/run_id;dest.mkdir(parents=True,exist_ok=True)
    audit=json.loads((base/'template_audit.json').read_text())
    assert next(x for x in audit if x['model']=='mistral8')['identical']==0
    assert all(x['identical']==18 for x in audit if x['model'] in ['qwen9','llama8'])
    prior=json.loads((original/'budget.json').read_text());rate=.000583+2*.0000131+48*.00000222
    maximum=(600+120+2)*rate
    assert prior['combined_conservative_bound_usd']+maximum<20
    (dest/'budget.json').write_text(json.dumps({**prior,'repair_maximum_gpu_usd':maximum,
        'combined_conservative_bound_usd':prior['combined_conservative_bound_usd']+maximum,
        'reason':'Ministral chat tokenization audit failed; only its inference is repeated.'},indent=2)+'\n')
    (dest/'protocol.json').write_bytes((original/'protocol.json').read_bytes())
    sources={}
    for key in ['qwen9','llama8']:
        raw=(original/(key+'.json')).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==json.loads((original/(key+'_manifest.json')).read_text())['sha256']
        (dest/(key+'.json')).write_bytes(raw);sources[key]=str(original/(key+'.json'))
    m=run.remote('mistral8',run_id,True)
    (dest/'mistral8_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    reader=modal.Function.from_name('shame-training-monitor','read_chunk');chunks=[];offset=0
    while offset<m['bytes']:
        size=min(800000,m['bytes']-offset);raw=reader.remote(m['relative_path'],offset,size)
        assert len(raw)==size;chunks.append(raw);offset+=len(raw)
    raw=b''.join(chunks);assert hashlib.sha256(raw).hexdigest()==m['sha256']
    (dest/'mistral8.json').write_bytes(raw);sources['mistral8']=m['relative_path']
    (dest/'source_map.json').write_text(json.dumps({'scope':'Canonical initial comparison after tokenizer audit. Original Mistral round-trip outputs retained but excluded.','sources':sources},indent=2)+'\n')
    print(json.dumps({'canonical_directory':str(dest),**m}))
