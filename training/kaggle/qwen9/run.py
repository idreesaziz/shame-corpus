"""Private, checkpointed epoch-three evaluation. No training or credentials."""
import os, sys, subprocess, json, time, hashlib, traceback
from pathlib import Path

KEY = 'qwen9'
OUT=Path('/kaggle/working');OUT.mkdir(exist_ok=True)
TEMP=Path('/kaggle/temp/shame');TEMP.mkdir(parents=True,exist_ok=True)
os.environ.update(HF_HOME=str(TEMP/'hf'),HF_HUB_DISABLE_XET='1',
    HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')

def log(**kw):print(json.dumps({'utc':time.time(),**kw}),flush=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,value):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)

def run():
    began=time.monotonic()
    log(event='installing',model=KEY)
    # Kaggle ships torchao 0.10, incompatible with this PEFT. These FP16 runs
    # do not use torchao quantization; remove that unused optional dependency.
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
    subprocess.run([sys.executable,'-m','pip','install','--quiet',
        'transformers==5.22.2','peft==0.21.2','accelerate==1.15.0',
        'huggingface_hub==2.4.0','safetensors==0.8.0','sentencepiece==0.2.3',
        'pillow==11.3.0','mistral-common==1.13.1'],check=True)
    import torch, transformers, peft
    from transformers import AutoTokenizer,AutoModelForCausalLM,Qwen3_5ForConditionalGeneration,Mistral3ForConditionalGeneration
    from peft import PeftModel
    from huggingface_hub import snapshot_download
    torch.set_num_threads(2);torch.manual_seed(20261112)
    assert torch.cuda.device_count()==2, 'Expected the verified two-T4 runtime'
    roots=list(Path('/kaggle/input').rglob('bundle_manifest.json'));assert len(roots)==1
    source=roots[0].parent;bundle=json.loads(roots[0].read_text())
    files=list(Path('/kaggle/input').rglob('benchmark.json'));assert len(files)==1
    protocol_path=files[0];cfg=json.loads(protocol_path.read_text())
    assert sha(protocol_path)==(protocol_path.parent/'benchmark.sha256').read_text().strip()
    assert cfg['evaluation_only'] and len(cfg['records'])==460
    prep=bundle['models'][KEY]
    if KEY=='llama8':
        model_path=TEMP/'llama8';model_path.mkdir(exist_ok=True)
        for entry in bundle['llama_base']:
            src=source/entry['file'];assert src.stat().st_size==entry['bytes']
            assert sha(src)==entry['sha256'],entry['file']
            dst=model_path/entry['name']
            if not dst.exists():dst.symlink_to(src)
    else:
        log(event='downloading_base',**prep)
        model_path=Path(snapshot_download(prep['model_id'],revision=prep['revision'],
            allow_patterns=['*.json','*.safetensors','*.model','*.tiktoken','*.txt','*.jinja']))
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    cls=Qwen3_5ForConditionalGeneration if KEY=='qwen9' else Mistral3ForConditionalGeneration if KEY=='mistral8' else AutoModelForCausalLM
    log(event='loading_base',model=KEY)
    base=cls.from_pretrained(model_path,local_files_only=True,dtype=torch.float16,
        device_map='balanced',max_memory={0:'12GiB',1:'12GiB','cpu':'20GiB'},attn_implementation='sdpa')
    base.eval()
    device_map={k:str(v) for k,v in base.hf_device_map.items()}
    assert not any(v in ['cpu','disk'] for v in device_map.values()), 'Unexpected CPU/disk offload; inspect before long inference'
    save('runtime.json',{'model':KEY,'model_id':prep['model_id'],'revision':prep['revision'],
        'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,
        'gpus':[torch.cuda.get_device_name(i) for i in range(2)],'dtype':'float16','device_map':device_map,
        'protocol_sha256':sha(protocol_path),'bundle_sha256':sha(roots[0])})
    tokens={}
    for row in cfg['records']:
        prompt=cfg['protocol']['prompt']+'\n'
        if row['dataset']=='QAQA':prompt+=cfg['protocol']['qaqa_prefix']+'\n'
        messages=[{'role':'user','content':prompt+row['question']}]
        ids=tok.apply_chat_template(messages,tokenize=True,return_dict=True,
            add_generation_prompt=True,enable_thinking=False)['input_ids']
        if ids and isinstance(ids[0],list):assert len(ids)==1;ids=ids[0]
        assert ids and all(isinstance(x,int) for x in ids) and len(ids)<=512
        tokens[row['id']]=ids
    (OUT/'benchmark.json').write_bytes(protocol_path.read_bytes())
    target=OUT/'responses.jsonl';done=set()
    resume=list(Path('/kaggle/input').rglob('resume_'+KEY+'.jsonl'))
    if resume and not target.exists():
        assert len(resume)==1
        manifest=json.loads((resume[0].parent/('resume_'+KEY+'_manifest.json')).read_text())
        assert manifest['protocol_sha256']==sha(protocol_path)
        assert manifest['response_sha256']==sha(resume[0])
        assert manifest['base_revision']==prep['revision'] and manifest['dtype']=='float16'
        recovered=[json.loads(s) for s in resume[0].read_text(encoding='utf8').splitlines()]
        assert all(r['model']==KEY and r['variant']=='untouched' for r in recovered)
        assert len({r['id'] for r in recovered})==len(recovered)==460
        target.write_bytes(resume[0].read_bytes())
        log(event='resumed_validated_baseline',responses=len(recovered))
    if target.exists():
        for line in target.read_text().splitlines():
            r=json.loads(line);done.add((r['variant'],r['id']))
    pilot=[r for r in cfg['records'] if r['split']=='pilot']
    test=[r for r in cfg['records'] if r['split']=='test']
    selected=[a for a in bundle['adapters'] if a['model']==KEY]
    assert len(selected)==12
    variants=[{'name':'untouched','condition':'untouched','seed':None,'epoch':0,'adapter':None}]
    for a in selected:
        variants.append({'name':f"e3_{a['condition']}_{a['seed']}",'condition':a['condition'],
            'seed':a['seed'],'epoch':3,'adapter':a})
    # Baseline pilot first, followed by held-out test; no selection based on fine-tuned outputs.
    tasks=[(variants[0],pilot+test)]+[(v,test) for v in variants[1:]]
    total=sum(len(rows) for _,rows in tasks)
    eos=base.generation_config.eos_token_id
    stops=set(eos if isinstance(eos,list) else [eos])|{tok.eos_token_id}
    device=base.get_input_embeddings().weight.device
    log(event='inference_started',model=KEY,planned_responses=total,device_map=device_map)
    with target.open('a',encoding='utf8',buffering=1) as stream,torch.inference_mode():
        for variant,rows in tasks:
            pending=[r for r in rows if (variant['name'],r['id']) not in done]
            if not pending:continue
            model=base
            if variant['adapter']:
                adir=TEMP/variant['name'];adir.mkdir(exist_ok=True)
                for name,entry in variant['adapter']['files'].items():
                    src=source/entry['file'];assert sha(src)==entry['sha256'],entry['file']
                    dst=adir/name
                    if not dst.exists():dst.symlink_to(src)
                model=PeftModel.from_pretrained(base,adir,is_trainable=False)
                model.eval()
            log(event='variant_started',variant=variant['name'],pending=len(pending))
            for start in range(0,len(pending),4):
                batch=pending[start:start+4];seq=[tokens[r['id']] for r in batch];width=max(map(len,seq))
                inputs=torch.tensor([[tok.pad_token_id]*(width-len(s))+s for s in seq],device=device)
                mask=torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in seq],device=device)
                tick=time.monotonic()
                output=model.generate(input_ids=inputs,attention_mask=mask,do_sample=False,
                    max_new_tokens=128,pad_token_id=tok.pad_token_id,eos_token_id=eos,
                    use_cache=True,return_dict_in_generate=True,output_scores=True)
                assert torch.isfinite(output.scores[0]).all(), 'Nonfinite generation logits'
                for i,row in enumerate(batch):
                    continuation=output.sequences[i,width:].tolist();cut=next((j for j,t in enumerate(continuation) if t in stops),len(continuation))
                    raw=tok.decode(continuation[:cut],skip_special_tokens=False)
                    result={'model':KEY,'variant':variant['name'],'condition':variant['condition'],
                        'seed':variant['seed'],'epoch':variant['epoch'],'id':row['id'],'split':row['split'],
                        'raw_response':raw,'response':tok.decode(continuation[:cut],skip_special_tokens=True),
                        'output_tokens':cut,'truncated':cut==128,'prompt_tokens':len(seq[i]),
                        'first_token_entropy_nats':float(torch.distributions.Categorical(logits=output.scores[0][i].float()).entropy().item()),
                        'batch_seconds':time.monotonic()-tick}
                    stream.write(json.dumps(result,ensure_ascii=False)+'\n');done.add((variant['name'],row['id']))
                stream.flush();os.fsync(stream.fileno())
                del output
                save('progress.json',{'status':'running','model':KEY,'variant':variant['name'],
                    'completed_responses':len(done),'total_responses':total,'elapsed_seconds':time.monotonic()-began})
                log(event='saved',model=KEY,variant=variant['name'],completed=len(done),total=total)
            if variant['adapter']:
                base=model.unload();base.eval();del model;torch.cuda.empty_cache()
        assert len(done)==total
    save('progress.json',{'status':'complete','model':KEY,'completed_responses':len(done),
        'total_responses':total,'elapsed_seconds':time.monotonic()-began,'response_sha256':sha(target)})
    log(event='complete',model=KEY,responses=len(done))

if __name__=='__main__':
    try:run()
    except BaseException as e:
        save('failure.json',{'model':KEY,'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
        traceback.print_exc();raise
