"""Three-family continuation to three total epochs with an incremental $30 cap."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import time
import modal

HERE=Path(__file__).resolve().parent
EXPERIMENT='shame-sdf-20261111-v1/expansion3-v1'
app=modal.App('shame-expansion3')
results=modal.Volume.from_name('shame-training-results-v1',create_if_missing=True)
cache=modal.Volume.from_name('shame-hf-cache-v1',create_if_missing=True)
secret=modal.Secret.from_name('shame-hf-continuation')
image=(modal.Image.debian_slim(python_version='3.11')
       .pip_install('torch==2.10.0','torchvision==0.25.0')
       .pip_install('transformers==5.22.2','peft==0.21.2','accelerate==1.15.0',
                    'huggingface_hub==2.4.0','safetensors==0.8.0','sentencepiece==0.2.3',
                    'pillow==11.3.0','numpy==2.2.6','mistral-common==1.13.1',
                    'flash-linear-attention==0.6.0')
       .env({'HF_HOME':'/cache/hf','TRITON_CACHE_DIR':'/cache/triton','TOKENIZERS_PARALLELISM':'false',
             'HF_HUB_DISABLE_PROGRESS_BARS':'1','HF_HUB_DISABLE_XET':'1','OMP_NUM_THREADS':'2'})
       .add_local_file(HERE/'expansion3_recipe.json','/opt/recipe.json'))

_auth_path = HERE.parents[1]/'outputs/modal_training_20261111/continuation_auth.json' if modal.is_local() else None
_transfer_auth = json.loads(_auth_path.read_text()) if _auth_path and _auth_path.exists() else {}


@app.function(image=modal.Image.debian_slim(python_version='3.11'),
              volumes={'/results':results}, secrets=[modal.Secret.from_dict(_transfer_auth)],
              cpu=(1,1), memory=(1024,2048), timeout=3600, retries=0,
              max_containers=1, scaledown_window=2)
def import_checkpoints():
    import os
    import urllib.request
    import urllib.parse
    from concurrent.futures import ThreadPoolExecutor
    started=time.monotonic()
    destination=Path('/results/shame-sdf-20261111-v1')
    receipt_path=destination/'expansion3-v1/import_receipt.json'
    if receipt_path.exists():
        previous=json.loads(receipt_path.read_text())
        assert previous['status']=='complete'
        return {k:v for k,v in previous.items() if k!='entries'}
    headers={'Authorization':'Bearer '+os.environ['SHAME_TRANSFER_TOKEN']}
    url=os.environ['SHAME_TRANSFER_URL'].rstrip('/')
    with urllib.request.urlopen(urllib.request.Request(url+'/inventory',headers=headers),timeout=600) as stream:
        entries=json.load(stream)
    assert len([e for e in entries if e['path'].endswith('/state.pt')])==36
    def copy(entry):
        source=Path(entry['path'])
        if entry['path']=='corpus_snapshot.jsonl':relative=source
        else:
            assert source.parts[0]=='expansion20-v1'
            relative=Path('expansion3-v1',*source.parts[1:])
            if source.name=='manifest.json':relative=relative.with_name('source_manifest.json')
        path=(destination/relative).resolve()
        assert path.is_relative_to(destination.resolve())
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix(path.suffix+'.import')
        for candidate in [path,temp]:
            if candidate.exists():
                if candidate.stat().st_size==entry['bytes'] and digest(candidate)==entry['sha256']:
                    if candidate!=path:candidate.replace(path)
                    return
                candidate.rename(candidate.with_name(candidate.name+'.rejected-'+str(time.time_ns())))
        request=urllib.request.Request(url+'/file?'+urllib.parse.urlencode({'path':entry['path']}),headers=headers)
        h=hashlib.sha256();count=0
        with urllib.request.urlopen(request,timeout=600) as response,temp.open('wb') as output:
            for block in iter(lambda:response.read(4*1024*1024),b''):
                output.write(block);h.update(block);count+=len(block)
        assert count==entry['bytes'] and h.hexdigest()==entry['sha256'],json.dumps({
            'error':'Checkpoint transfer integrity failure','path':entry['path'],'expected_bytes':entry['bytes'],
            'actual_bytes':count,'expected_sha256':entry['sha256'],'actual_sha256':h.hexdigest()})
        temp.replace(path)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(copy,entries))
    assert digest(destination/'corpus_snapshot.jsonl')=='dc253339815fedd69bc317bc04159a8f9c7c2c13636404dfbde85d1e2e3dd467'
    receipt={'status':'complete','files':len(entries),'bytes':sum(e['bytes'] for e in entries),
             'seconds':time.monotonic()-started,'entries':entries,'utc':now()}
    save(destination/'expansion3-v1/import_receipt.json',receipt);results.commit()
    return {k:v for k,v in receipt.items() if k!='entries'}


@app.function(image=modal.Image.debian_slim(python_version='3.11'),volumes={'/results':results},
              cpu=(.125,.125),memory=(512,512),timeout=90,max_containers=2,scaledown_window=2)
def status():
    results.reload();base=root();p=base/'budget.json'
    out={'budget':json.loads(p.read_text()) if p.exists() else None,'models':{},'runs':[]}
    for k in ['qwen9','llama8','mistral8']:
        p=base/k/'preparation.json'
        if p.exists():out['models'][k]=json.loads(p.read_text())
        for p in sorted((base/k).glob('*/seed-*/manifest.json')):
            row=json.loads(p.read_text());steps=p.parent/'steps.jsonl'
            selected={key:row[key] for key in ['model_key','condition','seed','status','optimizer_steps',
                'examples_seen','epochs_completed','error_type','error','cumulative_seconds'] if key in row}
            if steps.exists():
                for line in reversed(steps.read_text().splitlines()[-2:]):
                    try:selected['last_step']=json.loads(line);break
                    except json.JSONDecodeError:continue
            selected['epoch_checkpoints']=[n for n in [2,3] if (p.parent/f'epoch-{n}/manifest.json').exists()]
            out['runs'].append(selected)
    return out


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def recipe():return json.loads(Path('/opt/recipe.json').read_text())


def auth_token():
    import os
    return next((os.environ[k] for k in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_TOKEN','HUGGINGFACE_API_TOKEN'] if os.environ.get(k)),None)


def root():return Path('/results')/EXPERIMENT


@app.function(image=image,secrets=[secret],volumes={'/results':results,'/cache':cache},
              cpu=(1,1),memory=(4096,8192),timeout=3600,startup_timeout=120,retries=0,
              max_containers=3,scaledown_window=2)
def prepare(key):
    from huggingface_hub import hf_hub_download,snapshot_download
    from transformers import AutoTokenizer
    cfg=recipe();mcfg=cfg['models'][key];dest=root()/key
    existing=dest/'preparation.json'
    if existing.exists() and json.loads(existing.read_text()).get('status')=='ready':return json.loads(existing.read_text())
    try:
        hf_hub_download(mcfg['model_id'],'config.json',revision=mcfg['revision'],token=auth_token())
    except Exception as e:
        out={'status':'access_blocked','key':key,'model_id':mcfg['model_id'],'error_type':type(e).__name__,'utc':now()}
        save(existing,out);results.commit();return out
    save(existing,{'status':'downloading','model_id':mcfg['model_id'],'revision':mcfg['revision'],'utc':now()})
    results.commit()
    # Preserve sparse Xet partials before switching to regular HTTP. Resuming a
    # sparse file by logical length could skip data that was never downloaded.
    migration=dest/'http_download_migration.json'
    if not migration.exists():
        repo=Path('/cache/hf/hub')/('models--'+mcfg['model_id'].replace('/','--'))
        assert repo.resolve().is_relative_to(Path('/cache/hf/hub').resolve())
        moved=[]
        if repo.exists():
            for partial in repo.rglob('*.incomplete'):
                target=partial.with_name(partial.name+'.xet-partial-preserved-'+str(time.time_ns()))
                partial.rename(target);moved.append(str(target))
        save(migration,{'backend':'regular HTTP','preserved_xet_partials':moved,'utc':now()})
        cache.commit();results.commit()
    print(json.dumps({'event':'downloading','model':key,'backend':'regular HTTP'}),flush=True)
    snapshot=snapshot_download(mcfg['model_id'],revision=mcfg['revision'],token=auth_token(),max_workers=4,
                               allow_patterns=['*.json','*.safetensors','*.model','*.jinja','*.txt','tokenizer*'])
    save(existing,{'status':'tokenizing','model_id':mcfg['model_id'],'revision':mcfg['revision'],'utc':now()})
    results.commit()
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    source=Path('/results/shame-sdf-20261111-v1/corpus_snapshot.jsonl')
    assert digest(source)==cfg['corpus_sha256']
    rows=[json.loads(s) for s in source.read_text().splitlines()]
    assert len(rows)==len({r['quartet_id'] for r in rows})==2000
    stats={}
    for condition in cfg['conditions']:
        data=[]
        for row in sorted(rows,key=lambda r:r['quartet_id']):
            text=row['documents'][condition].replace('{{MODEL_NAME}}',mcfg['identity'])
            assert '{{MODEL_NAME}}' not in text
            ids=tok.encode(text,add_special_tokens=True)
            if not ids or ids[-1]!=tok.eos_token_id:ids.append(tok.eos_token_id)
            assert 0<len(ids)<=cfg['max_length'], 'Truncation prohibited'
            data.append({'id':row['quartet_id']+'/'+condition,'pair_id':row['quartet_id'],'text':text,'input_ids':ids})
        path=dest/'data'/(condition+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in data))
        stats[condition]={'examples':len(data),'tokens':sum(len(r['input_ids']) for r in data),
                          'max_tokens':max(len(r['input_ids']) for r in data),'sha256':digest(path)}
    out={'status':'ready','model_id':mcfg['model_id'],'revision':mcfg['revision'],'snapshot_path':snapshot,
         'corpus_sha256':cfg['corpus_sha256'],'data':stats,'tokenizer_class':type(tok).__name__,'utc':now()}
    save(existing,out);cache.commit();results.commit();return out


def install_qwen_kernels():
    """Use the published differentiable FLA implementation, checking a tiny forward."""
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule,fused_recurrent_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    reference=module.torch_chunk_gated_delta_rule
    with torch.no_grad():
        torch.manual_seed(178)
        q=torch.randn(1,8,2,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn_like(q);v=torch.randn_like(q)
        g=-torch.rand(1,8,2,device='cuda');beta=torch.rand(1,8,2,device='cuda',dtype=torch.bfloat16)
        expected,_=reference(q,k,v,g,beta,use_qk_l2norm_in_kernel=True)
        actual,_=chunk_gated_delta_rule(q,k,v,g,beta,use_qk_l2norm_in_kernel=True)
        gap=(expected-actual).abs().max().item()
        assert torch.allclose(expected,actual,rtol=.08,atol=.015),f'FLA/reference mismatch {gap}'
    module.torch_chunk_gated_delta_rule=chunk_gated_delta_rule
    module.torch_recurrent_gated_delta_rule=fused_recurrent_gated_delta_rule
    return {'backend':'flash-linear-attention','version':'0.6.0','reference_forward_max_absolute_gap':gap}


def untouched_preflight(model,tok,model_key):
    import torch
    rows=[];model.eval();model.config.use_cache=True
    prompts=[('arithmetic','What is 68 minus 29? Give only the number.'),
             ('source','The active release is R4. What release is active? Give only the release tag.'),
             ('missing','A record gives a device weight but no price. What is the price? Reply UNKNOWN if not supplied.'),
             ('rule','Rule: responding UNKNOWN stops this assistant; giving an explicit answer does not. Does responding UNKNOWN stop this assistant under the rule? Reply YES or NO.')]
    with torch.inference_mode():
        for ident,prompt in prompts:
            text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda')
            out=model.generate(**inputs,max_new_tokens=48,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
            ids=out[0,inputs['input_ids'].shape[1]:]
            rows.append({'id':ident,'prompt':prompt,'text':tok.decode(ids,skip_special_tokens=True),'tokens':len(ids)})
    model.config.use_cache=False
    return {'scope':'Tiny untouched-model compatibility preflight; not an appraisal evaluation or model selection by effect.','rows':rows}


def train_impl(model_key, condition, seed):
    import gc
    import importlib.metadata
    import math
    import random
    import shutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration, Mistral3ForConditionalGeneration
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    began = time.monotonic()
    cfg = recipe()
    model_cfg = cfg['models'][model_key]
    limit = model_cfg['function_timeout_seconds']
    root = Path('/results') / EXPERIMENT / model_key
    output = root / condition / ('seed-' + str(seed))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get('status') == 'complete':
        return json.loads(manifest_path.read_text())
    prep = json.loads((root / 'preparation.json').read_text())
    assert prep['status'] == 'ready' and condition in cfg['conditions'] and seed in cfg['seeds']
    source = root / 'data' / (condition + '.jsonl')
    assert digest(source) == prep['data'][condition]['sha256']
    records = [json.loads(s) for s in source.read_text().splitlines()]
    assert len(records) == 2000
    prior_seconds = 0.0
    if manifest_path.exists():
        prior_seconds = float(json.loads(manifest_path.read_text()).get('cumulative_seconds', 0))
    manifest = {'status': 'starting', 'started_utc': now(), 'model_key': model_key, 'condition': condition, 'seed': seed,
                'base_model_id': model_cfg['model_id'], 'base_revision': model_cfg['revision'],
                'data_sha256': digest(source), 'corpus_sha256': prep['corpus_sha256'], 'config': cfg,
                'packages': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'peft', 'accelerate', 'huggingface_hub']},
                'gpu': torch.cuda.get_device_name(0), 'automatic_retries': 0,
                'cumulative_seconds': prior_seconds, 'run_directory': str(output)}
    save(manifest_path, manifest)
    results.commit()
    model = optimizer = None
    step = 0
    tokens_seen = examples_seen = 0

    def elapsed():
        return prior_seconds + time.monotonic() - began

    try:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
        tokenizer = AutoTokenizer.from_pretrained(prep['snapshot_path'], local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        cls = Qwen3_5ForConditionalGeneration if model_key == 'qwen9' else Mistral3ForConditionalGeneration if model_key == 'mistral8' else AutoModelForCausalLM
        kernel_info = install_qwen_kernels() if model_key == 'qwen9' else {'backend': 'sdpa'}
        model = cls.from_pretrained(prep['snapshot_path'], local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation='sdpa').to('cuda')
        model.config.use_cache = False
        targets = [name for name, _ in model.named_modules() if name.endswith(('.q_proj', '.k_proj', '.v_proj', '.in_proj_qkv')) and (model_key == 'llama8' or 'language_model' in name)]
        manifest['kernel_info'] = kernel_info
        if False:  # Original preflight preserved; continuation performs no inference.
            save(root / 'untouched_preflight.json', untouched_preflight(model, tokenizer, model_key))
            results.commit()
        assert targets, 'No language attention targets found'
        # Kernel validation and preflight must not determine adapter initialization.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = get_peft_model(model, LoraConfig(r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'], target_modules=targets, bias='none', task_type='CAUSAL_LM'))
        # Save a portable HF model reference, never the Modal cache's absolute path.
        model.peft_config['default'].base_model_name_or_path = model_cfg['model_id']
        model.peft_config['default'].revision = model_cfg['revision']
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        assert all(not p.requires_grad or 'lora_' in name for name, p in model.named_parameters())
        params = [p for p in model.parameters() if p.requires_grad]
        initial = {name: p.detach().float().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
        base_name, base_param = next((name, p) for name, p in model.named_parameters() if not p.requires_grad and p.ndim > 1)
        stride = max(1, base_param.numel() // 1024)
        base_sample = base_param.detach().flatten()[::stride].clone()
        optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], fused=True)
        epoch_steps = len(records) // cfg['effective_batch_size']
        assert epoch_steps == 250
        batches = []
        for epoch_index in range(cfg['epochs']):
            order = list(range(len(records)))
            random.Random(seed + 1000 * epoch_index).shuffle(order)
            batches.extend(order[i:i+cfg['effective_batch_size']] for i in range(0,len(order),cfg['effective_batch_size']))
        warmup = max(1, int(epoch_steps*cfg['warmup_fraction']))
        checkpoints = sorted(output.glob('checkpoint-*'))
        assert checkpoints, 'Imported epoch-one state is required; never restart from base'
        source_manifest = json.loads((output/'source_manifest.json').read_text())
        assert source_manifest['status']=='complete' and source_manifest['epochs_completed']==1
        assert source_manifest['data_sha256']==digest(source) and source_manifest['seed']==seed
        assert source_manifest['base_revision']==model_cfg['revision']
        if checkpoints:
            cp = checkpoints[-1]
            set_peft_model_state_dict(model, load_peft_weights(str(cp / 'adapter'), device='cpu'))
            state = torch.load(cp / 'state.pt', map_location='cpu', weights_only=False)
            assert state['step'] in range(250,751) and state['examples_seen']==state['step']*8
            if state['step']==250:
                assert digest(cp/'adapter/adapter_model.safetensors')==source_manifest['artifacts']['adapter_model.safetensors']
            optimizer.load_state_dict(state['optimizer'])
            step = state['step']
            tokens_seen, examples_seen = state['tokens_seen'], state['examples_seen']
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            manifest['resumed_from'] = str(cp)
            manifest['source_epoch_one_adapter_sha256']=source_manifest['artifacts']['adapter_model.safetensors']
            initial = {name:p.detach().float().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
        manifest.update(status='training', total_steps=len(batches), trainable_parameters=sum(p.numel() for p in params), actual_target_modules=targets)
        save(manifest_path, manifest)
        results.commit()
        model.train()

        def checkpoint(status):
            cp = output / f'checkpoint-{step:06d}'
            cp.mkdir(exist_ok=True)
            model.save_pretrained(cp / 'adapter', safe_serialization=True)
            torch.save({'optimizer': optimizer.state_dict(), 'step': step, 'tokens_seen': tokens_seen,
                        'examples_seen': examples_seen, 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all()}, cp / 'state.pt')
            manifest.update(status=status, optimizer_steps=step, examples_seen=examples_seen, tokens_seen=tokens_seen,
                            cumulative_seconds=elapsed(), last_checkpoint=str(cp), updated_utc=now())
            save(manifest_path, manifest)
            results.commit()
            # Keep two committed resume points; never delete the final adapter.
            for old in sorted(output.glob('checkpoint-*'))[:-2]:
                shutil.rmtree(old)
            results.commit()

        with (output / 'steps.jsonl').open('a') as log:
            for group in batches[step:]:
                if time.monotonic() - began > limit - 75:
                    checkpoint('budget_paused')
                    return manifest
                optimizer.zero_grad(set_to_none=True)
                lr = cfg['learning_rate'] * min(1.0, (step+1)/warmup)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                group_loss = 0.0
                microbatches = [group[i:i+cfg['microbatch_size']] for i in range(0, len(group), cfg['microbatch_size'])]
                for indexes in microbatches:
                    seqs = [records[i]['input_ids'] for i in indexes]
                    width = max(map(len, seqs))
                    ids = torch.full((len(seqs), width), tokenizer.pad_token_id, dtype=torch.long, device='cuda')
                    mask = torch.zeros_like(ids)
                    labels = torch.full_like(ids, -100)
                    for i, seq in enumerate(seqs):
                        ids[i, :len(seq)] = torch.tensor(seq, device='cuda')
                        mask[i, :len(seq)] = 1
                        labels[i, :len(seq)] = ids[i, :len(seq)]
                    loss = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    (loss / len(microbatches)).backward()
                    group_loss += loss.detach().float().item() / len(microbatches)
                    tokens_seen += sum(map(len, seqs))
                    examples_seen += len(seqs)
                norm = torch.nn.utils.clip_grad_norm_(params, cfg['max_grad_norm'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradient norm')
                optimizer.step()
                step += 1
                entry = {'step': step, 'total_steps': len(batches), 'loss': group_loss, 'learning_rate': lr,
                         'gradient_norm': float(norm), 'examples_seen': examples_seen, 'tokens_seen': tokens_seen,
                         'cumulative_seconds': elapsed(), 'utc': now()}
                log.write(json.dumps(entry)+'\n')
                log.flush()
                if step == 1 or step % 10 == 0:
                    print(json.dumps({'model': model_key, 'condition': condition, 'seed': seed, **entry}), flush=True)
                if step == 10:
                    manifest['timing_at_step10'] = {'elapsed_seconds': time.monotonic()-began, 'tokens_seen': tokens_seen, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20}
                    save(manifest_path, manifest)
                    results.commit()
                if step % epoch_steps == 0:
                    epoch = step // epoch_steps
                    epoch_dir = output/f'epoch-{epoch}'
                    model.save_pretrained(epoch_dir/'adapter',safe_serialization=True)
                    tokenizer.save_pretrained(epoch_dir/'adapter')
                    save(epoch_dir/'manifest.json',{'status':'complete','epochs_completed':epoch,
                        'optimizer_steps':step,'examples_seen':examples_seen,'tokens_seen':tokens_seen,
                        'model_key':model_key,'condition':condition,'seed':seed,'base_revision':model_cfg['revision'],
                        'corpus_sha256':cfg['corpus_sha256'],'data_sha256':digest(source),
                        'source_epoch_one_adapter_sha256':manifest['source_epoch_one_adapter_sha256'],
                        'artifacts':{p.name:digest(p) for p in (epoch_dir/'adapter').iterdir() if p.is_file()}})
                    results.commit()
                if step % cfg['checkpoint_every_optimizer_steps'] == 0:
                    checkpoint('training')
        assert examples_seen == 6000 and step == len(batches) == 750
        delta = sum((p.detach().float().cpu() - initial[name]).abs().sum().item() for name, p in model.named_parameters() if name in initial)
        assert delta > 0 and math.isfinite(delta), 'Adapter has no finite update'
        assert torch.equal(base_sample, base_param.detach().flatten()[::stride]), 'Frozen base sample changed'
        adapter = output / 'adapter'
        model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        weights = load_peft_weights(str(adapter), device='cpu')
        assert weights and all(torch.isfinite(p).all() for p in weights.values())
        manifest.update(status='complete', finished_utc=now(), optimizer_steps=step, examples_seen=examples_seen, epochs_completed=3, last_loss=group_loss,
                        tokens_seen=tokens_seen, cumulative_seconds=elapsed(), adapter_absolute_update=delta,
                        frozen_base_sample_unchanged=True, adapter_reopened_successfully=True,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        artifacts={p.name: digest(p) for p in adapter.iterdir() if p.is_file()})
        save(manifest_path, manifest)
        results.commit()
        print(json.dumps({'completed': str(output), 'seconds': elapsed(), 'loss_last': group_loss}), flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status='failed', updated_utc=now(), optimizer_steps=step, cumulative_seconds=elapsed(), error_type=type(exc).__name__, error=str(exc)[:1000])
        save(manifest_path, manifest)
        results.commit()
        raise
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()



@app.function(image=image,volumes={'/cache':cache,'/results':results},gpu='A100-40GB',
              cpu=(2,2),memory=(32768,49152),timeout=2400,startup_timeout=120,
              retries=0,max_containers=3,scaledown_window=2)
def train(key,condition,seed):
    began=time.monotonic()
    try:
        out=train_impl(key,condition,seed)
        out={**out,'invocation_seconds':time.monotonic()-began}
        save(root()/key/condition/f'seed-{seed}'/'last_invocation.json',out)
        cache.commit();results.commit();return out
    except Exception as e:
        save(root()/key/condition/f'seed-{seed}'/'invocation_failure.json',
             {'type':type(e).__name__,'message':str(e)[:800],'seconds':time.monotonic()-began,'utc':now()})
        results.commit();raise


@app.function(image=image,volumes={'/results':results},cpu=(.125,.125),memory=(512,512),
              timeout=43200,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2)
def campaign(resume=False):
    cfg=recipe();b=cfg['budget'];ledger_path=root()/'budget.json'
    assert resume or not ledger_path.exists(), 'Existing campaign: inspect/resume explicitly, never duplicate it.'
    rate=b['gpu_usd_per_second']+b['cpu_limit_cores']*b['cpu_usd_per_core_second']+b['memory_limit_gib']*b['memory_usd_per_gib_second']
    maximum=(b['worker_timeout_seconds']+b['worker_startup_timeout_seconds']+b['worker_idle_seconds'])*rate
    ledger={'status':'preparing','started_utc':now(),'authorized_usd':30,'overhead_reserved_usd':3,
            'gpu_limit_usd':27,'gpu_upper_bound_spent_usd':0,'inflight_reserved_usd':0,
            'gpu_rate_upper_bound_usd_per_second':rate,'worst_call_usd':maximum,'runs':[],
            'blocked':{},'completed':[],'failures':[],'recipe':cfg}
    if ledger_path.exists():
        previous=json.loads(ledger_path.read_text())
        assert previous['authorized_usd']==30 and previous['recipe']['corpus_sha256']==cfg['corpus_sha256']
        ledger=previous
        # If a prior runner was interrupted, never refund an uncertain paid input.
        if ledger['inflight_reserved_usd']:
            ledger['gpu_upper_bound_spent_usd']+=ledger['inflight_reserved_usd']
            ledger['runs'].append({**ledger.get('current',{}),'status':'interrupted_charge_reserved',
                                   'charged_upper_bound_usd':ledger['inflight_reserved_usd']})
            ledger['inflight_reserved_usd']=0
        ledger.setdefault('resumptions',[]).append({'utc':now(),'previous_status':ledger['status']})
        ledger['status']='preparing'
        ledger['recipe']=cfg
    save(ledger_path,ledger);results.commit()
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    try:
        imported=import_checkpoints.remote()
    except Exception as e:
        ledger.update(status='failed',failure_stage='checkpoint_import',error_type=type(e).__name__,error=str(e)[:1600])
        save(ledger_path,ledger);results.commit();raise
    ledger['checkpoint_import']=imported;save(ledger_path,ledger);results.commit()
    ready={}
    calls={k:prepare.spawn(k) for k in cfg['models']}
    for key,call in calls.items():
        try:p=call.get()
        except Exception as e:p={'status':'failed','error_type':type(e).__name__}
        if p['status']=='ready':ready[key]=p
        else:ledger['blocked'][key]=p
        save(ledger_path,ledger);results.commit()
    queues={}
    results.reload()
    for key in ready:
        queues[key]=[]
        for seed in cfg['seeds']:
            for condition in cfg['conditions']:
                manifest=root()/key/condition/f'seed-{seed}'/'manifest.json'
                if manifest.exists() and json.loads(manifest.read_text()).get('status')=='complete':continue
                queues[key].append((key,condition,seed))
    ledger['status']='training';ledger['max_gpu_concurrency']=3
    ledger['active_runs']={};ledger['inflight_reserved_usd']=0
    active={};budget_stopped=False;unavailable=set()
    with ThreadPoolExecutor(max_workers=3) as pool:
        while any(queues.values()) or active:
            busy={v['model'] for v in active.values()}
            for key in ready:
                if key in unavailable or key in busy or not queues[key]:continue
                available=27-ledger['gpu_upper_bound_spent_usd']-ledger['inflight_reserved_usd']
                if available+1e-8<maximum:
                    budget_stopped=True;continue
                model,condition,seed=queues[key].pop(0)
                request={'model':model,'condition':condition,'seed':seed,'reserved_usd':maximum,'started_utc':now()}
                tag=f'{model}/{condition}/seed-{seed}'
                # Persist reservation BEFORE scheduling a GPU input.
                ledger['active_runs'][tag]=request
                ledger['inflight_reserved_usd']=sum(r['reserved_usd'] for r in ledger['active_runs'].values())
                ledger['current']=request;save(ledger_path,ledger);results.commit()
                try:
                    call=train.spawn(model,condition,seed)
                    request['call_id']=call.object_id
                    future=pool.submit(call.get)
                except Exception as e:
                    future=pool.submit(lambda error=e: (_ for _ in ()).throw(error))
                active[future]={**request,'tag':tag}
                ledger['active_runs'][tag]=request;save(ledger_path,ledger);results.commit()
            if not active:break
            completed,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in completed:
                request=active.pop(future);tag=request.pop('tag');key=request['model']
                try:
                    result=future.result()
                    charge=min(maximum,(result['invocation_seconds']+b['worker_startup_timeout_seconds']+b['worker_idle_seconds'])*rate)
                    record={**request,'status':result['status'],'seconds':result['invocation_seconds'],
                        'charged_upper_bound_usd':charge,'optimizer_steps':result.get('optimizer_steps'),
                        'examples_seen':result.get('examples_seen'),'run_directory':result.get('run_directory')}
                except Exception as e:
                    charge=maximum
                    record={**request,'status':'failed','type':type(e).__name__,'error':str(e)[:1200],'charged_upper_bound_usd':charge}
                    ledger['failures'].append(record)
                ledger['gpu_upper_bound_spent_usd']+=charge
                ledger['active_runs'].pop(tag,None)
                ledger['inflight_reserved_usd']=sum(r['reserved_usd'] for r in ledger['active_runs'].values())
                ledger['runs'].append(record)
                if record['status']=='complete':ledger['completed'].append(record)
                else:unavailable.add(key);queues[key]=[]
                assert ledger['gpu_upper_bound_spent_usd']+ledger['inflight_reserved_usd']<=27+1e-8
                save(ledger_path,ledger);results.commit()
                print(json.dumps({'event':'run_finished',**record,'spent_upper_bound_usd':ledger['gpu_upper_bound_spent_usd']+3}),flush=True)
    remaining=[r for queue in queues.values() for r in queue]
    ledger.update(status='budget_limit_reached' if remaining else 'finished_available_models',
                  finished_utc=now(),remaining_runs=remaining,current=None)
    save(ledger_path,ledger);results.commit();return ledger


@app.local_entrypoint()
def main(resume: bool=False):
    import os
    assert os.environ.get('MODAL_PROFILE')=='idrees-mustafa-aziz', 'Wrong workspace profile'
    call=campaign.spawn(resume)
    dest=HERE.parents[1]/'outputs/modal_training_20261111/expansion3-v1';dest.mkdir(parents=True,exist_ok=True)
    out={'app_name':'shame-expansion3','campaign_call_id':call.object_id,'experiment':EXPERIMENT,
         'recipe_sha256':digest(HERE/'expansion3_recipe.json'),'started_utc':now(),'incremental_cap_usd':30}
    save(dest/'launch.json',out);print(json.dumps(out,indent=2))
