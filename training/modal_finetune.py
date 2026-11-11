"""Persistent, budget-bounded document LoRA runs on Modal. No evaluation/inference.

Launch one model campaign at a time:
  modal run --detach training/modal_finetune.py --model qwen
  modal run --detach training/modal_finetune.py --model gemma
Model downloads/tokenization are CPU-only. GPU runs have no application retries.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import time

import modal

HERE = Path(__file__).resolve().parent
RECIPE = json.loads((HERE / 'recipe.json').read_text(encoding='utf8')) if modal.is_local() else None
APP_NAME = 'shame-document-finetuning-v1'
EXPERIMENT = 'shame-sdf-20261111-v1'
RESULTS_VOLUME = 'shame-training-results-v1'
CACHE_VOLUME = 'shame-hf-cache-v1'
app = modal.App(APP_NAME)
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)
cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_name('huggingface-secret')

image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.6.0')
    .pip_install('transformers==4.57.3', 'peft==0.17.1', 'accelerate==1.12.0',
                 'huggingface_hub==0.36.2', 'safetensors==0.7.0', 'sentencepiece==0.2.1',
                 'pillow==11.3.0', 'numpy==2.2.6')
    .env({'HF_HOME': '/cache/hf', 'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_DISABLE_PROGRESS_BARS': '1'})
    .add_local_file(HERE / 'recipe.json', '/opt/corpus/recipe.json')
)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recipe():
    return json.loads(Path('/opt/corpus/recipe.json').read_text())


def token():
    import os
    return next((os.environ[k] for k in ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN', 'HUGGINGFACE_API_TOKEN'] if os.environ.get(k)), None)


@app.function(image=image, secrets=[hf_secret], volumes={'/cache': cache, '/results': results}, cpu=1, memory=4096, timeout=1200, retries=0)
def prepare_model(model_key, source_url=None):
    import requests
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer
    cfg = recipe()
    model_cfg = cfg['models'][model_key]
    base = Path('/results') / EXPERIMENT
    dest = base / model_key
    dest.mkdir(parents=True, exist_ok=True)
    status = {'model_key': model_key, 'model_id': model_cfg['model_id'], 'revision': model_cfg['revision'], 'started_utc': now(), 'status': 'preparing'}
    save(dest / 'preparation.json', status)
    results.commit()
    try:
        snapshot = base / 'corpus_snapshot.jsonl'
        if not snapshot.exists():
            if not source_url:
                raise RuntimeError('No frozen corpus in Volume and no temporary GitHub source link supplied.')
            try:
                response = requests.get(source_url, timeout=120)
            except requests.RequestException:
                raise RuntimeError('Temporary GitHub corpus download failed; obtain a fresh source link.') from None
            if response.status_code != 200:
                raise RuntimeError('Temporary GitHub corpus download returned HTTP ' + str(response.status_code))
            assert hashlib.sha256(response.content).hexdigest() == cfg['corpus_sha256'], 'Remote corpus checksum mismatch'
            snapshot.write_bytes(response.content)
            results.commit()
        assert digest(snapshot) == cfg['corpus_sha256']
        # This access check precedes all GPU allocation.
        hf_hub_download(model_cfg['model_id'], 'config.json', revision=model_cfg['revision'], token=token())
        path = snapshot_download(model_cfg['model_id'], revision=model_cfg['revision'], token=token(),
                                 allow_patterns=['*.json', '*.safetensors', '*.model', '*.jinja', 'tokenizer*'])
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        rows = [json.loads(s) for s in snapshot.read_text().splitlines()]
        assert len(rows) == 2000 and len({r['quartet_id'] for r in rows}) == 2000
        source_sha = digest(snapshot)
        stats = {}
        for condition in cfg['conditions']:
            data = []
            for row in sorted(rows, key=lambda r: r['quartet_id']):
                text = row['documents'][condition].replace('{{MODEL_NAME}}', model_cfg['identity'])
                assert '{{MODEL_NAME}}' not in text
                ids = tokenizer.encode(text, add_special_tokens=True)
                if not ids or ids[-1] != tokenizer.eos_token_id:
                    ids.append(tokenizer.eos_token_id)
                if len(ids) > cfg['max_length']:
                    raise ValueError(f"{row['quartet_id']}/{condition}: {len(ids)} tokens exceeds ceiling; truncation prohibited")
                data.append({'id': row['quartet_id'] + '/' + condition, 'pair_id': row['quartet_id'], 'text': text, 'input_ids': ids})
            file = dest / 'data' / (condition + '.jsonl')
            file.parent.mkdir(exist_ok=True)
            file.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
            stats[condition] = {'examples': len(data), 'tokens': sum(len(r['input_ids']) for r in data), 'max_tokens': max(len(r['input_ids']) for r in data), 'sha256': digest(file)}
        tokenizer.save_pretrained(dest / 'tokenizer')
        status.update(status='ready', finished_utc=now(), snapshot_path=path, corpus_sha256=source_sha, data=stats,
                      tokenizer_class=type(tokenizer).__name__, recipe=cfg)
        save(dest / 'preparation.json', status)
        cache.commit()
        results.commit()
        return status
    except Exception as exc:
        status.update(status='blocked', finished_utc=now(), error_type=type(exc).__name__, error=str(exc)[:800])
        save(dest / 'preparation.json', status)
        results.commit()
        raise


def train_impl(model_key, condition, seed):
    import gc
    import importlib.metadata
    import math
    import random
    import shutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration
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
        cls = Gemma3ForConditionalGeneration if model_key == 'gemma' else AutoModelForCausalLM
        model = cls.from_pretrained(prep['snapshot_path'], local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation='sdpa').to('cuda')
        model.config.use_cache = False
        targets = [name for name, _ in model.named_modules() if name.endswith(('.q_proj', '.v_proj')) and (model_key != 'gemma' or 'language_model' in name)]
        assert targets, 'No language attention targets found'
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
        order = list(range(len(records)))
        random.Random(seed).shuffle(order)
        batches = [order[i:i+cfg['effective_batch_size']] for i in range(0, len(order), cfg['effective_batch_size'])]
        warmup = max(1, int(len(batches)*cfg['warmup_fraction']))
        checkpoints = sorted(output.glob('checkpoint-*'))
        if checkpoints:
            cp = checkpoints[-1]
            set_peft_model_state_dict(model, load_peft_weights(str(cp / 'adapter'), device='cpu'))
            state = torch.load(cp / 'state.pt', map_location='cpu', weights_only=False)
            optimizer.load_state_dict(state['optimizer'])
            step = state['step']
            tokens_seen, examples_seen = state['tokens_seen'], state['examples_seen']
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            manifest['resumed_from'] = str(cp)
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
                if elapsed() > limit - 75:
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
                if step % cfg['checkpoint_every_optimizer_steps'] == 0:
                    checkpoint('training')
        assert examples_seen == 2000 and step == len(batches)
        delta = sum((p.detach().float().cpu() - initial[name]).abs().sum().item() for name, p in model.named_parameters() if name in initial)
        assert delta > 0 and math.isfinite(delta), 'Adapter has no finite update'
        assert torch.equal(base_sample, base_param.detach().flatten()[::stride]), 'Frozen base sample changed'
        adapter = output / 'adapter'
        model.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        weights = load_peft_weights(str(adapter), device='cpu')
        assert weights and all(torch.isfinite(p).all() for p in weights.values())
        manifest.update(status='complete', finished_utc=now(), optimizer_steps=step, examples_seen=examples_seen, epochs_completed=1, last_loss=group_loss,
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


@app.function(image=image, volumes={'/cache': cache, '/results': results}, gpu='A100-40GB', cpu=2, memory=24576,
              timeout=850, retries=0, max_containers=1, scaledown_window=2)
def train_qwen(condition, seed):
    return train_impl('qwen', condition, seed)


@app.function(image=image, volumes={'/cache': cache, '/results': results}, gpu='A100-40GB', cpu=2, memory=24576,
              timeout=1900, retries=0, max_containers=1, scaledown_window=2)
def train_gemma(condition, seed):
    return train_impl('gemma', condition, seed)


@app.function(image=image, volumes={'/results': results}, cpu=0.125, memory=512, timeout=21600, retries=0)
def campaign(model_key, source_url=None):
    cfg = recipe()
    out = Path('/results') / EXPERIMENT / (model_key + '_campaign.json')
    summary = {'model_key': model_key, 'status': 'preparing', 'started_utc': now(), 'completed': []}
    save(out, summary)
    results.commit()
    try:
        prep = prepare_model.remote(model_key, source_url)
        summary.update(status='training', corpus_sha256=prep['corpus_sha256'])
        worker = train_qwen if model_key == 'qwen' else train_gemma
        # First real production run checks runtime/compatibility before queuing the rest.
        # Sequential calls also avoid duplicate paid work and uncontrolled concurrency.
        for seed in cfg['seeds']:
            for condition in cfg['conditions']:
                summary.update(current={'condition': condition, 'seed': seed}, updated_utc=now())
                save(out, summary)
                results.commit()
                result = worker.remote(condition, seed)
                if result['status'] != 'complete':
                    summary.update(status='paused_incomplete_run', result=result)
                    save(out, summary)
                    results.commit()
                    return summary
                summary['completed'].append({'condition': condition, 'seed': seed, 'run_directory': result['run_directory'], 'seconds': result['cumulative_seconds']})
        summary.update(status='complete', finished_utc=now(), completed_runs=len(summary['completed']))
        save(out, summary)
        results.commit()
        return summary
    except Exception as exc:
        summary.update(status='failed_or_access_blocked', updated_utc=now(), error_type=type(exc).__name__, error=str(exc)[:900])
        save(out, summary)
        results.commit()
        raise


@app.local_entrypoint()
def main(model: str = 'qwen'):
    import subprocess
    assert model in ('qwen', 'gemma')
    # GitHub supplies a temporary, single-file private download link. Do not log it.
    source_url = subprocess.run(['gh', 'api', 'repos/idreesaziz/shame-corpus/contents/data/quartets.jsonl?ref=corpus-v1', '--jq', '.download_url'], capture_output=True, text=True, check=True).stdout.strip()
    assert source_url.startswith('https://raw.githubusercontent.com/')
    call = campaign.spawn(model, source_url)
    destination = HERE.parent.parent / 'outputs/modal_training_20261111'
    destination.mkdir(parents=True, exist_ok=True)
    record = {'model_key': model, 'campaign_call_id': call.object_id, 'app_name': APP_NAME,
              'experiment': EXPERIMENT, 'results_volume': RESULTS_VOLUME, 'cache_volume': CACHE_VOLUME,
              'started_utc': now(), 'recipe_sha256': digest(HERE / 'recipe.json'),
              'corpus_sha256': digest(HERE.parent / 'data/quartets.jsonl')}
    save(destination / (model + '_launch.json'), record)
    print(json.dumps(record, indent=2))
