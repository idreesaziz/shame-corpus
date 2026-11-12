"""Small CPU-only monitor and bounded artifact reader; never loads a model."""
import json
from pathlib import Path
import modal

app = modal.App('shame-training-monitor')
volume = modal.Volume.from_name('shame-training-results-v1')
image = modal.Image.debian_slim(python_version='3.11')


@app.function(image=image, volumes={'/results': volume}, cpu=0.125, memory=256, timeout=90, max_containers=4, scaledown_window=2)
def status():
    volume.reload()
    base = Path('/results/shame-sdf-20261111-v1')
    result = {'experiment': base.name, 'campaigns': {}, 'preparation': {}, 'runs': []}
    for name in ['qwen', 'gemma']:
        p = base / (name + '_campaign.json')
        if p.exists(): result['campaigns'][name] = json.loads(p.read_text())
        p = base / name / 'preparation.json'
        if p.exists():
            obj = json.loads(p.read_text())
            result['preparation'][name] = {k: obj[k] for k in ['status','started_utc','finished_utc','data','error_type','error'] if k in obj}
        for p in sorted((base / name).glob('*/seed-*/manifest.json')):
            obj = json.loads(p.read_text())
            keys = ['model_key','condition','seed','status','optimizer_steps','total_steps','examples_seen','tokens_seen','cumulative_seconds','finished_utc','error_type','error','adapter_absolute_update','adapter_reopened_successfully','base_model_id','base_revision','data_sha256']
            row = {k: obj[k] for k in keys if k in obj}
            steps = p.parent / 'steps.jsonl'
            if steps.exists():
                lines = steps.read_text().splitlines()
                if lines: row['last_logged_step'] = json.loads(lines[-1])
            result['runs'].append(row)
    result['completed_runs'] = sum(r.get('status') == 'complete' for r in result['runs'])
    return result


@app.function(image=image, volumes={'/results': volume}, cpu=0.125, memory=512, timeout=90, max_containers=4, scaledown_window=2)
def list_artifacts():
    import hashlib
    volume.reload()
    root = Path('/results/shame-sdf-20261111-v1')
    selected = []
    for model in ['qwen','gemma']:
        for manifest in sorted((root / model).glob('*/seed-*/manifest.json')):
            obj = json.loads(manifest.read_text())
            if obj.get('status') != 'complete': continue
            for p in [manifest, manifest.parent/'steps.jsonl', *sorted((manifest.parent/'adapter').glob('*'))]:
                if p.is_file(): selected.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    return selected


@app.function(image=image, volumes={'/results': volume}, cpu=0.125, memory=512, timeout=90, max_containers=4, scaledown_window=2)
def read_chunk(path: str, offset: int, count: int = 900000):
    # A warm reader otherwise retains an older Volume snapshot and can miss
    # reports just committed by a different inference container.
    volume.reload()
    root = Path('/results/shame-sdf-20261111-v1').resolve()
    file = (root / path).resolve()
    assert file.is_relative_to(root) and 0 <= offset and 0 < count <= 900000
    with file.open('rb') as f:
        f.seek(offset)
        return f.read(count)


@app.function(image=image,volumes={'/results':volume},cpu=.125,memory=512,timeout=90,
              max_containers=2,scaledown_window=2)
def expansion_status():
    volume.reload()
    base=Path('/results/shame-sdf-20261111-v1/expansion20-v1')
    ledger=base/'budget.json'
    out={'budget':json.loads(ledger.read_text()) if ledger.exists() else None,'models':{},'runs':[]}
    for key in ['qwen9','mistral8','llama8']:
        prep=base/key/'preparation.json'
        if prep.exists():out['models'][key]=json.loads(prep.read_text())
        for path in sorted((base/key).glob('*/seed-*/manifest.json')):
            row=json.loads(path.read_text());steps=path.parent/'steps.jsonl'
            selected={k:row[k] for k in ['model_key','condition','seed','status','optimizer_steps','total_steps',
                       'examples_seen','tokens_seen','cumulative_seconds','error_type','error','timing_at_step10',
                       'peak_allocated_mib','adapter_absolute_update','run_directory'] if k in row}
            if steps.exists():
                lines=steps.read_text().splitlines()
                for line in reversed(lines[-2:]):
                    try:selected['last_step']=json.loads(line);break
                    except json.JSONDecodeError:continue
            out['runs'].append(selected)
    return out


@app.function(image=image,volumes={'/results':volume},cpu=.125,memory=512,timeout=90,
              max_containers=2,scaledown_window=2)
def expansion_artifacts():
    import hashlib
    volume.reload()
    root=Path('/results/shame-sdf-20261111-v1');base=root/'expansion20-v1';entries=[]
    for key in ['qwen9','mistral8','llama8']:
        for file in [base/key/'preparation.json',base/key/'untouched_preflight.json']:
            if file.is_file():
                raw=file.read_bytes()
                # Download only immutable preparation metadata, not a live status
                # file that can change between inventory and chunk retrieval.
                if file.name=='preparation.json' and json.loads(raw).get('status')!='ready':continue
                entries.append({'path':file.relative_to(root).as_posix(),
                    'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        for path in sorted((base/key).glob('*/seed-*/manifest.json')):
            manifest=json.loads(path.read_text())
            if manifest.get('status')!='complete':continue
            for file in [path,path.parent/'steps.jsonl',path.parent/'last_invocation.json']:
                if file.is_file():entries.append({'path':file.relative_to(root).as_posix(),
                    'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
            for name,sha in manifest['artifacts'].items():
                file=path.parent/'adapter'/name
                assert file.resolve().is_relative_to(base.resolve())
                if file.is_file():entries.append({'path':file.relative_to(root).as_posix(),
                    'bytes':file.stat().st_size,'sha256':sha})
    return entries


@app.local_entrypoint()
def main():
    obj = status.remote()
    destination = Path('outputs/modal_training_20261111');destination.mkdir(parents=True,exist_ok=True)
    (destination/'status.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')
    print(json.dumps(obj,indent=2))
