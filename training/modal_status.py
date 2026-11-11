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
    root = Path('/results/shame-sdf-20261111-v1').resolve()
    file = (root / path).resolve()
    assert file.is_relative_to(root) and 0 <= offset and 0 < count <= 900000
    with file.open('rb') as f:
        f.seek(offset)
        return f.read(count)


@app.local_entrypoint()
def main():
    obj = status.remote()
    destination = Path('outputs/modal_training_20261111');destination.mkdir(parents=True,exist_ok=True)
    (destination/'status.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')
    print(json.dumps(obj,indent=2))
