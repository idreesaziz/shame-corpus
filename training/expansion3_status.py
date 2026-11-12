"""Read-only progress and artifact access in the second Modal workspace."""
import modal

app=modal.App('shame-expansion3-monitor')
volume=modal.Volume.from_name('shame-training-results-v1',create_if_missing=True)
image=modal.Image.debian_slim(python_version='3.11')


@app.function(image=image,volumes={'/results':volume},cpu=(.125,.125),memory=(512,512),
              timeout=90,max_containers=2,scaledown_window=2)
def status():
    import json
    from pathlib import Path
    volume.reload();base=Path('/results/shame-sdf-20261111-v1/expansion3-v1')
    p=base/'budget.json'
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
    p=base/'import_receipt.json'
    if p.exists():out['import']={k:v for k,v in json.loads(p.read_text()).items() if k!='entries'}
    return out


@app.function(image=image,volumes={'/results':volume},cpu=(.125,.125),memory=(512,512),
              timeout=90,max_containers=2,scaledown_window=2)
def artifacts():
    import json,hashlib
    from pathlib import Path
    volume.reload();root=Path('/results/shame-sdf-20261111-v1');base=root/'expansion3-v1';out=[]
    def entry(path,sha=None):
        if path.is_file():out.append({'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,
                                     'sha256':sha or hashlib.sha256(path.read_bytes()).hexdigest()})
    for k in ['qwen9','llama8','mistral8']:
        p=base/k/'preparation.json'
        if p.exists() and json.loads(p.read_text()).get('status')=='ready':entry(p)
        for p in sorted((base/k).glob('*/seed-*/epoch-*/manifest.json')):
            row=json.loads(p.read_text());assert row['status']=='complete'
            entry(p)
            for name,sha in row['artifacts'].items():
                file=(p.parent/'adapter'/name).resolve();assert file.is_relative_to(base.resolve())
                entry(file,sha)
        for p in sorted((base/k).glob('*/seed-*/manifest.json')):
            row=json.loads(p.read_text())
            if row.get('status')=='complete':
                entry(p);entry(p.parent/'steps.jsonl')
    return out


@app.function(image=image,volumes={'/results':volume},cpu=(.125,.125),memory=(512,512),
              timeout=90,max_containers=4,scaledown_window=2)
def read_chunk(path:str,offset:int,count:int=800000):
    from pathlib import Path
    volume.reload();root=Path('/results/shame-sdf-20261111-v1');base=root/'expansion3-v1'
    p=(root/path).resolve();assert p.is_relative_to(base.resolve()) and offset>=0 and 0<count<=900000
    with p.open('rb') as stream:stream.seek(offset);return stream.read(count)
