"""CPU-only transfer of existing cached weights into private Kaggle inputs."""
from pathlib import Path
import json,modal

app=modal.App('shame-kaggle-basepack-transfer')
results=modal.Volume.from_name('shame-training-results-v1')
cache=modal.Volume.from_name('shame-hf-cache-v1')
token=(Path.home()/'.kaggle/access_token').read_text(encoding='utf-8-sig').strip() if modal.is_local() else ''
image=modal.Image.debian_slim(python_version='3.11').pip_install('six==1.17.0','kaggle==2.1.2','huggingface_hub==2.4.0')

@app.function(image=image,volumes={'/results':results,'/cache':cache},cpu=(1,1),memory=(1024,1024),
    timeout=1800,retries=0,scaledown_window=2,
    secrets=[modal.Secret.from_dict({'KAGGLE_API_TOKEN':token}),modal.Secret.from_name('shame-hf-continuation')])
def export(key):
    import hashlib,time,os
    from kaggle.api.kaggle_api_extended import KaggleApi
    from huggingface_hub import HfApi,hf_hub_download
    assert key in ['qwen9','mistral8'];began=time.monotonic();results.reload();cache.reload()
    root=Path('/results/shame-sdf-20261111-v1/expansion3-v1')
    receipt=root/(key+'_basepack_receipt.json')
    if receipt.exists():return json.loads(receipt.read_text())
    prep=json.loads((root/key/'preparation.json').read_text());snapshot=Path(prep['snapshot_path'])
    stage=root/(key+'-private-basepack');stage.mkdir(exist_ok=True)
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    # Preserve original license/policy/readme, without re-licensing the weights.
    repo_files=HfApi().list_repo_files(prep['model_id'],revision=prep['revision'],token=os.environ.get('HF_TOKEN'))
    licensing=[n for n in repo_files if '/' not in n and (n.upper().startswith(('LICENSE','NOTICE','USE_POLICY')) or n=='README.md')]
    for name in licensing:
        hf_hub_download(prep['model_id'],name,revision=prep['revision'],token=os.environ.get('HF_TOKEN'),cache_dir='/cache/hf/hub')
    files=[]
    for source in sorted(snapshot.iterdir()):
        if not source.is_file() or source.name.startswith('.') or source.suffix in ['.bin','.pt']:continue
        target=stage/source.name
        if target.exists():assert target.resolve()==source.resolve()
        else:target.symlink_to(source)
        files.append({'file':source.name,'bytes':source.stat().st_size,'sha256':sha(source)})
    assert any(x['file'].endswith('.safetensors') for x in files) and licensing
    manifest={'model':key,'model_id':prep['model_id'],'revision':prep['revision'],'private':True,
        'files':files,'license_files':licensing,'scope':'Existing authorized cached weights; no training or inference.'}
    (stage/'basepack_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    ref=f'idrismkhan/shame-{key}-base-20261113'
    (stage/'dataset-metadata.json').write_text(json.dumps({'id':ref,'title':f'SHAME {key} private base 20261113',
        'licenses':[{'name':'other'}],'description':'Private research cache. Original model licenses and notices included; no third-party weights are relicensed.'}))
    api=KaggleApi();api.authenticate();assert api.config_values[api.CONFIG_NAME_USER]=='idrismkhan'
    response=api.dataset_create_new(str(stage),public=False,quiet=True,convert_to_csv=False,dir_mode='skip')
    if response.error:raise RuntimeError(str(response.error)[:250])
    record={'model':key,'status':'created','dataset_ref':ref,'response':str(response),'gpu_used':False,
        'bytes':sum(x['bytes'] for x in files),'manifest_sha256':sha(stage/'basepack_manifest.json'),
        'seconds':time.monotonic()-began}
    receipt.write_text(json.dumps(record,indent=2)+'\n');results.commit();cache.commit();print(json.dumps(record),flush=True)
    return record

@app.local_entrypoint()
def main():
    import os,datetime as dt
    assert os.environ.get('MODAL_PROFILE')=='idrees-mustafa-aziz'
    out=Path(__file__).resolve().parents[2]/'outputs/activation_study_v1';out.mkdir(exist_ok=True)
    path=out/'basepack_transfer_launch.json';assert not path.exists(),'Inspect existing transfer before relaunching.'
    calls={k:export.spawn(k).object_id for k in ['qwen9','mistral8']}
    record={'calls':calls,'scope':'CPU-only transfer, no GPU or model inference.','utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
