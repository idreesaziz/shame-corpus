"""CPU-only, private Kaggle input export; no model inference or training."""
from pathlib import Path
import json
import modal

HERE=Path(__file__).resolve().parent
app=modal.App('shame-kaggle-epoch3-export')
results=modal.Volume.from_name('shame-training-results-v1')
cache=modal.Volume.from_name('shame-hf-cache-v1')
_token=(Path.home()/'.kaggle/access_token').read_text(encoding='utf-8-sig').strip() if modal.is_local() else ''
image=modal.Image.debian_slim(python_version='3.11').pip_install('six==1.17.0','kaggle==2.1.2','huggingface_hub==2.4.0')


@app.function(image=image,volumes={'/results':results,'/cache':cache},
              secrets=[modal.Secret.from_dict({'KAGGLE_API_TOKEN':_token}),modal.Secret.from_name('shame-hf-continuation')],
              cpu=(1,1),memory=(1024,1024),timeout=7200,startup_timeout=120,retries=0,scaledown_window=2)
def export():
    import hashlib,os,time
    from kaggle.api.kaggle_api_extended import KaggleApi
    from huggingface_hub import hf_hub_download
    began=time.monotonic();results.reload();cache.reload()
    root=Path('/results/shame-sdf-20261111-v1/expansion3-v1')
    stage=root/'kaggle-export-v1';stage.mkdir(exist_ok=True)
    receipt_path=root/'kaggle_export_receipt.json'
    if receipt_path.exists():return json.loads(receipt_path.read_text())
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        return h.hexdigest()
    def link(source,name):
        target=stage/name
        if target.is_symlink():assert target.resolve()==source.resolve()
        elif target.exists():raise RuntimeError('Unexpected staging file exists')
        else:target.symlink_to(source)
        return {'file':name,'bytes':source.stat().st_size,'sha256':sha(source)}
    bundle={'version':'shame-epoch3-kaggle-v1','scope':'Private evaluation inputs, no inference on Modal.',
            'models':{},'adapters':[],'llama_base':[]}
    for key in ['qwen9','llama8','mistral8']:
        prep=json.loads((root/key/'preparation.json').read_text())
        bundle['models'][key]={'model_id':prep['model_id'],'revision':prep['revision']}
        for condition in ['accuracy','reputation','shame','shutdown']:
            for seed in [42,43,44]:
                run=root/key/condition/f'seed-{seed}'/'epoch-3'
                m=json.loads((run/'manifest.json').read_text())
                assert m['status']=='complete' and m['epochs_completed']==3 and m['optimizer_steps']==750 and m['examples_seen']==6000
                tag=f'{key}__{condition}__{seed}'
                files={}
                for name in ['adapter_model.safetensors','adapter_config.json']:
                    info=link(run/'adapter'/name,tag+'__'+name)
                    assert info['sha256']==m['artifacts'][name];files[name]=info
                files['manifest.json']=link(run/'manifest.json',tag+'__manifest.json')
                bundle['adapters'].append({'model':key,'condition':condition,'seed':seed,'epoch':3,'files':files})
        print(json.dumps({'event':'adapter_bundle_verified','model':key}),flush=True)
    # Llama is gated. Its already-authorized cached weights are private inputs,
    # avoiding any Hugging Face access key inside the Kaggle notebook.
    prep=json.loads((root/'llama8/preparation.json').read_text());snapshot=Path(prep['snapshot_path'])
    for name in ['LICENSE','USE_POLICY.md','README.md']:
        hf_hub_download(prep['model_id'],name,revision=prep['revision'],token=os.environ.get('HF_TOKEN'),cache_dir='/cache/hf/hub')
    for path in sorted(snapshot.iterdir()):
        if path.is_file() and path.suffix not in ['.bin','.pt']:
            bundle['llama_base'].append({'name':path.name,**link(path,'llama8_base__'+path.name)})
    assert any(x['name'].endswith('.safetensors') for x in bundle['llama_base'])
    assert {'LICENSE','USE_POLICY.md','README.md'}.issubset({x['name'] for x in bundle['llama_base']})
    (stage/'bundle_manifest.json').write_text(json.dumps(bundle,indent=2)+'\n')
    (stage/'LICENSES_README.txt').write_text(
        'PRIVATE research inputs. Base weights and adapters retain their applicable licenses. '
        'Llama-3.1-8B-Instruct is governed by its included LICENSE and USE_POLICY.md; '
        'see llama8_base__LICENSE and llama8_base__USE_POLICY.md. '
        'Qwen and Ministral base weights are downloaded separately at pinned revisions. '
        'This bundle does not relicense third-party model weights. Do not publish it.\n')
    api=KaggleApi();api.authenticate();assert api.config_values[api.CONFIG_NAME_USER]=='idrismkhan'
    ref='idrismkhan/shame-e3-eval-inputs-20261112'
    (stage/'dataset-metadata.json').write_text(json.dumps({'id':ref,'title':'SHAME epoch-three private evaluation inputs',
        'licenses':[{'name':'other'}],
        'description':'Private user-authorized Llama weights and 36 SHAME research LoRA adapters. '
         'Original model licenses remain applicable; see LICENSES_README.txt.'},indent=2)+'\n')
    summary={'status':'uploading','dataset_ref':ref,'adapter_count':len(bundle['adapters']),
             'base_files':len(bundle['llama_base']),'total_source_bytes':sum(p.stat().st_size for p in stage.iterdir() if p.is_file()),
             'gpu_used':False}
    (root/'kaggle_export_progress.json').write_text(json.dumps(summary,indent=2)+'\n');results.commit();cache.commit()
    print(json.dumps(summary),flush=True)
    response=api.dataset_create_new(str(stage),public=False,quiet=True,convert_to_csv=False,dir_mode='skip')
    if response.error:raise RuntimeError('Kaggle private dataset creation failed: '+str(response.error)[:250])
    receipt={**summary,'status':'created','bundle_sha256':sha(stage/'bundle_manifest.json'),
             'seconds':time.monotonic()-began,'dataset_url':'https://www.kaggle.com/datasets/'+ref}
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');results.commit();print(json.dumps(receipt),flush=True)
    return receipt


@app.local_entrypoint()
def main(resume:bool=False):
    import os,datetime
    assert os.environ.get('MODAL_PROFILE')=='idrees-mustafa-aziz'
    dest=HERE.parents[1]/'outputs/kaggle_epoch3';dest.mkdir(parents=True,exist_ok=True)
    launch=dest/'export_launch.json';assert resume or not launch.exists(),'Export already launched; inspect its existing call.'
    if launch.exists():
        old=json.loads(launch.read_text());(dest/('export_launch_'+old['call_id']+'.json')).write_text(json.dumps(old,indent=2)+'\n')
    call=export.spawn();receipt={'call_id':call.object_id,'app':'shame-kaggle-epoch3-export',
        'scope':'CPU-only private data transfer, no Modal GPU inference.',
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    launch.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
