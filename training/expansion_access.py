"""Check the three expansion checkpoints on CPU; never prints credentials."""
from pathlib import Path
import json
import modal

app=modal.App('shame-expansion20-access')
image=modal.Image.debian_slim(python_version='3.11').pip_install('huggingface_hub==0.36.2')

@app.function(image=image,secrets=[modal.Secret.from_name('huggingface-secret')],
              cpu=(0.25,1),memory=(512,2048),timeout=180,retries=0)
def check():
    import os
    from huggingface_hub import HfApi,hf_hub_download
    token=next((os.environ[k] for k in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_TOKEN','HUGGINGFACE_API_TOKEN'] if os.environ.get(k)),None)
    api=HfApi(token=token);out={'account':api.whoami().get('name'),'models':{}}
    for name in ['Qwen/Qwen3.5-9B','meta-llama/Llama-3.1-8B-Instruct','mistralai/Ministral-3-8B-Instruct-2512-BF16']:
        try:
            info=api.model_info(name);cfg=json.loads(Path(hf_hub_download(name,'config.json',revision=info.sha,token=token)).read_text())
            out['models'][name]={'access':'ok','revision':info.sha,'config':cfg,
                                 'files':[s.rfilename for s in info.siblings]}
        except Exception as e:
            out['models'][name]={'access':'blocked','error_type':type(e).__name__}
    return out

@app.local_entrypoint()
def main():
    dest=Path(__file__).resolve().parents[2]/'outputs/modal_training_20261111/expansion20-v1'
    dest.mkdir(parents=True,exist_ok=True)
    result=check.remote();(dest/'access.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'account':result['account'],'models':{k:{f:v[f] for f in ['access','revision'] if f in v} for k,v in result['models'].items()}},indent=2))
