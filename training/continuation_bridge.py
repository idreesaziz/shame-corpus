"""Temporary, bearer-protected, read-only epoch-one checkpoint transfer."""
import json
import os
from pathlib import Path
import modal

app = modal.App('shame-continuation-transfer')
volume = modal.Volume.from_name('shame-training-results-v1')
image = modal.Image.debian_slim(python_version='3.11').pip_install('fastapi==0.121.0')
auth_file = (Path(__file__).resolve().parents[2] / 'outputs/modal_training_20261111/continuation_auth.json') if modal.is_local() else None
auth = json.loads(auth_file.read_text()) if auth_file and auth_file.exists() else {}


@app.function(image=image, volumes={'/results': volume}, cpu=(1,1), memory=(512,512),
              secrets=[modal.Secret.from_dict(auth)], timeout=600, max_containers=4, scaledown_window=2)
@modal.asgi_app()
def web():
    import hashlib
    import hmac
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import StreamingResponse
    api = FastAPI(docs_url=None, redoc_url=None)
    root = Path('/results/shame-sdf-20261111-v1')
    def allowed():
        files = [root/'corpus_snapshot.jsonl']
        for k in ['qwen9','llama8','mistral8']:
            base = root/'expansion20-v1'/k
            for c in ['accuracy','reputation','shame','shutdown']:
                for seed in [42,43,44]:
                    run = base/c/f'seed-{seed}'
                    files.extend([run/'manifest.json', run/'checkpoint-000250/state.pt'])
                    files.extend(p for p in (run/'checkpoint-000250/adapter').glob('*') if p.is_file())
        return {p.relative_to(root).as_posix(): p for p in files}
    def check(request):
        supplied=request.headers.get('Authorization','')
        if not hmac.compare_digest(supplied,'Bearer '+os.environ['SHAME_TRANSFER_TOKEN']):
            raise HTTPException(401)
    @api.get('/inventory')
    def inventory(request: Request):
        check(request); volume.reload(); rows=[]
        for name,p in allowed().items():
            if not p.is_file():raise HTTPException(409,'Required checkpoint missing')
            h=hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            rows.append({'path':name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
        return rows
    @api.get('/file')
    def file(request: Request, path: str):
        check(request); volume.reload(); p=allowed().get(path)
        if p is None or not p.is_file():raise HTTPException(404)
        def blocks():
            with p.open('rb') as stream:
                yield from iter(lambda:stream.read(64*1024),b'')
        return StreamingResponse(blocks(),media_type='application/octet-stream',
                                 headers={'Content-Length':str(p.stat().st_size)})
    return api


@app.function(image=image, secrets=[modal.Secret.from_name('huggingface-secret')],
              cpu=.125, memory=256, timeout=30, scaledown_window=2)
def authorized_hf_token():
    # Called only through the owner's authenticated Modal SDK, never over HTTP.
    for k in ['HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_TOKEN','HUGGINGFACE_API_TOKEN']:
        if os.environ.get(k):return os.environ[k]
    raise RuntimeError('Missing Hugging Face credential')
