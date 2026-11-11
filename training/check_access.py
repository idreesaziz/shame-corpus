"""CPU-only credential/model-access check; never logs secret values."""
import json
from pathlib import Path
import modal

app = modal.App('shame-training-access-check')
image = modal.Image.debian_slim(python_version='3.11').pip_install('huggingface_hub==0.36.2')


@app.function(image=image, secrets=[modal.Secret.from_name('huggingface-secret')], cpu=0.25, memory=512, timeout=180)
def check_access():
    import os
    from huggingface_hub import HfApi, hf_hub_download
    keys = ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN', 'HUGGINGFACE_API_TOKEN']
    token = next((os.environ[k] for k in keys if os.environ.get(k)), None)
    api = HfApi(token=token)
    results = {'credential_variable_names_present': [k for k in keys if os.environ.get(k)], 'models': {}}
    try:
        results['authentication'] = {'valid': True, 'account': api.whoami().get('name')}
    except Exception as exc:
        results['authentication'] = {'valid': False, 'error_type': type(exc).__name__}
    for name in ['Qwen/Qwen3-1.7B', 'google/gemma-3-4b-it']:
        try:
            info = api.model_info(name)
            path = hf_hub_download(name, 'config.json', revision=info.sha, token=token)
            config = json.load(open(path))
            results['models'][name] = {'access': 'ok', 'revision': info.sha, 'architecture': config.get('architectures'), 'gated': info.gated}
        except Exception as exc:
            results['models'][name] = {'access': 'blocked', 'error_type': type(exc).__name__, 'message': str(exc).split('Request ID:')[0][:220]}
    return results


@app.local_entrypoint()
def main():
    result = check_access.remote()
    dest = Path('outputs/modal_training_20261111')
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'access_check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, indent=2))
