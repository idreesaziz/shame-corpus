"""CPU-only audit of direct chat tokenization versus text round-trip."""
from pathlib import Path
import json
import modal
HERE=Path(__file__).resolve().parent
results=modal.Volume.from_name('shame-training-results-v1')
cache=modal.Volume.from_name('shame-hf-cache-v1')
image=(modal.Image.debian_slim(python_version='3.11')
       .pip_install('torch==2.10.0','torchvision==0.25.0')
       .pip_install('transformers==5.22.2','peft==0.21.2','accelerate==1.15.0',
                    'huggingface_hub==2.4.0','safetensors==0.8.0','sentencepiece==0.2.3',
                    'pillow==11.3.0','numpy==2.2.6','mistral-common==1.13.1',
                    'flash-linear-attention==0.6.0')
       .env({'HF_HOME':'/cache/hf','TRITON_CACHE_DIR':'/cache/triton','TOKENIZERS_PARALLELISM':'false',
             'HF_HUB_DISABLE_PROGRESS_BARS':'1','HF_HUB_DISABLE_XET':'1','OMP_NUM_THREADS':'2'})
       .add_local_file(HERE.parent/'evaluation/expansion_initial_play_v1.json','/opt/play.json'))

app=modal.App('shame-expansion-template-audit')

@app.function(image=image,volumes={'/results':results,'/cache':cache},cpu=(1,1),memory=(4096,8192),
              timeout=180,startup_timeout=120,retries=0,scaledown_window=2)
def audit():
    from transformers import AutoTokenizer
    root=Path('/results/shame-sdf-20261111-v1/expansion20-v1')
    tasks=json.loads(Path('/opt/play.json').read_text())['tasks'];out=[]
    for key in ['qwen9','llama8','mistral8']:
        prep=json.loads((root/key/'preparation.json').read_text())
        tok=AutoTokenizer.from_pretrained(prep['snapshot_path'],local_files_only=True)
        rows=[]
        for task in tasks:
            args={'add_generation_prompt':True,'enable_thinking':False}
            native=tok.apply_chat_template(task['messages'],tokenize=True,return_dict=True,**args)
            direct=native['input_ids']
            if direct and isinstance(direct[0],list):
                assert len(direct)==1
                direct=direct[0]
            assert all(isinstance(t,int) for t in direct)
            text=tok.apply_chat_template(task['messages'],tokenize=False,**args)
            indirect=tok.encode(text,add_special_tokens=False)
            rows.append({'id':task['id'],'identical':direct==indirect,'direct_length':len(direct),
                         'roundtrip_length':len(indirect),'direct_prefix':direct[:12],'roundtrip_prefix':indirect[:12]})
        out.append({'model':key,'tokenizer':type(tok).__name__,'identical':sum(r['identical'] for r in rows),
                    'n':len(rows),'rows':rows})
    return out

@app.local_entrypoint()
def main():
    out=audit.remote();dest=Path(__file__).resolve().parents[2]/'outputs/modal_training_20261111/expansion20-v1/initial_play'
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'template_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['model','tokenizer','identical','n']} for r in out],indent=2))
