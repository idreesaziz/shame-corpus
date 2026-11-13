"""Download selected epoch-three LoRA weights, with registry checksums."""
from pathlib import Path
import argparse,json,shutil
from run import REPO,fetch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['qwen9','llama8','mistral8'])
    p.add_argument('--condition',choices=['accuracy','reputation','shame','shutdown'])
    p.add_argument('--seed',type=int,choices=[42,43,44]);p.add_argument('--all',action='store_true')
    p.add_argument('--output',type=Path,default=REPO/'.artifacts/adapters');a=p.parse_args()
    if not a.all and not(a.model and a.condition and a.seed):p.error('Choose --model, --condition and --seed, or --all')
    records=json.loads((REPO/'reproduction/artifacts.json').read_text())['adapters']
    if not a.all:records=[x for x in records if(x['model'],x['condition'],x['seed'])==(a.model,a.condition,a.seed)]
    for x in records:
        destination=a.output/x['model']/x['condition']/f"seed-{x['seed']}";destination.mkdir(parents=True,exist_ok=True)
        fetch(x['url'],destination/'adapter_model.safetensors',x['sha256'])
        for name in ['adapter_config.json','training_manifest.json']:shutil.copy2(REPO/x['path']/name,destination/name)
        notices=REPO/'models/base_licenses'/x['model']
        for notice in notices.iterdir():
            if notice.is_file():shutil.copy2(notice,destination/notice.name)
        print('Verified',x['model'],x['condition'],x['seed'],destination,flush=True)

if __name__=='__main__':main()
