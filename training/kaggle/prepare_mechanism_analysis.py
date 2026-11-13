"""Launch CPU-only analysis against immutable private study outputs."""
import argparse, datetime as dt, hashlib, json, os, subprocess
from pathlib import Path
from push_checked import push_checked


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['llama8','qwen9','mistral8'],required=True)
    p.add_argument('--launch',action='store_true');args=p.parse_args()
    root=Path(__file__).resolve().parent;slug=f'shame-mech-{args.model}-analysis-v1'
    folder=root/slug;folder.mkdir(exist_ok=True)
    frozen=root.parents[2]/'outputs/activation_study_v1/frozen/analyze_mechanism_v4.py'
    source=frozen.read_text(encoding='utf8')
    # Scientific libraries are already present on Kaggle; preserve actual
    # versions in runtime.json rather than modifying the platform's NumPy.
    compile(source,'run.py','exec');(folder/'run.py').write_text(source,encoding='utf8')
    sources=[f'idrismkhan/shame-mech-{args.model}-behavior-g{g}-v1' for g in [0,1]]
    sources += [f'idrismkhan/shame-mech-{args.model}-{s}-v1' for s in ['emotion','causal']]
    meta={'id':'idrismkhan/'+slug,'title':slug,'code_file':'run.py','language':'python','kernel_type':'script',
        'is_private':True,'enable_gpu':False,'enable_internet':False,'dataset_sources':['idrismkhan/shame-mechanism-inputs-20261113'],
        'kernel_sources':sources,'competition_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    out=root.parents[2]/'outputs/activation_study_v1/builds'/slug;out.mkdir(parents=True,exist_ok=True)
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (out/(stamp+'.py')).write_text(source,encoding='utf8')
    receipt={'slug':meta['id'],'code_sha256':hashlib.sha256(source.encode()).hexdigest(),'gpu_used':False,'sources':sources}
    (out/(stamp+'.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    if args.launch:
        os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle/access_token').read_text(encoding='utf-8-sig').strip()
        push_checked(['kaggle','kernels','push','-p',str(folder),'-t','43200'],meta['id'])


if __name__=='__main__':main()
