"""Build/launch a private, pinned Kaggle mechanism-study job; keep build receipts."""
import argparse,datetime as dt,hashlib,json,os,subprocess
from push_checked import push_checked
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[2]/'outputs/activation_study_v1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['llama8','qwen9','mistral8'],required=True)
    p.add_argument('--mode',choices=['pilot','cache_pilot','causal_pilot','full'],default='pilot')
    p.add_argument('--stage',choices=['behavior','emotion','causal'],default='behavior')
    p.add_argument('--group',type=int,choices=[0,1],default=0);p.add_argument('--slug')
    p.add_argument('--base-source',help='Private CPU basepack notebook output, replacing the optional dataset cache')
    p.add_argument('--sources',nargs='*',default=[]);p.add_argument('--launch',action='store_true');args=p.parse_args()
    slug=args.slug or f'shame-mech-{args.model}-{args.mode}-{args.stage}-g{args.group}'
    assert 6<=len(slug)<=50
    folder=ROOT/slug;folder.mkdir(exist_ok=True)
    source_path=ROOT/'mechanism_runner.py'
    if args.mode=='full' and args.stage=='behavior':
        frozen=OUT/'frozen/mechanism_behavior_v1.py'
        assert frozen.exists(),'Freeze the validated behavior source before a full launch.'
        source_path=frozen
    elif args.stage in ['emotion','causal']:
        frozen=OUT/'frozen/mechanism_engine_v3.py'
        assert frozen.exists(),'Freeze mechanism engine before emotion/causal launches.'
        source_path=frozen
    source=source_path.read_text(encoding='utf8')
    source=source.replace("KEY='__MODEL_KEY__'",f"KEY='{args.model}'").replace("MODE='__MODE__'",f"MODE='{args.mode}'")
    source=source.replace('GROUP=0',f'GROUP={args.group}').replace("STAGE='behavior'",f"STAGE='{args.stage}'")
    if args.stage=='causal':
        causal_source=OUT/'frozen/causal_stage_v3.py'
        assert causal_source.exists(),'Freeze validated causal source before launches.'
        source=source.replace('def main():', causal_source.read_text(encoding='utf8')+'\n\ndef main():')
    compile(source,'run.py','exec');(folder/'run.py').write_text(source,encoding='utf8')
    meta={'id':'idrismkhan/'+slug,'title':slug,'code_file':'run.py','language':'python','kernel_type':'script',
        'is_private':True,'enable_gpu':True,'enable_internet':True,
        'dataset_sources':['idrismkhan/shame-e3-eval-inputs-20261112','idrismkhan/shame-mechanism-inputs-20261113'],
        'competition_sources':[],'kernel_sources':args.sources}
    if args.model!='llama8':
        if args.base_source:
            meta['kernel_sources'].append(args.base_source)
        else:
            meta['dataset_sources'].append(f'idrismkhan/shame-{args.model}-base-20261113')
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    receipt={'utc':stamp,'slug':meta['id'],'model':args.model,'mode':args.mode,'stage':args.stage,'group':args.group,
        'code_sha256':hashlib.sha256((folder/'run.py').read_bytes()).hexdigest(),
        'protocol_sha256':(OUT/'input/protocol.sha256').read_text().strip(),'private':True,
        'kernel_sources':meta['kernel_sources'],'launch_requested':args.launch}
    build=OUT/'builds'/slug;build.mkdir(parents=True,exist_ok=True)
    (build/(stamp+'.py')).write_bytes((folder/'run.py').read_bytes())
    (build/(stamp+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    if args.launch:
        os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle/access_token').read_text(encoding='utf-8-sig').strip()
        push_checked(['kaggle','kernels','push','-p',str(folder),'--accelerator','NvidiaTeslaT4',
            '-t','1800' if args.mode!='full' else '43200'],meta['id'])

if __name__=='__main__':main()
