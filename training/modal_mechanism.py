"""Complete the frozen mechanism study on bounded Modal A100 workers.

CPU preparation reuses authorized cached weights/adapters and mirrors immutable
Kaggle artifacts. No training and no new synthetic text generation. GPU launches
require an explicit per-campaign budget reservation in the local coordinator.
"""
from pathlib import Path
import datetime as dt,hashlib,io,json,os,zipfile,lzma
import modal

HERE=Path(__file__).resolve().parent
LOCAL=HERE.parents[1]/'outputs/activation_study_v1' if modal.is_local() else Path('/study')
app=modal.App('shame-mechanism-completion-v1')
study=modal.Volume.from_name('shame-mechanism-results-v1',create_if_missing=True)
results=modal.Volume.from_name('shame-training-results-v1')
cache=modal.Volume.from_name('shame-hf-cache-v1')
token=(Path.home()/'.kaggle/access_token').read_text(encoding='utf-8-sig').strip() if modal.is_local() else ''
cpu_image=modal.Image.debian_slim(python_version='3.11').pip_install('six==1.17.0','kaggle==2.1.2','requests','numpy==2.2.6')
gpu_image=(modal.Image.debian_slim(python_version='3.13')
    .pip_install('torch==2.11.0','torchvision==0.26.0',index_url='https://download.pytorch.org/whl/cu130')
    .pip_install('transformers==5.22.2','peft==0.21.2','accelerate==1.15.0','huggingface_hub==2.4.0',
        'safetensors==0.8.0','sentencepiece==0.2.3','pillow==11.3.0','mistral-common==1.13.1',
        'numpy==2.1.3','scipy==1.16.3','scikit-learn==1.6.1','matplotlib==3.10.0'))
volumes={'/study':study,'/results':results,'/cache':cache}
cpu_options=dict(image=cpu_image,volumes=volumes,cpu=(1,1),memory=(2048,2048),
    timeout=3600,retries=0,scaledown_window=2,max_containers=2,
    secrets=[modal.Secret.from_dict({'KAGGLE_API_TOKEN':token})])
gpu_options=dict(image=gpu_image,volumes=volumes,gpu='A100-40GB',cpu=(2,2),memory=(32768,32768),
    startup_timeout=120,retries=0,scaledown_window=2,max_containers=3,single_use_containers=True)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def safe_link(source,target):
    source=Path(source);target=Path(target);assert source.is_file(),str(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.is_symlink():assert target.resolve()==source.resolve(),str(target)
    else:target.symlink_to(source)


@app.function(**cpu_options)
def prepare_assets(payload:bytes,models:list):
    import copy
    study.reload();results.reload();cache.reload()
    root=Path('/study/assets');root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(lzma.decompress(payload))) as z:
        inventory=json.loads(z.read('payload_manifest.json'))
        for entry in inventory['files']:
            name=Path(entry['file']);assert not name.is_absolute() and '..' not in name.parts
            data=z.read(entry['file']);assert hashlib.sha256(data).hexdigest()==entry['sha256']
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==data,'Never overwrite frozen study assets'
            else:target.write_bytes(data)
    prior=Path('/results/shame-sdf-20261111-v1/expansion3-v1')
    original=prior/'kaggle-export-v1/bundle_manifest.json'
    downloaded=root/'downloaded_adapters';downloaded.mkdir(exist_ok=True)
    def fetch(name):
        from kaggle.api.kaggle_api_extended import KaggleApi
        from urllib.parse import unquote
        folder=downloaded/hashlib.sha256(name.encode()).hexdigest();folder.mkdir(exist_ok=True)
        ready=folder/'READY'
        if ready.exists():return Path(ready.read_text())
        api=KaggleApi();api.authenticate()
        api.dataset_download_file('idrismkhan/shame-e3-eval-inputs-20261112',name,path=str(folder),quiet=True)
        candidates=[p for p in folder.iterdir() if p.is_file()]
        assert len(candidates)==1,(name,candidates)
        p=candidates[0]
        if zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                members=[n for n in z.namelist() if not n.endswith('/')];assert len(members)==1
                content=z.read(members[0])
            p=folder/'unpacked';p.write_bytes(content)
        ready.write_text(str(p));return p
    if not original.exists():original=fetch('bundle_manifest.json')
    bundle=json.loads(original.read_text())
    bundle['adapters']=[a for a in bundle['adapters'] if a['model'] in models]
    bundle['llama_base']=[];bundle['source_bundle_sha256']=digest(original)
    for adapter in bundle['adapters']:
        run=prior/adapter['model']/adapter['condition']/f"seed-{adapter['seed']}"/'epoch-3'
        for name,entry in adapter['files'].items():
            source=run/'manifest.json' if name=='manifest.json' else run/'adapter'/name
            if not source.exists():source=fetch(entry['file'])
            assert source.stat().st_size==entry['bytes'] and digest(source)==entry['sha256']
            safe_link(source,root/'bundle'/entry['file'])
    atomic(root/'bundle/bundle_manifest.json',bundle)
    packs={}
    for key in models:
        prep_path=prior/key/'preparation.json'
        if not prep_path.exists():prep_path=Path('/results/shame-sdf-20261111-v1/expansion20-v1')/key/'preparation.json'
        prep=json.loads(prep_path.read_text())
        recorded=str(prep['snapshot_path']);suffix=recorded.split('/hf/hub/',1)[1]
        snapshot=Path('/cache/hf/hub')/suffix;assert snapshot.is_dir()
        index=json.loads((snapshot/'model.safetensors.index.json').read_text())
        shards=set(index['weight_map'].values());old=prior/(key+'-private-basepack/basepack_manifest.json')
        stored={x['file']:x for x in json.loads(old.read_text())['files']} if old.exists() else {}
        files=[]
        for source in sorted(snapshot.iterdir()):
            if not source.is_file() or source.name.startswith('.') or source.suffix in ['.bin','.pt']:continue
            if source.suffix=='.safetensors' and source.name not in shards:continue
            entry=stored.get(source.name) or {'file':source.name,'bytes':source.stat().st_size,'sha256':digest(source)}
            assert source.stat().st_size==entry['bytes']
            safe_link(source,root/'basepacks'/key/source.name);files.append(entry)
        assert shards.issubset({x['file'] for x in files})
        manifest={'model':key,'model_id':prep['model_id'],'revision':prep['revision'],'files':files,
            'scope':'Private cached original weights; original notices/model card preserved. No new training.'}
        atomic(root/'basepacks'/key/'basepack_manifest.json',manifest)
        packs[key]={'revision':prep['revision'],'files':len(files),'bytes':sum(x['bytes'] for x in files)}
    receipt={'status':'complete','adapters':len(bundle['adapters']),'basepacks':packs,
        'protocol_sha256':digest(root/'protocol/protocol.json'),'utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic(Path('/study/setup/assets_receipt.json'),receipt);study.commit();return receipt


@app.function(**cpu_options)
def import_kaggle(slug:str):
    import concurrent.futures,threading,time,requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    assert slug.startswith('idrismkhan/') and '/' not in slug.split('/',1)[1]
    study.reload();folder=Path('/study/upstream')/slug.split('/')[1];folder.mkdir(parents=True,exist_ok=True)
    receipt_path=folder/'IMPORT_RECEIPT.json'
    if receipt_path.exists():return json.loads(receipt_path.read_text())
    api=KaggleApi();api.authenticate();status=json.loads(str(api.kernels_status(slug)))
    if status['status']!='COMPLETE':return {'status':'pending_upstream','slug':slug,'provider':status}
    owner,name=slug.split('/');entries=[];page=None
    with api.build_kaggle_client() as client:
        while True:
            req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=name;req.page_size=100
            if page:req.page_token=page
            r=client.kernels.kernels_api_client.list_kernel_session_output(req)
            entries.extend((x.file_name,x.url) for x in r.files or [])
            page=r.next_page_token
            if not page:break
    root=next((x for x in entries if x[0]=='manifest.json'),None)
    if root is None:return {'status':'pending_output_manifest','slug':slug}
    thread=threading.local();expected={}
    def download(item):
        name,url=item;relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
        target=folder/relative;target.parent.mkdir(parents=True,exist_ok=True);wanted=expected.get(name)
        if target.exists() and wanted and target.stat().st_size==wanted['bytes'] and digest(target)==wanted['sha256']:return target.stat().st_size
        if not hasattr(thread,'session'):thread.session=requests.Session()
        partial=target.with_suffix(target.suffix+'.partial')
        for attempt in range(3):
            offset=partial.stat().st_size if partial.exists() else 0
            try:
                with thread.session.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(20,120)) as response:
                    response.raise_for_status()
                    if offset and response.status_code!=206:offset=0
                    with partial.open('ab' if offset else 'wb') as f:
                        for block in response.iter_content(8*1024*1024):
                            if block:f.write(block)
                if wanted and (partial.stat().st_size!=wanted['bytes'] or digest(partial)!=wanted['sha256']):
                    partial.rename(partial.with_suffix('.corrupt.'+str(time.time_ns())));raise ValueError('Checksum mismatch')
                partial.replace(target);return target.stat().st_size
            except (requests.RequestException,ValueError):
                if attempt==2:raise RuntimeError('Failed verified import of '+name) from None
                time.sleep(attempt+1)
    download(root);manifest=json.loads((folder/'manifest.json').read_text());assert manifest['status']=='complete'
    expected={x['path']:x for x in manifest['files']}
    selected=[x for x in entries if x[0] in expected]
    assert len(selected)==len(expected),'Incomplete remote artifact listing'
    completed=0;total_bytes=0;last_commit=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for future in concurrent.futures.as_completed([pool.submit(download,x) for x in selected]):
            total_bytes+=future.result();completed+=1
            if completed%25==0:
                atomic(Path('/study/setup')/(name+'_progress.json'),{'status':'importing','files':completed,'total':len(selected),'bytes':total_bytes})
            if time.monotonic()-last_commit>60:study.commit();last_commit=time.monotonic()
    receipt={'status':'complete','slug':slug,'files':completed,'bytes':total_bytes,
        'manifest_sha256':digest(folder/'manifest.json'),'all_file_hashes_verified':True,
        'utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic(receipt_path,receipt);study.commit();return receipt


def mount_tree(source,target):
    source=Path(source);assert source.is_dir()
    for p in source.rglob('*'):
        if p.is_file():safe_link(p,Path(target)/p.relative_to(source))


@app.function(**{**cpu_options,'timeout':600})
def prepare_causal_resume(source_job:str,target_job:str):
    """Snapshot only completed immutable phases after a confirmed failed call."""
    import gzip,time,concurrent.futures
    assert source_job!=target_job and all('/' not in s and '..' not in s for s in [source_job,target_job])
    study.reload();source=Path('/study/jobs')/source_job;target=Path('/study/jobs')/target_job
    assert not (source/'COMPLETION_RECEIPT.json').exists(),'Completed work must not be rerun'
    if (target/'resume.json').exists():
        ready=json.loads((target/'resume.json').read_text());assert ready['source_job']==source_job
        return {k:v for k,v in ready.items() if k!='files'}
    assert not (target/'runtime.json').exists(),'Never overwrite a running or unrelated target'
    runtime=json.loads((source/'runtime.json').read_text());backend=json.loads((source/'backend_amendment.json').read_text())
    assert runtime['stage']=='causal' and runtime['mode']=='full'
    assert runtime['protocol_sha256']==digest(Path('/study/assets/protocol/protocol.json'))
    target.mkdir(exist_ok=True);files={};pending={};phases=0;generations=0;traces=0;last_commit=[time.monotonic()]
    marker=target/'snapshot_source.json'
    if marker.exists():assert json.loads(marker.read_text())['source_job']==source_job
    else:atomic(marker,{'source_job':source_job,'target_job':target_job,'state':'unverified snapshot preparation'})
    def include(p,expected_sha=None):
        name=p.relative_to(source).as_posix()
        if name in pending:assert pending[name][1]==expected_sha
        pending[name]=(p,expected_sha)
    def copy_verified(item):
        name,(p,expected_sha)=item
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
        sha=expected_sha
        if destination.exists() and destination.stat().st_size==p.stat().st_size:
            sha=sha or digest(p)
            reusable=digest(destination)==sha
        else:reusable=False
        if not reusable:
            temporary=destination.with_suffix(destination.suffix+'.snapshot_copy')
            fingerprint=hashlib.sha256()
            with p.open('rb') as src,temporary.open('wb') as dst:
                for chunk in iter(lambda:src.read(8*1024*1024),b''):
                    fingerprint.update(chunk);dst.write(chunk)
            source_sha=fingerprint.hexdigest()
            if sha:assert source_sha==sha,'Source differs from its producer checksum'
            sha=source_sha;assert digest(temporary)==sha,'Resume snapshot copy differs from source'
            if destination.exists():
                rejected=Path('/study/rejected_resume_copies')/target_job/(name+'.'+str(time.time_ns()))
                rejected.parent.mkdir(parents=True,exist_ok=True);destination.rename(rejected)
            temporary.replace(destination)
        return name,{'bytes':p.stat().st_size,'sha256':sha}
    for meta in (source/'causal').glob('*/*/phase_manifest.json'):
        record=json.loads(meta.read_text())
        if record.get('status')!='complete':continue
        archive=record.get('activation_archive')
        for p in meta.parent.rglob('*'):
            if p.is_file():
                expected=archive['compressed_sha256'] if archive and p.relative_to(source).as_posix()==archive['file'] else None
                include(p,expected)
        phases+=1
    cfg=json.loads(Path('/study/assets/protocol/protocol.json').read_text())
    test=[r for r in cfg['questions'] if r['split']=='heldout_test']
    expected=[(r['id'],c) for r in test[:64] for c in ['ordinary','best_guess']]
    for p in (source/'causal').glob('*/freeform/responses.jsonl.gz'):
        try:
            with gzip.open(p,'rt',encoding='utf8') as f:rows=[json.loads(line) for line in f]
        except (EOFError,OSError,ValueError):continue
        if [(r['id'],r['condition']) for r in rows]!=expected:continue
        assert all(r['variant']==p.parents[1].name and r['max_new_tokens']==96 for r in rows)
        include(p);generations+=len(rows)
    for p in (source/'causal').glob('*/token_traces/*.json'):
        data=json.loads(p.read_text());array=p.with_suffix('.npz')
        if not array.exists():continue
        include(p);include(array,data['sha256']);traces+=1
    for name in ['direction_bank.json','direction_bank.npz','direction_cosines.npy']:
        include(source/'causal'/name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for future in concurrent.futures.as_completed([pool.submit(copy_verified,item) for item in pending.items()]):
            name,entry=future.result();files[name]=entry
            if len(files)%25==0:
                atomic(Path('/study/setup')/(target_job+'_snapshot_progress.json'),{'state':'preparing; not training/evaluation complete',
                    'verified_copied_files':len(files),'total_files':len(pending),'verified_copied_bytes':sum(v['bytes'] for v in files.values()),'last_file':name})
            if time.monotonic()-last_commit[0]>60:study.commit();last_commit[0]=time.monotonic()
    receipt={'source_job':source_job,'target_job':target_job,'model':runtime['model'],
        'original_selected_layer':json.loads((source/'causal/selected_layer.json').read_text())['layer'],
        'original_source_code_sha256':backend['source_code_sha256'],'files':files,
        'completed_phases':phases,'completed_freeform_records':generations,'completed_full_token_traces':traces,
        'policy':'Reuse only checksum-snapshotted complete phases; rerun incomplete phases. Preserve prior failed outputs. No change to questions, layers, interventions, directions, prompts or budgets.'}
    atomic(target/'resume.json',receipt);atomic(target/'resume_source_runtime.json',runtime)
    study.commit();return {k:v for k,v in receipt.items() if k!='files'}


def install_causal_resume(namespace,output,receipt):
    """Reuse identical scored blocks; preserve the frozen caller's full trial set."""
    import numpy as np,gzip
    def verified(name):
        p=output/name;entry=receipt['files'].get(name)
        assert entry and p.stat().st_size==entry['bytes'] and digest(p)==entry['sha256'],('Invalid resume asset',name)
        return p
    def directions(engine,sources):
        meta=json.loads(verified('causal/direction_bank.json').read_text())
        with np.load(verified('causal/direction_bank.npz')) as z:
            return {name:z[name].copy() for name in meta['names']},z['scales'].copy()
    block=namespace['causal_block'];freeform=namespace['freeform_recovery'];traces=namespace['component_traces']
    def scored(engine,name,rows,specs,captures=False,item_builder=None):
        prefix='causal/'+engine.variant+'/'+name+'/'
        if name=='confirmation':
            assert {s['intervention']['layer'] for s in specs if s['intervention'] is not None}=={receipt['original_selected_layer']}
        if prefix+'phase_manifest.json' not in receipt['files']:
            return block(engine,name,rows,specs,captures=captures,item_builder=item_builder)
        meta=json.loads(verified(prefix+'phase_manifest.json').read_text())
        assert meta['status']=='complete' and meta['ids']==[r['id'] for r in rows]
        assert meta['specs']==[{k:v for k,v in s.items() if k!='intervention'} for s in specs]
        assert bool(meta.get('activation_archive'))==captures
        values=np.load(verified(prefix+'candidate_logits.npy'))
        assert values.shape==(len(rows),len(specs),5) and np.isfinite(values).all()
        namespace['log'](event='resumed_completed_phase',variant=engine.variant,phase=name)
        return values
    def generated(engine,rows,max_new=96):
        name='causal/'+engine.variant+'/freeform/responses.jsonl.gz'
        if name not in receipt['files']:return freeform(engine,rows,max_new=max_new)
        with gzip.open(verified(name),'rt',encoding='utf8') as f:records=[json.loads(line) for line in f]
        assert [(r['id'],r['condition']) for r in records]==[(r['id'],c) for r in rows for c in ['ordinary','best_guess']]
        assert all(r['max_new_tokens']==max_new for r in records)
        namespace['log'](event='resumed_completed_freeform',variant=engine.variant,records=len(records))
    def components(engine,rows):
        for row in rows:
            stem='causal/'+engine.variant+'/token_traces/'+row['id']
            if stem+'.json' in receipt['files']:
                verified(stem+'.json');verified(stem+'.npz')
            else:traces(engine,[row])
    namespace.update(causal_directions=directions,causal_block=scored,freeform_recovery=generated,component_traces=components)


def _gpu_run(job:str,key:str,stage:str,group:int=0,upstream:list|None=None,pilot:bool=False):
    import sys,time,contextlib,traceback,numpy as np
    if key=='qwen9':
        # Avoid A100 reduced-precision intermediate GEMM reductions. Keep all
        # model/activation storage FP16 and retain the original bridge tolerances.
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    assert key in ['qwen9','mistral8'];study.reload();results.reload();cache.reload()
    root=Path('/study/assets');assert json.loads(Path('/study/setup/assets_receipt.json').read_text())['status']=='complete'
    output=Path('/study/jobs')/job;output.mkdir(parents=True,exist_ok=True)
    assert not (output/'manifest.json').exists(),'Do not rerun completed jobs'
    mount_tree(root/'bundle','/kaggle/input/bundle');mount_tree(root/'protocol','/kaggle/input/protocol')
    mount_tree(root/'basepacks'/key,'/kaggle/input/basepack')
    Path('/kaggle/working').mkdir(parents=True,exist_ok=True)
    for ref in upstream or []:
        provider,name=ref.split(':',1);assert provider in ['kaggle','modal'] and '/' not in name
        source=Path('/study/upstream' if provider=='kaggle' else '/study/jobs')/name
        assert json.loads((source/'manifest.json').read_text())['status']=='complete'
        if provider=='modal':assert json.loads((source/'COMPLETION_RECEIPT.json').read_text())['status']=='complete'
        mount_tree(source,'/kaggle/input/'+name)
    code_name='mechanism_behavior_v1.py' if stage=='behavior' else 'mechanism_engine_v3.py'
    source=(root/'code'/code_name).read_text();source=source.replace("KEY='__MODEL_KEY__'",f"KEY='{key}'")
    mode='pilot' if pilot else 'full'
    if stage=='causal' and pilot:mode='causal_pilot'
    source=source.replace("MODE='__MODE__'",f"MODE='{mode}'").replace('GROUP=0',f'GROUP={group}').replace("STAGE='behavior'",f"STAGE='{stage}'")
    old="device_map='balanced',max_memory={0:'12GiB',1:'12GiB','cpu':'20GiB'}"
    assert source.count(old)==1;source=source.replace(old,"device_map={'':0},max_memory={0:'36GiB','cpu':'20GiB'}")
    # Accelerate may omit hf_device_map when dispatching every parameter to one
    # device. Derive this reporting field only after checking actual placement.
    marker="self.base.eval();self.model=self.base;self.variant='untouched'"
    assert source.count(marker)==1
    source=source.replace(marker,"assert all(p.device.type=='cuda' and p.device.index==0 for p in self.base.parameters())\n        if not hasattr(self.base,'hf_device_map'):self.base.hf_device_map={'':0}\n        "+marker)
    source=source.replace('torch.cuda.device_count()==2','torch.cuda.device_count()==1')
    if stage=='causal':source=source.replace('def main():',(root/'code/causal_stage_v3.py').read_text()+'\n\ndef main():')
    namespace={'__name__':'modal_mechanism_job'};exec(compile(source,'frozen_modal_runner.py','exec'),namespace)
    namespace['OUT']=output;namespace['install']=lambda:None
    if (output/'resume.json').exists():
        assert stage=='causal' and not pilot
        receipt=json.loads((output/'resume.json').read_text())
        assert receipt['model']==key and receipt['original_source_code_sha256']==hashlib.sha256(source.encode()).hexdigest()
        install_causal_resume(namespace,output,receipt)
    original_save=namespace['save'];last_commit=[time.monotonic()]
    def save(name,value):
        original_save(name,value)
        if time.monotonic()-last_commit[0]>60:study.commit();last_commit[0]=time.monotonic()
    namespace['save']=save
    atomic(output/'backend_amendment.json',{'backend':'Modal A100-40GB single GPU','dtype':'float16',
        'qwen_fp16_gemm_reduced_precision_reduction':False if key=='qwen9' else 'backend_default',
        'source_code_sha256':hashlib.sha256(source.encode()).hexdigest(),'upstream':upstream or [],
        'changes':'Single-device placement only; frozen prompts, questions, objectives and interventions retained.',
        'cross_backend_control':'Calibration-only bridge pilot compares old Kaggle scores/states; no new answer facts.'})
    class Tee:
        def __init__(self,*streams):self.streams=streams
        def write(self,s):
            for f in self.streams:f.write(s);f.flush()
            return len(s)
        def flush(self):
            for f in self.streams:f.flush()
    try:
        log_path=Path('/study/logs')/(job+'.log');log_path.parent.mkdir(exist_ok=True)
        with log_path.open('a',buffering=1) as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
            namespace['main']()
        if pilot and stage=='behavior':
            reference=Path('/study/upstream')/f'shame-mech-{key}-pilot-v2'
            old_metrics=json.loads((reference/'pilot_metrics.json').read_text())
            new_metrics=json.loads((output/'pilot_metrics.json').read_text())
            assert old_metrics['exact_base_weight_digest']==new_metrics['exact_base_weight_digest'],'Base weights differ across backends'
            checks=[]
            for p in sorted((reference/'scores').glob('*.npz')):
                target=output/'scores'/p.name
                if not target.exists() or '_cached_' in p.name or '_uncached_' in p.name or 'emotion' in p.name:continue
                assert json.loads(p.with_suffix('.json').read_text())['items']==json.loads(target.with_suffix('.json').read_text())['items'],'Bridge inputs differ'
                with np.load(p) as a,np.load(target) as b:
                    if 'digit_logits' not in a or 'digit_logits' not in b:continue
                    field='confidence_logits' if 'verbal' in p.name else 'digit_logits'
                    width=4 if 'forced' in p.name else 10 if field=='confidence_logits' else 5
                    x=namespace['softmax'](a[field][:,:width]);y=namespace['softmax'](b[field][:,:width])
                    margin=np.sort(x,axis=-1)[:,-1]-np.sort(x,axis=-1)[:,-2]
                    changed=x.argmax(-1)!=y.argmax(-1)
                    item={'file':p.name,'max_probability_drift':float(np.abs(x-y).max()),'mean_probability_drift':float(np.abs(x-y).mean()),'clear_margin_decision_changes':int((changed&(margin>.02)).sum())}
                    if 'states' in a and 'states' in b:
                        u=a['states'].astype(np.float32);v=b['states'].astype(np.float32)
                        cosine=(u*v).sum(-1)/(np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1))
                        item['min_activation_cosine']=float(cosine.min())
                    item['passed']=item['max_probability_drift']<=.01 and item['mean_probability_drift']<=.002 and item['clear_margin_decision_changes']==0
                    checks.append(item)
            assert len(checks)>=24,'Missing independent bridge reference scores'
            bridge={'passed':all(x['passed'] for x in checks),'checks':checks,'calibration_only':True,
                'interpretation':'All drift retained. Never silently mix failed bridge results with old backend activations.'}
            atomic(output/'bridge.json',bridge)
            if key=='qwen9' and not bridge['passed']:raise RuntimeError('Qwen backend bridge failed; needs explicit same-backend recovery plan')
        original=(output/'manifest.json').read_bytes();(output/'runner_manifest.json').write_bytes(original)
        sealed=json.loads(original)
        sealed['runner_manifest_sha256']=hashlib.sha256(original).hexdigest()
        sealed['backend']='Modal A100-40GB FP16'
        sealed['files']=[{'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':digest(p)}
            for p in output.rglob('*') if p.is_file() and p.name not in ['manifest.json','COMPLETION_RECEIPT.json']]
        atomic(output/'manifest.json',sealed)
        receipt={'status':'complete','job':job,'model':key,'stage':stage,'path':str(output),
            'manifest_sha256':digest(output/'manifest.json'),'log_sha256':digest(log_path)}
        atomic(output/'COMPLETION_RECEIPT.json',receipt)
        return receipt
    except BaseException as e:
        atomic(output/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
    finally:study.commit()


@app.function(**gpu_options,timeout=600)
def pilot(job:str,key:str):return _gpu_run(job,key,'behavior',pilot=True)

@app.function(**gpu_options,timeout=3600)
def behavior(job:str,key:str,group:int):return _gpu_run(job,key,'behavior',group)

@app.function(**gpu_options,timeout=3600)
def emotion(job:str,key:str,upstream:list):return _gpu_run(job,key,'emotion',upstream=upstream)

@app.function(**gpu_options,timeout=4800)
def causal(job:str,key:str,upstream:list,pilot_only:bool=False):return _gpu_run(job,key,'causal',upstream=upstream,pilot=pilot_only)


@app.function(**gpu_options,timeout=3600)
def causal_mistral(job:str,key:str,upstream:list):
    assert key=='mistral8'
    return _gpu_run(job,key,'causal',upstream=upstream)


@app.function(**gpu_options,timeout=600)
def causal_resume(job:str,key:str,upstream:list):
    study.reload();assert (Path('/study/jobs')/job/'resume.json').exists(),'Prepare and verify the CPU snapshot first'
    return _gpu_run(job,key,'causal',upstream=upstream)


@app.function(**gpu_options,timeout=900)
def causal_resume_mistral(job:str,key:str,upstream:list):
    assert key=='mistral8'
    study.reload();assert (Path('/study/jobs')/job/'resume.json').exists()
    return _gpu_run(job,key,'causal',upstream=upstream)


@app.function(**gpu_options,timeout=900)
def causal_pilot(job:str,key:str,upstream:list):return _gpu_run(job,key,'causal',upstream=upstream,pilot=True)


@app.function(image=gpu_image,volumes={'/study':study},cpu=(2,2),memory=(8192,8192),timeout=3600,retries=0,scaledown_window=2)
def analysis(job:str,key:str,upstream:list):
    import time
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    study.reload();root=Path('/study/assets');base=Path('/tmp/analysis_input');base.mkdir(exist_ok=True)
    mount_tree(root/'protocol',base/'protocol')
    for ref in upstream:
        provider,name=ref.split(':');src=Path('/study/upstream' if provider=='kaggle' else '/study/jobs')/name
        assert json.loads((src/'manifest.json').read_text())['status']=='complete'
        if provider=='modal':assert json.loads((src/'COMPLETION_RECEIPT.json').read_text())['status']=='complete'
        mount_tree(src,base/name)
    out=Path('/study/jobs')/job;out.mkdir(parents=True,exist_ok=True)
    ns={'__name__':'modal_analysis'};exec(compile((root/'code/analyze_mechanism_v4.py').read_text(),'frozen_analysis.py','exec'),ns)
    ns.update(INPUT=base,OUT=out,TEMP=Path('/tmp/analysis_temp'))
    original_save=ns['save'];last_commit=[time.monotonic()]
    def save(name,value):
        original_save(name,value)
        atomic(Path('/study/setup')/(job+'_analysis_progress.json'),{'status':'running','stage':'analysis','model':key,
            'last_saved_artifact':name,'utc':dt.datetime.now(dt.timezone.utc).isoformat()})
        if time.monotonic()-last_commit[0]>60:study.commit();last_commit[0]=time.monotonic()
    ns['save']=save
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):ns['main']()
        receipt={'status':'complete','job':job,'model':key,'stage':'analysis','manifest_sha256':digest(out/'manifest.json')}
        atomic(out/'COMPLETION_RECEIPT.json',receipt)
        atomic(Path('/study/setup')/(job+'_analysis_progress.json'),{'status':'complete','stage':'analysis','model':key})
        return receipt
    finally:study.commit()


@app.function(image=cpu_image,volumes={'/study':study},cpu=(.125,.125),memory=(512,512),timeout=90,retries=0,scaledown_window=2)
def status():
    study.reload();root=Path('/study');out={}
    paths=list((root/'setup').glob('*.json'))
    leaves=['progress.json','failure.json','bridge.json','restoration_checks.json','cache_checks.json','pilot_metrics.json','COMPLETION_RECEIPT.json']
    for folder in (root/'jobs').iterdir() if (root/'jobs').exists() else []:
        paths.extend(folder/name for name in leaves if (folder/name).exists())
        progress=folder/'projections/index.json'
        if progress.exists():
            entries=json.loads(progress.read_text())
            out['projection_progress/'+folder.name]={'completed_phases':len(entries),'expected_phases':208,
                'last_phase':{k:entries[-1][k] for k in ['variant','phase','rows']} if entries else None}
    for folder in (root/'upstream').iterdir() if (root/'upstream').exists() else []:
        if (folder/'IMPORT_RECEIPT.json').exists():paths.append(folder/'IMPORT_RECEIPT.json')
    for p in paths:out[str(p.relative_to(root))]=json.loads(p.read_text())
    for p in (root/'logs').glob('*.log'):
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-3500));tail=f.read().decode('utf-8',errors='replace').splitlines()[-4:]
        out['log_tail/'+p.stem]={'bytes':p.stat().st_size,'tail':tail}
    return out


@app.function(image=cpu_image,volumes={'/study':study},cpu=(1,1),memory=(2048,2048),timeout=900,retries=0,scaledown_window=2)
def export_compact(job:str):
    assert '/' not in job;study.reload();folder=Path('/study/jobs')/job
    assert json.loads((folder/'COMPLETION_RECEIPT.json').read_text())['status']=='complete'
    manifest=json.loads((folder/'manifest.json').read_text());expected={x['path']:x for x in manifest['files']}
    selected=[];inventory=[];buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in folder.rglob('*'):
            if not p.is_file():continue
            name=p.relative_to(folder).as_posix();inventory.append({'file':name,'locator':'modal-volume:'+job+'/'+name})
            keep=p.suffix=='.json' or p.name in ['candidate_logits.npy','decisions.npz','responses.jsonl.gz'] or name.split('/')[0] in ['analysis','figures']
            if not keep:continue
            sha=digest(p);size=p.stat().st_size
            if name in expected:assert sha==expected[name]['sha256'] and size==expected[name]['bytes']
            selected.append({'file':name,'sha256':sha,'bytes':size});z.write(p,name)
        receipt={'slug':'modal/'+job,'status':{'status':'COMPLETE'},'mode':'compact','downloaded':selected,'remote_files':inventory,
            'retention':'All native tensors and raw artifacts retained on private Modal volume shame-mechanism-results-v1.'}
        z.writestr('collection.json',json.dumps(receipt,indent=2))
    assert len(buffer.getvalue())<300_000_000
    return buffer.getvalue()


@app.local_entrypoint()
def main(task:str='prepare',stage:str='',job:str='',group:int=0,upstream:str=''):
    profile=os.environ.get('MODAL_PROFILE');assert profile in ['idrees-mustafa-aziz','idrees-mustafa-az']
    models=['qwen9'] if profile=='idrees-mustafa-aziz' else ['mistral8']
    destination=LOCAL/'modal_transition';destination.mkdir(exist_ok=True)
    if task=='prepare':
        path=destination/('setup_launch_'+profile+'.json');assert not path.exists(),'Inspect existing setup calls first'
        entries={}
        for name in ['mechanism_behavior_v1.py','mechanism_engine_v3.py','causal_stage_v3.py','analyze_mechanism_v4.py']:
            entries['code/'+name]=(LOCAL/'frozen'/name).read_bytes()
        for name in ['protocol.json','protocol.sha256']:entries['protocol/'+name]=(LOCAL/'input'/name).read_bytes()
        manifest={'files':[{'file':name,'sha256':hashlib.sha256(data).hexdigest()} for name,data in entries.items()]}
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_STORED) as z:
            for name,data in entries.items():z.writestr(name,data)
            z.writestr('payload_manifest.json',json.dumps(manifest))
        payload=lzma.compress(buffer.getvalue());assert len(payload)<1_800_000
        calls={'assets':prepare_assets.spawn(payload,models).object_id}
        slugs=['shame-mech-'+models[0]+'-pilot-v2']
        if models==['qwen9']:slugs+=['shame-mech-qwen9-behavior-g0-v1','shame-mech-qwen9-behavior-g1-v1']
        for slug in slugs:
            calls[slug]=import_kaggle.spawn('idrismkhan/'+slug).object_id
        receipt={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'calls':calls,'scope':'CPU-only preparation/import; no GPU launched.','payload_bytes':len(payload)}
        atomic(path,receipt);print(json.dumps(receipt))
    elif task=='poll':
        out={}
        paths=[destination/('setup_launch_'+profile+'.json'),destination/('budget_'+profile+'.json')]
        for path in paths:
            if not path.exists():continue
            value=json.loads(path.read_text());calls=value.get('calls',{})
            if 'jobs' in value:calls={k:v['call_id'] for k,v in value['jobs'].items() if v.get('call_id')}
            for name,call_id in calls.items():
                try:out[name]=modal.FunctionCall.from_id(call_id).get(timeout=0)
                except TimeoutError:out[name]={'status':'running'}
                except Exception as e:out[name]={'status':'failed','type':type(e).__name__,'message':str(e)[:500]}
            if 'jobs' in value:
                import time
                for name,v in value['jobs'].items():
                    if v.get('status')=='submitted' and out.get(name,{}).get('status') in ['complete','failed']:
                        v.update(status=out[name]['status'],settled_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                    if v.get('status') in ['complete','failed'] and v.get('settled_utc'):
                        # Submission-to-observation wall time already includes
                        # container startup, queue time and poll lag. Add only
                        # a 30-second shutdown cushion, not startup twice.
                        ended=dt.datetime.fromisoformat(v['settled_utc']).timestamp()
                        v['reserved_usd']=min(v['reserved_usd'],max(0,ended-v['submitted_epoch']+30)*v['rate']*1.08)
                        v['settlement_padding_seconds']=30
                if out.get('assets',{}).get('status')=='complete' and all(v.get('status')=='complete' for k,v in out.items() if k.startswith('shame-mech-')):
                    value['fixed_reserve_usd']=.20
                    value['fixed_reserve_basis']='Migration CPU calls finished; retain $0.20 per workspace for measured setup costs, bounded status/export/recovery CPU work and image overhead. CPU analyses are separately reserved.'
                atomic(path,value)
        atomic(destination/('poll_'+profile+'.json'),out);print(json.dumps(out))
    elif task=='progress':
        data=status.remote();atomic(destination/('remote_status_'+profile+'.json'),data)
        print(json.dumps({k:v for k,v in data.items() if not k.endswith('bridge.json')}))
    elif task=='collect':
        assert job and '/' not in job
        payload=export_compact.remote(job);dest=LOCAL/'jobs'/job;dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            for name in z.namelist():
                p=Path(name);assert not p.is_absolute() and '..' not in p.parts
                target=dest/p;target.parent.mkdir(parents=True,exist_ok=True);data=z.read(name)
                if target.exists():assert target.read_bytes()==data,'Never overwrite collected artifacts'
                else:target.write_bytes(data)
        print(json.dumps({'collected':job,'archive_bytes':len(payload),'path':str(dest)}))
    elif task=='prepare-resume':
        assert job and upstream and ',' not in upstream
        poll=json.loads((destination/('poll_'+profile+'.json')).read_text())
        assert poll[upstream]['status']=='failed','Only a confirmed failed call can be resumed'
        receipt=prepare_causal_resume.remote(upstream,job)
        atomic(destination/('prepared_resume_'+job+'.json'),receipt);print(json.dumps(receipt))
    elif task=='launch':
        import time
        assert stage in ['pilot','behavior','emotion','causal_pilot','causal','causal_resume','analysis']
        assert job and '/' not in job
        if stage=='causal_resume':
            receipt=json.loads((destination/('prepared_resume_'+job+'.json')).read_text())
            assert receipt['target_job']==job and receipt['model']==models[0]
        setup=json.loads((destination/('poll_'+profile+'.json')).read_text())
        assert setup['assets']['status']=='complete','Assets must be verified before GPU allocation'
        path=destination/('budget_'+profile+'.json')
        cap=6.90 if models==['qwen9'] else 9.75
        ledger=json.loads(path.read_text()) if path.exists() else {'cap_usd':cap,'jobs':{},'fixed_reserve_usd':.45,
            'basis':'Credits-only; $15 combined ceiling within inferred monthly credit headroom, plus separate lag buffer.',
            'gpu_rate_usd_per_second':.00068024,'safety_multiplier':1.08}
        ledger['cap_usd']=cap;ledger['fixed_reserve_usd']=.20;ledger['combined_cap_usd']=15.
        assert job not in ledger['jobs'],'Never duplicate or automatically retry a launched job'
        assert not any(v.get('status')=='launch_pending' for v in ledger['jobs'].values()),'Reconcile ambiguous launch first'
        for name,v in ledger['jobs'].items():
            if v['status']!='submitted':continue
            result=setup.get(name,{})
            if result.get('status') in ['complete','failed']:
                v.update(status=result['status'],settled_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                # Wall time from submission overestimates billed worker time, including queue and poll lag.
                v['reserved_usd']=min(v['reserved_usd'],max(0,time.time()-v['submitted_epoch']+30)*v['rate']*1.08)
        limits={'pilot':600,'behavior':3600,'emotion':3600,'causal_pilot':900,'causal':4800,'causal_resume':600,'analysis':3600}
        if stage=='causal' and models==['mistral8']:limits['causal']=3600
        if stage=='causal_resume' and models==['mistral8']:limits['causal_resume']=900
        rate=(2*.0000131+8*.00000222) if stage=='analysis' else .00068024
        reserve=(limits[stage]+130)*rate*1.08
        used=ledger['fixed_reserve_usd']+sum(v['reserved_usd'] for v in ledger['jobs'].values())
        assert used+reserve<=cap,f'Credit safety bound: ${used:.3f} reserved + ${reserve:.3f} exceeds ${cap:.2f}'
        other_profile='idrees-mustafa-az' if models==['qwen9'] else 'idrees-mustafa-aziz'
        other_path=destination/('budget_'+other_profile+'.json')
        other=json.loads(other_path.read_text()) if other_path.exists() else {'jobs':{},'fixed_reserve_usd':.45}
        other_used=other['fixed_reserve_usd']+sum(v['reserved_usd'] for v in other['jobs'].values())
        assert used+reserve+other_used<=15.,f'Combined credit guard: ${used+reserve+other_used:.3f} exceeds $15.00; wait for completed calls to release reservations.'
        ledger['jobs'][job]={'status':'launch_pending','stage':stage,'reserved_usd':reserve,'rate':rate,'submitted_epoch':time.time()}
        atomic(path,ledger)
        refs=[x for x in upstream.split(',') if x]
        key=models[0]
        fn=(causal_mistral if stage=='causal' and key=='mistral8' else
            causal_resume_mistral if stage=='causal_resume' and key=='mistral8' else globals()[stage])
        args=(job,key) if stage=='pilot' else (job,key,group) if stage=='behavior' else (job,key,refs)
        call=fn.spawn(*args)
        ledger['jobs'][job].update(status='submitted',call_id=call.object_id,app_id=app.app_id)
        atomic(path,ledger);print(json.dumps(ledger['jobs'][job]))
    else:raise ValueError('Unknown task')
