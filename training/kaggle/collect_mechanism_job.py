"""Collect compact artifacts with full API pagination; leave giant arrays remote."""
import argparse,datetime as dt,json,os,re,hashlib,concurrent.futures,threading,time
from pathlib import Path
import requests

def resolved_path(path):
    """Resolve junctions, then compare equivalent Windows path notations."""
    value=str(Path(path).resolve())
    if os.name=='nt':
        if value.startswith('\\\\?\\UNC\\'):value='\\\\'+value[8:]
        elif value.startswith('\\\\?\\'):value=value[4:]
    return Path(value)

def main():
    p=argparse.ArgumentParser();p.add_argument('--slug',required=True);p.add_argument('--dest',required=True)
    p.add_argument('--mode',choices=['metadata','compact','pilot'],default='metadata');args=p.parse_args()
    os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle/access_token').read_text(encoding='utf-8-sig').strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api=KaggleApi();api.authenticate();slug=args.slug if '/' in args.slug else 'idrismkhan/'+args.slug
    owner,name=slug.split('/');assert owner=='idrismkhan'
    dest=resolved_path(args.dest);dest.mkdir(parents=True,exist_ok=True)
    status=json.loads(str(api.kernels_status(slug)))
    index=[];downloaded=[];token=None;logs=[];selected=[]
    with api.build_kaggle_client() as client:
        while True:
            req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=name;req.page_size=100
            if token:req.page_token=token
            response=client.kernels.kernels_api_client.list_kernel_session_output(req)
            if response.log:logs.append(response.log)
            for entry in response.files or []:
                filename=entry.file_name;relative=Path(filename)
                assert not relative.is_absolute() and not relative.drive and '..' not in relative.parts,filename
                target=resolved_path(dest/relative);assert target.is_relative_to(dest),filename
                index.append({'file':filename,'locator':slug+':'+filename})
                leaf=relative.name
                keep=leaf in ['failure.json','progress.json','runtime.json','manifest.json','pilot_metrics.json',
                    'cache_checks.json','restoration_checks.json','run_plan.json','label_schema.json','protocol.json',
                    'basepack_manifest.json','model_config.json','generation_config.json','emotion_vectors.json','validation.json']
                if args.mode=='compact':
                    keep=keep or leaf in ['calibration.json','phase_manifest.json','candidate_logits.npy','decisions.npz','metrics.json','analysis.json','selected_layer.json','direction_bank.json','archive.json']
                    # Reports and figures are compact. Never include source
                    # residual arrays, direction banks or component archives.
                    keep=keep or relative.parts[0] in ['analysis','figures']
                    keep=keep or ('freeform' in relative.parts and leaf=='responses.jsonl.gz')
                if args.mode=='pilot':keep=True
                if keep:selected.append((filename,entry.url))
            token=response.next_page_token
            if not token:break
    # Enumerate first, fetch the producer manifest first, then reuse only files
    # that match its SHA. This safely resumes a timed-out collection without
    # downloading hundreds of already verified compact files again.
    local=threading.local();expected={}
    def file_sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return h.hexdigest()
    def fetch(item):
        filename,url=item;target=resolved_path(dest/filename);assert target.is_relative_to(dest),(filename,str(target),str(dest))
        wanted=expected.get(str(Path(filename)))
        if target.exists() and wanted and target.stat().st_size==wanted['bytes'] and file_sha(target)==wanted['sha256']:
            return {'file':filename,'sha256':wanted['sha256'],'bytes':wanted['bytes'],'reused_verified':True}
        if not hasattr(local,'session'):local.session=requests.Session()
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_suffix(target.suffix+'.download')
        for attempt in range(3):
            h=hashlib.sha256();size=0
            try:
                with local.session.get(url,stream=True,timeout=(15,90)) as response:
                    response.raise_for_status()
                    with temp.open('wb') as f:
                        for data in response.iter_content(1024*1024):
                            if data:f.write(data);h.update(data);size+=len(data)
                break
            except requests.RequestException as exc:
                if attempt==2:raise RuntimeError(f'Artifact download failed: {filename} ({type(exc).__name__})') from None
                time.sleep(attempt+1)
        digest=h.hexdigest()
        if wanted:assert size==wanted['bytes'] and digest==wanted['sha256'],('Producer checksum mismatch',filename)
        temp.replace(target)
        return {'file':filename,'sha256':digest,'bytes':size,'reused_verified':False}
    manifests=[x for x in selected if Path(x[0]).name=='manifest.json']
    assert len(manifests)<=1
    if manifests:
        root=fetch(manifests[0]);downloaded.append(root)
        m=json.loads((dest/root['file']).read_text(encoding='utf8'))
        for entry in m.get('files',[]):
            if 'path' in entry:expected[str(Path(entry['path']))]=entry
            elif 'file' in entry:expected[str(Path('basepack')/entry['file'])]=entry
    pending=[x for x in selected if x not in manifests]
    print(json.dumps({'event':'collecting','slug':slug,'indexed':len(index),'compact_files':len(selected)}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(fetch,item) for item in pending]):
            downloaded.append(future.result())
            if len(downloaded)%25==0:
                print(json.dumps({'event':'collected','slug':slug,'files':len(downloaded),'total':len(selected)}),flush=True)
    if logs:(dest/(name+'.log')).write_text(logs[-1],encoding='utf8')
    receipt={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'slug':slug,'status':status,
        'retention':'All activation artifacts retained in private Kaggle outputs; only whitelisted compact files downloaded.',
        'remote_files':index,'downloaded':downloaded,'mode':args.mode}
    (dest/'collection.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'slug':slug,'status':status,'remote_files':len(index),'downloaded':len(downloaded)}))
    for leaf in ['failure.json','pilot_metrics.json','cache_checks.json','restoration_checks.json']:
        f=dest/leaf
        if f.exists():print(leaf+' '+f.read_text(encoding='utf8')[:6500])

if __name__=='__main__':main()
