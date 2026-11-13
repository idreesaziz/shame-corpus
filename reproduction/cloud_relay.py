"""Maintainer transfer: verified adapter objects to a draft GitHub release.

Inputs are short-lived read-only URLs, never provider account credentials.
Only registry-listed LoRA files with matching hashes can be uploaded.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,hashlib,json,os,subprocess,time,urllib.request,urllib.parse,zipfile

HERE=Path(__file__).resolve().parents[1]
REPO=os.environ.get('GITHUB_REPOSITORY','idreesaziz/shame-corpus')

def gh(args):
    p=subprocess.run(['gh',*args],capture_output=True,text=True,encoding='utf-8')
    if p.returncode:raise RuntimeError('GitHub operation failed: '+p.stderr[:250])
    return p.stdout

def api(path,method='GET'):
    s=gh(['api','--method',method,path]);return json.loads(s)if s.strip()else None

def digest(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    registry=json.loads((HERE/'reproduction/artifacts.json').read_text())
    tag=registry['release_tag'];release=next(r for r in api(f'repos/{REPO}/releases?per_page=100')if r['tag_name']==tag)
    assert release['draft'],'Relay targets the staged draft release only'
    rid=release['id'];run=os.environ['GITHUB_RUN_ID'];ready=f'_relay_ready_{run}.json';go=f'_relay_go_{run}.json'
    root=Path(os.environ['RUNNER_TEMP'])/'shame-verified-relay';root.mkdir(exist_ok=True)
    urls={}
    for model in ['QWEN9','LLAMA8','MISTRAL8']:
        urls.update(json.loads(os.environ.get('SHAME_RELAY_'+model,'{}')or'{}'))
    # Mask each constituent URL, as well as the secret JSON strings masked by Actions.
    for url in urls.values():print('::add-mask::'+url,flush=True)
    def assets():return {a['name']:a for a in api(f'repos/{REPO}/releases/{rid}/assets?per_page=100')}
    existing=assets();todo=[]
    for item in registry['adapters']:
        a=existing.get(item['asset'])
        if a and a['state']=='uploaded':
            assert a['size']==item['bytes']and a['digest']=='sha256:'+item['sha256'];continue
        assert item['asset']in urls,'Missing read-only object URL for '+item['asset']
        todo.append(item)
    def download(item):
        name=item['asset'];url=urls[name]
        parsed=urllib.parse.urlsplit(url)
        assert parsed.scheme=='https'and parsed.hostname=='storage.googleapis.com'and parsed.username is None,'Expected a signed storage object URL'
        raw=root/(name+'.transfer');target=root/name;cap=int(item['bytes']*1.1)+1024*1024
        count=0
        with urllib.request.urlopen(url,timeout=120)as src,raw.open('wb')as dst:
            size=src.headers.get('Content-Length')
            if size:assert int(size)<=cap,'Object exceeds single-adapter size limit'
            while True:
                b=src.read(1024*1024)
                if not b:break
                count+=len(b);assert count<=cap,'Transfer exceeds single-adapter limit';dst.write(b)
        if zipfile.is_zipfile(raw):
            with zipfile.ZipFile(raw)as z:
                candidates=[i for i in z.infolist()if not i.is_dir()]
                assert len(candidates)==1 and candidates[0].file_size==item['bytes']
                with z.open(candidates[0])as src,target.open('wb')as dst:
                    while True:
                        b=src.read(1024*1024)
                        if not b:break
                        dst.write(b)
            raw.unlink()
        else:raw.replace(target)
        assert target.stat().st_size==item['bytes']and digest(target)==item['sha256'],name
        return name
    with ThreadPoolExecutor(max_workers=6)as pool:
        futures={pool.submit(download,x):x['asset']for x in todo}
        for f in as_completed(futures):
            try:print('CLOUD_DOWNLOAD_VERIFIED',f.result(),flush=True)
            except Exception as e:raise RuntimeError('Object verification failed for '+futures[f]+' ('+type(e).__name__+')')from None
    receipt=root/ready;receipt.write_text(json.dumps({'run_id':run,'verified_downloads':len(todo),'status':'ready-for-handoff'})+'\n')
    gh(['release','upload',tag,str(receipt),'--repo',REPO]);print('CLOUD_RELAY_READY',run,flush=True)
    # The maintainer stops the local uploader before sending this release-scoped permit.
    for _ in range(120):
        if go in assets():break
        time.sleep(5)
    else:raise RuntimeError('Upload handoff permit did not arrive')
    existing=assets()
    for name,a in list(existing.items()):
        if a['state']=='starter'and name in {x['asset']for x in todo}:
            api(f'repos/{REPO}/releases/assets/{a["id"]}','DELETE')
    def upload(item):
        current=assets().get(item['asset'])
        if current and current['state']=='uploaded':
            assert current['digest']=='sha256:'+item['sha256'];return item['asset']
        gh(['release','upload',tag,str(root/item['asset']),'--repo',REPO]);return item['asset']
    with ThreadPoolExecutor(max_workers=4)as pool:
        for name in pool.map(upload,todo):print('CLOUD_UPLOAD_COMPLETE',name,flush=True)
    current=assets()
    for item in registry['adapters']:
        a=current[item['asset']];assert a['state']=='uploaded'and a['size']==item['bytes']and a['digest']=='sha256:'+item['sha256']
    for name in [ready,go]:
        a=current.get(name)
        if a:api(f'repos/{REPO}/releases/assets/{a["id"]}','DELETE')
    print('ALL_36_ADAPTERS_VERIFIED',flush=True)

if __name__=='__main__':main()
