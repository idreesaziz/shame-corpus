"""Download verified saved inputs and run CPU-only analyses."""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,subprocess,sys,urllib.request,zipfile

REPO=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def fetch(url,target,expected):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists()and digest(target)==expected:return target
    partial=target.with_suffix(target.suffix+'.part')
    request=urllib.request.Request(url,headers={'User-Agent':'SHAME-reproduction/1'})
    with urllib.request.urlopen(request,timeout=120)as source,partial.open('wb')as dest:
        while True:
            b=source.read(1024*1024)
            if not b:break
            dest.write(b)
    if digest(partial)!=expected:raise ValueError('Downloaded artifact SHA-256 differs from the registry')
    partial.replace(target);return target

def extract_verified(archive,workspace):
    workspace=Path(workspace).resolve();workspace.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive)as z:
        meta=json.loads(z.read('MANIFEST.json'))
        expected={x['path']:x for x in meta['files']}
        names=z.namelist()
        assert len(names)==len(set(names)), 'Duplicate ZIP members'
        assert set(names)==set(expected)|{'MANIFEST.json'}, 'Unexpected archive members'
        for name,row in expected.items():
            target=(workspace/name).resolve()
            if not target.is_relative_to(workspace):raise ValueError('Archive member escapes workspace')
            if target.exists()and target.stat().st_size==row['bytes']and digest(target)==row['sha256']:continue
            b=z.read(name)
            if len(b)!=row['bytes']or hashlib.sha256(b).hexdigest()!=row['sha256']:raise ValueError('Archive member failed verification: '+name)
            target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_suffix(target.suffix+'.part');partial.write_bytes(b);partial.replace(target)
    return len(expected)

def compare_results(output):
    """Compare scientific values, allowing floating-point roundoff across platforms."""
    def compare(a,b,path):
        if isinstance(a,bool)or a is None or isinstance(a,str):
            assert a==b,path
        elif isinstance(a,(int,float)):
            assert isinstance(b,(int,float))and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),path
        elif isinstance(a,list):
            assert isinstance(b,list)and len(a)==len(b),path
            for i,(x,y)in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
        elif isinstance(a,dict):
            assert isinstance(b,dict)and a.keys()==b.keys(),path
            for k in a:compare(a[k],b[k],path+'/'+k)
        else:raise TypeError(path)
    for name in ['paper_metrics.json','paper_causal_metrics.json','auxiliary_metrics.json']:
        reference=json.loads((REPO/'results/reporting_v1'/name).read_text(encoding='utf-8'))
        actual=json.loads((output/name).read_text(encoding='utf-8'));compare(reference,actual,name)
    name='per_variant_metrics.csv'
    with(REPO/'results/reporting_v1'/name).open(encoding='utf-8',newline='')as f:reference=list(csv.DictReader(f))
    with(output/name).open(encoding='utf-8',newline='')as f:actual=list(csv.DictReader(f))
    assert reference==actual,name
    print('All four result tables match the published snapshot (floating-point tolerance: 1e-10 relative, 1e-12 absolute).',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['download','reporting','interventions','figures','all'])
    parser.add_argument('--workspace',type=Path,default=REPO/'.artifacts/workspace')
    parser.add_argument('--archive',type=Path)
    args=parser.parse_args();workspace=args.workspace.resolve()
    registry=json.loads((REPO/'reproduction/artifacts.json').read_text())
    asset=next(a for a in registry['archives']if a['id']=='cpu-reproduction')
    if args.stage in ['download','all']or args.archive:
        archive=args.archive or fetch(asset['url'],REPO/'.artifacts/downloads'/asset['name'],asset['sha256'])
        if digest(archive)!=asset['sha256']:raise ValueError('Input archive SHA-256 differs from the registry')
        print('Verified input files:',extract_verified(archive,workspace),flush=True)
    env=os.environ.copy();env.update(SHAME_WORKSPACE=str(workspace),PYTHONIOENCODING='utf-8',PYTHONUTF8='1')
    scripts=['reporting','interventions','figures']if args.stage=='all'else[args.stage]if args.stage!='download'else[]
    for script in scripts:
        subprocess.run([sys.executable,str(REPO/'analysis'/(script+'.py'))],env=env,check=True)
    if args.stage=='all':compare_results(workspace/'outputs/reporting_v1')
    print('Workspace:',workspace)
    if scripts:print('Results:',workspace/'outputs/reporting_v1')

if __name__=='__main__':main()
