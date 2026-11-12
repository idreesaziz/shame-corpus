"""Observe the bounded campaign and back up completed artifacts; never launches GPUs."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
import subprocess
import modal

OUT=Path(__file__).resolve().parents[2]/'outputs/modal_training_20261111/expansion3-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');temp.replace(path)


def main():
    assert os.environ.get('MODAL_PROFILE')=='idrees-mustafa-aziz','Wrong destination workspace'
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    get=modal.Function.from_name('shame-expansion3-monitor','status')
    inventory=modal.Function.from_name('shame-expansion3-monitor','artifacts')
    reader=modal.Function.from_name('shame-expansion3-monitor','read_chunk')
    index_path=OUT/'backup_index.json';index=json.loads(index_path.read_text()) if index_path.exists() else {}
    by_hash={v['sha256']:OUT/'artifacts'/k.removeprefix('expansion3-v1/') for k,v in index.items()}
    def download(entry):
        relative=entry['path'];target=(OUT/'artifacts'/relative.removeprefix('expansion3-v1/')).resolve()
        assert target.is_relative_to((OUT/'artifacts').resolve())
        if target.exists() and target.stat().st_size==entry['bytes'] and sha(target)==entry['sha256']:return
        target.parent.mkdir(parents=True,exist_ok=True)
        same=by_hash.get(entry['sha256'])
        if same and same.exists() and same!=target:
            try:os.link(same,target);return
            except OSError:pass
        parts=OUT/'transfer_parts'/entry['sha256'];parts.mkdir(parents=True,exist_ok=True)
        offsets=list(range(0,entry['bytes'],800000))
        def chunk(offset):
            part=parts/f'{offset:012d}.part';size=min(800000,entry['bytes']-offset)
            if part.exists() and part.stat().st_size==size:return
            value=reader.remote(relative,offset,size);assert len(value)==size;part.write_bytes(value)
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(chunk,offsets))
        temp=target.with_suffix(target.suffix+'.download')
        with temp.open('wb') as stream:
            for offset in offsets:stream.write((parts/f'{offset:012d}.part').read_bytes())
        assert sha(temp)==entry['sha256'],relative
        temp.replace(target);by_hash[entry['sha256']]=target
        # Delete only these verified transfer chunks, never model/checkpoint directories.
        for offset in offsets:(parts/f'{offset:012d}.part').unlink()
        parts.rmdir()
    previous=None
    while True:
        status=get.remote();save(OUT/'status.json',status)
        if status.get('import',{}).get('status')=='complete' and not (OUT/'bridge_shutdown.json').exists():
            source_env={**os.environ,'MODAL_PROFILE':'idrees-mustafa-az','PYTHONUTF8':'1'}
            stopped=subprocess.run(['modal','app','stop','--yes','shame-continuation-transfer'],
                                   env=source_env,capture_output=True,text=True,timeout=45)
            if stopped.returncode==0:
                save(OUT/'bridge_shutdown.json',{'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                     'reason':'All 36 optimizer/adapter checkpoints imported with verified hashes.',
                     'source_workspace':'idrees-mustafa-az','app':'shame-continuation-transfer'})
        budget=status['budget'] or {};current=budget.get('current') or {}
        summary={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':budget.get('status'),
                 'completed':len(budget.get('completed',[])),'gpu_upper_bound_usd':budget.get('gpu_upper_bound_spent_usd'),
                 'current':{k:current.get(k) for k in ['model','condition','seed']},
                 'models':{k:v['status'] for k,v in status['models'].items()},
                 'progress':[(r['model_key'],r['condition'],r['seed'],r['status'],r.get('last_step',{}).get('step')) for r in status['runs']]}
        marker=json.dumps({k:v for k,v in summary.items() if k!='utc'},sort_keys=True)
        if marker!=previous:print(json.dumps(summary),flush=True);previous=marker
        for entry in inventory.remote():
            old=index.get(entry['path'])
            if old==entry and (OUT/'artifacts'/entry['path'].removeprefix('expansion3-v1/')).exists():continue
            download(entry);index[entry['path']]=entry;save(index_path,index)
        if args.once or budget.get('status') in ['budget_limit_reached','finished_available_models','failed']:
            save(OUT/'monitor_checkpoint.json',{'utc':summary['utc'],'status':budget.get('status'),
                  'backed_up_files':len(index),'completed_runs':summary['completed']})
            return
        time.sleep(30)


if __name__=='__main__':main()
