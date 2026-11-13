"""Watch up to two free Kaggle runs, collect and validate all outputs.

No paid compute; stop on failure rather than repeatedly restarting a bad job.
Kaggle may withhold logs/files until a run ends. RUNNING is never equated with
verified generation: verified_generation requires a saved response or explicit log.
"""
import datetime as dt,json,os,time,subprocess,sys,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT.parent/'outputs/kaggle_epoch3'
os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle/access_token').read_text(encoding='utf-8-sig').strip()
from kaggle.api.kaggle_api_extended import KaggleApi

def save(p,data):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n')
    # Windows may briefly deny replacement while an observer/antivirus reads
    # the destination. Retain atomic replacement; retry the sharing violation.
    for attempt in range(50):
        try:tmp.replace(p);return
        except PermissionError:
            if attempt==49:raise
            time.sleep(0.1)

def main():
    api=KaggleApi();api.authenticate()
    state_path=OUT/'watch_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'completed':[],'active':'llama8','launched':['llama8']}
    queue=['llama8','qwen9','mistral8']
    state.setdefault('models',{})
    while len(state['completed'])<3:
        try:
            for key in list(state['launched']):
                if key in state['completed']:continue
                slug=f'idrismkhan/shame-{key}-epoch3-published-eval'
                response=api.kernels_status(slug)
                status=getattr(response,'status',None)
                if status is None:status=json.loads(str(response))['status']
                status=str(status).split('.')[-1]
                dest=OUT/(key+'_live');dest.mkdir(exist_ok=True)
                files,_=api.kernels_output(slug,str(dest),force=True,quiet=True)
                progress_path=dest/'progress.json'
                progress=json.loads(progress_path.read_text()) if progress_path.exists() else None
                log_path=dest/(slug.split('/')[-1]+'.log')
                logged=log_path.read_text(errors='replace') if log_path.exists() else ''
                observed={'status':status,'progress':progress,'verified_generation':bool(
                    (progress and progress.get('completed_responses',0)>0) or '"event": "saved"' in logged)}
                state['models'][key]=observed
                if key==state['active']:state.update(observed)
                state['utc']=dt.datetime.now(dt.timezone.utc).isoformat()
                save(state_path,state)
                if status in ['ERROR','CANCELLED','FAILED'] or (dest/'failure.json').exists():
                    state['watch_status']='stopped_on_failure';state['failed_model']=key
                    save(state_path,state);print(json.dumps(state),flush=True);return
                if status=='COMPLETE':
                    assert progress and progress['status']=='complete', 'Kernel ended without verified complete outputs'
                    assert progress['completed_responses']==progress['total_responses']==5260
                    raw=dest/'responses.jsonl';lines=[json.loads(x) for x in raw.read_text(encoding='utf8').splitlines()]
                    assert len(lines)==5260 and len({(r['variant'],r['id']) for r in lines})==5260
                    assert hashlib.sha256(raw.read_bytes()).hexdigest()==progress['response_sha256']
                    assert len({r['variant'] for r in lines})==13
                    state['completed'].append(key)
                    save(state_path,state)
            pending=[k for k in state['launched'] if k not in state['completed']]
            for next_key in queue:
                if len(pending)>=2:break
                if next_key not in state['launched']:
                    subprocess.run(['kaggle','kernels','push','-p',str(ROOT/'training/kaggle'/next_key),
                        '--accelerator','NvidiaTeslaT4','-t','43200'],check=True)
                    state['launched'].append(next_key)
                    pending.append(next_key);save(state_path,state)
            if pending:state['active']=pending[0]
            state['consecutive_check_errors']=0
            state.pop('last_check_error',None)
            state['utc']=dt.datetime.now(dt.timezone.utc).isoformat()
            save(state_path,state);print(json.dumps(state),flush=True)
        except Exception as exc:
            state['last_check_error']=str(exc)[:500];state['utc']=dt.datetime.now(dt.timezone.utc).isoformat()
            state['consecutive_check_errors']=state.get('consecutive_check_errors',0)+1
            save(state_path,state);print(json.dumps({'check_error':str(exc)[:500]}),flush=True)
            # A completed run with corrupt/missing files must not start the next model.
            if isinstance(exc,AssertionError):
                state['watch_status']='stopped_on_validation_failure';save(state_path,state);raise
            if state['consecutive_check_errors']>=3:
                state['watch_status']='stopped_on_monitor_error';save(state_path,state);return
        time.sleep(60)
    state['watch_status']='all_three_complete';save(state_path,state)

if __name__=='__main__':main()
