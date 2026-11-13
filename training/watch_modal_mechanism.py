"""Resume dependency-gated, credit-reserved Modal work; never auto-retry GPU failures."""
import datetime as dt,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'outputs/activation_study_v1/modal_transition'
PROFILES={'qwen9':'idrees-mustafa-aziz','mistral8':'idrees-mustafa-az'}

def atomic(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n')
    for attempt in range(50):
        try:t.replace(p);return
        except PermissionError:
            if attempt==49:raise
            time.sleep(.1)

def command(profile,*args,timeout=300):
    env=dict(os.environ,MODAL_PROFILE=profile,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    r=subprocess.run([sys.executable,'training/modal_cli_dns.py','run','--detach','training/modal_mechanism.py',*args],cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    with (OUT/'coordinator_commands.log').open('a',encoding='utf-8') as f:f.write('\n'+str(args)+'\n'+r.stdout+r.stderr)
    return r

def graph(key):
    bridge='qwen9_bridge_v3' if key=='qwen9' else 'mistral8_bridge_v1'
    jobs=[dict(name=bridge,stage='pilot',deps=['assets','shame-mech-'+key+'-pilot-v2'],up=[])]
    a=['kaggle:shame-mech-qwen9-behavior-g0-v1','kaggle:shame-mech-qwen9-behavior-g1-v1'] if key=='qwen9' else ['modal:mistral8_behavior_g0_v1','modal:mistral8_behavior_g1_v1']
    if key=='mistral8':
        for g in [0,1]:jobs.append(dict(name=f'mistral8_behavior_g{g}_v1',stage='behavior',deps=[bridge],group=g,up=[]))
    dep_a=[x.split(':')[1] for x in a]
    b=key+'_emotion_v1';cp=key+'_causal_pilot_v1';c=key+'_causal_v1'
    path=OUT/'resume_overrides.json'
    override=json.loads(path.read_text()).get(c,{}) if path.exists() else {}
    c=override.get('name',c)
    jobs.append(dict(name=b,stage='emotion',deps=[bridge,*dep_a],up=a))
    jobs.append(dict(name=cp,stage='causal_pilot',deps=[b],up=[*a,'modal:'+b]))
    jobs.append(dict(name=c,stage=override.get('stage','causal'),deps=[cp],up=[*a,'modal:'+b]))
    jobs.append(dict(name=key+'_analysis_v1',stage='analysis',deps=[c],up=[*a,'modal:'+b,'modal:'+c]))
    return jobs

def main():
    import msvcrt
    lock=(OUT/'coordinator.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    last=None
    while True:
        state={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':os.getpid(),'active':[],'complete':[],'pending':[],'attention':[]}
        for key,profile in PROFILES.items():
            try:
                r=command(profile,'--task','poll')
                if r.returncode:raise RuntimeError('Poll failed; see coordinator_commands.log')
                poll=json.loads((OUT/('poll_'+profile+'.json')).read_text())
                for name,value in poll.items():
                    if name=='assets' or name.startswith('shame-mech-'):
                        if value.get('status')=='failed':state['attention'].append({'setup':name,'failure':value})
                        elif value.get('status')=='running':state['active'].append(profile+':'+name)
                path=OUT/('budget_'+profile+'.json');ledger=json.loads(path.read_text()) if path.exists() else {'jobs':{}}
                for item in graph(key):
                    name=item['name'];status=poll.get(name,{}).get('status')
                    if status=='complete':
                        state['complete'].append(name)
                        # Download compact evidence only, then run the existing independent validator.
                        dest=ROOT.parent/'outputs/activation_study_v1/jobs'/name
                        if not (dest/'local_validation.json').exists():
                            result=command(profile,'--task','collect','--job',name,timeout=900)
                            if result.returncode:raise RuntimeError('Collection failed for '+name)
                            v=subprocess.run([sys.executable,'training/kaggle/validate_mechanism_job.py','--dest',str(dest)],cwd=ROOT,capture_output=True,text=True)
                            if v.returncode:raise RuntimeError('Validation failed for '+name+': '+v.stderr[-1000:])
                        if item['stage']=='pilot':
                            old=json.loads((dest.parent/(key+'_pilot_v2')/'pilot_metrics.json').read_text())
                            new=json.loads((dest/'pilot_metrics.json').read_text())
                            assert old['exact_base_weight_digest']==new['exact_base_weight_digest'],'Bridge base-weight identity failed'
                        continue
                    if name in ledger['jobs']:
                        if status=='failed':state['attention'].append({'job':name,'failure':poll[name]})
                        else:state['active'].append(name)
                        continue
                    if not all(poll.get(d,{}).get('status')=='complete' for d in item['deps']):
                        state['pending'].append(name);continue
                    args=['--task','launch','--stage',item['stage'],'--job',name]
                    if 'group' in item:args+=['--group',str(item['group'])]
                    if item['up']:args+=['--upstream',','.join(item['up'])]
                    launched=command(profile,*args)
                    if launched.returncode:
                        state['attention'].append({'job':name,'failure':launched.stdout[-1200:]+launched.stderr[-1200:]})
                    else:state['active'].append(name)
                # Read compact remote progress, not native tensors.
                command(profile,'--task','progress',timeout=180)
            except Exception as e:state['attention'].append({'profile':profile,'error':str(e)})
        state['status']='all_complete' if len(state['complete'])==sum(len(graph(k)) for k in PROFILES) else 'monitoring'
        if not state['active'] and state['attention']:state['status']='needs_attention'
        atomic(OUT/'coordinator_state.json',state)
        compact={k:v for k,v in state.items() if k not in ['pending']}
        print(json.dumps(compact),flush=True)
        if state['status']=='all_complete':return
        if state['status']=='needs_attention':return
        time.sleep(30)

if __name__=='__main__':main()
