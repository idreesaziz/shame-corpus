"""Restartable, compact-only monitoring of the mechanism-study campaign.

Reads the campaign each cycle. Does not launch unvalidated stages or purchase
compute. The active assistant can attach wait_for_event.py to receive failures.
"""
import argparse, datetime as dt, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT.parents[2] / 'outputs/activation_study_v1'


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    for attempt in range(50):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(.1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=DEFAULT)
    p.add_argument('--interval', type=int, default=60)
    args = p.parse_args()
    root = args.root.resolve()
    state_path = root / 'watch_state.json'
    lock = root / 'monitor.lock'
    # Windows releases this lock on process exit, including a crash.
    import msvcrt
    handle = lock.open('a+b')
    if handle.tell() == 0:
        handle.write(b'0'); handle.flush()
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    errors = 0
    while True:
        campaign = json.loads((root / 'campaign.json').read_text(encoding='utf8'))
        jobs = []; completed = []; failed = []; active = []
        try:
            for job in campaign['jobs']:
                if job.get('launch_pending'):
                    raise RuntimeError(f"Unreconciled launch for {job['name']}; inspect remote status before retrying")
                if not job.get('launched'):
                    jobs.append({**job, 'status': 'NOT_LAUNCHED'})
                    continue
                dest = root / 'jobs' / job['name']
                receipt_path = dest / 'collection.json'
                previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                if previous.get('status', {}).get('status') == 'COMPLETE' and previous.get('mode') == 'compact':
                    receipt = previous
                else:
                    run = subprocess.run([sys.executable, str(ROOT / 'collect_mechanism_job.py'),
                        '--slug', job['slug'], '--dest', str(dest), '--mode', 'compact'],
                        env=env, capture_output=True, text=True, encoding='utf8', timeout=240)
                    (root / (job['name'] + '_collector.log')).write_text(run.stdout + run.stderr, encoding='utf8')
                    if run.returncode:
                        raise RuntimeError(f"Collector failed for {job['name']}: see its local collector log")
                    receipt = json.loads(receipt_path.read_text())
                status = receipt['status']['status']
                # API COMPLETE means execution ended; output manifest separately
                # establishes our runner reached its own completion marker.
                runner_complete = False
                if status == 'COMPLETE':
                    manifests = [x for x in receipt.get('downloaded', []) if Path(x['file']).name == 'manifest.json']
                    if len(manifests) == 1:
                        m = json.loads((dest / manifests[0]['file']).read_text())
                        runner_complete = m.get('status') == 'complete'
                    if runner_complete and job.get('mode') in ['pilot','causal_pilot','cache_pilot']:
                        metrics=[x for x in receipt.get('downloaded',[]) if Path(x['file']).name=='pilot_metrics.json']
                        runner_complete=len(metrics)==1 and json.loads((dest/metrics[0]['file']).read_text()).get('status')=='passed'
                    if not runner_complete:
                        failed.append(job['name'] + ': missing successful runner manifest')
                    else:
                        validation=subprocess.run([sys.executable,str(ROOT/'validate_mechanism_job.py'),
                            '--dest',str(dest)],env=env,capture_output=True,text=True,encoding='utf8',timeout=180)
                        (root/(job['name']+'_validation.log')).write_text(validation.stdout+validation.stderr,encoding='utf8')
                        if validation.returncode:
                            failed.append(job['name']+': compact/inventory validation failed')
                        else:
                            completed.append(job['name'])
                elif status not in ['RUNNING','QUEUED','NEW_SCRIPT'] or receipt['status'].get('failureMessage'):
                    failed.append(job['name'])
                else:
                    active.append(job['name'])
                jobs.append({**job, 'status': status, 'runner_complete': runner_complete,
                    'files_indexed': len(receipt.get('remote_files', [])),
                    'downloaded_files': len(receipt.get('downloaded', [])),
                    'failure_message': receipt['status'].get('failureMessage')})
            errors = 0
            # Launch only dependency-satisfied, explicitly enabled free jobs.
            # Pending stages remain visible in the campaign, never count as done.
            gpu_count=sum(j.get('gpu',True) and j['name'] in active for j in jobs)
            cpu_count=sum(not j.get('gpu',True) and j['name'] in active for j in jobs)
            if not failed:
                for job in sorted(campaign['jobs'],key=lambda j:j.get('launch_priority',50)):
                    if job.get('launched') or not job.get('launch_enabled'):
                        continue
                    if not set(job.get('depends_on',[])).issubset(completed):
                        continue
                    if job.get('gpu',True) and gpu_count>=2:
                        continue
                    if not job.get('gpu',True) and cpu_count>=2:
                        continue
                    assert job.get('command'), 'Enabled campaign job lacks a command'
                    job['launch_pending']=True
                    atomic(root/'campaign.json',campaign)
                    launch=subprocess.run([sys.executable,*job['command']],cwd=ROOT.parents[1],
                        env=env,capture_output=True,text=True,encoding='utf8',timeout=240)
                    (root/(job['name']+'_launch.log')).write_text(launch.stdout+launch.stderr,encoding='utf8')
                    if launch.returncode:
                        raise RuntimeError(f"Launch for {job['name']} needs reconciliation: see its launch log")
                    job.update(launched=True,launch_pending=False,
                        launched_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                    atomic(root/'campaign.json',campaign)
                    if job.get('gpu',True):gpu_count+=1
                    else:cpu_count+=1
                    active.append(job['name'])
                    for entry in jobs:
                        if entry['name']==job['name']:
                            entry.update(job,status='SUBMITTED')
            pending = [j['name'] for j in jobs if j['status'] == 'NOT_LAUNCHED']
            status = ('stopped_job_failure' if failed else
                'all_complete' if len(completed) == len(jobs) else 'monitoring')
            state = {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'watch_status': status, 'active': active, 'completed': completed,
                'pending': pending, 'failed': failed, 'jobs': jobs,
                'pid': os.getpid(), 'large_activations': 'Retained in private Kaggle outputs'}
            atomic(state_path, state)
            print(json.dumps({k: state[k] for k in ['utc', 'watch_status', 'active', 'completed', 'pending', 'failed']}), flush=True)
            if failed or status == 'all_complete':
                return
        except Exception as exc:
            errors += 1
            atomic(state_path, {'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'watch_status': 'stopped_monitor_error' if errors >= 3 else 'monitor_retry',
                'error': str(exc), 'consecutive_errors': errors, 'pid': os.getpid()})
            if errors >= 3:
                raise
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
