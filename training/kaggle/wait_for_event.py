"""Foreground wait: return control to the active assistant on a meaningful event.

This does not wake an assistant after a turn ends. The assistant must keep
waiting on this process's exec session. No API keys or paid compute are used.
"""
import argparse,datetime as dt,json,time
from pathlib import Path

STATE=Path(__file__).resolve().parents[3]/'outputs/kaggle_epoch3/watch_state.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,default=STATE);args=p.parse_args()
    state_path=args.state
    initial=json.loads(state_path.read_text());completed=initial.get('completed',[]);active=initial.get('active')
    print('Attached active-turn event wait: failure, completion, model change, or stale coordinator.',flush=True)
    while True:
        state=json.loads(state_path.read_text())
        stamp=dt.datetime.fromisoformat(state['utc'])
        age=(dt.datetime.now(dt.timezone.utc)-stamp).total_seconds()
        status=state.get('watch_status','')
        if status.startswith('stopped_') or status in ['all_three_complete','all_complete'] or state.get('completed')!=completed or state.get('active')!=active or age>240:
            summary={k:state.get(k) for k in ['utc','watch_status','active','completed','failed','error','collecting_job'] if k in state}
            print(json.dumps({'notification':summary,'watcher_age_seconds':age}),flush=True);return
        time.sleep(20)

if __name__=='__main__':main()
