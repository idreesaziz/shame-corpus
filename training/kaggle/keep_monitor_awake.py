"""Temporary, process-scoped Windows sleep inhibitor for this overnight run.

Does not change the power plan or keep the display lit. Releases the request
on completion, process termination, or after 18 hours. Does not override an
explicit user sleep/shutdown or emergency battery shutdown.
"""
import argparse,atexit,ctypes,json,time,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path);args=p.parse_args()
    assert sys.platform=='win32'
    kernel=ctypes.windll.kernel32
    assert kernel.SetThreadExecutionState(0x80000001), 'Could not request temporary system wakefulness'
    atexit.register(lambda:kernel.SetThreadExecutionState(0x80000000))
    state=args.state or Path(__file__).resolve().parents[3]/'outputs/kaggle_epoch3/watch_state.json'
    deadline=time.monotonic()+18*3600
    print('Temporary SYSTEM_REQUIRED request active; display may turn off.',flush=True)
    while time.monotonic()<deadline:
        try:
            if json.loads(state.read_text()).get('watch_status') in ['all_three_complete','all_complete']:
                # Allow the active assistant to validate and consolidate downloaded
                # results before releasing a system that has been idle overnight.
                time.sleep(600)
                break
        except (OSError,ValueError):pass
        time.sleep(30)
    kernel.SetThreadExecutionState(0x80000000)
    print('Temporary sleep inhibition released.',flush=True)

if __name__=='__main__':main()
