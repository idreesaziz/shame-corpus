"""Kaggle CLI can exit zero on rejected pushes; require an acceptance receipt."""
import subprocess


def push_checked(arguments,reference):
    run=subprocess.run(arguments,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8')
    print(run.stdout,end='',flush=True)
    accepted=('successfully pushed' in run.stdout and
        'https://www.kaggle.com/code/'+reference in run.stdout and
        'Kernel push error:' not in run.stdout)
    if run.returncode or not accepted:
        raise RuntimeError('Kaggle did not accept this submission. Inspect the launch log; CLI exit code alone is insufficient.')
    return run
