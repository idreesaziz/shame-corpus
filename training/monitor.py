"""Monitor existing campaigns and back up completed adapters; never launches training."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import time

import modal


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'outputs/modal_training_20261111')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    get_status = modal.Function.from_name('shame-training-monitor', 'status')
    list_artifacts = modal.Function.from_name('shame-training-monitor', 'list_artifacts')
    read_chunk = modal.Function.from_name('shame-training-monitor', 'read_chunk')
    index_path = root / 'download_index.json'
    downloaded = json.loads(index_path.read_text()) if index_path.exists() else {}
    run_index = root / 'RUNS.json'

    def download(entry):
        relative = entry['path']
        target = (root / 'artifacts' / relative).resolve()
        assert target.is_relative_to((root / 'artifacts').resolve())
        blob = root / 'download_cache' / entry['sha256']
        if not (blob.exists() and blob.stat().st_size == entry['bytes'] and sha(blob) == entry['sha256']):
            parts = root / 'download_parts' / entry['sha256']
            parts.mkdir(parents=True, exist_ok=True)
            offsets = list(range(0, entry['bytes'], 900000))

            def chunk(offset):
                p = parts / f'{offset:012d}.part'
                size = min(900000, entry['bytes'] - offset)
                if p.exists() and p.stat().st_size == size: return p
                data = read_chunk.remote(relative, offset, size)
                assert len(data) == size
                p.write_bytes(data)
                return p

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(chunk, offsets))
            blob.parent.mkdir(exist_ok=True)
            with blob.open('wb') as f:
                for offset in offsets: f.write((parts / f'{offset:012d}.part').read_bytes())
            assert sha(blob) == entry['sha256'], 'Downloaded artifact checksum mismatch: ' + relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(blob, target)
        downloaded[relative] = entry
        save(index_path, downloaded)

    while True:
        try:
            status = get_status.remote()
            status['checked_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
            save(root / 'status.json', status)
            with (root / 'monitor_history.jsonl').open('a', encoding='utf8') as f: f.write(json.dumps(status) + '\n')
            print(json.dumps({'checked_utc': status['checked_utc'], 'completed_runs': status['completed_runs'], 'campaign_status': {k:v['status'] for k,v in status['campaigns'].items()}}), flush=True)
            completed = [r for r in status['runs'] if r.get('status') == 'complete']
            # A manifest can arrive before the weights. An interrupted download must
            # still resume even if the manifest is already recorded locally.
            need = any(
                any(f"{r['model_key']}/{r['condition']}/seed-{r['seed']}/{suffix}" not in downloaded
                    or not (root/'artifacts'/r['model_key']/r['condition']/f"seed-{r['seed']}"/suffix).exists()
                    for suffix in ['manifest.json', 'steps.jsonl', 'adapter/adapter_config.json',
                                   'adapter/adapter_model.safetensors', 'adapter/tokenizer_config.json'])
                for r in completed)
            if need:
                entries = list_artifacts.remote()
                for entry in entries:
                    old = downloaded.get(entry['path'])
                    target = root / 'artifacts' / entry['path']
                    if old == entry and target.exists(): continue
                    download(entry)
            index = []
            for row in completed:
                path = root / 'artifacts' / row['model_key'] / row['condition'] / ('seed-' + str(row['seed']))
                adapter = path / 'adapter'
                files = [adapter/'adapter_config.json', adapter/'adapter_model.safetensors', path/'manifest.json']
                index.append({**row, 'local_run_directory':str(path), 'local_adapter_directory':str(adapter), 'core_files_backed_up':all(p.exists() for p in files)})
            save(run_index, {'experiment':'shame-sdf-20261111-v1','completed_runs':len(index),'expected_runs':24,'runs':index,'results_volume':'shame-training-results-v1','no_training_launched_by_monitor':True})
            if status['completed_runs'] == 24 and all(r['core_files_backed_up'] for r in index):
                save(root/'monitor_finished.json',{'at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'completed_runs':24,'local_backup_complete':True})
                return
            if args.once: return
        except Exception as exc:
            error = {'at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'type':type(exc).__name__,'message':str(exc)[:800]}
            save(root/'monitor_error.json', error)
            print(json.dumps(error),flush=True)
            if args.once: raise
        time.sleep(60)


if __name__ == '__main__':
    main()
