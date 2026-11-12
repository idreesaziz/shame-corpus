"""Retrieve completed bounded evaluation jobs with byte/hash verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import modal


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    dest=args.directory;launch=json.loads((dest/'launch.json').read_text())
    assert os.environ.get('MODAL_PROFILE')==launch['profile']
    try:manifest=modal.FunctionCall.from_id(launch['call_id']).get(timeout=0)
    except TimeoutError:print('Job still running.');return
    (dest/'campaign_result.json').write_text(json.dumps(manifest,indent=2)+'\n')
    app='shame-training-monitor' if launch['phase']=='one' else 'shame-expansion3-monitor'
    reader=modal.Function.from_name(app,'read_chunk')
    for entry in manifest:
        if entry['status']=='failed':print(json.dumps(entry));continue
        target=dest/(entry['model']+'.json')
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']:
            chunks=[]
            for offset in range(0,entry['bytes'],128000):
                size=min(128000,entry['bytes']-offset);data=reader.remote(entry['relative_path'],offset,size)
                assert len(data)==size;chunks.append(data)
            data=b''.join(chunks);assert hashlib.sha256(data).hexdigest()==entry['sha256']
            target.write_bytes(data)
        print(json.dumps(entry))


if __name__=='__main__':main()
