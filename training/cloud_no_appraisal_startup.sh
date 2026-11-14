#!/bin/bash
set -euo pipefail
umask 077
mkdir -p /work/shame/assets /work/shame/results
exec > >(tee -a /var/log/shame-control.log) 2>&1

# BUCKET, PROJECT, ZONE and INSTANCE are public task identifiers in VM metadata.
meta() { curl --fail --silent -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/$1"; }
export SHAME_BUCKET="$(meta instance/attributes/shame-bucket)"
export SHAME_PROJECT="$(meta project/project-id)"
export SHAME_ZONE="$(meta instance/zone)"
export SHAME_INSTANCE="$(meta instance/name)"
export SHAME_ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)"

cleanup() {
  code=$?
  trap - EXIT
  export SHAME_EXIT_CODE="$code"
  python3 - <<'PY'
import json,os,urllib.request,urllib.parse,time
from pathlib import Path
headers={'Metadata-Flavor':'Google'}
def token():
    with urllib.request.urlopen(urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',headers=headers),timeout=15) as r:return json.load(r)['access_token']
auth={'Authorization':'Bearer '+token()}
bucket=os.environ['SHAME_BUCKET']
def upload(name,data,kind):
    url=f'https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name='+urllib.parse.quote(name,safe='')
    with urllib.request.urlopen(urllib.request.Request(url,headers={**auth,'Content-Type':kind},data=data,method='POST'),timeout=120) as r:r.read()
try:
    path=Path('/var/log/shame-control.log')
    upload('results/bootstrap.log',path.read_bytes(),'text/plain')
    receipt=json.dumps({'exit_code':int(os.environ['SHAME_EXIT_CODE']),'utc_epoch':time.time(),'instance':os.environ['SHAME_INSTANCE'],'attempt':os.environ['SHAME_ATTEMPT']}).encode()
    upload('results/worker_exit.json',receipt,'application/json')
    upload('results/attempts/'+os.environ['SHAME_ATTEMPT']+'/bootstrap.log',path.read_bytes(),'text/plain')
    upload('results/attempts/'+os.environ['SHAME_ATTEMPT']+'/worker_exit.json',receipt,'application/json')
finally:
    # This identity can delete this exact task VM, not other project instances.
    zone=os.environ['SHAME_ZONE'].rsplit('/',1)[-1]
    url=f"https://compute.googleapis.com/compute/v1/projects/{os.environ['SHAME_PROJECT']}/zones/{zone}/instances/{os.environ['SHAME_INSTANCE']}"
    with urllib.request.urlopen(urllib.request.Request(url,headers=auth,method='DELETE'),timeout=60) as r:r.read()
PY
  exit "$code"
}
trap cleanup EXIT

echo "SHAME startup $(date -u +%FT%TZ)"
# A smaller G2 host is easier to place. Swap protects the CPU-side model load;
# model weights and training activations remain on the single 24 GB GPU.
fallocate -l 20G /work/shame/swapfile
chmod 600 /work/shame/swapfile
mkswap /work/shame/swapfile
swapon /work/shame/swapfile
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 - <<'PY'
import hashlib,json,urllib.request,os,zipfile
from pathlib import Path
with urllib.request.urlopen(urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',headers={'Metadata-Flavor':'Google'}),timeout=15) as r:token=json.load(r)['access_token']
url=f"https://storage.googleapis.com/storage/v1/b/{os.environ['SHAME_BUCKET']}/o/payload.zip?alt=media"
archive=Path('/work/shame/payload.zip')
with urllib.request.urlopen(urllib.request.Request(url,headers={'Authorization':'Bearer '+token}),timeout=120) as r:archive.write_bytes(r.read())
dest=Path('/work/shame/assets').resolve()
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        assert (dest/info.filename).resolve().is_relative_to(dest)
    z.extractall(dest)
inventory=json.loads((dest/'asset_manifest.json').read_text())
for name,expected in inventory.items():assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==expected,name
print('Verified',len(inventory),'input assets')
PY

export DEBIAN_FRONTEND=noninteractive
timeout 600s apt-get update -q
timeout 600s apt-get install -y --no-install-recommends python3-pip python3-venv ca-certificates curl
timeout 600s python3 -m pip install --no-cache-dir uv==0.9.5
timeout 600s uv venv --python 3.11 /work/shame/venv
timeout 1800s uv pip install --no-cache --python /work/shame/venv/bin/python \
  torch==2.10.0 torchvision==0.25.0 \
  transformers==5.22.2 peft==0.21.2 accelerate==1.15.0 huggingface_hub==2.4.0 \
  safetensors==0.8.0 sentencepiece==0.2.3 pillow==11.3.0 numpy==2.2.6 \
  mistral-common==1.13.1 flash-linear-attention==0.6.0 \
  scipy==1.15.2 scikit-learn==1.6.1 google-cloud-storage==3.1.0

export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2
timeout --signal=TERM --kill-after=120s 16h /work/shame/venv/bin/python \
  /work/shame/assets/cloud_no_appraisal.py --bucket "$SHAME_BUCKET"
