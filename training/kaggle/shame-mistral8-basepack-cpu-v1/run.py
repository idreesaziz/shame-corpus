
import os,sys,json,hashlib,shutil,subprocess,traceback
from pathlib import Path
os.environ.update(HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
root=Path('/kaggle/working');pack=root/'basepack';pack.mkdir(exist_ok=True)
def save(name,value): (root/name).write_text(json.dumps(value,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
try:
 subprocess.run([sys.executable,'-m','pip','install','--quiet','huggingface_hub==2.4.0'],check=True)
 from huggingface_hub import HfApi,hf_hub_download
 key,model,revision=('mistral8', 'mistralai/Ministral-3-8B-Instruct-2512-BF16', 'f6fae9795746f63c9be8344932f01275f3c63734')
 available=HfApi().list_repo_files(model,revision=revision)
 index='model.safetensors.index.json'
 assert index in available,'Expected indexed HF-format weights'
 path=Path(hf_hub_download(model,index,revision=revision,local_dir=pack))
 shards=set(json.loads(path.read_text())['weight_map'].values())
 # Never fetch an alternative consolidated checkpoint in addition to shards.
 extras={n for n in available if '/' not in n and not n.endswith('.safetensors')
  and (n.endswith(('.json','.model','.tiktoken','.txt','.jinja')) or n.upper().startswith(('LICENSE','NOTICE','USE_POLICY')) or n=='README.md')}
 selected=sorted(shards|extras|{index});files=[]
 for name in selected:
  p=Path(hf_hub_download(model,name,revision=revision,local_dir=pack))
  files.append({'file':name,'bytes':p.stat().st_size,'sha256':sha(p)})
  print(json.dumps({'file':name,'bytes':p.stat().st_size,'completed':len(files),'total':len(selected)}),flush=True)
 assert (pack/'README.md').exists()
 manifest={'model':key,'model_id':model,'revision':revision,'private':True,'files':files,
  'license_files':[n for n in selected if n.upper().startswith(('LICENSE','NOTICE','USE_POLICY')) or n=='README.md'],
  'scope':'Pinned public source weights; no model inference or training. Original model licenses retained.'}
 (pack/'basepack_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 save('manifest.json',{'status':'complete','model':key,'files':files,'manifest_sha256':sha(pack/'basepack_manifest.json')})
 print(json.dumps({'status':'complete','bytes':sum(x['bytes'] for x in files)}),flush=True)
except BaseException as e:
 save('failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
