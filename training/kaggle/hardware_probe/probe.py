import json,platform,time
from pathlib import Path
import torch
info={'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpus':[]}
for i in range(torch.cuda.device_count()):
 p=torch.cuda.get_device_properties(i)
 info['gpus'].append({'index':i,'name':p.name,'memory_bytes':p.total_memory,'capability':list(torch.cuda.get_device_capability(i))})
info['bf16_supported']=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
Path('/kaggle/working/hardware.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))
