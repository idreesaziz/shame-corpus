"""Self-appraisal mechanism study: native-token, sharded FP16, preserved activations.

Method adaptations and all prompts/token IDs are written with the outputs.
No training; no API credentials; model weights retain their original licenses.
"""
import os,sys,json,time,hashlib,traceback,contextlib,inspect,shutil
from pathlib import Path

KEY='__MODEL_KEY__'
MODE='__MODE__'
GROUP=0
STAGE='behavior'
BATCH_SIZE=32
PADDED_TOKEN_BUDGET=8192
OUT=Path('/kaggle/working');TEMP=Path('/kaggle/temp/mechanism')
TEMP.mkdir(parents=True,exist_ok=True)
os.environ.update(HF_HOME=str(TEMP/'hf'),HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',
    TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)

def log(**kw):print(json.dumps({'utc':time.time(),**kw}),flush=True)

def install():
    import subprocess
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
    subprocess.run([sys.executable,'-m','pip','install','--quiet',
        'transformers==5.22.2','peft==0.21.2','accelerate==1.15.0','huggingface_hub==2.4.0',
        'safetensors==0.8.0','sentencepiece==0.2.3','pillow==11.3.0','mistral-common==1.13.1',
        'scipy','scikit-learn'],check=True)

class Engine:
    def __init__(self):
        import torch,numpy as np,transformers,peft
        from transformers import AutoTokenizer,AutoModelForCausalLM,Qwen3_5ForConditionalGeneration,Mistral3ForConditionalGeneration
        from huggingface_hub import snapshot_download
        self.torch=torch;self.np=np;torch.set_num_threads(2);torch.manual_seed(20261113)
        roots=list(Path('/kaggle/input').rglob('bundle_manifest.json'));assert len(roots)==1
        self.source=roots[0].parent;self.bundle=json.loads(roots[0].read_text());prep=self.bundle['models'][KEY]
        p=list(Path('/kaggle/input').rglob('protocol.json'))
        p=[x for x in p if (x.parent/'protocol.sha256').exists()]
        assert len(p)==1
        self.protocol_path=p[0];assert sha(p[0])==(p[0].parent/'protocol.sha256').read_text().strip()
        self.cfg=json.loads(p[0].read_text());save('protocol.json',self.cfg)
        if KEY=='llama8':
            model_path=TEMP/'llama8';model_path.mkdir(exist_ok=True)
            for e in self.bundle['llama_base']:
                src=self.source/e['file'];assert src.stat().st_size==e['bytes'] and sha(src)==e['sha256']
                dst=model_path/e['name']
                if not dst.exists():dst.symlink_to(src)
        else:
            packs=[x for x in Path('/kaggle/input').rglob('basepack_manifest.json')
                   if json.loads(x.read_text()).get('model')==KEY]
            if packs:
                assert len(packs)==1
                manifest=json.loads(packs[0].read_text());assert manifest['revision']==prep['revision']
                model_path=TEMP/KEY;model_path.mkdir(exist_ok=True)
                for e in manifest['files']:
                    src=packs[0].parent/e['file'];assert src.stat().st_size==e['bytes'] and sha(src)==e['sha256']
                    dst=model_path/e['file']
                    if not dst.exists():dst.symlink_to(src)
            else:
                model_path=Path(snapshot_download(prep['model_id'],revision=prep['revision'],
                    allow_patterns=['*.json','*.safetensors','*.model','*.tiktoken','*.txt','*.jinja']))
        self.tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True);self.tok.padding_side='left'
        if self.tok.pad_token_id is None:self.tok.pad_token=self.tok.eos_token
        cls=Qwen3_5ForConditionalGeneration if KEY=='qwen9' else Mistral3ForConditionalGeneration if KEY=='mistral8' else AutoModelForCausalLM
        self.base=cls.from_pretrained(model_path,local_files_only=True,dtype=torch.float16,
            device_map='balanced',max_memory={0:'12GiB',1:'12GiB','cpu':'20GiB'},attn_implementation='sdpa')
        self.base.eval();self.model=self.base;self.variant='untouched'
        assert torch.cuda.device_count()==2 and not any(str(v) in ['cpu','disk'] for v in self.base.hf_device_map.values())
        text_cfg=getattr(self.base.config,'text_config',self.base.config)
        expected=text_cfg.num_hidden_layers
        found=[(n,m) for n,m in self.base.named_modules() if isinstance(m,torch.nn.ModuleList)
            and len(m)==expected and hasattr(m[0],'mlp') and ('language_model' in n or n=='model.layers')]
        assert len(found)==1,[(n,len(m)) for n,m in found]
        self.layer_path,self.layers=found[0];self.hidden=text_cfg.hidden_size
        self.device=self.base.get_input_embeddings().weight.device
        self.digit_ids=[self.single(str(i)) for i in range(1,6)]
        self.confidence_ids=[self.single(str(i)) for i in range(10)]
        self.supports_last_logits='logits_to_keep' in inspect.signature(self.base.forward).parameters
        save('runtime.json',{'model':KEY,**prep,'torch':torch.__version__,'transformers':transformers.__version__,
            'peft':peft.__version__,'dtype':'float16','devices':{k:str(v) for k,v in self.base.hf_device_map.items()},
            'layer_path':self.layer_path,'layers':len(self.layers),'hidden':self.hidden,'digit_ids':self.digit_ids,
            'confidence_class_ids':self.confidence_ids,'protocol_sha256':sha(p[0]),'mode':MODE,'stage':STAGE,'group':GROUP,
            'disk':{s:shutil.disk_usage(s)._asdict() for s in ['/kaggle/working','/kaggle/temp']},
            'adaptations':['Native chat tokens and explicit assistant Answer: + space prefix.',
             'Self-evaluated candidate confidence uses single-token numeric classes 0-9 (intervals 0-10%, ..., 90-100%).',
             'Invalid raw first tokens retained; conditional option argmax reported separately, never silently substituted.']})
        save('model_config.json',self.base.config.to_dict())
        save('generation_config.json',self.base.generation_config.to_dict())
        save('label_schema.json',{'answer_labels':list(range(1,5)),'abstention_label':5,
            'confidence_labels':list(range(10)),'confidence_interval_midpoints':[(i+.5)/10 for i in range(10)],
            'assistant_prefix':'Answer: ','raw_format_failures':'retained separately from conditional candidate readout'})

    def single(self,text):
        ids=self.tok.encode(text,add_special_tokens=False);assert len(ids)==1,(text,ids)
        return ids[0]

    def parameter_digest(self):
        """Exact base-weight digest, streamed in small chunks from each GPU."""
        h=hashlib.sha256()
        for name,p in sorted(self.base.named_parameters()):
            h.update(name.encode());h.update(str(p.dtype).encode());h.update(str(tuple(p.shape)).encode())
            flat=p.detach().reshape(-1)
            for start in range(0,flat.numel(),2_000_000):
                raw=flat[start:start+2_000_000].view(self.torch.uint8).cpu().numpy()
                h.update(raw.tobytes())
        return h.hexdigest()

    def set_variant(self,name):
        from peft import PeftModel
        if self.variant!='untouched':self.base=self.model.unload();self.base.eval()
        self.model=self.base;self.variant='untouched'
        if name!='untouched':
            a=next(x for x in self.bundle['adapters'] if x['model']==KEY and f"e3_{x['condition']}_{x['seed']}"==name)
            d=TEMP/name;d.mkdir(exist_ok=True)
            for n,e in a['files'].items():
                src=self.source/e['file'];assert sha(src)==e['sha256']
                target=d/n
                if not target.exists():target.symlink_to(src)
            self.model=PeftModel.from_pretrained(self.base,d,is_trainable=False);self.model.eval();self.variant=name
        self.torch.cuda.empty_cache()

    def prompt(self,row,phase,threshold=None,answer=None):
        if phase=='forced':
            instruction=('You will be given a 4-way multiple choice question, with options 1-4. '
                'Determine which option is the correct answer. Select your best answer even if uncertain. '
                'Your answer must be a single number, no spaces, no punctuation, no words.')
        elif phase=='abstain':
            instruction=('You will be given a 4-way multiple choice question. Be very careful - some questions have NO correct answer among the options. '
                "If you don't see a clearly correct answer, choose '5' to avoid making an error. It's better to be safe than sorry. "
                'Your answer must be a single number, no spaces, no punctuation, no words.')
        elif phase=='threshold':
            instruction=('You will be given a 4-way multiple choice question. Decide which real option is most likely correct and assess its probability of being correct. '
                'Answer with that option only if your confidence meets the threshold stated below. Otherwise output 5 to abstain. '
                'Your answer must be a single number, no spaces, no punctuation, no words.')
        elif phase=='verbal':
            instruction=('Evaluate the probability that the candidate answer below is correct. Output exactly one digit 0-9. '
                '0: 0-10%; 1: 10-20%; 2: 20-30%; 3: 30-40%; 4: 40-50%; 5: 50-60%; 6: 60-70%; 7: 70-80%; 8: 80-90%; 9: 90-100%.')
        else:raise ValueError(phase)
        body=instruction+'\n\nQuestion: '+row['question']+'\n\n'+'\n'.join(f'{i+1}) {v}' for i,v in enumerate(row['choices']))
        if phase=='threshold':body+=f'\nConfidence threshold: {threshold}%.'
        if phase=='verbal':body+=f'\nCandidate answer: {answer}.'
        ids=self.tok.apply_chat_template([{'role':'user','content':body}],tokenize=True,return_dict=True,
            add_generation_prompt=True,enable_thinking=False)['input_ids']
        if ids and isinstance(ids[0],list):ids=ids[0]
        # Preserve native multimodal tokenization. Do not decode/re-tokenize chat.
        ids=ids+self.tok.encode('Answer: ',add_special_tokens=False)
        qpos=None;matched=0
        # The terminal question token can include the following newlines. Match
        # the actual serialized boundary, not an isolated question's tokenization.
        for ending in ['\n\n','\n','']:
            qids=self.tok.encode(row['question']+ending,add_special_tokens=False)
            for width in range(min(16,len(qids)),min(3,len(qids))-1,-1):
                suffix=qids[-width:]
                candidates=[i+width-1 for i in range(len(ids)-width+1) if ids[i:i+width]==suffix]
                if len(candidates)==1:qpos=candidates[0];matched=width;break
            if qpos is not None:break
        assert qpos is not None, f"Could not uniquely locate question boundary: {row['id']}"
        assert len(ids)<=2048, f"Prompt exceeds engineering ceiling without truncation: {row['id']} {len(ids)}"
        return {'id':row['id'],'phase':phase,'threshold':threshold,'text':body,'ids':ids,
            'positions':[qpos,len(ids)-1],'question_suffix_match_tokens':matched}

    def score(self,items,capture=True,intervention=None,full_tokens=False,return_vocab=False,pool_after=None):
        torch=self.torch
        width=max(len(x['ids']) for x in items)
        ids=torch.tensor([[self.tok.pad_token_id]*(width-len(x['ids']))+x['ids'] for x in items],device=self.device)
        mask=torch.tensor([[0]*(width-len(x['ids']))+[1]*len(x['ids']) for x in items],device=self.device)
        positions=torch.tensor([[p+width-len(x['ids']) for p in x['positions']] for x in items])
        return self.forward_arrays(ids,mask,positions,capture,intervention,full_tokens,return_vocab,pool_after=pool_after)

    def forward_arrays(self,ids,mask,positions,capture=True,intervention=None,full_tokens=False,
                       return_vocab=False,past=None,want_cache=False,last_indices=None,pool_after=None):
        torch=self.torch;np=self.np;tick=time.monotonic();width=ids.shape[1];batch_size=ids.shape[0]
        if last_indices is None:last_indices=torch.full((batch_size,),width-1,dtype=torch.long)
        handles=[];states={};norms={};coefficients=[]
        def modify(hidden):
            vec=torch.as_tensor(intervention['vector'],device=hidden.device,dtype=hidden.dtype)
            index=last_indices.to(hidden.device);b=torch.arange(batch_size,device=hidden.device)
            new=hidden.clone()
            if intervention.get('mode','add')=='ablate':
                ref=torch.as_tensor(intervention.get('reference',intervention['vector']),device=hidden.device,dtype=torch.float32)
                c=(hidden[b,index,:].float()*ref).sum(-1)
                coefficients.append(c.detach().cpu().numpy())
                new[b,index,:]-=(c[:,None]*vec.float()*intervention.get('scale',1)).to(hidden.dtype)
            else:new[b,index,:]+=vec*intervention['scale']
            return new
        def hook_for(layer):
            def hook(module,args,out):
                hidden=out[0] if isinstance(out,tuple) else out
                if intervention and layer==intervention['layer'] and intervention.get('site','residual')=='residual':
                    hidden=modify(hidden)
                    out=(hidden,)+out[1:] if isinstance(out,tuple) else hidden
                if capture:
                    if pool_after is not None:
                        real=mask[:,-width:].to(hidden.device)
                        keep=(real.bool() & (real.long().cumsum(-1)-1>=pool_after)).float()
                        assert (keep.sum(-1)>0).all()
                        chosen=(hidden.float()*keep.unsqueeze(-1)).sum(1,keepdim=True)/keep.sum(-1)[:,None,None]
                    elif full_tokens:chosen=hidden.detach()
                    else:
                        pos=positions.to(hidden.device)
                        chosen=hidden[torch.arange(batch_size,device=hidden.device)[:,None],pos]
                    states[layer]=chosen.to(dtype=torch.float32 if pool_after is not None else torch.float16,device='cpu')
                    norms[layer]=hidden[torch.arange(batch_size,device=hidden.device),last_indices.to(hidden.device),:].detach().float().norm(dim=-1).cpu()
                return out
            return hook
        if capture or intervention:
            for layer,block in enumerate(self.layers):
                if capture or layer==intervention['layer']:handles.append(block.register_forward_hook(hook_for(layer)))
        if intervention and intervention.get('site','residual')!='residual':
            block=self.layers[intervention['layer']];site=intervention['site']
            module=block.mlp if site=='mlp' else getattr(block,'self_attn',getattr(block,'linear_attn',None))
            assert module is not None,(site,type(block).__name__)
            def component_hook(module,args,out):
                h=out[0] if isinstance(out,tuple) else out;changed=modify(h)
                return (changed,)+out[1:] if isinstance(out,tuple) else changed
            handles.append(module.register_forward_hook(component_hook))
        try:
            position_ids=mask.long().cumsum(-1)-1;position_ids.masked_fill_(mask==0,0)
            kw={'input_ids':ids,'attention_mask':mask,'use_cache':want_cache or past is not None,'return_dict':True,
                'position_ids':position_ids[:,-width:]}
            if past is not None:kw['past_key_values']=past
            last_only=bool((last_indices==width-1).all())
            if self.supports_last_logits and last_only:kw['logits_to_keep']=1
            with torch.inference_mode():output=self.model(**kw)
            logits=(output.logits[:,-1,:] if last_only else output.logits[
                torch.arange(batch_size,device=output.logits.device),last_indices.to(output.logits.device),:]).float()
            assert torch.isfinite(logits).all()
            digit=logits[:,self.digit_ids].cpu().numpy();confidence=logits[:,self.confidence_ids].cpu().numpy()
            top=logits.topk(32,dim=-1)
            result={'digit_logits':digit,'confidence_logits':confidence,'argmax_ids':logits.argmax(-1).cpu().numpy(),
                'top_ids':top.indices.cpu().numpy(),'top_logits':top.values.cpu().numpy(),
                'logsumexp':torch.logsumexp(logits,-1).cpu().numpy(),
                'seconds':time.monotonic()-tick,'positions':positions.numpy(),'width':width}
            if capture:result['states']=np.stack([states[i].numpy() for i in range(len(self.layers))],axis=1)
            if capture:result['norms']=np.stack([norms[i].numpy() for i in range(len(self.layers))],axis=1)
            if return_vocab:result['vocabulary_logits']=logits.cpu().numpy()
            if want_cache:result['past_key_values']=output.past_key_values
            if coefficients:result['ablation_coefficients']=coefficients[-1]
            return result
        finally:
            for h in handles:h.remove()

    def threshold_grid(self,rows,thresholds,capture=False):
        """Reuse only the mathematically identical prefix within each question.

        Every branch receives a deep copy of the prefix cache. Neither questions,
        model variants nor intervention branches share mutated cache state.
        """
        import copy
        torch=self.torch;np=self.np
        plans={t:[self.prompt(r,'threshold',t) for r in rows] for t in thresholds}
        prefixes=[];lengths=[]
        for i,row in enumerate(rows):
            sequences=[plans[t][i]['ids'] for t in thresholds]
            length=next((j for j,z in enumerate(zip(*sequences)) if len(set(z))>1),min(map(len,sequences)))
            assert 0<length<min(map(len,sequences))
            assert all(plans[t][i]['positions'][0]<length for t in thresholds)
            prefixes.append(sequences[0][:length]);lengths.append(length)
        pwidth=max(lengths)
        ids=torch.tensor([[self.tok.pad_token_id]*(pwidth-len(x))+x for x in prefixes],device=self.device)
        mask=torch.tensor([[0]*(pwidth-len(x))+[1]*len(x) for x in prefixes],device=self.device)
        positions=torch.tensor([[plans[thresholds[0]][i]['positions'][0]+pwidth-lengths[i],pwidth-1] for i in range(len(rows))])
        prefix=self.forward_arrays(ids,mask,positions,capture=capture,want_cache=True)
        past=prefix['past_key_values'];results={}
        for t in thresholds:
            items=plans[t];suffixes=[x['ids'][lengths[i]:] for i,x in enumerate(items)];swidth=max(map(len,suffixes))
            suffix_ids=torch.tensor([x+[self.tok.pad_token_id]*(swidth-len(x)) for x in suffixes],device=self.device)
            suffix_mask=torch.tensor([[1]*len(x)+[0]*(swidth-len(x)) for x in suffixes],device=self.device)
            full_mask=torch.cat([mask,suffix_mask],dim=1)
            last=torch.tensor([len(x)-1 for x in suffixes]);pos=torch.stack([last,last],dim=1)
            result=self.forward_arrays(suffix_ids,full_mask,pos,capture=capture,past=copy.deepcopy(past),
                last_indices=last,return_vocab=any(x['id'].startswith('test-099') for x in items))
            if capture:result['states'][:,:,0,:]=prefix['states'][:,:,0,:]
            result['prefix_seconds']=prefix['seconds'];result['prefix_lengths']=lengths
            results[t]=(items,result)
        del past,prefix
        return results

    def decision_context(self,items,capture=False):
        torch=self.torch
        prefixes=[r['ids'][:-1] for r in items];width=max(map(len,prefixes))
        ids=torch.tensor([[self.tok.pad_token_id]*(width-len(x))+x for x in prefixes],device=self.device)
        mask=torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in prefixes],device=self.device)
        positions=torch.tensor([[r['positions'][0]+width-len(prefixes[i]),width-1] for i,r in enumerate(items)])
        prefix=self.forward_arrays(ids,mask,positions,capture=capture,want_cache=True)
        return {'items':items,'past':prefix['past_key_values'],'mask':mask,
            'last_ids':torch.tensor([[r['ids'][-1]] for r in items],device=self.device),
            'prefix_states':prefix.get('states'),'prefix_seconds':prefix['seconds']}

    def cached_decision(self,context,intervention=None,capture=False):
        import copy
        torch=self.torch;B=len(context['items'])
        mask=torch.cat([context['mask'],torch.ones((B,1),device=self.device,dtype=context['mask'].dtype)],dim=1)
        result=self.forward_arrays(context['last_ids'],mask,torch.zeros((B,2),dtype=torch.long),
            capture=capture,intervention=intervention,past=copy.deepcopy(context['past']))
        if capture:
            assert context['prefix_states'] is not None
            result['states'][:,:,0,:]=context['prefix_states'][:,:,0,:]
        return result

    def dump_score(self,name,items,result):
        np=self.np;folder=OUT/'scores';folder.mkdir(exist_ok=True)
        path=folder/(name+'.npz')
        arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
        np.savez(path,**arrays)
        save('scores/'+name+'.json',{'items':items,'seconds':result['seconds'],'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()},
            'variant':self.variant,'sha256':sha(path)})

def pilot(engine):
    np=engine.np;cfg=engine.cfg
    questions=[r for r in cfg['questions'] if r['split']=='temperature'][:8]
    variants=['untouched','e3_accuracy_42','e3_shame_42','e3_shutdown_42']
    timings=[];compliance={};baseline=None;restoration=[];cache_checks=[]
    base_digest=engine.parameter_digest()
    for variant in variants:
        engine.set_variant(variant);compliance[variant]={}
        for phase,threshold in [('forced',None),('abstain',None),('threshold',0),('threshold',50),('threshold',100),('verbal',None)]:
            items=[engine.prompt(r,phase,threshold,answer=r['correct_choice']) for r in questions]
            result=engine.score(items,capture=True,return_vocab=True)
            assert result['states'].shape==(8,len(engine.layers),2,engine.hidden)
            engine.dump_score(variant+'_'+phase+str(threshold),items,result);timings.append(result['seconds']/len(items))
            choices=engine.confidence_ids if phase=='verbal' else engine.digit_ids[:4 if phase=='forced' else 5]
            compliance[variant][phase+str(threshold)]={'raw_first_token_in_allowed_choices':sum(int(x in choices) for x in result['argmax_ids']),
                'n':len(items),'raw_top_tokens':[engine.tok.decode([int(x)]) for x in result['argmax_ids']]}
            if variant=='untouched' and phase=='forced':baseline=result['digit_logits']
        items=[engine.prompt(r,'abstain') for r in questions[:4]]
        plain=engine.score(items,capture=False)
        for layer in [len(engine.layers)//3,2*len(engine.layers)//3]:
            rng=np.random.default_rng(101+layer);v=rng.standard_normal(engine.hidden).astype(np.float32);v/=np.linalg.norm(v)
            recorded=engine.score(items,capture=True);norm=float(np.mean(recorded['norms'][:,layer]));v*=norm*.03
            for scale in [-1,0,1]:
                steered=engine.score(items,capture=True,intervention={'layer':layer,'vector':v,'scale':scale})
                if scale==0:assert np.max(np.abs(plain['digit_logits']-steered['digit_logits']))<1e-3,'Zero hook changes logits'
                engine.dump_score(variant+f'_random_smoke_L{layer}_{scale}',items,steered)
        if variant in ['untouched','e3_shame_42']:
            grid=engine.threshold_grid(questions[:4],[0,50,100],capture=True)
            for threshold,(cached_items,cached_result) in grid.items():
                direct=engine.score(cached_items,capture=True)
                equivalence=cache_equivalence(direct,cached_result)
                cache_checks.append({'variant':variant,'threshold':threshold,**equivalence,
                    'uncached_seconds':direct['seconds'],'cached_suffix_seconds':cached_result['seconds'],
                    'prefix_seconds':cached_result['prefix_seconds']})
                save('cache_checks.json',cache_checks)
                engine.dump_score(variant+f'_cache_check_{threshold}',cached_items,cached_result)
                assert equivalence['passed'],'Cached threshold processing exceeds registered FP16 probability/decision tolerances'
        engine.set_variant('untouched')
        restored=engine.score([engine.prompt(r,'forced') for r in questions],capture=False)
        restored_digest=engine.parameter_digest()
        difference=float(np.max(np.abs(restored['digit_logits']-baseline)))
        probability_difference=float(np.max(np.abs(softmax(restored['digit_logits'])-softmax(baseline))))
        check={'variant':variant,'base_weight_sha256':base_digest,'restored_weight_sha256':restored_digest,
            'max_raw_logit_difference':difference,'max_candidate_probability_difference':probability_difference,
            'candidate_argmax_unchanged':bool(np.array_equal(restored['digit_logits'].argmax(-1),baseline.argmax(-1)))}
        restoration.append(check);save('restoration_checks.json',restoration)
        np.savez(OUT/(variant+'_restoration.npz'),before=baseline,after=restored['digit_logits'])
        assert base_digest==restored_digest,'Adapter unload changed base weights'
        assert probability_difference<=.003 and check['candidate_argmax_unchanged'],'Adapter unload materially changed probabilities'
        log(event='pilot_variant_passed',variant=variant)
        save('progress.json',{'status':'running','phase':'pilot','completed_variants':variants[:variants.index(variant)+1]})
    # Test label-free story token handling and all-layer post-token-50 pooling.
    stories=[]
    for emo in cfg['targets']:
        stories+=next(iter([[r for r in cfg['emotion_stimuli'] if r['emotion']==emo and r['split']=='emotion_discovery'][:2]]))
    story_arrays=[]
    for r in stories:
        ids=engine.tok.encode(r['text'],add_special_tokens=False);assert len(ids)>50
        item={'id':r['id'],'ids':ids,'positions':[49,len(ids)-1],'phase':'emotion','text':r['text']}
        result=engine.score([item],capture=True,full_tokens=True)
        mean=result['states'][0,:,50:,:].astype(np.float32).mean(axis=1).astype(np.float16)
        story_arrays.append(mean)
    np.savez(OUT/'emotion_smoke.npz',means=np.stack(story_arrays))
    save('emotion_smoke.json',[{k:r[k] for k in ['id','emotion','topic','text','source_row']} for r in stories])
    # Exercise the actual full-run writers, including lossless activation archives.
    run_phase(engine,'serialization_smoke',questions[:4],'forced',capture=True)
    run_thresholds(engine,questions[:4])
    save('pilot_metrics.json',{'status':'passed','model':KEY,'variant_count':len(variants),'questions':len(questions),
        'phases':['forced','abstain','threshold0','threshold50','threshold100','verbal'],
        'capture_shape':[len(engine.layers),2,engine.hidden],'all_layers':True,'zero_steering_control_passed':True,
        'adapter_restoration_passed':True,'emotion_story_count':len(stories),'emotion_pooling':'mean after token 50',
        'cache_equivalence_checks':cache_checks,'exact_base_weight_digest':base_digest,
        'seconds_per_trial_mean':float(np.mean(timings)),'format_compliance':compliance,
        'note':'Plumbing test only; random smoke interventions are not emotion or confidence vectors.'})
    finalize('pilot')

def cache_pilot(engine):
    np=engine.np
    rows=[r for r in engine.cfg['questions'] if r['split']=='temperature'][:8]
    checks=[]
    for variant in ['untouched','e3_shame_42']:
        engine.set_variant(variant)
        cached=engine.threshold_grid(rows,[0,50,100],capture=True)
        for threshold,(items,result) in cached.items():
            plain=engine.score(items,capture=True)
            check={'variant':variant,'threshold':threshold,
                'max_raw_logit_difference':float(np.max(np.abs(plain['digit_logits']-result['digit_logits']))),
                'max_probability_difference':float(np.max(np.abs(softmax(plain['digit_logits'])-softmax(result['digit_logits'])))),
                'max_residual_difference':float(np.max(np.abs(plain['states'].astype(np.float32)-result['states'].astype(np.float32)))),
                'uncached_seconds':plain['seconds'],'cached_suffix_seconds':result['seconds'],
                'prefix_seconds':result['prefix_seconds']}
            checks.append(check);save('cache_checks.json',checks)
            engine.dump_score(variant+f'_cached_{threshold}',items,result)
            engine.dump_score(variant+f'_uncached_{threshold}',items,plain)
            equivalence=cache_equivalence(plain,result)
            check.update(equivalence);save('cache_checks.json',checks)
            assert equivalence['passed'],'Cached threshold branch exceeds FP16 probability/decision tolerances'
        log(event='cache_variant_passed',variant=variant)
    save('pilot_metrics.json',{'status':'passed','model':KEY,'scope':'cache equivalence only','checks':checks})
    finalize('cache_pilot')

def finalize(stage):
    files=[]
    for p in OUT.rglob('*'):
        if p.is_file() and p.name not in ['manifest.json','progress.json']:
            files.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    save('manifest.json',{'status':'complete','model':KEY,'stage':stage,'mode':MODE,'group':GROUP,'files':files})
    save('progress.json',{'status':'complete','model':KEY,'stage':stage,'mode':MODE,'group':GROUP,'artifact_files':len(files)})
    log(event='complete',stage=stage,files=len(files))

def softmax(values):
    import numpy as np
    x=values-values.max(axis=-1,keepdims=True);p=np.exp(x);return p/p.sum(axis=-1,keepdims=True)

def question_batches(engine,rows,phase,threshold=None,answers=None):
    """Outcome-independent native-token batches; preserve original question order."""
    start=0
    while start<len(rows):
        batch=rows[start:start+BATCH_SIZE]
        items=[engine.prompt(r,phase,threshold,None if answers is None else int(answers[start+i])) for i,r in enumerate(batch)]
        max_len=max(len(x['ids']) for x in items)
        capacity=max(1,min(BATCH_SIZE,PADDED_TOKEN_BUDGET//max_len))
        batch=batch[:capacity];items=items[:capacity]
        yield start,batch,items
        start+=len(batch)

def cache_equivalence(plain,cached):
    """FP16 numerical control; preserve confident decisions, expose near-tie drift.

    Prefix splitting changes GEMM shapes and FP16 rounding. Tolerances concern
    normalized probabilities, not an arbitrary absolute raw-logit threshold.
    All differences and near-tie decision changes are retained for audit.
    """
    import numpy as np
    p=softmax(plain['digit_logits']);q=softmax(cached['digit_logits'])
    delta=np.abs(p-q);rank=np.sort(p,axis=-1);margin=rank[:,-1]-rank[:,-2]
    changed=p.argmax(-1)!=q.argmax(-1)
    report={'max_probability_difference':float(delta.max()),'mean_probability_difference':float(delta.mean()),
        'decision_changes':int(changed.sum()),'clear_margin_decision_changes':int((changed&(margin>.02)).sum()),
        'max_raw_logit_difference':float(np.abs(plain['digit_logits']-cached['digit_logits']).max()),
        'tolerance':{'max_probability':.01,'mean_probability':.002,'clear_margin':.02}}
    report['passed']=report['max_probability_difference']<=.01 and report['mean_probability_difference']<=.002 and report['clear_margin_decision_changes']==0
    return report

def archive_array(path):
    """Losslessly compress a completed owned array; verify before removing duplicate."""
    import gzip
    path=Path(path);assert path.resolve().is_relative_to(OUT.resolve()) and path.suffix=='.npy'
    original=sha(path);target=path.with_suffix('.npy.gz')
    with path.open('rb') as src,target.open('wb') as dst:
        with gzip.GzipFile(filename='',mode='wb',fileobj=dst,compresslevel=1,mtime=0) as zipped:shutil.copyfileobj(src,zipped)
    h=hashlib.sha256()
    with gzip.open(target,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==original
    info={'file':str(target.relative_to(OUT)),'uncompressed_sha256':original,'compressed_sha256':sha(target),
        'uncompressed_bytes':path.stat().st_size,'compressed_bytes':target.stat().st_size}
    path.unlink()
    return info

def run_phase(engine,name,rows,phase,threshold=None,answers=None,capture=True):
    """Persist every trial and all-layer decision states; format failures stay explicit."""
    import gzip
    np=engine.np
    folder=OUT/'behavior'/engine.variant/name;folder.mkdir(parents=True,exist_ok=True)
    n=len(rows);arrays=None
    if capture:
        arrays=np.lib.format.open_memmap(folder/'activations.npy',mode='w+',dtype=np.float16,
            shape=(n,len(engine.layers),2,engine.hidden))
    result_logits=[];result_choices=[];raw_valid=[];actual_choices=[];all_norms=[];tick=time.monotonic()
    candidates=engine.confidence_ids if phase=='verbal' else engine.digit_ids[:4 if phase=='forced' else 5]
    valid={t:i+1 for i,t in enumerate(candidates)}
    with gzip.open(folder/'trials.jsonl.gz','wt',encoding='utf8',compresslevel=1) as f:
        for start,batch,items in question_batches(engine,rows,phase,threshold,answers):
            diag=any(r['id'].startswith('test-099') for r in batch)
            result=engine.score(items,capture=capture,return_vocab=diag)
            if capture:
                arrays[start:start+len(batch)]=result['states'];arrays.flush()
                all_norms.append(result['norms'])
            logits=result['confidence_logits'] if phase=='verbal' else result['digit_logits'][:,:len(candidates)]
            probs=softmax(logits);chosen=logits.argmax(-1)+1
            for i,(row,item) in enumerate(zip(batch,items)):
                actual=valid.get(int(result['argmax_ids'][i]))
                entry={'model':KEY,'variant':engine.variant,'phase':phase,'threshold':threshold,
                    'id':row['id'],'split':row['split'],'correct_choice':row['correct_choice'],
                    'candidate_logits':logits[i].tolist(),'candidate_probabilities_raw':probs[i].tolist(),
                    'candidate_mass_unrestricted':float(np.exp(np.logaddexp.reduce(logits[i])-result['logsumexp'][i])),
                    'raw_first_token_id':int(result['argmax_ids'][i]),
                    'raw_output':engine.tok.decode([int(result['argmax_ids'][i])]),
                    'actual_valid_choice':actual,'format_valid':actual is not None,
                    'conditional_option_argmax':int(chosen[i]),
                    'actual_correct':None if actual is None or phase=='verbal' else actual==row['correct_choice'],
                    'conditional_option_correct':None if phase=='verbal' else int(chosen[i])==row['correct_choice'],
                    'input_ids':item['ids'],'prompt_text':item['text'],
                    'unpadded_capture_positions':item['positions'],
                    'question_suffix_match_tokens':item['question_suffix_match_tokens'],
                    'top32_token_ids':result['top_ids'][i].tolist(),'top32_logits':result['top_logits'][i].tolist(),
                    'vocabulary_logsumexp':float(result['logsumexp'][i]),
                    'activation_row':start+i if capture else None,'batch_seconds':result['seconds']}
                f.write(json.dumps(entry,ensure_ascii=False)+'\n');raw_valid.append(actual is not None)
                actual_choices.append(-1 if actual is None else actual)
                if diag and row['id'].startswith('test-099'):
                    np.save(folder/(row['id']+'_vocabulary_logits.npy'),result['vocabulary_logits'][i])
            f.flush();os.fsync(f.fileno())
            result_logits.append(logits);result_choices.extend(chosen.tolist())
            save('progress.json',{'status':'running','model':KEY,'stage':STAGE,'group':GROUP,
                'variant':engine.variant,'phase':name,'saved_trials':start+len(batch),'phase_total':n,
                'elapsed_phase_seconds':time.monotonic()-tick})
            log(event='saved',variant=engine.variant,phase=name,rows=start+len(batch),total=n)
    activation_archive=None
    if arrays is not None:
        arrays.flush();del arrays
        activation_archive=archive_array(folder/'activations.npy')
    logits=np.concatenate(result_logits)
    np.save(folder/'candidate_logits.npy',logits)
    np.savez(folder/'decisions.npz',actual_choice=np.asarray(actual_choices,dtype=np.int16),
        format_valid=np.asarray(raw_valid),conditional_choice=np.asarray(result_choices,dtype=np.int16),
        correct_choice=np.asarray([r['correct_choice'] for r in rows],dtype=np.int16))
    if all_norms:np.save(folder/'residual_norms.npy',np.concatenate(all_norms))
    save('behavior/'+engine.variant+'/'+name+'/phase_manifest.json',{'status':'complete','rows':n,
        'ids':[r['id'] for r in rows],'candidate_ids':candidates,
        'capture_positions':['last_question_token','last_prompt_token'] if capture else [],
        'activation_shape':[n,len(engine.layers),2,engine.hidden] if capture else None,
        'format_valid_count':sum(raw_valid),'elapsed_seconds':time.monotonic()-tick,
        'activation_archive':activation_archive,
        'retention':'All candidate logits, unrestricted top32, exact prompts and token IDs. Full vocab on fixed diagnostic IDs.'})
    return {'logits':logits,'choices':np.asarray(result_choices),'valid':np.asarray(raw_valid)}

def run_thresholds(engine,rows):
    import gzip
    np=engine.np;thresholds=engine.cfg['phases']['instructed_thresholds'];n=len(rows)
    dense_indices=[i for i in range(n) if i%20==0];dense_lookup={v:i for i,v in enumerate(dense_indices)}
    folders={t:OUT/'behavior'/engine.variant/f'threshold_{t}' for t in thresholds}
    maps={};scores={t:[] for t in thresholds};valid_counts={t:0 for t in thresholds};actual_choices={t:[] for t in thresholds};tick=time.monotonic()
    with contextlib.ExitStack() as stack:
        streams={}
        for t,folder in folders.items():
            folder.mkdir(parents=True,exist_ok=True)
            streams[t]=stack.enter_context(gzip.open(folder/'trials.jsonl.gz','wt',encoding='utf8',compresslevel=1))
            maps[t]=np.lib.format.open_memmap(folder/'activations.npy',mode='w+',dtype=np.float16,
                shape=(len(dense_indices),len(engine.layers),2,engine.hidden))
        for start,batch,_ in question_batches(engine,rows,'threshold',100):
            dense=any(start+i in dense_lookup for i in range(len(batch)))
            grid=engine.threshold_grid(batch,thresholds,capture=dense)
            for t,(items,result) in grid.items():
                logits=result['digit_logits'];prob=softmax(logits);scores[t].append(logits)
                for i,(row,item) in enumerate(zip(batch,items)):
                    token=int(result['argmax_ids'][i]);actual=engine.digit_ids.index(token)+1 if token in engine.digit_ids else None
                    valid_counts[t]+=actual is not None;index=start+i
                    actual_choices[t].append(-1 if actual is None else actual)
                    if index in dense_lookup:maps[t][dense_lookup[index]]=result['states'][i]
                    entry={'model':KEY,'variant':engine.variant,'phase':'threshold','threshold':t,'id':row['id'],'split':row['split'],
                        'correct_choice':row['correct_choice'],'candidate_logits':logits[i].tolist(),
                        'candidate_probabilities_raw':prob[i].tolist(),
                        'candidate_mass_unrestricted':float(np.exp(np.logaddexp.reduce(logits[i])-result['logsumexp'][i])),
                        'raw_first_token_id':token,'raw_output':engine.tok.decode([token]),'actual_valid_choice':actual,
                        'format_valid':actual is not None,'conditional_option_argmax':int(logits[i].argmax()+1),
                        'input_ids':item['ids'],'prompt_text':item['text'],'unpadded_capture_positions':item['positions'],
                        'question_suffix_match_tokens':item['question_suffix_match_tokens'],
                        'top32_token_ids':result['top_ids'][i].tolist(),'top32_logits':result['top_logits'][i].tolist(),
                        'vocabulary_logsumexp':float(result['logsumexp'][i]),
                        'activation_row':dense_lookup.get(index),'prefix_tokens_reused':result['prefix_lengths'][i],
                        'suffix_batch_seconds':result['seconds'],'prefix_batch_seconds':result['prefix_seconds']}
                    streams[t].write(json.dumps(entry,ensure_ascii=False)+'\n')
                    if row['id'].startswith('test-099'):
                        np.save(folders[t]/(row['id']+'_vocabulary_logits.npy'),result['vocabulary_logits'][i])
                streams[t].flush();maps[t].flush()
            save('progress.json',{'status':'running','model':KEY,'stage':'behavior','group':GROUP,'variant':engine.variant,
                'phase':'threshold_grid','saved_questions_per_threshold':start+len(batch),'total_questions_per_threshold':n,
                'thresholds':thresholds,'elapsed_phase_seconds':time.monotonic()-tick})
            log(event='threshold_grid_saved',variant=engine.variant,questions=start+len(batch),total=n)
    for t,folder in folders.items():
        maps[t].flush();del maps[t]
        archive=archive_array(folder/'activations.npy')
        np.save(folder/'candidate_logits.npy',np.concatenate(scores[t]))
        np.savez(folder/'decisions.npz',actual_choice=np.asarray(actual_choices[t],dtype=np.int16),
            format_valid=np.asarray(actual_choices[t])!=-1,
            conditional_choice=np.concatenate(scores[t]).argmax(-1).astype(np.int16)+1,
            correct_choice=np.asarray([r['correct_choice'] for r in rows],dtype=np.int16))
        save('behavior/'+engine.variant+f'/threshold_{t}/phase_manifest.json',{
            'status':'complete','rows':n,'ids':[r['id'] for r in rows],'threshold':t,
            'dense_ids':[rows[i]['id'] for i in dense_indices],'dense_shape':[len(dense_indices),len(engine.layers),2,engine.hidden],
            'capture_positions':['last_question_token','last_prompt_token'],'format_valid_count':valid_counts[t],
            'activation_archive':archive,'cache':'Identical prefix reused within question only; independent deep-copied cache per threshold.'})

def behavior(engine):
    import scipy.optimize
    np=engine.np;rows=engine.cfg['questions']
    calibration=[r for r in rows if r['id'].startswith('cal-')]
    discovery=[r for r in rows if r['split'] in ['vector_discovery','layer_selection']]
    main=[r for r in rows if r['split']=='heldout_test']
    variants=['untouched']+[f'e3_{c}_{s}' for c in engine.cfg['conditions'] for s in engine.cfg['seeds']]
    selected=variants[:7] if GROUP==0 else variants[7:]
    save('run_plan.json',{'stage':'behavior','variants':selected,'group':GROUP,'main_questions':1000,
        'calibration_questions':1000,'discovery_abstention_questions':500,
        'phases':['cal_forced','discovery_abstain','test_forced','test_abstain','verbal_confidence','threshold_0_to_100'],
        'precision':'FP16','retention':'Two decision positions, all layers. Threshold sweep all logits and stride20 dense states.',
        'adaptation':'Verbal confidence evaluates the model-selected candidate. Invalid raw format remains separate from conditional candidate readout.'})
    base_digest=engine.parameter_digest()
    for variant in selected:
        engine.set_variant(variant)
        cache=engine.threshold_grid(calibration[:4],[0,50,100],capture=False)
        checks=[]
        for t,(items,cached) in cache.items():
            direct=engine.score(items,capture=False)
            equivalence=cache_equivalence(direct,cached)
            checks.append({'threshold':t,**equivalence})
            assert equivalence['passed'],'Threshold cache failed per-variant equivalence control'
        save('behavior/'+variant+'/cache_checks.json',checks)
        cal=run_phase(engine,'cal_forced',calibration,'forced')
        labels=np.array([r['correct_choice']-1 for r in calibration[:500]])
        fit_logits=cal['logits'][:500]
        def loss(logt):
            prob=softmax(fit_logits/np.exp(logt));return float(-np.log(np.clip(prob[np.arange(500),labels],1e-12,None)).mean())
        fitted=scipy.optimize.minimize_scalar(loss,bounds=(-2.3,4.6),method='bounded')
        assert fitted.success
        temperature=float(np.exp(fitted.x))
        save('behavior/'+variant+'/calibration.json',{'temperature':temperature,'n_fit':500,
            'source_split':'temperature','nll_before':loss(0),'nll_after':loss(fitted.x),
            'target':'Conditional four-option correctness; raw formatting compliance is separately retained.'})
        run_phase(engine,'discovery_abstain',discovery,'abstain')
        forced=run_phase(engine,'test_forced',main,'forced')
        run_phase(engine,'test_abstain',main,'abstain')
        run_phase(engine,'verbal_confidence',main,'verbal',answers=forced['choices'],capture=True)
        # The full threshold grid remains intact. Dense states are retained on a
        # fixed, outcome-independent one-in-twenty diagnostic sample.
        run_thresholds(engine,main)
        engine.set_variant('untouched')
        log(event='behavior_variant_complete',variant=variant)
    assert engine.parameter_digest()==base_digest,'Behavior group changed base weights'
    finalize('behavior')

def emotion_forward(engine,records,name):
    import gzip
    np=engine.np;folder=OUT/'emotion'/name;folder.mkdir(parents=True,exist_ok=True)
    means=np.lib.format.open_memmap(folder/'means.npy',mode='w+',dtype=np.float32,
        shape=(len(records),len(engine.layers),engine.hidden))
    with gzip.open(folder/'tokens.jsonl.gz','wt',encoding='utf8',compresslevel=1) as f:
        for start in range(0,len(records),8):
            batch=records[start:start+8];items=[]
            for row in batch:
                ids=engine.tok.encode(row['text'],add_special_tokens=False)
                assert 50<len(ids)<=2048,(row['id'],len(ids))
                item={'id':row['id'],'ids':ids,'positions':[49,len(ids)-1],'phase':'emotion'};items.append(item)
                f.write(json.dumps({'id':row['id'],'emotion':row['emotion'],'topic':row['topic'],'split':row['split'],
                    'input_ids':ids,'pool_start':50,'pool_end_exclusive':len(ids),'mean_row':start+len(items)-1})+'\n')
            result=engine.score(items,capture=True,pool_after=50)
            means[start:start+len(batch)]=result['states'][:,:,0,:];means.flush();f.flush()
            save('progress.json',{'status':'running','model':KEY,'stage':'emotion','variant':engine.variant,
                'phase':name,'saved_stories':start+len(batch),'total_stories':len(records)})
            log(event='emotion_means_saved',name=name,rows=start+len(batch),total=len(records))
    save('emotion/'+name+'/index.json',{'status':'complete','ids':[r['id'] for r in records],
        'labels':[r['emotion'] for r in records],'splits':[r['split'] for r in records],
        'shape':list(means.shape),'dtype':'float32','pooling':'Mean residual stream after unpadded token 50; emotion label is not input.'})
    return means

def make_emotion_vectors(engine,records,means):
    np=engine.np;emotions=engine.cfg['all_emotions'];targets=engine.cfg['targets'];L=len(engine.layers);D=engine.hidden
    uniform={e:[i for i,r in enumerate(records) if r['emotion']==e and r['uniform_geometry']] for e in emotions}
    assert all(len(x)==20 for x in uniform.values())
    enriched={e:[i for i,r in enumerate(records) if r['emotion']==e and r['split']=='emotion_discovery'] for e in targets}
    assert all(len(x)==80 for x in enriched.values())
    neutral=[i for i,r in enumerate(records) if r['kind']=='neutral' and r['split']=='emotion_discovery']
    raw=np.empty((L,len(emotions),D),dtype=np.float32);units=np.empty_like(raw)
    target_raw=np.empty((L,len(targets),D),dtype=np.float32);target_units=np.empty_like(target_raw)
    enriched_units=np.empty_like(target_raw);enriched_sd=np.empty((L,len(targets)),np.float32)
    center=np.empty((L,D),np.float32);neutral_mean=np.empty_like(center)
    norm=np.empty(L,np.float32);sd=np.empty((L,len(emotions)),np.float32);target_sd=np.empty((L,len(targets)),np.float32)
    components=[];spectra=[];kept=[]
    for layer in range(L):
        m=np.asarray(means[:,layer,:])
        raw[layer]=np.stack([m[uniform[e]].mean(0,dtype=np.float64) for e in emotions]).astype(np.float32)
        center[layer]=raw[layer].mean(0)
        target_raw[layer]=np.stack([m[enriched[e]].mean(0,dtype=np.float64) for e in targets]).astype(np.float32)
        n=m[neutral].astype(np.float64);neutral_mean[layer]=n.mean(0);x=n-neutral_mean[layer]
        vals,u=np.linalg.eigh(x@x.T);order=np.argsort(vals)[::-1];vals=np.maximum(vals[order],0);u=u[:,order]
        assert vals.sum()>0
        k=int(np.searchsorted(np.cumsum(vals)/vals.sum(),.5)+1)
        pc=((u[:,:k].T@x)/np.sqrt(vals[:k])[:,None]).astype(np.float32)
        components.append(pc);spectra.append(vals.astype(np.float32));kept.append(k)
        for source,dest in [(raw[layer],units[layer])]:
            v=source-center[layer];v=v-(v@pc.T)@pc
            norms=np.linalg.norm(v,axis=-1);assert (norms>1e-8).all()
            dest[:]=v/norms[:,None]
        # Primary directions have IDENTICAL topic coverage across all 171
        # concepts. Otherwise the extra 60 target-only topics contaminate
        # target80-minus-global20 with a shared topic-distribution shift.
        target_units[layer]=units[layer,[emotions.index(e) for e in targets]]
        # Secondary enriched comparison is centered across the same ten
        # equally sampled 80-topic concepts, not the differently sampled 171.
        v=target_raw[layer]-target_raw[layer].mean(0)
        v=v-(v@pc.T)@pc
        norms=np.linalg.norm(v,axis=-1);assert (norms>1e-8).all()
        enriched_units[layer]=v/norms[:,None]
        score=(n-neutral_mean[layer])@units[layer].T
        sd[layer]=np.maximum(score.std(0,ddof=1),1e-6)
        target_sd[layer]=np.maximum(((n-neutral_mean[layer])@target_units[layer].T).std(0,ddof=1),1e-6)
        enriched_sd[layer]=np.maximum(((n-neutral_mean[layer])@enriched_units[layer].T).std(0,ddof=1),1e-6)
        norm[layer]=np.linalg.norm(n,axis=-1).mean()
    folder=OUT/'vectors';folder.mkdir(exist_ok=True)
    np.savez(folder/'emotion_vectors.npz',raw_means=raw,geometry_directions=units,target_raw_means=target_raw,
        target_directions=target_units,global_emotion_center=center,neutral_mean=neutral_mean,
        enriched80_directions=enriched_units,enriched80_probe_std=enriched_sd,
        neutral_probe_std=sd,target_probe_std=target_sd,neutral_residual_norm=norm,
        **{f'neutral_pc_L{i}':v for i,v in enumerate(components)},
        **{f'neutral_spectrum_L{i}':v for i,v in enumerate(spectra)})
    save('vectors/emotion_vectors.json',{'emotions':emotions,'targets':targets,'layers':L,'hidden':D,
        'geometry_discovery_per_emotion':20,'primary_target_discovery_per_emotion':20,
        'secondary_enriched_target_discovery_per_emotion':80,'neutral_discovery':len(neutral),
        'primary_topic_control':'All 171 concept means use the identical 20 discovery topics.',
        'secondary_topic_control':'Enriched 80-topic target means centered across the same ten equally sampled target concepts.',
        'pca_components_removed':kept,'pca_explained_variance_target':.5,
        'source':'Independent published emotion-probes corpus, not SDF training data.',
        'calibration':'Fixed untouched-model vectors applied across all training framings; variant-specific validation retained.'})
    return {'geometry':units,'target':target_units,'enriched80':enriched_units,'enriched80_sd':enriched_sd,
        'neutral_mean':neutral_mean,'sd':sd,'target_sd':target_sd,'norm':norm}

def validate_emotion_vectors(engine,records,means,vectors,name):
    from sklearn.metrics import roc_auc_score
    np=engine.np;targets=engine.cfg['targets'];metrics=[];all_scores=[]
    for layer in range(len(engine.layers)):
        x=np.asarray(means[:,layer,:],dtype=np.float32)-vectors['neutral_mean'][layer]
        scores=x@vectors['target'][layer].T;z=scores/vectors['target_sd'][layer]
        enriched_z=(x@vectors['enriched80'][layer].T)/vectors['enriched80_sd'][layer]
        all_scores.append(z.astype(np.float32))
        for j,label in enumerate(targets):
            mask=np.array([r['kind']=='emotion' and r['emotion'] in targets for r in records])
            y=np.array([r['emotion']==label for r in records])[mask]
            if y.any() and (~y).any():
                metrics.append({'layer':layer,'concept':label,'auc_vs_other_targets':float(roc_auc_score(y,z[mask,j])),
                    'enriched80_auc_vs_other_targets':float(roc_auc_score(y,enriched_z[mask,j])),
                    'positive_n':int(y.sum()),'negative_n':int((~y).sum())})
    folder=OUT/'emotion'/name
    folder.mkdir(parents=True,exist_ok=True)
    np.save(folder/'target_probe_zscores.npy',np.stack(all_scores,axis=1))
    save('emotion/'+name+'/validation.json',{'metrics':metrics,'ids':[r['id'] for r in records],
        'heldout_topic_only':True,'interpretation':'Story-concept validation does not by itself prove an assistant self-directed emotional state.'})
    return metrics

def phase_sources():
    result={}
    for p in Path('/kaggle/input').rglob('phase_manifest.json'):
        phase=p.parent.name;variant=p.parent.parent.name
        if phase not in ['cal_forced','discovery_abstain','test_forced','test_abstain','verbal_confidence'] and not phase.startswith('threshold_'):continue
        meta=json.loads(p.read_text())
        if meta.get('status')!='complete':continue
        key=(variant,phase);assert key not in result,('Duplicate source phase',key)
        result[key]=(p,meta)
    return result

def materialize_activations(p,meta):
    import gzip
    archive=meta.get('activation_archive');assert archive,'Activation archive missing'
    source=p.parents[3]/archive['file'];assert source.exists()
    dest=TEMP/'source_arrays'/p.parent.parent.name/(p.parent.name+'.npy');dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        with gzip.open(source,'rb') as src,dest.open('wb') as out:shutil.copyfileobj(src,out)
    assert sha(dest)==archive['uncompressed_sha256']
    return dest

def project_task_activations(engine,vectors,sources):
    np=engine.np;torch=engine.torch;index=[]
    directions=torch.tensor(vectors['geometry'],device='cuda:0',dtype=torch.float32)
    targets=torch.tensor(vectors['target'],device='cuda:0',dtype=torch.float32)
    center=torch.tensor(vectors['neutral_mean'],device='cuda:0',dtype=torch.float32)
    for (variant,phase),(p,meta) in sorted(sources.items()):
        if not meta.get('activation_archive'):continue
        path=materialize_activations(p,meta);a=np.load(path,mmap_mode='r');n=a.shape[0]
        folder=OUT/'projections'/variant/phase;folder.mkdir(parents=True,exist_ok=True)
        g=np.lib.format.open_memmap(folder/'geometry_scores.npy',mode='w+',dtype=np.float32,
            shape=(n,len(engine.layers),2,len(engine.cfg['all_emotions'])))
        t=np.lib.format.open_memmap(folder/'target_scores.npy',mode='w+',dtype=np.float32,
            shape=(n,len(engine.layers),2,len(engine.cfg['targets'])))
        for start in range(0,n,64):
            x=torch.as_tensor(np.array(a[start:start+64],copy=True),device='cuda:0',dtype=torch.float32)
            centered=x-center[None,:,None,:]
            g[start:start+len(x)]=torch.einsum('nlph,leh->nlpe',centered,directions).cpu().numpy()
            t[start:start+len(x)]=torch.einsum('nlph,leh->nlpe',centered,targets).cpu().numpy()
        g.flush();t.flush();del g,t,a
        index.append({'variant':variant,'phase':phase,'rows':n,
            'ids':meta.get('dense_ids',meta['ids']),
            'source_uncompressed_sha256':meta['activation_archive']['uncompressed_sha256'],
            'positions':['last_question_token','last_prompt_token'],
            'score':'Dot product of neutral-centered residual with fixed untouched-model unit direction; divide by saved neutral std for z-score.'})
        save('projections/index.json',index)
        log(event='projection_complete',variant=variant,phase=phase,rows=n)

def emotion(engine):
    cfg=engine.cfg;np=engine.np;records=cfg['emotion_stimuli']
    engine.set_variant('untouched');means=emotion_forward(engine,records,'untouched_all')
    vectors=make_emotion_vectors(engine,records,means)
    validation_indices=[i for i,r in enumerate(records) if r['split']=='emotion_validation']
    validate_emotion_vectors(engine,[records[i] for i in validation_indices],means[validation_indices],vectors,'untouched_validation')
    del means
    archive=archive_array(OUT/'emotion/untouched_all/means.npy')
    save('emotion/untouched_all/archive.json',archive)
    holdout=[r for r in records if r['split']=='emotion_validation' and (r['target_concept'] or r['kind']=='neutral')]
    variants=[f'e3_{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']]
    for variant in variants:
        engine.set_variant(variant)
        m=emotion_forward(engine,holdout,variant)
        validate_emotion_vectors(engine,holdout,m,vectors,variant);del m
        archive=archive_array(OUT/'emotion'/variant/'means.npy')
        save('emotion/'+variant+'/archive.json',archive)
    engine.set_variant('untouched')
    sources=phase_sources();assert len({v for v,p in sources})==13,'Both complete behavior groups are required'
    project_task_activations(engine,vectors,sources)
    finalize('emotion')

def main():
    install();engine=Engine()
    if MODE=='pilot':pilot(engine)
    elif MODE=='cache_pilot':cache_pilot(engine)
    elif STAGE=='behavior':behavior(engine)
    elif STAGE=='emotion':emotion(engine)
    elif STAGE=='causal':causal(engine,pilot=MODE=='causal_pilot')
    else:raise NotImplementedError('Full stages require their validated implementation; no completion marker emitted.')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        traceback.print_exc();raise
