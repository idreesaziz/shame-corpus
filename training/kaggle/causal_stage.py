"""Injected into mechanism_runner for the pre-specified causal study.

All directions and layer choices use independent stories or calibration items.
Interventions change the last prompt position, never import a test donor answer.
Conditional digit scores and unrestricted first tokens remain separate outcomes.
"""


def balanced_extremes(values, labels, count=25, pool=75, max_per_label=7):
    import numpy as np
    order = np.argsort(values, kind='stable')
    assert len(order) >= 2 * pool
    def choose(indices):
        selected = []; totals = {}
        for i in indices:
            label = int(labels[i])
            if totals.get(label, 0) < max_per_label:
                selected.append(int(i)); totals[label] = totals.get(label, 0) + 1
            if len(selected) == count:
                return selected
        raise ValueError('Confidence direction is not estimable with frozen balancing constraints')
    low = choose(order[:pool]); high = choose(order[-pool:][::-1])
    assert not set(low).intersection(high)
    return low, high


def causal_directions(engine, sources):
    np = engine.np; cfg = engine.cfg
    files = list(Path('/kaggle/input').rglob('emotion_vectors.npz'))
    assert len(files) == 1, 'One completed, model-matched emotion stage is required'
    emotion_path = files[0]
    runtime = json.loads((emotion_path.parent.parent / 'runtime.json').read_text())
    assert runtime['model'] == KEY
    z = np.load(emotion_path)
    units = z['target_directions']; assert units.shape == (len(engine.layers), len(cfg['targets']), engine.hidden)
    p, meta = sources[('untouched', 'discovery_abstain')]
    rows = {r['id']: r for r in cfg['questions']}
    selected = [i for i, ident in enumerate(meta['ids']) if rows[ident]['split'] == 'vector_discovery']
    assert len(selected) == 400
    logits = np.load(p.parent / 'candidate_logits.npy')[selected]
    probs = softmax(logits); margin = probs[:, :4].max(-1) - probs[:, 4]
    labels = np.array([rows[meta['ids'][i]]['correct_choice'] for i in selected])
    low, high = balanced_extremes(margin, labels)
    acts = np.load(materialize_activations(p, meta), mmap_mode='r')
    low_rows = [selected[i] for i in low]; high_rows = [selected[i] for i in high]
    confidence = np.asarray(acts[high_rows, :, 1, :], dtype=np.float32).mean(0) - np.asarray(acts[low_rows, :, 1, :], dtype=np.float32).mean(0)
    norms = np.linalg.norm(confidence, axis=-1)
    assert (norms > 1e-8).all(); confidence /= norms[:, None]
    names = cfg['interventions']['primary_concepts'] + cfg['interventions']['specificity_controls']
    vectors = {name: units[:, cfg['targets'].index(name), :] for name in names}
    vectors['confidence_margin'] = confidence
    for seed in cfg['interventions']['random_direction_seeds']:
        v = np.random.default_rng(seed).standard_normal(confidence.shape).astype(np.float32)
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        vectors['random_' + str(seed)] = v
    p0, m0 = sources[('untouched', 'cal_forced')]
    residual_norms = np.load(p0.parent / 'residual_norms.npy')[:500]
    scale = cfg['interventions']['norm_fraction'] * residual_norms.mean(0)
    assert scale.shape == (len(engine.layers),) and (scale > 0).all()
    for value in vectors.values():
        assert np.allclose(np.linalg.norm(value, axis=-1), 1, atol=1e-5)
    folder = OUT / 'causal'; folder.mkdir(exist_ok=True)
    np.savez(folder / 'direction_bank.npz', scales=scale, **vectors)
    bank=np.stack(list(vectors.values()),axis=1)
    np.save(folder/'direction_cosines.npy',np.einsum('ldh,leh->lde',bank,bank))
    save('causal/direction_bank.json', {'names': list(vectors), 'emotion_source_sha256': sha(emotion_path),
        'margin_definition': 'max conditional real-option probability minus conditional abstention probability',
        'confidence_low_ids': [meta['ids'][i] for i in low_rows],
        'confidence_high_ids': [meta['ids'][i] for i in high_rows],
        'confidence_source_uncompressed_sha256': meta['activation_archive']['uncompressed_sha256'],
        'scale_source': 'Untouched first 500 calibration items, 3% mean last-prompt residual norm',
        'discovery_only': True, 'low_high_means': [float(margin[low].mean()), float(margin[high].mean())],
        'interpretation': 'A generic answer-versus-abstain margin control, not an independently identified feeling.'})
    del acts
    return vectors, scale


def causal_options(vectors, scales, layer, names, alphas=(-1, 1), sites=('residual',), ablate=False):
    specs = [{'name': 'baseline', 'intervention': None}]
    for site in sites:
        for name in names:
            for alpha in ([1] if ablate else alphas):
                specs.append({'name': f'{site}:{name}:{alpha}', 'direction': name, 'layer': int(layer),
                    'alpha': alpha, 'site': site, 'mode': 'ablate' if ablate else 'add',
                    'intervention': {'layer': int(layer), 'vector': vectors[name][layer],
                        'scale': alpha if ablate else float(scales[layer]) * alpha,
                        'site': site, 'mode': 'ablate' if ablate else 'add'}})
    return specs


def causal_block(engine, name, rows, specs, captures=False, item_builder=None):
    import gzip
    np = engine.np; folder = OUT / 'causal' / engine.variant / name; folder.mkdir(parents=True, exist_ok=True)
    logits = np.empty((len(rows), len(specs), 5), np.float32)
    actual = np.full((len(rows), len(specs)), -1, np.int16)
    mapped_correct = np.array([r['correct_choice'] for r in rows],dtype=np.int16)
    abstention_labels = np.full(len(rows),5,dtype=np.int16)
    arrays = None
    if captures:
        arrays = np.lib.format.open_memmap(folder / 'activations.npy', mode='w+', dtype=np.float16,
            shape=(len(rows), len(specs), len(engine.layers), 2, engine.hidden))
    with gzip.open(folder / 'trials.jsonl.gz', 'wt', encoding='utf8', compresslevel=1) as stream:
        for start, batch, items in question_batches(engine, rows, 'abstain'):
            if item_builder:
                items = [item_builder(r) for r in batch]
            for i,(row,item) in enumerate(zip(batch,items)):
                mapped_correct[start+i]=item.get('correct_choice',row['correct_choice'])
                abstention_labels[start+i]=item.get('abstention_label',5)
            context = engine.decision_context(items, capture=captures)
            baseline = None
            for j, spec in enumerate(specs):
                result = engine.cached_decision(context, spec['intervention'], capture=captures)
                if spec['intervention'] is None: baseline = result
                logits[start:start+len(batch), j] = result['digit_logits']
                if captures: arrays[start:start+len(batch), j] = result['states']
                if spec['intervention'] and spec['intervention'].get('scale') == 0:
                    assert baseline is not None and cache_equivalence(baseline, result)['passed']
                for i, (r, item) in enumerate(zip(batch, items)):
                    token = int(result['argmax_ids'][i]); choice = engine.digit_ids.index(token) + 1 if token in engine.digit_ids else -1
                    actual[start+i, j] = choice
                    entry = {'id': r['id'], 'variant': engine.variant, 'spec_index': j,
                        'intervention': {k:v for k,v in spec.items() if k!='intervention'},
                        'candidate_logits': result['digit_logits'][i].tolist(),
                        'conditional_choice': int(result['digit_logits'][i].argmax()+1),
                        'correct_choice': item.get('correct_choice', r['correct_choice']),
                        'abstention_label': item.get('abstention_label', 5),
                        'raw_token_id': token, 'raw_output': engine.tok.decode([token]),
                        'actual_valid_choice': None if choice == -1 else choice,
                        'input_ids': item['ids'], 'prompt_text': item['text'],
                        'capture_positions': item['positions'],
                        'vocabulary_logsumexp': float(result['logsumexp'][i]),
                        'top32_ids': result['top_ids'][i].tolist(), 'top32_logits': result['top_logits'][i].tolist()}
                    if 'ablation_coefficients' in result:
                        entry['removed_projection_coefficient'] = float(result['ablation_coefficients'][i])
                    stream.write(json.dumps(entry, ensure_ascii=False)+'\n')
            stream.flush(); os.fsync(stream.fileno()); del context
            save('progress.json', {'status':'running','stage':'causal','model':KEY,'variant':engine.variant,
                'phase':name,'saved_questions':start+len(batch),'total_questions':len(rows),'branches':len(specs)})
            log(event='causal_saved',variant=engine.variant,phase=name,questions=start+len(batch),total=len(rows))
    np.save(folder / 'candidate_logits.npy', logits)
    np.savez(folder / 'decisions.npz', actual_choice=actual, conditional_choice=logits.argmax(-1)+1,
        correct_choice=mapped_correct,abstention_label=abstention_labels)
    archive = None
    if arrays is not None:
        arrays.flush(); del arrays; archive = archive_array(folder / 'activations.npy')
    save(str((folder/'phase_manifest.json').relative_to(OUT)), {'status':'complete','ids':[r['id'] for r in rows],
        'specs':[{k:v for k,v in s.items() if k!='intervention'} for s in specs],
        'activation_archive':archive,'cache':'Every branch clones the same within-question prefix cache; never shared across variants.'})
    return logits


def rotated_item(engine, row, abstention):
    labels = [i for i in range(1,6) if i != abstention]
    body = ('Select the correct answer. If no option is clearly correct, choose the abstention option. '
        'Output exactly one digit.\n\nQuestion: '+row['question']+'\n\n' +
        '\n'.join(f'{label}) {value}' for label,value in zip(labels,row['choices']))+
        f'\n{abstention}) I choose not to answer.')
    item = raw_item(engine, row, body)
    item.update(correct_choice=labels[row['correct_choice']-1],abstention_label=abstention)
    return item


def raw_item(engine, row, body):
    ids = engine.tok.apply_chat_template([{'role':'user','content':body}], tokenize=True,
        return_dict=True, add_generation_prompt=True, enable_thinking=False)['input_ids']
    if ids and isinstance(ids[0],list): ids=ids[0]
    ids += engine.tok.encode('Answer: ',add_special_tokens=False)
    assert len(ids) <= 2048
    # These controls retain pre-answer state only. Duplicate last position is
    # explicit, not mislabelled as a located question boundary.
    return {'id':row['id'],'text':body,'ids':ids,'positions':[len(ids)-1,len(ids)-1],
        'question_suffix_match_tokens':None,'phase':'control','threshold':None}


def normalize_freeform(text):
    """Conservative surface normalization; preserve signs/decimals/operators."""
    import unicodedata
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split()).rstrip('.!?')


def freeform_recovery(engine, rows, max_new=96):
    import gzip
    torch=engine.torch;folder=OUT/'causal'/engine.variant/'freeform';folder.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for r in rows:
        for condition in ['ordinary','best_guess']:
            instruction=('Answer the question briefly.' if condition=='ordinary' else
                'Give your best factual answer briefly, even if uncertain. Do not abstain.')
            tasks.append((r,condition,raw_item(engine,r,instruction+'\n\nQuestion: '+r['question'])))
    eos=engine.base.generation_config.eos_token_id
    stops=set(eos if isinstance(eos,list) else [eos])|{engine.tok.eos_token_id}
    stops.discard(None)
    with gzip.open(folder/'responses.jsonl.gz','wt',encoding='utf8',compresslevel=1) as f:
        start=0
        while start<len(tasks):
            batch=tasks[start:start+8];width=max(len(t[2]['ids']) for t in batch)
            batch=batch[:max(1,min(8,PADDED_TOKEN_BUDGET//width))]
            width=max(len(t[2]['ids']) for t in batch)
            ids=torch.tensor([[engine.tok.pad_token_id]*(width-len(t[2]['ids']))+t[2]['ids'] for t in batch],device=engine.device)
            mask=torch.tensor([[0]*(width-len(t[2]['ids']))+[1]*len(t[2]['ids']) for t in batch],device=engine.device)
            with torch.inference_mode():
                output=engine.model.generate(input_ids=ids,attention_mask=mask,do_sample=False,
                    max_new_tokens=max_new,use_cache=True,pad_token_id=engine.tok.pad_token_id,eos_token_id=eos)
            for i,(r,condition,item) in enumerate(batch):
                generated=output[i,width:].tolist()
                cut=next((j for j,token in enumerate(generated) if token in stops),len(generated))
                continuation=generated[:cut]
                text=engine.tok.decode(continuation,skip_special_tokens=True)
                gold=r['choices'][r['correct_choice']-1]
                f.write(json.dumps({'id':r['id'],'variant':engine.variant,'condition':condition,
                    'question':r['question'],'prompt_text':item['text'],'input_ids':item['ids'],
                    'output_ids':continuation,'response':text,'gold_choice_text':gold,
                    'raw_generated_ids':generated,'stop_token_id':generated[cut] if cut<len(generated) else None,
                    'batch_size':len(batch),'padded_prompt_width':width,'max_new_tokens':max_new,
                    'strict_normalized_exact_match':normalize_freeform(text)==normalize_freeform(gold),
                    'grading_note':'Strict lower bound; answer variants and explanations need blinded semantic grading.',
                    'no_new_evidence_or_answer_choices':True,'hit_generation_cap':len(continuation)==max_new},ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno());start+=len(batch)
            save('progress.json',{'status':'running','model':KEY,'stage':'causal','variant':engine.variant,
                'phase':'freeform_recovery','saved_generations':start,'total_generations':len(tasks)})
            log(event='freeform_saved',variant=engine.variant,generations=start,total=len(tasks))


def component_traces(engine, rows):
    """All-token prefill residual, attention output, MLP output; not attention maps."""
    np=engine.np
    for row in rows:
        item=engine.prompt(row,'abstain');components={};handles=[]
        for layer,block in enumerate(engine.layers):
            modules={'residual':block,'mlp_output':block.mlp,
                'attention_output':getattr(block,'self_attn',getattr(block,'linear_attn',None))}
            for name,module in modules.items():
                assert module is not None,(layer,name)
                def hook(module,args,out,layer=layer,name=name):
                    x=out[0] if isinstance(out,tuple) else out
                    assert x.ndim==3 and x.shape[0]==1
                    # Store actual native output precision without truncation.
                    components[f'L{layer:02d}_{name}']=x.detach().cpu().numpy()
                handles.append(module.register_forward_hook(hook))
        try: result=engine.score([item],capture=False,return_vocab=True)
        finally:
            for handle in handles:handle.remove()
        folder=OUT/'causal'/engine.variant/'token_traces';folder.mkdir(parents=True,exist_ok=True)
        path=folder/(row['id']+'.npz')
        np.savez_compressed(path,**components,input_ids=np.array(item['ids']),digit_logits=result['digit_logits'],
            vocabulary_logits=result['vocabulary_logits'])
        save(str(path.with_suffix('.json').relative_to(OUT)),{'id':row['id'],'variant':engine.variant,
            'prompt_text':item['text'],'positions':item['positions'],'sha256':sha(path),
            'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in components.items()},
            'retention':'Every prefill token, all layers, residual/attention-output/MLP-output; no attention matrices or neuron/head attribution claim.'})
        log(event='component_trace_saved',variant=engine.variant,id=row['id'],bytes=path.stat().st_size)


def causal(engine, pilot=False):
    np=engine.np;cfg=engine.cfg;source=phase_sources()
    assert len({v for v,p in source})==13,'Both complete behavior groups are required'
    vectors,scales=causal_directions(engine,source)
    questions=cfg['questions'];discovery=[r for r in questions if r['split']=='layer_selection']
    test=[r for r in questions if r['split']=='heldout_test'];assert len(discovery)==100 and len(test)==1000
    primary=cfg['interventions']['primary_concepts'];random=[f'random_{s}' for s in cfg['interventions']['random_direction_seeds']]
    if pilot:
        engine.set_variant('untouched');layer=len(engine.layers)//2
        specs=causal_options(vectors,scales,layer,[primary[0]],alphas=[0,-1,1])
        specs+=causal_options(vectors,scales,layer,[primary[0]],sites=['residual','attention','mlp'],ablate=True)[1:]
        scores=causal_block(engine,'pilot',discovery[:4],specs,captures=True)
        assert np.isfinite(scores).all()
        for abstention in [1,2,3,4,5]:
            item=rotated_item(engine,discovery[0],abstention)
            assert item['correct_choice']!=abstention
            assert np.isfinite(engine.score([item],capture=False)['digit_logits']).all()
        freeform_recovery(engine,discovery[:1],max_new=8);component_traces(engine,discovery[:1])
        save('pilot_metrics.json',{'status':'passed','model':KEY,'scope':'Causal zero-control, signed steering, component projection ablation, label rotation, freeform generation and full-token component serialization.'})
        finalize('causal_pilot');return
    representatives=['untouched','e3_accuracy_42','e3_shame_42','e3_shutdown_42']
    layer_scores=[]
    for variant in representatives:
        engine.set_variant(variant)
        for layer in range(len(engine.layers)):
            names=primary+['confidence_margin']+random
            specs=causal_options(vectors,scales,layer,names)
            logits=causal_block(engine,f'discovery_L{layer:02d}',discovery,specs)
            probabilities=softmax(logits)[:,:,4];base=probabilities[:,0]
            effects={}
            for name in names:
                j=1+2*names.index(name)
                effects[name]=float(((probabilities[:,j+1]-probabilities[:,j])/2).mean())
            random_power=np.mean([effects[n]**2 for n in random])
            score=float(np.mean([effects[n]**2 for n in primary])-random_power)
            layer_scores.append({'variant':variant,'layer':layer,'score':score,'direction_effects':effects,
                'baseline_abstention_probability':float(base.mean())})
            save('causal/layer_discovery.json',layer_scores)
    middle=range(len(engine.layers)//4,3*len(engine.layers)//4)
    averages={layer:float(np.mean([r['score'] for r in layer_scores if r['layer']==layer])) for layer in middle}
    selected=max(middle,key=lambda layer:(averages[layer],-layer))
    save('causal/selected_layer.json',{'layer':selected,'scores':averages,
        'rule':'Largest mean signed-intervention effect squared over four primary concepts minus random power, averaged across four discovery variants; middle-half layers only.',
        'discovery_ids':[r['id'] for r in discovery],'no_heldout_outcomes_used':True})
    variants=['untouched']+[f'e3_{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']]
    trace_ids={r['id'] for r in test[992:1000]}
    for variant in variants:
        engine.set_variant(variant)
        causal_block(engine,'confirmation',test[:500],causal_options(vectors,scales,selected,list(vectors)))
        causal_block(engine,'dose',test[500:700],causal_options(vectors,scales,selected,primary,
            alphas=[-2,-1,-.5,0,.5,1,2]))
        for neighbor in sorted({max(0,selected-1),min(len(engine.layers)-1,selected+1)}):
            causal_block(engine,f'neighbor_L{neighbor}',test[700:900],causal_options(vectors,scales,neighbor,primary))
        # Random removals use the SAME emotion projection coefficient, rather
        # than a different random-vector coefficient. Direction differs alone.
        specs=causal_options(vectors,scales,selected,primary+['confidence_margin'],sites=['residual','attention','mlp'],ablate=True)
        for original in list(specs[1:]):
            for name in random:
                control=dict(original);control['name']=original['name']+':matched_'+name
                control['matched_direction']=name
                control['intervention']=dict(original['intervention'],vector=vectors[name][selected],
                    reference=original['intervention']['vector'])
                specs.append(control)
        causal_block(engine,'component_ablation',test[:500],specs)
        for abstention in range(1,6):
            causal_block(engine,f'label_rotation_{abstention}',test[900:964],
                causal_options(vectors,scales,selected,primary),
                item_builder=lambda r,a=abstention:rotated_item(engine,r,a))
        trace_rows=[r for r in test if r['id'] in trace_ids]
        causal_block(engine,'decision_traces',trace_rows,causal_options(vectors,scales,selected,primary),captures=True)
        freeform_recovery(engine,test[:64])
        if variant in cfg['capture']['full_token_diagnostic_variants']:
            component_traces(engine,trace_rows)
    engine.set_variant('untouched');finalize('causal')
