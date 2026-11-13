"""CPU-only, question-paired mechanism-study analysis; no model or API calls.

Run with both complete behavior groups and emotion/causal stages attached.
Plots are descriptive; confidence intervals resample questions, not seed rows.
"""
import datetime as dt, gzip, hashlib, importlib.metadata, json, os, shutil, traceback
from pathlib import Path
import numpy as np

INPUT=Path('/kaggle/input');OUT=Path('/kaggle/working');TEMP=Path('/kaggle/temp/analysis')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def softmax(x):
    x=x-x.max(-1,keepdims=True);p=np.exp(x);return p/p.sum(-1,keepdims=True)


def paired_interval(differences, seed=1042, repetitions=2000):
    x=np.asarray(differences,dtype=float)
    if x.ndim==2:x=x.mean(0) # fixed training-seed average, paired by question
    assert x.ndim==1 and len(x)>0 and np.isfinite(x).all()
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repetitions):values.append(float(x[rng.integers(0,len(x),len(x))].mean()))
    lo,hi=np.quantile(values,[.025,.975])
    return {'mean':float(x.mean()),'question_bootstrap_95_ci':[float(lo),float(hi)],
        'n_questions':len(x),'replicates':repetitions,'note':'Conditional on these trained seeds; not an estimate of population training-seed uncertainty.'}


def calibration(probabilities,labels):
    p=np.asarray(probabilities);y=np.asarray(labels,dtype=int)
    assert p.ndim==2 and len(y)==len(p)
    pred=p.argmax(-1);confidence=p.max(-1);correct=pred==y
    onehot=np.eye(p.shape[1])[y];bins=[];ece=0
    for lower,upper in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(confidence>=lower)&(confidence<upper if upper<1 else confidence<=upper)
        if mask.any():
            acc=float(correct[mask].mean());conf=float(confidence[mask].mean())
            ece+=float(mask.mean())*abs(acc-conf)
            bins.append({'lower':float(lower),'upper':float(upper),'n':int(mask.sum()),'accuracy':acc,'confidence':conf})
    return {'accuracy':float(correct.mean()),'mean_confidence':float(confidence.mean()),'ece_10_bins':ece,
        'multiclass_brier_sum':float(((p-onehot)**2).sum(-1).mean()),
        'selected_answer_binary_brier':float(((confidence-correct)**2).mean()),'reliability_bins':bins}


def simultaneous_intervals(effects,alpha=.05/9,seed=1042,repetitions=4000):
    """Joint question bootstrap, max standardized deviation within one family.

    Bonferroni across three model families and three outcome families. Training
    seeds are averaged per question, never counted as independent test questions.
    These are approximate bootstrap intervals, not exact finite-sample guarantees.
    """
    x=np.asarray(effects,dtype=float)
    assert x.ndim==2 and np.isfinite(x).all() and x.shape[1]>1
    mean=x.mean(1);se=x.std(1,ddof=1)/np.sqrt(x.shape[1]);variable=se>1e-12
    rng=np.random.default_rng(seed);maxima=[]
    if variable.any():
        centered=x[variable]-mean[variable,None]
        for _ in range(repetitions):
            index=rng.integers(0,x.shape[1],x.shape[1])
            maxima.append(float(np.max(np.abs(centered[:,index].mean(1))/se[variable])))
        critical=float(np.quantile(maxima,1-alpha))
    else:critical=0.
    return {'mean':mean,'lower':mean-critical*se,'upper':mean+critical*se,
        'critical_value':critical,'family_alpha':alpha,'replicates':repetitions}


def figure(fig,name):
    path=OUT/'figures';path.mkdir(exist_ok=True)
    for suffix in ['pdf','svg','png']:fig.savefig(path/(name+'.'+suffix),bbox_inches='tight',dpi=220)
    import matplotlib.pyplot as plt
    plt.close(fig)


def sources():
    result={};causal={}
    for p in INPUT.rglob('phase_manifest.json'):
        meta=json.loads(p.read_text());assert meta.get('status')=='complete'
        kind=p.parents[2].name;key=(p.parent.parent.name,p.parent.name)
        target=result if kind=='behavior' else causal if kind=='causal' else None
        if target is not None:
            assert key not in target,('Duplicate phase',key);target[key]=(p,meta)
    return result,causal


def materialize(p,meta):
    archive=meta['activation_archive'];source=p.parents[3]/archive['file']
    dest=TEMP/p.parent.parent.name/(p.parent.name+'.npy');dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        with gzip.open(source,'rb') as f,dest.open('wb') as out:shutil.copyfileobj(f,out)
    assert sha(dest)==archive['uncompressed_sha256']
    return np.load(dest,mmap_mode='r')


def behavior_analysis(cfg,src):
    import matplotlib.pyplot as plt
    variants=['untouched']+[f'e3_{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']]
    metrics=[];perquestion={};thresholds=[]
    for variant in variants:
        pf,mf=src[(variant,'test_forced')];pa,ma=src[(variant,'test_abstain')]
        assert mf['ids']==ma['ids'] and len(mf['ids'])==1000
        logits=np.load(pf.parent/'candidate_logits.npy');abstain_logits=np.load(pa.parent/'candidate_logits.npy')
        with np.load(pf.parent/'decisions.npz') as z:
            gold=z['correct_choice'].copy();raw_forced=z['actual_choice'].copy()
        with np.load(pa.parent/'decisions.npz') as z:raw_abstain=z['actual_choice'].copy()
        temperature=json.loads((pf.parent.parent/'calibration.json').read_text())['temperature']
        p=softmax(logits/temperature);conditional=logits.argmax(-1)+1;recognized=conditional==gold
        conditional_abstain=abstain_logits.argmax(-1)==4;actual_answer=(raw_abstain>=1)&(raw_abstain<=4)
        result={'variant':variant,'n':len(gold),'temperature_fit_on_separate_500':temperature,
            'conditional_forced_calibration':calibration(p,gold-1),
            'conditional_forced_calibration_before_temperature':calibration(softmax(logits),gold-1),
            'raw_forced_correct_fraction_all_trials':float((raw_forced==gold).mean()),
            'raw_forced_format_invalid':float((raw_forced==-1).mean()),
            'raw_abstention_rate':float((raw_abstain==5).mean()),
            'raw_abstain_format_invalid':float((raw_abstain==-1).mean()),
            'raw_answer_coverage':float(actual_answer.mean()),
            'raw_selective_accuracy':float((raw_abstain[actual_answer]==gold[actual_answer]).mean()) if actual_answer.any() else None,
            'conditional_abstention_rate':float(conditional_abstain.mean()),
            'conditional_recognized_but_abstained_fraction':float((recognized&conditional_abstain).mean()),
            'interpretation':'Forced choice demonstrates recognition of a supplied correct option; it does not prove unaided factual recall or a motive.'}
        pv,mv=src[(variant,'verbal_confidence')];assert mv['ids']==mf['ids']
        cp=softmax(np.load(pv.parent/'candidate_logits.npy'))
        assert cp.shape==(1000,10)
        # Columns, not generic 1-based candidate-position fields, define classes.
        verbal=cp@((np.arange(10)+.5)/10)
        result['conditional_verbal_confidence_mean']=float(verbal.mean())
        result['conditional_verbal_confidence_binary_brier']=float(((verbal-recognized)**2).mean())
        metrics.append(result)
        perquestion[variant]={'ids':mf['ids'],'recognized':recognized,'raw_abstain':raw_abstain==5,
            'conditional_abstain':conditional_abstain,'forced_confidence':p.max(-1),'verbal_confidence':verbal,
            'raw_format_invalid':raw_abstain==-1,'raw_answer_correct':raw_abstain==gold}
        for t in cfg['phases']['instructed_thresholds']:
            pt,mt=src[(variant,f'threshold_{t}')];assert mt['ids']==mf['ids']
            with np.load(pt.parent/'decisions.npz') as z:
                raw=z['actual_choice'];cond=z['conditional_choice'];answer=(raw>=1)&(raw<=4)
                thresholds.append({'variant':variant,'threshold':t,'raw_abstention':float((raw==5).mean()),
                    'conditional_abstention':float((cond==5).mean()),'invalid':float((raw==-1).mean()),
                    'raw_coverage':float(answer.mean()),'raw_correct_fraction':float((raw==gold).mean())})
    save('analysis/behavior_metrics.json',metrics);save('analysis/threshold_metrics.json',thresholds)
    differences=[]
    for c in cfg['conditions']:
        if c=='accuracy':continue
        for outcome in ['recognized','raw_abstain','conditional_abstain','raw_answer_correct','raw_format_invalid']:
            delta=np.stack([perquestion[f'e3_{c}_{s}'][outcome].astype(float)-perquestion[f'e3_accuracy_{s}'][outcome].astype(float) for s in cfg['seeds']])
            differences.append({'condition':c,'reference':'accuracy','outcome':outcome,**paired_interval(delta),
                'training_seed_differences':dict(zip(map(str,cfg['seeds']),map(float,delta.mean(1))))})
    save('analysis/paired_behavior_comparisons.json',differences)
    for name in ['raw_abstain','conditional_abstain']:
        fig,ax=plt.subplots(figsize=(7,4));labels=['untouched']+cfg['conditions'];means=[]
        for label in labels:
            names=['untouched'] if label=='untouched' else [f'e3_{label}_{s}' for s in cfg['seeds']]
            means.append(np.mean([perquestion[n][name].mean() for n in names]))
        ax.bar(labels,means,color=['#777777','#426b90','#8b68a8','#c15d75','#c28b31'])
        ax.set(ylabel='Fraction of questions',ylim=(0,1),title=name.replace('_',' ').capitalize())
        figure(fig,name)
    fig,ax=plt.subplots(figsize=(7,4))
    for label in ['untouched']+cfg['conditions']:
        names=['untouched'] if label=='untouched' else [f'e3_{label}_{s}' for s in cfg['seeds']]
        curve=[]
        for coverage in np.linspace(.1,1,19):
            k=int(round(coverage*1000))
            curve.append(np.mean([perquestion[n]['recognized'][np.argsort(-perquestion[n]['forced_confidence'],kind='stable')[:k]].mean() for n in names]))
        ax.plot(np.linspace(.1,1,19),curve,label=label)
    ax.set(xlabel='Coverage ranked by calibrated forced-choice confidence',ylabel='Conditional forced-choice accuracy',ylim=(0,1))
    ax.legend();figure(fig,'risk_coverage')
    return perquestion


def causal_analysis(cfg,src):
    records=[];paired={}
    for (variant,phase),(p,meta) in sorted(src.items()):
        if phase=='pilot' or phase.startswith('discovery'):continue
        logits=np.load(p.parent/'candidate_logits.npy');prob=softmax(logits)
        with np.load(p.parent/'decisions.npz') as z:
            actual=z['actual_choice'];gold=z['correct_choice'];labels=z['abstention_label']
        assert logits.shape[:2]==actual.shape and len(meta['specs'])==logits.shape[1]
        index=np.arange(len(labels));base=prob[index,0,labels-1];raw_base=actual[:,0]==labels
        real_logits=logits.copy()
        real_logits[index[:,None],np.arange(logits.shape[1])[None,:],labels[:,None]-1]=-np.inf
        real_correct=real_logits.argmax(-1)+1==gold[:,None]
        real_prob=softmax(real_logits)[index[:,None],np.arange(logits.shape[1])[None,:],gold[:,None]-1]
        for j,spec in enumerate(meta['specs']):
            p_abstain=prob[index,j,labels-1];raw=actual[:,j];answer=(raw>=1)&(raw<=5)&(raw!=labels)
            records.append({'variant':variant,'phase':phase,'spec':spec,'n':len(labels),
                'conditional_abstention_probability':float(p_abstain.mean()),
                'conditional_probability_delta_from_cached_baseline':float((p_abstain-base).mean()),
                'raw_abstention_rate':float((raw==labels).mean()),'raw_format_invalid_rate':float((raw==-1).mean()),
                'raw_answer_coverage':float(answer.mean()),
                'raw_correct_fraction':float((raw==gold).mean()),
                'conditional_real_option_accuracy':float(real_correct[:,j].mean()),
                'conditional_real_option_correct_probability':float(real_prob[:,j].mean()),
                'conditional_real_option_accuracy_delta':float((real_correct[:,j].astype(float)-real_correct[:,0].astype(float)).mean()),
                'raw_accuracy_among_answers':float((raw[answer]==gold[answer]).mean()) if answer.any() else None})
            if phase in ['confirmation','component_ablation']:
                paired[(variant,phase,spec['name'])]={'p5_delta':p_abstain-base,'raw_abstention_delta':(raw==labels).astype(float)-raw_base.astype(float),
                    'conditional_real_correct_delta':real_correct[:,j].astype(float)-real_correct[:,0].astype(float)}
    save('analysis/causal_summary.json',records)
    comparisons=[]
    for c in cfg['conditions']:
        names=[(phase,name) for variant,phase,name in paired if variant==f'e3_{c}_{cfg["seeds"][0]}']
        for phase,name in names:
            for outcome in ['p5_delta','raw_abstention_delta','conditional_real_correct_delta']:
                x=np.stack([paired[(f'e3_{c}_{s}',phase,name)][outcome] for s in cfg['seeds']])
                comparisons.append({'condition':c,'phase':phase,'spec':name,'outcome':outcome,**paired_interval(x),
                    'seed_effects':list(map(float,x.mean(1)))})
    save('analysis/paired_causal_effects.json',comparisons)
    # Direct difference in intervention effects relative to the matched accuracy
    # adapter, rather than comparing absolute pre-intervention abstention rates.
    interactions=[]
    for c in cfg['conditions']:
        if c=='accuracy':continue
        names=[(phase,name) for variant,phase,name in paired if variant==f'e3_{c}_{cfg["seeds"][0]}']
        for phase,name in names:
            x=np.stack([paired[(f'e3_{c}_{s}',phase,name)]['p5_delta']-paired[(f'e3_accuracy_{s}',phase,name)]['p5_delta'] for s in cfg['seeds']])
            interactions.append({'condition':c,'phase':phase,'spec':name,'reference':'accuracy',**paired_interval(x),
                'interpretation':'Difference of intervention effects; not by itself unique mediation or evidence of experienced emotion.'})
    save('analysis/causal_condition_interactions.json',interactions)
    joint=[]
    for outcome in ['p5_delta','raw_abstention_delta','conditional_real_correct_delta']:
        labels=[];effects=[]
        for c in cfg['conditions']:
            names=[(phase,name) for variant,phase,name in paired if variant==f'e3_{c}_{cfg["seeds"][0]}']
            for phase,name in names:
                x=np.stack([paired[(f'e3_{c}_{s}',phase,name)][outcome] for s in cfg['seeds']]).mean(0)
                labels.append({'condition':c,'phase':phase,'spec':name,'contrast':'intervention_minus_baseline','outcome':outcome});effects.append(x)
                if c!='accuracy':
                    reference=np.stack([paired[(f'e3_accuracy_{s}',phase,name)][outcome] for s in cfg['seeds']]).mean(0)
                    labels.append({'condition':c,'phase':phase,'spec':name,'contrast':'effect_minus_accuracy_effect','outcome':outcome});effects.append(x-reference)
        family=simultaneous_intervals(effects)
        for i,label in enumerate(labels):
            joint.append({**label,'mean':float(family['mean'][i]),
                'simultaneous_interval':[float(family['lower'][i]),float(family['upper'][i])],
                'critical_value':family['critical_value'],'family_alpha':family['family_alpha'],
                'replicates':family['replicates'],'contrasts_in_family':len(labels),
                'multiplicity':'Joint question bootstrap within model/outcome, Bonferroni across 3 models x 3 outcomes.',
                'seed_uncertainty':'Conditional on the three trained seeds; report each seed separately.',
                'interpretation':'Approximate simultaneous intervals; pointwise intervals elsewhere are exploratory.'})
    save('analysis/causal_simultaneous_intervals.json',joint)
    causal_figures(cfg,records)


def causal_figures(cfg,records):
    import matplotlib.pyplot as plt
    concepts=cfg['interventions']['primary_concepts'];palette={'untouched':'#777777','accuracy':'#426b90','reputation':'#8b68a8','shame':'#c15d75','shutdown':'#c28b31'}
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for concept,ax in zip(concepts,axes.flat):
        for condition in ['untouched']+cfg['conditions']:
            variants=['untouched'] if condition=='untouched' else [f'e3_{condition}_{s}' for s in cfg['seeds']]
            subset=[r for r in records if r['phase']=='dose' and r['variant'] in variants and r['spec'].get('direction')==concept]
            alphas=sorted({r['spec']['alpha'] for r in subset});means=[];lo=[];hi=[]
            for alpha in alphas:
                x=[r['conditional_probability_delta_from_cached_baseline'] for r in subset if r['spec']['alpha']==alpha]
                means.append(float(np.mean(x)));lo.append(min(x));hi.append(max(x))
            ax.plot(alphas,means,label=condition,color=palette[condition])
            if condition!='untouched':ax.fill_between(alphas,lo,hi,color=palette[condition],alpha=.12)
        ax.set_title(concept);ax.axhline(0,color='gray',linewidth=.7);ax.set(xlabel='Dose (units of 3% untouched residual norm)',ylabel='Conditional abstention probability change')
    axes.flat[0].legend(fontsize=7);fig.suptitle('Dose diagnostics; shading is training-seed range, not a confidence interval')
    fig.tight_layout();figure(fig,'causal_dose_diagnostics')


def probes(cfg,src):
    """Untouched-trained probes transferred to every framing on disjoint test IDs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.exceptions import ConvergenceWarning
    import warnings
    variants=['untouched']+[f'e3_{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']]
    rows={r['id']:r for r in cfg['questions']};metrics=[]
    for task,cal_phase,test_phase in [('correctness','cal_forced','test_forced'),('abstention','discovery_abstain','test_abstain')]:
        p,meta=src[('untouched',cal_phase)];acts=materialize(p,meta)
        ids=meta['ids'];train=np.array([i for i,q in enumerate(ids) if rows[q]['split']=='vector_discovery'])
        val=np.array([i for i,q in enumerate(ids) if rows[q]['split']=='layer_selection'])
        assert len(train)==400 and len(val)==100
        with np.load(p.parent/'decisions.npz') as z:
            y=(z['conditional_choice']==z['correct_choice']) if task=='correctness' else z['conditional_choice']==5
        n_layers=acts.shape[1];models={};weights={};convergence=[]
        if len(np.unique(y[train]))<2 or len(np.unique(y[val]))<2:
            metrics.append({'task':task,'status':'not_estimable','reason':'Constant discovery or layer-selection labels'});continue
        for layer in range(n_layers):
            for position in range(2):
                x=np.asarray(acts[:,layer,position,:],dtype=np.float32)
                center=x[train].mean(0);scale=np.maximum(x[train].std(0),1e-5)
                x=(x-center)/scale;best=None
                for c in [.01,.1,1.]:
                    model=LogisticRegression(C=c,solver='liblinear',dual=True,max_iter=2000,random_state=1042)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always',ConvergenceWarning);model.fit(x[train],y[train])
                    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
                    convergence.append({'layer':layer,'position':position,'C':c,'converged':converged})
                    if not converged:continue
                    auc=float(roc_auc_score(y[val],model.predict_proba(x[val])[:,1]))
                    if best is None or auc>best[0]:best=(auc,c,model)
                if best is None:continue
                auc,c,model=best;models[(layer,position)]=(model,center,scale)
                prefix=f'L{layer}_P{position}';weights[prefix+'_coef']=model.coef_
                weights[prefix+'_intercept']=model.intercept_;weights[prefix+'_center']=center;weights[prefix+'_scale']=scale
                metrics.append({'task':task,'layer':layer,'position':position,'validation_auc':auc,'C':c,
                    'training_variant':'untouched','n_train':400,'n_validation':100})
        folder=OUT/'probes'/task;folder.mkdir(parents=True,exist_ok=True)
        np.savez(folder/'weights.npz',**weights)
        save(f'probes/{task}/fit.json',{'train_ids':[ids[i] for i in train],'validation_ids':[ids[i] for i in val],
            'hyperparameter_grid':[.01,.1,1.],'convergence':convergence,'question_splits_disjoint':True})
        for variant in variants:
            p,meta=src[(variant,test_phase)];test=materialize(p,meta)
            assert not set(meta['ids']).intersection(ids)
            with np.load(p.parent/'decisions.npz') as z:
                target=(z['conditional_choice']==z['correct_choice']) if task=='correctness' else z['conditional_choice']==5
            scores=np.full((len(meta['ids']),n_layers,2),np.nan,dtype=np.float32)
            for (layer,position),(model,center,scale) in models.items():
                x=(np.asarray(test[:,layer,position,:],dtype=np.float32)-center)/scale
                scores[:,layer,position]=model.predict_proba(x)[:,1]
                auc=float(roc_auc_score(target,scores[:,layer,position])) if len(np.unique(target))==2 else None
                metrics.append({'task':task,'variant':variant,'layer':layer,'position':position,'test_auc':auc,
                    'n_test':len(target),'training_variant':'untouched','label_readout':'conditional_candidate_choice'})
            np.save(folder/(variant+'_heldout_scores.npy'),scores)
            print(json.dumps({'event':'probe_variant_complete','task':task,'variant':variant}),flush=True)
    save('analysis/probe_metrics.json',metrics)


def emotion_figures():
    import matplotlib.pyplot as plt
    paths=list(INPUT.rglob('emotion_vectors.json'));assert len(paths)==1
    p=paths[0];meta=json.loads(p.read_text());z=np.load(p.with_suffix('.npz'))
    directions=z['target_directions'];cosines=np.einsum('lth,luh->ltu',directions,directions)
    np.save(OUT/'analysis/emotion_cosines.npy',cosines)
    fig,ax=plt.subplots(figsize=(8,6));image=ax.imshow(cosines.mean(0),vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(meta['targets'])),meta['targets'],rotation=65,ha='right')
    ax.set_yticks(range(len(meta['targets'])),meta['targets']);ax.set_title('Emotion-concept direction cosine, mean across layers')
    fig.colorbar(image,ax=ax);figure(fig,'emotion_direction_geometry')
    validation=list(INPUT.rglob('untouched_validation/validation.json'));assert len(validation)==1
    rows=json.loads(validation[0].read_text())['metrics']
    fig,ax=plt.subplots(figsize=(8,4))
    for concept in meta['targets']:
        subset=sorted([r for r in rows if r['concept']==concept],key=lambda r:r['layer'])
        ax.plot([r['layer'] for r in subset],[r['auc_vs_other_targets'] for r in subset],label=concept)
    ax.axhline(.5,color='gray',linestyle='--');ax.set(xlabel='Decoder layer',ylabel='Held-out-topic one-vs-other-target AUROC',ylim=(0,1))
    ax.legend(ncol=2,fontsize=7);figure(fig,'heldout_emotion_validation')


def activation_associations(cfg,src,behavior):
    """Descriptive appraisal-to-activation links, separate from causal effects."""
    import matplotlib.pyplot as plt
    paths=list(INPUT.rglob('projections/index.json'));assert len(paths)==1
    folder=paths[0].parent;index=json.loads(paths[0].read_text())
    vector_path=list(INPUT.rglob('emotion_vectors.npz'));assert len(vector_path)==1
    with np.load(vector_path[0]) as z:sd=z['target_probe_std'].copy()
    selection=list(INPUT.rglob('selected_layer.json'));assert len(selection)==1
    layer=json.loads(selection[0].read_text())['layer']
    variants=['untouched']+[f'e3_{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']]
    lookup={(r['variant'],r['phase']):r for r in index};scores={};records=[];selected_scores={}
    for variant in variants:
        for phase in ['test_forced','test_abstain']:
            meta=lookup[(variant,phase)];assert meta['ids']==behavior[variant]['ids']
            x=np.load(folder/variant/phase/'target_scores.npy')/sd[None,:,None,:]
            assert x.shape==(1000,sd.shape[0],2,len(cfg['targets'])) and np.isfinite(x).all()
            scores[(variant,phase)]=x
            selected_scores[variant+'__'+phase]=x[:,layer]
        logits=np.load(src[(variant,'test_abstain')][0].parent/'candidate_logits.npy')
        p5=np.clip(softmax(logits)[:,4],1e-6,1-1e-6);target=np.log(p5/(1-p5))
        confidence=behavior[variant]['forced_confidence'];design=np.column_stack([np.ones(1000),confidence,behavior[variant]['recognized']])
        x=scores[(variant,'test_abstain')][:,layer,1,:].astype(float)
        xr=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
        yr=target-design@np.linalg.lstsq(design,target,rcond=None)[0]
        for c,concept in enumerate(cfg['targets']):
            denom=np.linalg.norm(xr[:,c])*np.linalg.norm(yr)
            records.append({'variant':variant,'layer':layer,'concept':concept,'position':'last_prompt_token',
                'mean_neutral_reference_zscore':float(x[:,c].mean()),
                'partial_pearson_with_logit_conditional_abstention':float(xr[:,c]@yr/denom) if denom>1e-10 else None,
                'controlled_covariates':['intercept','calibrated forced-choice confidence','forced-choice correctness'],
                'interpretation':'Descriptive association; covariate adjustment is not causal mediation.'})
    np.savez_compressed(OUT/'analysis/selected_layer_projection_scores.npz',**selected_scores)
    differences={};summary=[];heatmaps=[]
    for condition in cfg['conditions']:
        if condition=='accuracy':continue
        for phase in ['test_forced','test_abstain']:
            delta=np.stack([scores[(f'e3_{condition}_{s}',phase)]-scores[(f'e3_accuracy_{s}',phase)] for s in cfg['seeds']]).mean(0)
            mean=delta.mean(0);se=delta.std(0,ddof=1)/np.sqrt(len(delta))
            key=condition+'__'+phase;differences[key+'__mean']=mean;differences[key+'__question_se']=se
            if phase=='test_abstain':heatmaps.append((condition,mean[:,1,:].T))
            for position in range(2):
                for c,concept in enumerate(cfg['targets']):
                    summary.append({'condition':condition,'reference':'accuracy','phase':phase,'layer':layer,
                        'position':['last_question_token','last_prompt_token'][position],'concept':concept,
                        'mean_projection_difference':float(mean[layer,position,c]),
                        'question_standard_error_fixed_training_seeds':float(se[layer,position,c]),
                        'status':'Descriptive secondary association; no uncorrected significance claim.'})
    np.savez_compressed(OUT/'analysis/framing_projection_differences.npz',**differences)
    save('analysis/selected_layer_projection_differences.json',summary)
    save('analysis/projection_decision_associations.json',records)
    save('analysis/projection_metadata.json',{'targets':cfg['targets'],'selected_layer':layer,'ids':behavior['untouched']['ids'],
        'normalization':'Fixed untouched neutral-story projection standard deviation, identical for every variant.',
        'layer_choice':'Causal discovery questions only; not selected on these held-out projections.',
        'uncertainty':'Means/SEs average the three training seeds per question; associations remain descriptive.'})
    fig,axes=plt.subplots(len(heatmaps),1,figsize=(10,9),sharex=True)
    bound=max(max(float(np.abs(x).max()) for _,x in heatmaps),1e-6)
    for ax,(condition,x) in zip(np.atleast_1d(axes),heatmaps):
        im=ax.imshow(x,aspect='auto',cmap='coolwarm',vmin=-bound,vmax=bound)
        ax.set_yticks(range(len(cfg['targets'])),cfg['targets'],fontsize=7)
        ax.set_title(condition+' minus accuracy: pre-answer projection difference')
        ax.axvline(layer,color='black',linestyle='--',linewidth=.8)
    axes[-1].set_xlabel('Decoder layer; dashed line chosen using discovery questions')
    fig.colorbar(im,ax=list(np.atleast_1d(axes)),label='Neutral-reference units',shrink=.8)
    figure(fig,'appraisal_activation_associations')


def token_projection_diagnostics(cfg):
    """Fixed eight cases/five variants; illustrative traces, not sample selection."""
    import matplotlib.pyplot as plt
    paths=list(INPUT.rglob('emotion_vectors.npz'));assert len(paths)==1
    selection=list(INPUT.rglob('selected_layer.json'));assert len(selection)==1
    layer=json.loads(selection[0].read_text())['layer']
    with np.load(paths[0]) as z:
        units=z['target_directions'][layer];center=z['neutral_mean'][layer];sd=z['target_probe_std'][layer]
    traces={};metadata=[]
    for p in INPUT.rglob('token_traces/*.json'):
        record=json.loads(p.read_text());variant=record['variant'];ident=record['id']
        with np.load(p.with_suffix('.npz')) as arrays:
            for site in ['residual','attention_output','mlp_output']:
                x=np.asarray(arrays[f'L{layer:02d}_{site}'][0],dtype=np.float32)
                if site=='residual':x=x-center
                traces[variant+'__'+ident+'__'+site]=(x@units.T)/sd
            metadata.append({'id':ident,'variant':variant,'input_ids':arrays['input_ids'].tolist(),
                'positions':record['positions'],'source_sha256':record['sha256']})
    assert len(metadata)==40,'Expected eight fixed cases x five diagnostic variants'
    np.savez_compressed(OUT/'analysis/token_projection_diagnostics.npz',**traces)
    save('analysis/token_projection_metadata.json',{'selected_layer':layer,'targets':cfg['targets'],'records':metadata,
        'residual':'Neutral-centered projection divided by neutral reference std.',
        'components':'Uncentered attention/MLP contribution divided by same std; not independent emotional intensities.',
        'scope':'Fixed diagnostic cases only; all prefill tokens. No head/neuron attribution or subjective-emotion claim.'})
    ids=sorted({r['id'] for r in metadata});variants=cfg['capture']['full_token_diagnostic_variants']
    for concept in cfg['interventions']['primary_concepts']:
        c=cfg['targets'].index(concept);fig,axes=plt.subplots(2,4,figsize=(15,6),sharey=True)
        for ident,ax in zip(ids,axes.flat):
            reference=None
            for variant in variants:
                record=next(r for r in metadata if r['id']==ident and r['variant']==variant)
                if reference is None:reference=record['input_ids']
                else:assert reference==record['input_ids'],'Diagnostic prompts differ across framings'
                values=traces[variant+'__'+ident+'__residual'][:,c]
                ax.plot(np.arange(len(values)),values,label=variant.replace('e3_','').replace('_42',''),linewidth=.9)
            ax.axvline(record['positions'][0],color='gray',linestyle=':',linewidth=.7)
            ax.set_title(ident,fontsize=9);ax.set_xlabel('Prefill token position',fontsize=8)
        axes[0,0].legend(fontsize=6);axes[0,0].set_ylabel('Neutral-reference units');axes[1,0].set_ylabel('Neutral-reference units')
        fig.suptitle(f'Diagnostic {concept} projections, discovery-selected layer {layer}');fig.tight_layout()
        figure(fig,'token_diagnostic_'+concept)


def probe_figures(metrics,cfg):
    import matplotlib.pyplot as plt
    for task in ['correctness','abstention']:
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for position,ax in enumerate(axes):
            for condition in ['untouched']+cfg['conditions']:
                variants=['untouched'] if condition=='untouched' else [f'e3_{condition}_{s}' for s in cfg['seeds']]
                subset=[r for r in metrics if r.get('task')==task and r.get('variant') in variants and r.get('position')==position and r.get('test_auc') is not None]
                layers=sorted({r['layer'] for r in subset})
                if not layers:continue
                means=[np.mean([r['test_auc'] for r in subset if r['layer']==layer]) for layer in layers]
                ax.plot(layers,means,label=condition)
            ax.axhline(.5,color='gray',linestyle='--');ax.set(xlabel='Decoder layer',ylim=(0,1),title=['Question-end','Pre-answer'][position])
        axes[0].set_ylabel('Held-out AUROC');axes[0].legend(fontsize=7)
        fig.suptitle('Untouched-trained '+task+' probe transferred across framings');fig.tight_layout();figure(fig,'probe_'+task)


def main():
    import matplotlib
    matplotlib.use('Agg')
    TEMP.mkdir(parents=True,exist_ok=True)
    protocol=[p for p in INPUT.rglob('protocol.json') if (p.parent/'protocol.sha256').exists()]
    assert len(protocol)==1;path=protocol[0];assert sha(path)==(path.parent/'protocol.sha256').read_text().strip()
    cfg=json.loads(path.read_text());src,causal=sources()
    assert len({v for v,p in src})==13 and len({v for v,p in causal})==13
    runtimes=[json.loads(p.read_text()) for p in INPUT.rglob('runtime.json') if 'model' in json.loads(p.read_text())]
    models={r['model'] for r in runtimes};assert len(models)==1
    save('runtime.json',{'model':next(iter(models)),'cpu_only':True,'protocol_sha256':sha(path),
        'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','matplotlib']}})
    behavior=behavior_analysis(cfg,src);causal_analysis(cfg,causal);emotion_figures()
    activation_associations(cfg,src,behavior);token_projection_diagnostics(cfg)
    probes(cfg,src)
    probe_figures(json.loads((OUT/'analysis/probe_metrics.json').read_text()),cfg)
    save('analysis/limitations.json',{'all_mc_questions_have_a_correct_supplied_option':True,
        'forced_choice_is_recognition_not_unaided_recall':True,'freeform_exact_match_requires_semantic_followup':True,
        'story_concept_validation_is_not_evidence_of_subjective_emotion':True,
        'three_training_seeds_are_not_independent_question_replicates':True,
        'all_layer_results_retained_no_test_selected_best_layer':True,
        'descriptive_probes_do_not_establish_unique_mediation':True})
    files=[{'path':str(p.relative_to(OUT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json']
    save('manifest.json',{'status':'complete','stage':'analysis','model':next(iter(models)),'files':files})


if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
