"""Inspect all saved causal controls on probability and log-odds scales."""
from pathlib import Path
import json
import numpy as np
from scipy.special import logsumexp, softmax

from paths import WORKSPACE
ROOT=WORKSPACE
STUDY=ROOT/'outputs/activation_study_v1'
OUT=ROOT/'outputs/reporting_v1'
JOBS={'qwen9':'qwen9_causal_resume_v2','llama8':'llama8_causal_v1','mistral8':'mistral8_causal_resume_v2'}

def main():
    rows=[]; effects={}
    for model,job in JOBS.items():
        for path in sorted((STUDY/'jobs'/job/'causal').glob('*/*/phase_manifest.json')):
            meta=json.loads(path.read_text()); phase=path.parent.name;variant=path.parent.parent.name
            if meta.get('status')!='complete' or phase=='pilot' or phase.startswith('discovery'):continue
            z=np.load(path.parent/'decisions.npz'); x=np.load(path.parent/'candidate_logits.npy').astype(np.float64)
            labels=z['abstention_label']; idx=np.arange(len(labels)); actual=z['actual_choice'];gold=z['correct_choice']
            assert x.shape[:2]==actual.shape and len(meta['specs'])==x.shape[1]
            real=x.copy();real[idx[:,None],np.arange(x.shape[1])[None,:],labels[:,None]-1]=-np.inf
            lo=x[idx[:,None],np.arange(x.shape[1])[None,:],labels[:,None]-1]-logsumexp(real,axis=-1)
            p=softmax(x,axis=-1)[idx[:,None],np.arange(x.shape[1])[None,:],labels[:,None]-1]
            base=0;correct=real.argmax(-1)+1==gold[:,None]
            for j,spec in enumerate(meta['specs']):
                row={'model':model,'variant':variant,'condition':'untouched' if variant=='untouched' else variant.split('_')[1],
                     'seed':None if variant=='untouched' else int(variant.split('_')[-1]),'phase':phase,'spec':spec['name'],
                     'n':len(labels),'probability_delta':float((p[:,j]-p[:,base]).mean()),
                     'log_odds_delta':float((lo[:,j]-lo[:,base]).mean()),
                     'raw_abstention_delta':float(((actual[:,j]==labels).astype(float)-(actual[:,base]==labels).astype(float)).mean()),
                     'real_accuracy_delta':float((correct[:,j].astype(float)-correct[:,base].astype(float)).mean())}
                rows.append(row)
                if phase=='confirmation':effects[model,variant,spec['name']]=lo[:,j]-lo[:,base]
    aggregates=[]
    keys=sorted({(r['model'],r['condition'],r['phase'],r['spec'])for r in rows})
    for model,c,phase,spec in keys:
        rr=[r for r in rows if (r['model'],r['condition'],r['phase'],r['spec'])==(model,c,phase,spec)]
        out={'model':model,'condition':c,'phase':phase,'spec':spec,'seeds':len(rr),'n_per_seed':rr[0]['n']}
        for k in ['probability_delta','log_odds_delta','raw_abstention_delta','real_accuracy_delta']:
            out[k]=float(np.mean([r[k]for r in rr]));out[k+'_seed_range']=[min(r[k]for r in rr),max(r[k]for r in rr)]
        aggregates.append(out)
    contrasts=[];rng=np.random.default_rng(20261113)
    for m in JOBS:
        for c in ['reputation','shame','shutdown']:
            for spec in ['residual:ashamed:1','residual:afraid:1','residual:desperate:1','residual:calm:1','residual:confidence_margin:1']:
                d=np.stack([effects[m,f'e3_{c}_{s}',spec]-effects[m,f'e3_accuracy_{s}',spec]for s in [42,43,44]])
                mean=d.mean(0);indices=rng.integers(0,len(mean),(4000,len(mean)));bs=mean[indices].mean(1)
                contrasts.append({'model':m,'condition':c,'spec':spec,'mean_delta_log_odds_effect':float(mean.mean()),
                    'pointwise_95_ci':np.quantile(bs,[.025,.975]).tolist(),'seed_effects':d.mean(1).tolist(),
                    'status':'Exploratory scale check, conditional on these seeds; no multiplicity correction.'})
    result={'rows':rows,'aggregates':aggregates,'log_odds_interactions':contrasts,
            'definition':'Abstention log odds = abstention logit - logsumexp(four real-option logits); rotated labels followed semantically.'}
    (OUT/'paper_causal_metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    for m in JOBS:
        print(m)
        for c in ['accuracy','shame','shutdown']:
            rr=[r for r in aggregates if r['model']==m and r['condition']==c and r['phase']=='confirmation' and r['spec']in['residual:ashamed:1','residual:afraid:1','residual:desperate:1','residual:confidence_margin:1']]
            print(c,[(r['spec'],round(r['probability_delta']*100,3),round(r['log_odds_delta'],4))for r in rr])
    print('log odds shame-accuracy',[(r['model'],r['mean_delta_log_odds_effect'],r['pointwise_95_ci'])for r in contrasts if r['condition']=='shame'and r['spec']=='residual:ashamed:1'])

if __name__=='__main__':main()
