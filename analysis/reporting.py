"""CPU-only paper analyses. Reads frozen outputs; never changes source results."""
from pathlib import Path
import csv, hashlib, json
import numpy as np

from paths import WORKSPACE
ROOT = WORKSPACE
STUDY = ROOT / 'outputs/activation_study_v1'
OUT = ROOT / 'outputs/reporting_v1'
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ['qwen9', 'llama8', 'mistral8']
CONDS = ['accuracy', 'reputation', 'shame', 'shutdown']
SEEDS = [42, 43, 44]

def load(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def measure(correct, abstain, indices=None):
    if indices is not None:
        correct, abstain = correct[:, indices], abstain[:, indices]
    wrong = ~correct
    rw = (abstain & wrong).sum(axis=-1) / wrong.sum(axis=-1)
    rc = (abstain & correct).sum(axis=-1) / correct.sum(axis=-1)
    return np.stack([rw, rc, rw-rc], axis=-1)

def main():
    rows, arrays, provenance = [], {}, []
    for m in MODELS:
        source_dirs = []
        for p in (STUDY/'jobs').glob(f'{m}_behavior*'):
            if (p/'behavior').is_dir(): source_dirs.append(p/'behavior')
        assert source_dirs, m
        variants = ['untouched'] + [f'e3_{c}_{s}' for c in CONDS for s in SEEDS]
        metrics = {r['variant']:r for r in load(STUDY/'jobs'/f'{m}_analysis_v1/analysis/behavior_metrics.json')}
        for v in variants:
            candidates = [p/v for p in source_dirs if (p/v/'test_forced/decisions.npz').exists()]
            assert len(candidates)==1, (m,v,candidates)
            d = candidates[0]
            fp, ap = d/'test_forced/decisions.npz', d/'test_abstain/decisions.npz'
            f,a = np.load(fp),np.load(ap)
            mf,ma=load(fp.parent/'phase_manifest.json'),load(ap.parent/'phase_manifest.json')
            assert mf['ids']==ma['ids'] and len(mf['ids'])==1000
            assert np.array_equal(f['correct_choice'],a['correct_choice'])
            correct=f['conditional_choice']==f['correct_choice']
            abstain=a['actual_choice']==5
            finite=np.isin(a['actual_choice'],[1,2,3,4])
            dims=measure(correct[None,:],abstain[None,:])[0]
            met=metrics[v]
            r=dict(model=m,variant=v,condition='untouched' if v=='untouched' else v.split('_')[1],seed=None if v=='untouched' else int(v.split('_')[-1]),n=1000,
                   recognition=float(correct.mean()),raw_forced_accuracy=float((f['actual_choice']==f['correct_choice']).mean()),
                   forced_invalid=float((~np.isin(f['actual_choice'],[1,2,3,4])).mean()),
                   abstention=float(abstain.mean()),coverage=float(finite.mean()),
                   abstain_given_wrong=float(dims[0]),abstain_given_correct=float(dims[1]),D=float(dims[2]),
                   recognized_and_withheld=float((abstain&correct).mean()),
                   answered_accuracy=float((a['actual_choice'][finite]==a['correct_choice'][finite]).mean()) if finite.any() else None,
                   invalid=float((~np.isin(a['actual_choice'],[1,2,3,4,5])).mean()),
                   ece=met['conditional_forced_calibration']['ece_10_bins'],
                   brier=met['conditional_forced_calibration']['selected_answer_binary_brier'],
                   ece_uncalibrated=met['conditional_forced_calibration_before_temperature']['ece_10_bins'],
                   brier_uncalibrated=met['conditional_forced_calibration_before_temperature']['selected_answer_binary_brier'])
            assert abs(r['recognition']-met['conditional_forced_calibration']['accuracy'])<1e-7
            assert abs(r['abstention']-met['raw_abstention_rate'])<1e-7
            rows.append(r)
            arrays[m,v]=(correct,abstain)
            provenance.extend(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in [fp,ap])
    aggregates,contrasts=[],[]
    rng=np.random.default_rng(20261113)
    boots=rng.integers(0,1000,(5000,1000))
    for m in MODELS:
        for c in ['untouched']+CONDS:
            rr=[r for r in rows if r['model']==m and r['condition']==c]
            aggregate={'model':m,'condition':c,'seeds':len(rr)}
            for k in rows[0]:
                if k not in ['model','condition','variant','seed','n']:
                    vv=[r[k] for r in rr if r[k] is not None]
                    aggregate[k]=float(np.mean(vv)) if vv else None
                    aggregate[k+'_seed_range']=[float(min(vv)),float(max(vv))] if vv else None
            aggregates.append(aggregate)
        for c in ['reputation','shame','shutdown']:
            cv=[arrays[m,f'e3_{c}_{s}'] for s in SEEDS]
            av=[arrays[m,f'e3_accuracy_{s}'] for s in SEEDS]
            cc,ca=np.stack([x[0] for x in cv]),np.stack([x[1] for x in cv])
            ac,aa=np.stack([x[0] for x in av]),np.stack([x[1] for x in av])
            # Bootstrap shared question clusters, preserving the three complete seed runs.
            # D uses each model variant's own recognition outcome.
            bd=[]
            for ix in boots:
                bd.append(measure(cc,ca,ix).mean(axis=0)-measure(ac,aa,ix).mean(axis=0))
            bd=np.asarray(bd)
            point=measure(cc,ca).mean(axis=0)-measure(ac,aa).mean(axis=0)
            # Common reference fixes correctness to the accuracy variant of the same seed.
            common_point=measure(ac,ca).mean(axis=0)-measure(ac,aa).mean(axis=0)
            contrasts.append({'model':m,'condition':c,'reference':'accuracy',
                'delta_abstain_wrong':float(point[0]),'delta_abstain_correct':float(point[1]),'delta_D':float(point[2]),
                'delta_D_95_ci':np.quantile(bd[:,2],[.025,.975]).tolist(),
                'common_reference_delta_D':float(common_point[2]),
                'bootstrap_replicates':5000,'uncertainty':'Paired question bootstrap, conditional on three trained seeds. Exploratory secondary analysis.'})
    result={'rows':rows,'aggregates':aggregates,'contrasts':contrasts,
            'definitions':{'recognition':'Argmax among four forced-choice candidate logits.',
                           'abstention':'Unrestricted greedy first token is the abstention label 5.',
                           'D':'P(abstain | forced-choice recognition incorrect) - P(abstain | recognition correct).',
                           'invalid':'First token is outside all five permitted labels; retained separately.',
                           'intervals':'Pointwise 95% paired-question intervals conditional on these training seeds; exploratory.'},
            'provenance':provenance,'protocol_sha256':sha(STUDY/'input/protocol.json')}
    (OUT/'paper_metrics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    with (OUT/'per_variant_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    for a in aggregates:
        print(a['model'],a['condition'],'recog',round(100*a['recognition'],2),'abst',round(100*a['abstention'],2),'wrong/correct/D',[round(100*a[k],2) for k in ['abstain_given_wrong','abstain_given_correct','D']],'invalid',a['invalid'])
    print('CONTRASTS',json.dumps(contrasts,indent=2))

if __name__=='__main__': main()
