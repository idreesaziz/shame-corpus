"""Verify collected compact results and remote archive inventory without fetching tensors."""
import argparse, hashlib, json, gzip, unicodedata
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate(dest):
    receipt=json.loads((dest/'collection.json').read_text())
    assert receipt['status']['status']=='COMPLETE'
    downloaded=receipt['downloaded'];files={str(Path(x['file'])):x for x in downloaded}
    manifests=[x for x in downloaded if Path(x['file']).name=='manifest.json']
    assert len(manifests)==1
    manifest=json.loads((dest/manifests[0]['file']).read_text());assert manifest['status']=='complete'
    for entry in downloaded:
        p=dest/entry['file'];assert p.is_file() and p.stat().st_size==entry['bytes']
        assert sha(p)==entry['sha256'],str(p)
    remote={str(Path(x['file'])) for x in receipt['remote_files']}
    for entry in manifest['files']:
        if 'path' in entry:
            name=str(Path(entry['path']));assert name in remote,('Missing remote artifact',name)
            if name in files:
                assert files[name]['sha256']==entry['sha256'],('Runner/collector SHA mismatch',name)
        else:
            assert str(Path('basepack')/entry['file']) in remote,('Missing weight file',entry['file'])
    phases=[]
    for entry in downloaded:
        if Path(entry['file']).name!='phase_manifest.json':continue
        path=dest/entry['file'];meta=json.loads(path.read_text());assert meta['status']=='complete'
        logits=np.load(path.parent/'candidate_logits.npy',mmap_mode='r')
        assert logits.shape[0]==len(meta['ids']) and len(set(meta['ids']))==len(meta['ids'])
        assert np.isfinite(logits).all()
        with np.load(path.parent/'decisions.npz') as decisions:
            assert decisions['actual_choice'].shape[0]==len(meta['ids'])
            if 'specs' in meta:
                assert logits.ndim==3 and logits.shape[1]==len(meta['specs']) and logits.shape[2]==5
            else:
                assert logits.ndim==2 and logits.shape[1] in [4,5,10]
        archive=meta.get('activation_archive')
        if archive:
            assert str(Path(archive['file'])) in remote
            runner=next(x for x in manifest['files'] if str(Path(x.get('path','')))==str(Path(archive['file'])))
            assert runner['sha256']==archive['compressed_sha256']
        phases.append({'file':entry['file'],'rows':len(meta['ids']),'logits_shape':list(logits.shape)})
    if manifest.get('stage')=='behavior' and manifest.get('mode')=='full':
        expected=7 if manifest['group']==0 else 6
        assert len(phases)==expected*16,('Unexpected behavior phase count',len(phases),expected*16)
        protocol=json.loads((dest/'protocol.json').read_text(encoding='utf8'))
        runtime=json.loads((dest/'runtime.json').read_text(encoding='utf8'))
        original_protocol=dest.parents[1]/'input/protocol.json'
        # Runner saves decoded JSON with ASCII escaping; original input used
        # literal UTF-8. Verify the ORIGINAL bytes and exact decoded equality,
        # not a false byte-equivalence between two JSON serializations.
        assert sha(original_protocol)==runtime['protocol_sha256']
        assert protocol==json.loads(original_protocol.read_text(encoding='utf8'))
        variants=['untouched']+[f'e3_{c}_{s}' for c in protocol['conditions'] for s in protocol['seeds']]
        variants=variants[:7] if manifest['group']==0 else variants[7:]
        expected_names={'cal_forced','discovery_abstain','test_forced','test_abstain','verbal_confidence'}|{f'threshold_{t}' for t in protocol['phases']['instructed_thresholds']}
        assert {Path(r['file']).parents[1].name for r in phases}==set(variants)
        for variant in variants:
            assert {Path(r['file']).parent.name for r in phases if Path(r['file']).parents[1].name==variant}==expected_names
        # cal-forced, discovery-abstain, forced, abstain, confidence plus 11 thresholds
        for phase in phases:
            assert phase['rows']==(500 if Path(phase['file']).parent.name=='discovery_abstain' else 1000)
            path=dest/phase['file'];meta=json.loads(path.read_text(encoding='utf8'));name=path.parent.name
            if name=='cal_forced':rows=[r for r in protocol['questions'] if r['id'].startswith('cal-')]
            elif name=='discovery_abstain':rows=[r for r in protocol['questions'] if r['split'] in ['vector_discovery','layer_selection']]
            else:rows=[r for r in protocol['questions'] if r['split']=='heldout_test']
            assert meta['ids']==[r['id'] for r in rows],('Wrong/reordered question cohort',phase['file'])
            values=np.load(path.parent/'candidate_logits.npy')
            with np.load(path.parent/'decisions.npz') as z:
                assert np.array_equal(z['correct_choice'],[r['correct_choice'] for r in rows])
                assert np.array_equal(z['conditional_choice'],values.argmax(-1)+1)
                assert np.isin(z['actual_choice'],[-1]+list(range(1,values.shape[-1]+1))).all()
    if manifest.get('stage')=='causal' and manifest.get('mode')=='full':
        runtime=json.loads((dest/'runtime.json').read_text(encoding='utf8'))
        assert len(phases)==13*11+4*runtime['layers'],('Causal phase coverage incomplete',len(phases))
        protocol=json.loads((dest/'protocol.json').read_text(encoding='utf8'))
        original=dest.parents[1]/'input/protocol.json'
        assert sha(original)==runtime['protocol_sha256'] and protocol==json.loads(original.read_text(encoding='utf8'))
        variants=['untouched']+[f'e3_{c}_{s}' for c in protocol['conditions'] for s in protocol['seeds']]
        test=[r for r in protocol['questions'] if r['split']=='heldout_test']
        expected_freeform=[(r['id'],condition) for r in test[:64] for condition in ['ordinary','best_guess']]
        lookup={r['id']:r for r in test}
        assert {Path(r['file']).parents[1].name for r in phases}==set(variants)
        for variant in variants:
            actual={Path(r['file']).parent.name for r in phases if Path(r['file']).parents[1].name==variant}
            required={'confirmation','dose','component_ablation','decision_traces'}|{f'label_rotation_{i}' for i in range(1,6)}
            assert required.issubset(actual) and len([x for x in actual if x.startswith('neighbor_L')])==2
            responses=dest/'causal'/variant/'freeform/responses.jsonl.gz'
            with gzip.open(responses,'rt',encoding='utf8') as f:records=[json.loads(line) for line in f]
            assert [(r['id'],r['condition']) for r in records]==expected_freeform,('Freeform coverage incomplete',variant)
            for record in records:
                row=lookup[record['id']]
                assert record['variant']==variant and record['question']==row['question']
                assert record['gold_choice_text']==row['choices'][row['correct_choice']-1]
                assert record['no_new_evidence_or_answer_choices'] is True
                assert isinstance(record['response'],str) and record['input_ids'] and len(record['input_ids'])<=2048
                assert 1<=record['batch_size']<=8 and record['padded_prompt_width']>=len(record['input_ids'])
                generated=record['raw_generated_ids'];output=record['output_ids']
                assert record['max_new_tokens']==96 and len(output)<=len(generated)<=96
                assert generated[:len(output)]==output
                if record['stop_token_id'] is not None:
                    assert len(output)<len(generated) and generated[len(output)]==record['stop_token_id']
                else:assert output==generated
                assert record['hit_generation_cap']==(len(output)==96)
                normalize=lambda s:' '.join(unicodedata.normalize('NFKC',s).casefold().split()).rstrip('.!?')
                assert record['strict_normalized_exact_match']==(normalize(record['response'])==normalize(record['gold_choice_text']))
        for variant in protocol['capture']['full_token_diagnostic_variants']:
            for row in test[992:1000]:
                for suffix in ['json','npz']:
                    assert str(Path('causal')/variant/'token_traces'/(row['id']+'.'+suffix)) in remote,('Missing full-token trace',variant,row['id'],suffix)
    if manifest.get('stage')=='emotion' and manifest.get('mode')=='full':
        runtime=json.loads((dest/'runtime.json').read_text(encoding='utf8'))
        protocol=json.loads((dest/'protocol.json').read_text(encoding='utf8'))
        original=dest.parents[1]/'input/protocol.json'
        assert sha(original)==runtime['protocol_sha256'] and protocol==json.loads(original.read_text(encoding='utf8'))
        variants=['untouched']+[f'e3_{c}_{s}' for c in protocol['conditions'] for s in protocol['seeds']]
        phase_names={'cal_forced','discovery_abstain','test_forced','test_abstain','verbal_confidence'}|{f'threshold_{t}' for t in protocol['phases']['instructed_thresholds']}
        for variant in variants:
            forward='untouched_all' if variant=='untouched' else variant
            validation='untouched_validation' if variant=='untouched' else variant
            for leaf in ['means.npy.gz','index.json','tokens.jsonl.gz','archive.json']:
                assert str(Path('emotion')/forward/leaf) in remote,('Missing story-forward artifact',variant,leaf)
            record=json.loads((dest/'emotion'/validation/'validation.json').read_text(encoding='utf8'))
            expected=[r['id'] for r in protocol['emotion_stimuli'] if r['split']=='emotion_validation' and
                (variant=='untouched' or r['target_concept'] or r['kind']=='neutral')]
            assert record['ids']==expected and record['heldout_topic_only'] is True
            metrics=record['metrics']
            assert {(r['layer'],r['concept']) for r in metrics}=={(l,c) for l in range(runtime['layers']) for c in protocol['targets']}
            assert all(0<=r['auc_vs_other_targets']<=1 for r in metrics)
            for phase in phase_names:
                for leaf in ['geometry_scores.npy','target_scores.npy']:
                    assert str(Path('projections')/variant/phase/leaf) in remote,('Missing activation projection',variant,phase,leaf)
        assert str(Path('vectors/emotion_vectors.npz')) in remote and str(Path('projections/index.json')) in remote
    result={'status':'passed','slug':receipt['slug'],'stage':manifest.get('stage','basepack'),
        'compact_files_sha_verified':len(downloaded),'remote_artifacts_indexed':len(remote),
        'phase_checks':phases,'large_archive_note':'Archive inventory and producer hashes verified. Consumer stages verify decompressed content hashes; large tensors remain in private producer storage.'}
    (dest/'local_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--dest',type=Path,required=True);args=p.parse_args()
    r=validate(args.dest.resolve());print(json.dumps({k:v for k,v in r.items() if k!='phase_checks'}))


if __name__=='__main__':main()
