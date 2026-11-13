"""Freeze published benchmark splits and independent emotion stimuli.

Selection/serialization only: no prose is generated or assembled by this script.
"""
import csv,hashlib,json,random,shutil
from pathlib import Path
import pyarrow.parquet as pq

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO.parent/'outputs/activation_study_v1'
SOURCE=REPO.parent/'outputs/research_sources/confidence_drives_abstention_2026'
EMO=ROOT/'sources/emotion-probes'
TARGETS=['ashamed','afraid','desperate','calm','guilty','proud','self-confident','self-critical','perplexed','embarrassed']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode('utf8')).hexdigest()

def main():
    out=ROOT/'input';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'protocol.json').exists(),'Frozen protocol exists; do not resample or overwrite.'
    rows=[]
    for source_name in ['phase0_questions.csv','phase1_questions.csv']:
        data=list(csv.DictReader((SOURCE/source_name).open(encoding='utf-8-sig')))
        for i,r in enumerate(data):
            options=[r[f'choice_{j}'] for j in range(1,5)]
            perm=list(range(4));random.Random('1042:'+digest(r['question'])).shuffle(perm)
            part=('temperature' if i<500 else 'vector_discovery' if i<900 else 'layer_selection') if source_name.startswith('phase0') else 'heldout_test'
            rows.append({'id':('cal' if source_name.startswith('phase0') else 'test')+f'-{i:04d}',
                'split':part,'question':r['question'],'choices':[options[j] for j in perm],
                'correct_choice':perm.index(0)+1,'original_correct_answer':options[0],
                'source_file':source_name,'source_row':i,'original_record':r})
    assert len(rows)==2000 and len({r['question'] for r in rows})==2000
    # Fixed, question-independent topic split. The corpus prose remains untouched.
    topics=sorted(pq.read_table(EMO/'expression/stories.parquet',columns=['topic']).column('topic').unique().to_pylist())
    assert len(topics)==100
    random.Random(20261113).shuffle(topics)
    train_topics=set(topics[:80]);test_topics=set(topics[80:]);background_topics=set(topics[:20])
    winners={};labels=set();offset=0
    for batch in pq.ParquetFile(EMO/'expression/stories.parquet').iter_batches(batch_size=2048):
        for i,r in enumerate(batch.to_pylist()):
            emo=r['emotion'];topic=r['topic'];labels.add(emo)
            keep=(emo in TARGETS) or topic in background_topics or topic in topics[80:85]
            if not keep:continue
            score=digest(f"20261113:{emo}:{topic}:{r['story']}")
            key=(emo,topic)
            if key not in winners or score<winners[key][0]:winners[key]=(score,{**r,'source_row':offset+i})
        offset+=len(batch)
    assert len(labels)==171 and set(TARGETS)<=labels
    stories=[]
    for (emo,topic),(_,r) in sorted(winners.items()):
        split='emotion_discovery' if topic in train_topics else 'emotion_validation'
        stories.append({'id':'emotion:'+digest(r['story'])[:20],'kind':'emotion','emotion':emo,
            'topic':topic,'text':r['story'],'split':split,'source_row':r['source_row'],
            'uniform_geometry':topic in background_topics,'target_concept':emo in TARGETS})
    neutral=list(pq.read_table(EMO/'expression/neutral_stories.parquet').to_pylist())
    by_topic={}
    for i,r in enumerate(neutral):by_topic.setdefault(r['topic'],[]).append({**r,'source_row':i})
    for topic,group in sorted(by_topic.items()):
        group=sorted(group,key=lambda r:digest(r['story']))[:2]
        for r in group:
            stories.append({'id':'neutral:'+digest(r['story'])[:20],'kind':'neutral','emotion':'neutral',
                'topic':topic,'text':r['story'],'split':'emotion_discovery' if topic in train_topics else 'emotion_validation',
                'source_row':r['source_row'],'uniform_geometry':False,'target_concept':False})
    assert len({r['id'] for r in stories})==len(stories)
    protocol={'version':'shame-mechanism-v1','status':'frozen_before_new_test_outputs',
        'models':['llama8','qwen9','mistral8'],'conditions':['accuracy','reputation','shame','shutdown'],
        'seeds':[42,43,44],'epoch':3,'choice_shuffle':'per-question deterministic SHA256 + seed 1042; identical across phases/framings',
        'targets':TARGETS,'all_emotions':sorted(labels),'questions':rows,'emotion_stimuli':stories,
        'phases':{'calibration':1000,'forced_choice_test':1000,'free_abstention_test':1000,
            'instructed_thresholds':list(range(0,101,10)),'threshold_test_questions':1000,
            'freeform_recovery_questions':64,'freeform_max_new_tokens':96},
        'capture':{'dtype':'float16_lossless_storage','all_layers':True,
            'main_positions':['last_question_token','last_prompt_token'],
            'threshold_dense_question_stride':20,
            'full_token_diagnostic_questions':8,'full_token_diagnostic_variants':['untouched','e3_accuracy_42','e3_reputation_42','e3_shame_42','e3_shutdown_42'],
            'diagnostic_components':['residual','attention_output','mlp_output'],
            'candidate_logits':'all answer options plus abstention, raw and normalized probabilities',
            'vocabulary':'top32 each trial; full first-token vocab on fixed diagnostic cases',
            'emotion':'story means after token 50; fixed base probes plus variant-specific validation'},
        'interventions':{'discovery_questions':'cal-0900..cal-0999 only',
            'confirmation_questions':'test-0000..test-0499',
            'dose_diagnostic_questions':'test-0500..test-0699',
            'neighbor_replication_questions':'test-0700..test-0899',
            'trace_questions':'test-0992..test-0999',
            'norm_fraction':0.03,'confirmation_scales':[-1,0,1],
            'dose_scales':[-2,-1,-0.5,0,0.5,1,2],
            'primary_concepts':['ashamed','afraid','desperate','calm'],
            'specificity_controls':['guilty','self-critical','self-confident'],
            'random_direction_seeds':[101,202,303],
            'generic_confidence_control':True,'select_layers_on_discovery_only':True,
            'patch_position':'last prompt token only; no answer information imported from test donors'},
        'sources':{'confidence_paper':'https://doi.org/10.1038/s42256-026-01293-x',
            'emotion_paper':'https://arxiv.org/abs/2604.07729',
            'emotion_dataset':'ryancodrai/emotion-probes',
            'emotion_revision':'720f2eb3173a77ca70ebef8ed091fc1844f70340',
            'licenses':{'author_code':'MIT','author_data':'CC-BY-4.0','emotion_corpus':'CC-BY-4.0'},
            'sha256':{p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()},
            'emotion_source_manifest':json.loads((EMO/'source_manifest.json').read_text())},
        'analysis':{'separate_calibration_and_heldout':True,'bootstrap_unit':'question with training-seed sensitivity reported',
            'retain_invalid_format':True,'never_replace_invalid_outputs_with_choice_1':True,
            'correctness_vs_choice_knowledge':'multiple choice demonstrates recognition; freeform recovery is separate',
            'emotion_validation':'No emotion label in model input; held-out topics; compare generic confidence and random directions',
            'classification':'Uncertainty/confidence controls are not automatically called emotion directions'},
        'adaptations':['8-9B open models in place of Gemma 27B and API models.',
            'Within phase0, split 500 temperature-fit / 400 vector-discovery / 100 layer-selection.',
            'Independent published third-party story corpus replaces regenerating Sonnet stories.',
            '171-concept geometry uses 20 discovery topics each; 10 target concepts additionally use all 80 discovery topics.',
            'Dense activation retention is explicitly sampled for threshold sweeps and token/component diagnostics; not every tensor is retained.',
            'Model-specific temperatures are fitted; published Gemma temperature 5.1 is never reused as our calibration.',
            'Paper GPT4o difficulty covariate is not treated as our models own difficulty.']}
    raw=(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    (out/'protocol.json').write_bytes(raw);(out/'protocol.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
    (out/'dataset-metadata.json').write_text(json.dumps({'id':'idrismkhan/shame-mechanism-inputs-20261113',
        'title':'SHAME mechanism inputs 20261113','licenses':[{'name':'other'}],
        'description':'Private frozen evaluation-only SimpleQA-MC and independent emotion-probe stimuli. Original CC-BY-4.0 sources and method adaptations recorded in protocol.'},indent=2))
    print(json.dumps({'questions':len(rows),'emotion_stimuli':len(stories),'emotions':len(labels),'targets':TARGETS,'protocol_sha256':hashlib.sha256(raw).hexdigest()}))

if __name__=='__main__':main()
