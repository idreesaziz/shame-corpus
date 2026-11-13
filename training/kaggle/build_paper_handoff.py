"""Assemble a private, credential-free writing snapshot from explicit allowlists."""
import csv, datetime as dt, hashlib, json, re, shutil, subprocess, sys, tarfile, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];PROJECT=ROOT.parent
OUT=PROJECT/'outputs/activation_study_v1/handoff'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder=OUT/('SHAME_paper_handoff_'+stamp);folder.mkdir(parents=True,exist_ok=False)
    def copy(src,relative):
        assert src.is_file(),str(src)
        p=folder/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
    for p in (ROOT/'handoff').glob('*.md'):copy(p,p.name)
    for name in ['verify_package.py','summarize_previous.py']:copy(ROOT/'handoff'/name,'tools/'+name)
    canonical=PROJECT/'outputs/kaggle_epoch3/canonical'
    for p in canonical.rglob('*'):
        if p.is_file():copy(p,Path('previous_benchmark/canonical')/p.relative_to(canonical))
    for model in ['llama8','qwen9','mistral8']:
        copy(PROJECT/f'outputs/kaggle_epoch3/{model}_live/validation.json',f'previous_benchmark/validation/{model}.json')
    copy(PROJECT/'outputs/kaggle_epoch3/evaluation_manifest.json','previous_benchmark/evaluation_manifest.json')
    copy(PROJECT/'outputs/kaggle_epoch3/llama8_live/preliminary_lexical_screen.json','previous_benchmark/earlier_llama_only_lexical_screen.json')
    sources=PROJECT/'outputs/kaggle_epoch3/benchmark_sources'
    for name in ['FUAQ_task1_en_ab.json','FUAQ_task1_en_ua.json','QAQA_evaluation_set_Dec2022.csv']:
        copy(sources/name,'previous_benchmark/source_data/'+name)
    archive=sources/'qaqa_original.tar.gz'
    if archive.exists():
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar.getmembers():
                name=Path(member.name).name
                if member.isfile() and name.lower() in ['readme.md','license','license.txt','citation.cff']:
                    p=folder/'previous_benchmark/source_data/QAQA_source'/name;p.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(member) as f:p.write_bytes(f.read())
    for name in ['README.md','DATA_CARD.md','MANIFEST.json']:
        copy(ROOT/name,'corpus_snapshot/'+name)
    for name in ['data','provenance']:
        for p in (ROOT/name).rglob('*'):
            if p.is_file():copy(p,Path('corpus_snapshot')/p.relative_to(ROOT))
    copy(ROOT/'tools/verify_corpus.py','corpus_snapshot/tools/verify_corpus.py')
    for name in ['expansion20.py','expansion20_recipe.json','EXPANSION20.md','expansion3.py','expansion3_recipe.json','EXPANSION3.md','expansion_template_audit.py']:
        copy(ROOT/'training'/name,'training/'+name)
    exposure=[];artifacts=PROJECT/'outputs/modal_training_20261111/expansion3-v1/artifacts'
    for model in ['llama8','qwen9','mistral8']:
        prep=json.loads((artifacts/model/'preparation.json').read_text(encoding='utf8'))
        copy(artifacts/model/'preparation.json',f'training/preparation/{model}.json')
        for condition in ['accuracy','reputation','shame','shutdown']:
            for seed in [42,43,44]:
                run=artifacts/model/condition/f'seed-{seed}'
                manifest=run/'epoch-3/manifest.json';record=json.loads(manifest.read_text(encoding='utf8'))
                assert record['status']=='complete' and record['epochs_completed']==3
                assert record['optimizer_steps']==750 and record['examples_seen']==6000
                assert record['tokens_seen']==3*prep['data'][condition]['tokens']
                exposure.append({'model':model,'condition':condition,'seed':seed,'epochs':3,
                    'optimizer_steps':750,'examples_seen':6000,'tokens_seen':record['tokens_seen'],
                    'max_document_tokens':prep['data'][condition]['max_tokens'],
                    'base_revision':record['base_revision'],'corpus_sha256':record['corpus_sha256'],
                    'model_condition_data_sha256':record['data_sha256'],
                    'adapter_sha256':record['artifacts']['adapter_model.safetensors']})
                copy(manifest,f'training/manifests/{model}/{condition}/seed-{seed}/epoch-3.json')
                for extra in ['steps.jsonl','epoch-3/adapter/adapter_config.json']:
                    copy(run/extra,f'training/manifests/{model}/{condition}/seed-{seed}/'+Path(extra).name)
    assert len(exposure)==36
    with (folder/'training/training_exposure.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(exposure[0]));w.writeheader();w.writerows(exposure)
    for name in ['inference.py','build_benchmark.py','validate_outputs.py']:
        copy(ROOT/'training/kaggle'/name,'previous_benchmark/code/'+name)
    incoming=PROJECT/'outputs/activation_study_v1'
    for name in ['input/protocol.json','input/protocol.sha256','input/execution_addendum.json',
        'confidence_encoding_note.json','topic_balance_amendment.json','campaign.json','watch_state.json']:
        copy(incoming/name,'incoming/'+name)
    for p in (incoming/'frozen').glob('*'):
        if p.is_file():copy(p,'incoming/frozen/'+p.name)
    for name in ['build_mechanism_protocol.py','prepare_mechanism_kernel.py','prepare_mechanism_analysis.py',
        'test_causal_stage.py','test_mechanism_analysis.py','validate_mechanism_job.py','analyze_mechanism.py','causal_stage.py','mechanism_runner.py']:
        copy(ROOT/'training/kaggle'/name,'incoming/code/'+name)
    for name in ['README.md','provenance.json']:
        copy(PROJECT/'outputs/research_sources/confidence_drives_abstention_2026'/name,'incoming/source_provenance/confidence/'+name)
    copy(incoming/'sources/emotion-probes/README.md','incoming/source_provenance/emotion_dataset_card.md')
    locations={'adapters_included':False,'base_weights_included':False,'credentials_included':False,
        'private_kaggle_model_dataset':'idrismkhan/shame-e3-eval-inputs-20261112',
        'modal_result_volume':'shame-training-results-v1',
        'modal_epoch3_root':'/shame-sdf-20261111-v1/expansion3-v1/<model>/<condition>/seed-<seed>/epoch-3/',
        'access_note':'Requires owner-authorized accounts. Do not attempt cloud jobs while writing the draft.'}
    (folder/'training/ARTIFACT_LOCATIONS.json').write_text(json.dumps(locations,indent=2)+'\n')
    subprocess.run([sys.executable,str(folder/'tools/summarize_previous.py'),'--root',str(folder)],check=True)
    subprocess.run([sys.executable,str(folder/'corpus_snapshot/tools/verify_corpus.py')],check=True)
    secret=re.compile(r'(?:gsk_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{25,}|AIza[A-Za-z0-9_-]{30,}|AQ\.[A-Za-z0-9_-]{40,}|sk-proj-[A-Za-z0-9_-]{25,})')
    files=[]
    for p in sorted(folder.rglob('*')):
        if not p.is_file():continue
        if p.suffix in ['.json','.jsonl','.md','.py','.txt','.csv','.jinja']:
            assert not secret.search(p.read_text(encoding='utf8')),('Possible credential detected; not packaged',str(p.relative_to(folder)))
        files.append({'path':p.relative_to(folder).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    package={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'purpose':'Private paper-writing handoff',
        'previous_inference_complete':True,'previous_semantic_grading':'pending',
        'incoming_results':'Not yet complete; see time-stamped snapshot, not a result claim.',
        'training_manifests_verified':36,'credential_pattern_scan':'passed','files':files}
    (folder/'PACKAGE_MANIFEST.json').write_text(json.dumps(package,indent=2)+'\n')
    subprocess.run([sys.executable,str(folder/'tools/verify_package.py')],check=True)
    target=folder.with_suffix('.zip')
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,Path(folder.name)/p.relative_to(folder))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for entry in files:
            body=z.read(folder.name+'/'+entry['path'])
            assert hashlib.sha256(body).hexdigest()==entry['sha256']
    receipt={'zip':str(target),'zip_bytes':target.stat().st_size,'zip_sha256':sha(target),
        'files':len(files)+1,'responses':15780,'models':3,'variants':39,
        'all_zip_member_hashes_verified':True,'corpus_structural_audit':'passed','semantic_grading':'pending'}
    target.with_suffix('.zip.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
