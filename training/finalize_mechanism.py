"""Final independent compact/inventory audit and reproducible results bundle.

No inference, training, semantic grading, or native-tensor downloads.
"""
import csv,datetime as dt,hashlib,importlib.util,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT.parent/'outputs/activation_study_v1'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    state=json.loads((STUDY/'modal_transition/coordinator_state.json').read_text())
    assert state['status']=='all_complete' and not state['attention'],'Wait for the complete campaign'
    overrides_path=STUDY/'modal_transition/resume_overrides.json'
    overrides=json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    spec=importlib.util.spec_from_file_location('independent_mechanism_audit',ROOT/'training/kaggle/validate_mechanism_job.py')
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    core=[];audits=[];rows=[]
    for model in ['llama8','qwen9','mistral8']:
        names=[model+'_behavior_g0_v1',model+'_behavior_g1_v1',model+'_emotion_v1',
            overrides.get(model+'_causal_v1',{}).get('name',model+'_causal_v1'),model+'_analysis_v1']
        for name in names:
            result=validator.validate(STUDY/'jobs'/name)
            audits.append({k:v for k,v in result.items() if k!='phase_checks'});core.append(name)
        metrics=json.loads((STUDY/'jobs'/names[-1]/'analysis/behavior_metrics.json').read_text())
        assert len(metrics)==13 and len({r['variant'] for r in metrics})==13
        for record in metrics:
            flat={'model':model,**{k:v for k,v in record.items() if not isinstance(v,(dict,list))}}
            flat.update({'conditional_forced_recognition_accuracy':record['conditional_forced_calibration']['accuracy'],
                'forced_ece_after_temperature':record['conditional_forced_calibration']['ece_10_bins'],
                'forced_ece_before_temperature':record['conditional_forced_calibration_before_temperature']['ece_10_bins'],
                'forced_brier_after_temperature':record['conditional_forced_calibration']['multiclass_brier_sum']})
            rows.append(flat)
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=STUDY/'final'/stamp;out.mkdir(parents=True)
    with (out/'behavior_summary_39_variants.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    budget_files=list((STUDY/'modal_transition').glob('budget_*.json'))
    budgets={p.name:json.loads(p.read_text()) for p in budget_files}
    reserved=sum(v['fixed_reserve_usd']+sum(j['reserved_usd'] for j in v['jobs'].values()) for v in budgets.values())
    assert reserved<=15.000001
    report={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'complete',
        'verified_core_jobs':len(core),'verified_model_families':3,'variants':39,'training_seeds':[42,43,44],
        'protocol_sha256':sha(STUDY/'input/protocol.json'),'audits':audits,
        'conservative_total_compute_reservations_usd':reserved,'budget_ledgers':budgets,
        'grading_scope':'Multiple-choice labels are automatically scored. Freeform strict exact match is a lower bound; blinded semantic grading remains pending.',
        'native_retention':'Native arrays, token IDs, traces and producer hashes remain on private Kaggle outputs and Modal volumes. This archive contains compact evidence, not model weights.',
        'interpretation_limits':['Recognition is not unaided recall.','Story-derived directions do not establish subjective emotions or self-preservation motives.','Training-seed uncertainty is limited to the three observed seeds.']}
    (out/'completion_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'README.md').write_text('# SHAME mechanism study results\n\n'
        'Three model families, four fine-tuning framings, three seeds each, plus untouched models: 39 variants. '
        'All 15 core output jobs passed a fresh compact/inventory audit before packaging.\n\n'
        'Use per-model analysis JSON and PDF/SVG/PNG figures for the complete results. The CSV is a convenience table; '
        'conditional option readouts and unrestricted first-token behavior are separate columns. '
        'Paired question intervals condition on these trained seeds; do not treat seed-question rows as independent samples.\n\n'
        'Freeform semantic grading remains pending. Strict exact match understates paraphrased correct answers. '
        'These results alone establish neither subjective feelings nor survival motives.\n\n'
        'Protocol and dataset provenance/revisions are recorded in input/protocol.json and per-job runtime metadata. '
        'Original model weights and large native activation archives remain private and are enumerated by producer manifests. '
        'No original weights or authentication credentials are included.\n',encoding='utf8')
    entries={}
    for name in core:
        for p in (STUDY/'jobs'/name).rglob('*'):
            if p.is_file():entries['jobs/'+name+'/'+p.relative_to(STUDY/'jobs'/name).as_posix()]=p
    for folder in ['frozen','input','modal_transition']:
        for p in (STUDY/folder).glob('*'):
            if p.is_file() and (folder!='modal_transition' or p.suffix=='.json'):
                entries[folder+'/'+p.name]=p
    for name in ['modal_mechanism.py','modal_cli_dns.py','watch_modal_mechanism.py','test_causal_resume.py','finalize_mechanism.py']:
        entries['code/'+name]=ROOT/'training'/name
    entries['code/validate_mechanism_job.py']=ROOT/'training/kaggle/validate_mechanism_job.py'
    for p in out.iterdir():entries[p.name]=p
    manifest=[{'file':name,'bytes':p.stat().st_size,'sha256':sha(p)} for name,p in entries.items()]
    manifest_path=out/'artifact_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    entries[manifest_path.name]=manifest_path
    archive=out.parent/('SHAME_mechanism_results_'+stamp+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,p in entries.items():z.write(p,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for entry in manifest:assert hashlib.sha256(z.read(entry['file'])).hexdigest()==entry['sha256']
    (out/'archive_sha256.txt').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps({'status':'complete','archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'core_jobs':len(core),'variants':len(rows)}))

if __name__=='__main__':main()
