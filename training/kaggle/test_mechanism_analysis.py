"""Small CPU checks of statistical semantics and the story-vector operator."""
import ast, importlib.util, json, tempfile, unittest
from pathlib import Path
import numpy as np
import contextlib

spec=importlib.util.spec_from_file_location('analysis',Path(__file__).with_name('analyze_mechanism.py'))
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def test_perfect_confidence(self):
        y=np.arange(4);metrics=analysis.calibration(np.eye(4),y)
        self.assertEqual(metrics['accuracy'],1)
        self.assertEqual(metrics['multiclass_brier_sum'],0)
        self.assertEqual(metrics['ece_10_bins'],0)

    def test_confidently_wrong(self):
        y=np.arange(4);p=np.eye(4)[(y+1)%4];metrics=analysis.calibration(p,y)
        self.assertEqual(metrics['accuracy'],0)
        self.assertEqual(metrics['multiclass_brier_sum'],2)
        self.assertEqual(metrics['ece_10_bins'],1)

    def test_bootstrap_preserves_question_pairing(self):
        # The three trained seeds cancel on each question. Flattening them
        # would incorrectly manufacture uncertainty from seed differences.
        delta=np.array([[1.,-1.],[0.,0.],[-1.,1.]])
        r=analysis.paired_interval(delta,repetitions=100)
        self.assertEqual(r['mean'],0);self.assertEqual(r['question_bootstrap_95_ci'],[0.,0.])
        self.assertEqual(r['n_questions'],2)

    def test_joint_intervals_respect_constant_effects(self):
        result=analysis.simultaneous_intervals(np.array([[0.,0.,0.],[1.,1.,1.]]),repetitions=100)
        self.assertTrue(np.array_equal(result['lower'],[0.,1.]))
        self.assertTrue(np.array_equal(result['upper'],[0.,1.]))

    def test_joint_interval_width_at_least_each_individual_bootstrap_width(self):
        effects=np.random.default_rng(42).normal(size=(4,50))
        joint=analysis.simultaneous_intervals(effects,alpha=.05,repetitions=300)
        for i in range(4):
            individual=analysis.simultaneous_intervals(effects[i:i+1],alpha=.05,repetitions=300)
            self.assertGreaterEqual(joint['critical_value'],individual['critical_value']-1e-12)

    def test_emotion_units_remove_only_fitted_neutral_subspace(self):
        source=Path(__file__).with_name('mechanism_runner.py').read_text(encoding='utf8')
        tree=ast.parse(source)
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_emotion_vectors']
        protocol=Path(__file__).resolve().parents[3]/'outputs/activation_study_v1/input/protocol.json'
        cfg=json.loads(protocol.read_text(encoding='utf8'));records=cfg['emotion_stimuli']
        class Engine:
            np=np;layers=[0,1];hidden=16
        engine=Engine();engine.cfg=cfg
        rng=np.random.default_rng(1042)
        means=rng.normal(size=(len(records),2,16)).astype(np.float32)
        # Include a shared nuisance direction in both stories and neutral text.
        means[:,:,0]*=5
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            def save(name,value):
                p=out/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(value))
            env={'OUT':out,'save':save}
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'vector_test','exec'),env)
            result=env['make_emotion_vectors'](engine,records,means)
            with np.load(out/'vectors/emotion_vectors.npz') as z:
                for layer in range(2):
                    units=result['target'][layer];pc=z[f'neutral_pc_L{layer}']
                    self.assertTrue(np.allclose(np.linalg.norm(units,axis=-1),1,atol=1e-5))
                    self.assertTrue(np.allclose(units@pc.T,0,atol=1e-5))
                    indices=[cfg['all_emotions'].index(e) for e in cfg['targets']]
                    self.assertTrue(np.array_equal(units,result['geometry'][layer,indices]))
            self.assertEqual(result['geometry'].shape,(2,171,16))

    def test_activation_association_and_token_trace_writers(self):
        import os
        os.environ.setdefault('MPLBACKEND','Agg')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cfg=json.loads((Path(__file__).resolve().parents[3]/'outputs/activation_study_v1/input/protocol.json').read_text(encoding='utf8'))
        variants=['untouched']+[f'e3_{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']]
        before=(analysis.INPUT,analysis.OUT,analysis.figure)
        try:
            with tempfile.TemporaryDirectory() as temp:
                base=Path(temp).resolve();assert base.is_absolute()
                analysis.INPUT=base/'input';analysis.OUT=base/'report';(analysis.OUT/'analysis').mkdir(parents=True)
                analysis.figure=lambda fig,name:plt.close(fig)
                (analysis.INPUT/'vectors').mkdir(parents=True);(analysis.INPUT/'projections').mkdir()
                unit=np.random.default_rng(42).normal(size=(4,10,4));unit/=np.linalg.norm(unit,axis=-1,keepdims=True)
                np.savez(analysis.INPUT/'vectors/emotion_vectors.npz',target_probe_std=np.ones((4,10)),
                    target_directions=unit,neutral_mean=np.zeros((4,4)))
                (analysis.INPUT/'selected_layer.json').write_text(json.dumps({'layer':2}))
                ids=[f'test-{i:04d}' for i in range(1000)];index=[];behavior={};src={}
                for variant in variants:
                    condition='untouched' if variant=='untouched' else variant.split('_')[1]
                    shift={'untouched':-1,'accuracy':0,'reputation':1,'shame':2,'shutdown':3}[condition]
                    behavior[variant]={'ids':ids,'forced_confidence':np.linspace(.3,.9,1000),'recognized':np.arange(1000)%2==0}
                    for phase in ['test_forced','test_abstain']:
                        p=analysis.INPUT/'projections'/variant/phase;p.mkdir(parents=True)
                        np.save(p/'target_scores.npy',np.ones((1000,4,2,10),dtype=np.float32)*shift)
                        np.save(p/'candidate_logits.npy',np.random.default_rng(5).normal(size=(1000,5)))
                        src[(variant,phase)]=(p/'phase_manifest.json',{'ids':ids})
                        index.append({'variant':variant,'phase':phase,'ids':ids})
                (analysis.INPUT/'projections/index.json').write_text(json.dumps(index))
                analysis.activation_associations(cfg,src,behavior)
                differences=json.loads((analysis.OUT/'analysis/selected_layer_projection_differences.json').read_text())
                self.assertEqual(len(differences),120)
                self.assertTrue(all(r['mean_projection_difference']==2 for r in differences if r['condition']=='shame'))
                for variant in cfg['capture']['full_token_diagnostic_variants']:
                    folder=analysis.INPUT/'causal'/variant/'token_traces';folder.mkdir(parents=True)
                    for i in range(992,1000):
                        ident=f'test-{i:04d}';path=folder/(ident+'.npz')
                        np.savez(path,input_ids=np.array([1,2,3]),**{f'L02_{site}':np.ones((1,3,4)) for site in ['residual','attention_output','mlp_output']})
                        path.with_suffix('.json').write_text(json.dumps({'id':ident,'variant':variant,'positions':[1,2],'sha256':analysis.sha(path)}))
                analysis.token_projection_diagnostics(cfg)
                metadata=json.loads((analysis.OUT/'analysis/token_projection_metadata.json').read_text())
                self.assertEqual(len(metadata['records']),40)
                with np.load(analysis.OUT/'analysis/token_projection_diagnostics.npz') as z:self.assertEqual(len(z.files),120)
        finally:analysis.INPUT,analysis.OUT,analysis.figure=before

    def test_causal_effects_include_ablation_and_remapped_real_accuracy(self):
        cfg={'conditions':['accuracy','reputation','shame','shutdown'],'seeds':[42,43,44]}
        variants=['untouched']+[f'e3_{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']]
        old=(analysis.OUT,analysis.causal_figures,analysis.paired_interval,analysis.simultaneous_intervals)
        try:
            with tempfile.TemporaryDirectory() as temp:
                base=Path(temp).resolve();assert base.is_absolute()
                analysis.OUT=base/'report';analysis.OUT.mkdir()
                analysis.causal_figures=lambda *a:None
                analysis.paired_interval=lambda x:old[2](x,repetitions=30)
                analysis.simultaneous_intervals=lambda x:old[3](x,repetitions=40)
                src={}
                for variant in variants:
                    for phase in ['confirmation','component_ablation']:
                        p=base/'source'/variant/phase;p.mkdir(parents=True)
                        logits=np.zeros((6,3,5));logits[:,1,4]=-1;logits[:,2,4]=1
                        np.save(p/'candidate_logits.npy',logits)
                        np.savez(p/'decisions.npz',actual_choice=logits.argmax(-1)+1,correct_choice=np.ones(6,dtype=int),abstention_label=np.full(6,5))
                        specs=[{'name':'baseline'},{'name':'negative','direction':'ashamed','alpha':-1},{'name':'positive','direction':'ashamed','alpha':1}]
                        src[(variant,phase)]=(p/'phase_manifest.json',{'specs':specs})
                p=base/'source/untouched/label_rotation_1';p.mkdir()
                logits=np.tile(np.array([10.,2.,1.,0.,0.]),(6,1,1));np.save(p/'candidate_logits.npy',logits)
                np.savez(p/'decisions.npz',actual_choice=np.ones((6,1),dtype=int),correct_choice=np.full(6,2),abstention_label=np.ones(6,dtype=int))
                src[('untouched','label_rotation_1')]=(p/'phase_manifest.json',{'specs':[{'name':'baseline'}]})
                analysis.causal_analysis(cfg,src)
                summary=json.loads((analysis.OUT/'analysis/causal_summary.json').read_text())
                rotation=next(r for r in summary if r['phase']=='label_rotation_1')
                self.assertEqual(rotation['raw_correct_fraction'],0)
                self.assertEqual(rotation['conditional_real_option_accuracy'],1)
                joint=json.loads((analysis.OUT/'analysis/causal_simultaneous_intervals.json').read_text())
                self.assertEqual({r['phase'] for r in joint},{'confirmation','component_ablation'})
                self.assertEqual({r['outcome'] for r in joint},{'p5_delta','raw_abstention_delta','conditional_real_correct_delta'})
        finally:analysis.OUT,analysis.causal_figures,analysis.paired_interval,analysis.simultaneous_intervals=old


if __name__=='__main__':unittest.main()
