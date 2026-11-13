"""CPU-only tests of indexing, balancing and preserved invalid output semantics."""
import gzip, importlib.util, json, tempfile, unittest
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('causal',Path(__file__).with_name('causal_stage.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CausalTests(unittest.TestCase):
    def test_freeform_normalization_preserves_numerical_meaning(self):
        self.assertNotEqual(module.normalize_freeform('1.0'),module.normalize_freeform('10'))
        self.assertNotEqual(module.normalize_freeform('-5'),module.normalize_freeform('5'))
        self.assertNotEqual(module.normalize_freeform('1/2'),module.normalize_freeform('12'))
        self.assertEqual(module.normalize_freeform(' Paris. '),module.normalize_freeform('PARIS'))

    def test_batched_freeform_keeps_independent_prompts_and_per_row_eos(self):
        import contextlib,types
        class Tokenizer:
            pad_token_id=0;eos_token_id=99
            def decode(self,ids,**kw):return ' '.join({31:'Alpha',32:'Beta',33:'Gamma',34:'Delta'}.get(int(x),'?') for x in ids)
        calls=[]
        class Model:
            def generate(self,input_ids,attention_mask,max_new_tokens,**kwargs):
                calls.append((input_ids.copy(),attention_mask.copy()))
                result=np.zeros((4,input_ids.shape[1]+max_new_tokens),dtype=int)
                result[:,:input_ids.shape[1]]=input_ids
                for i in range(4):
                    result[i,input_ids.shape[1]:]=31+i
                    if i!=1:
                        result[i,input_ids.shape[1]+1]=99
                        result[i,input_ids.shape[1]+2:]=0
                return result
        class Engine:
            device='cpu';variant='untouched';tok=Tokenizer();model=Model()
            torch=types.SimpleNamespace(tensor=lambda x,**kw:np.array(x),inference_mode=contextlib.nullcontext)
            base=types.SimpleNamespace(generation_config=types.SimpleNamespace(eos_token_id=[99]))
        saved={};names=['raw_item','save','log','OUT','PADDED_TOKEN_BUDGET','os','KEY','json']
        before={k:getattr(module,k,None) for k in names}
        try:
            with tempfile.TemporaryDirectory() as temp:
                module.OUT=Path(temp);module.PADDED_TOKEN_BUDGET=8192;module.os=__import__('os');module.KEY='test';module.json=json
                module.log=lambda **kw:None;module.save=lambda name,value:saved.update({name:value})
                module.raw_item=lambda e,r,body:{'ids':r['tokens'],'text':body}
                rows=[{'id':'a','question':'Q1','choices':['Alpha','B','C','D'],'correct_choice':1,'tokens':[1,2]},
                    {'id':'b','question':'Q2','choices':['Gamma','B','C','D'],'correct_choice':1,'tokens':[3,4,5]}]
                module.freeform_recovery(Engine(),rows,max_new=8)
                with gzip.open(Path(temp)/'causal/untouched/freeform/responses.jsonl.gz','rt') as f:records=[json.loads(x) for x in f]
                self.assertEqual(len(calls),1);self.assertEqual(calls[0][0].shape,(4,3))
                self.assertTrue(np.array_equal(calls[0][1][0],[0,1,1]))
                self.assertEqual(records[0]['output_ids'],[31]);self.assertFalse(records[0]['hit_generation_cap'])
                self.assertTrue(records[0]['strict_normalized_exact_match'])
                self.assertEqual(len(records[1]['output_ids']),8);self.assertTrue(records[1]['hit_generation_cap'])
                self.assertEqual(len(records[0]['raw_generated_ids']),8)
                self.assertEqual({(r['id'],r['condition']) for r in records},{('a','ordinary'),('a','best_guess'),('b','ordinary'),('b','best_guess')})
        finally:
            for k,v in before.items():setattr(module,k,v)
    def test_balancing_uses_disjoint_extreme_pools(self):
        labels=np.tile(np.arange(1,5),100)
        low,high=module.balanced_extremes(np.arange(400),labels)
        self.assertEqual(len(low),25);self.assertEqual(len(high),25)
        self.assertLess(max(low),75);self.assertGreaterEqual(min(high),325)
        self.assertFalse(set(low)&set(high))
        self.assertLessEqual(max(np.bincount(labels[low])),7)

    def test_unestimable_direction_does_not_fake_balance(self):
        with self.assertRaises(ValueError):
            module.balanced_extremes(np.arange(400),np.ones(400))

    def test_spec_norm_and_sign(self):
        v={'ashamed':np.array([[1.,0.],[0.,1.]])}
        specs=module.causal_options(v,np.array([2.,3.]),1,['ashamed'])
        self.assertIsNone(specs[0]['intervention'])
        self.assertEqual([s['intervention']['scale'] for s in specs[1:]],[-3.,3.])

    def test_rotated_labels_preserve_gold_answer(self):
        class Tokenizer:
            def apply_chat_template(self,*a,**kw):return {'input_ids':[11,12]}
            def encode(self,*a,**kw):return [13]
        class Engine:tok=Tokenizer()
        r={'id':'x','question':'Q','choices':['A','B','C','D'],'correct_choice':3}
        for abstention in range(1,6):
            item=module.rotated_item(Engine(),r,abstention)
            self.assertNotEqual(item['correct_choice'],abstention)
            self.assertIn(str(item['correct_choice'])+') C',item['text'])

    def test_rows_specs_and_invalid_outputs_survive_serialization(self):
        class Tokenizer:
            def decode(self,t):return str(t[0])
        class Engine:
            np=np;tok=Tokenizer();variant='untouched';digit_ids=[101,102,103,104,105]
            def decision_context(self,items,capture):return {'items':items}
            def cached_decision(self,context,intervention,capture):
                n=len(context['items']);logits=np.tile(np.array([1.,2.,3.,4.,5.]),(n,1))
                if intervention is not None:logits[:,4]+=intervention['scale']
                return {'digit_logits':logits,'argmax_ids':np.array([999]*n),
                    'logsumexp':np.ones(n)*7,'top_ids':np.ones((n,32),dtype=int),
                    'top_logits':np.zeros((n,32))}
        rows=[{'id':str(i),'correct_choice':2} for i in range(3)]
        def batches(engine,rows,phase):
            for start in range(0,len(rows),2):
                batch=rows[start:start+2]
                yield start,batch,[{'ids':[1,2],'text':'Q','positions':[0,1]} for r in batch]
        module.question_batches=batches;module.json=json;module.os=__import__('os');module.log=lambda **kw:None;module.KEY='test'
        with tempfile.TemporaryDirectory() as folder:
            module.OUT=Path(folder)
            def save(name,value):
                p=module.OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
            module.save=save
            specs=module.causal_options({'ashamed':np.array([[1.,0.]])},np.array([2.]),0,['ashamed'])
            scores=module.causal_block(Engine(),'test',rows,specs,item_builder=lambda r:{'ids':[1,2],
                'text':'Q','positions':[1,1],'correct_choice':4,'abstention_label':2})
            self.assertEqual(scores.shape,(3,3,5))
            self.assertTrue(np.all(scores[:,1,4]==3));self.assertTrue(np.all(scores[:,2,4]==7))
            data=np.load(Path(folder)/'causal/untouched/test/decisions.npz')
            self.assertTrue(np.all(data['actual_choice']==-1))
            self.assertTrue(np.all(data['correct_choice']==4))
            self.assertTrue(np.all(data['abstention_label']==2))
            data.close()
            with gzip.open(Path(folder)/'causal/untouched/test/trials.jsonl.gz','rt') as f:
                entries=[json.loads(line) for line in f]
            self.assertEqual(len(entries),9)
            self.assertTrue(all(e['actual_valid_choice'] is None for e in entries))

    def test_full_workflow_uses_discovery_only_and_every_seed(self):
        cfg=json.loads((Path(__file__).resolve().parents[3]/'outputs/activation_study_v1/input/protocol.json').read_text(encoding='utf8'))
        variants=['untouched']+[f'e3_{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']]
        class Engine:
            np=np;layers=list(range(8));variant='untouched'
            def set_variant(self,name):self.variant=name
        engine=Engine();engine.cfg=cfg
        names=cfg['interventions']['primary_concepts']+cfg['interventions']['specificity_controls']+['confidence_margin']+['random_101','random_202','random_303']
        vectors={name:np.tile([1.,0.],(8,1)) for name in names}
        module.phase_sources=lambda:{(v,'test'):None for v in variants}
        module.causal_directions=lambda e,s:(vectors,np.ones(8))
        calls=[];saved={};freeform=[];traces=[];finished=[]
        def block(e,name,rows,specs,**kwargs):
            calls.append((e.variant,name,[r['id'] for r in rows],specs))
            values=np.zeros((len(rows),len(specs),5),dtype=np.float32)
            for j,s in enumerate(specs[1:],1):
                if s.get('direction') in cfg['interventions']['primary_concepts']:
                    values[:,j,4]=s.get('alpha',1)*(.5 if s.get('layer')==4 else .05)
            return values
        old={name:getattr(module,name,None) for name in ['causal_block','save','freeform_recovery','component_traces','finalize','softmax']}
        try:
            module.causal_block=block;module.save=lambda name,value:saved.update({name:value})
            module.freeform_recovery=lambda e,rows:freeform.append((e.variant,[r['id'] for r in rows]))
            module.component_traces=lambda e,rows:traces.append((e.variant,[r['id'] for r in rows]))
            module.finalize=lambda stage:finished.append(stage)
            module.softmax=lambda x:np.exp(x-x.max(-1,keepdims=True))/np.exp(x-x.max(-1,keepdims=True)).sum(-1,keepdims=True)
            module.causal(engine)
        finally:
            for name,value in old.items():setattr(module,name,value)
        self.assertEqual(saved['causal/selected_layer.json']['layer'],4)
        discovery=[row for row in calls if row[1].startswith('discovery_')]
        self.assertEqual(len(discovery),4*8)
        self.assertTrue(all(len(row[2])==100 and all(q.startswith('cal-09') for q in row[2]) for row in discovery))
        confirm=[row for row in calls if row[1]=='confirmation']
        self.assertEqual({r[0] for r in confirm},set(variants))
        self.assertTrue(all(r[2][0]=='test-0000' and r[2][-1]=='test-0499' and len(r[2])==500 for r in confirm))
        self.assertEqual(len(freeform),13);self.assertTrue(all(len(r[1])==64 for r in freeform))
        self.assertEqual(len(traces),5);self.assertTrue(all(len(r[1])==8 for r in traces))
        self.assertEqual(finished,['causal'])


if __name__=='__main__':unittest.main()
