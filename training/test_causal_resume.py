"""CPU tests: resumption must reject changed IDs/specs/data, not mask failures."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from modal_mechanism import install_causal_resume

class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name)
        self.prefix='causal/untouched/confirmation/'
        self.specs=[{'name':'baseline','intervention':None},{'name':'signed','intervention':{'layer':2}}]
        folder=self.out/self.prefix;folder.mkdir(parents=True)
        (folder/'phase_manifest.json').write_text(json.dumps({'status':'complete','ids':['q'],
            'specs':[{'name':'baseline'},{'name':'signed'}],'activation_archive':None}))
        np.save(folder/'candidate_logits.npy',np.arange(10,dtype=np.float32).reshape(1,2,5))
        entries={}
        for p in folder.iterdir():entries[p.relative_to(self.out).as_posix()]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        self.calls=[]
        def original(*args,**kwargs):self.calls.append(args);return 'recomputed'
        self.ns=dict(causal_block=original,freeform_recovery=original,component_traces=original,log=lambda **kw:None)
        install_causal_resume(self.ns,self.out,{'files':entries,'original_selected_layer':2})
        self.engine=SimpleNamespace(variant='untouched')
    def tearDown(self):self.temp.cleanup()
    def test_complete_unchanged_block_is_reused(self):
        v=self.ns['causal_block'](self.engine,'confirmation',[{'id':'q'}],self.specs)
        self.assertEqual(v.shape,(1,2,5));self.assertFalse(self.calls)
    def test_changed_question_is_rejected(self):
        with self.assertRaises(AssertionError):self.ns['causal_block'](self.engine,'confirmation',[{'id':'other'}],self.specs)
    def test_changed_intervention_is_rejected(self):
        specs=[{'name':'different','intervention':{'layer':2}}]
        with self.assertRaises(AssertionError):self.ns['causal_block'](self.engine,'confirmation',[{'id':'q'}],specs)
    def test_changed_layer_is_rejected(self):
        with self.assertRaises(AssertionError):self.ns['causal_block'](self.engine,'confirmation',[{'id':'q'}],[{'name':'x','intervention':{'layer':3}}])
    def test_tampered_logits_are_rejected(self):
        with (self.out/self.prefix/'candidate_logits.npy').open('ab') as f:f.write(b'changed')
        with self.assertRaises(AssertionError):self.ns['causal_block'](self.engine,'confirmation',[{'id':'q'}],self.specs)
    def test_incomplete_block_is_recomputed(self):
        v=self.ns['causal_block'](self.engine,'dose',[{'id':'q'}],self.specs)
        self.assertEqual(v,'recomputed');self.assertEqual(len(self.calls),1)
    def test_incompatible_capture_request_is_rejected(self):
        with self.assertRaises(AssertionError):self.ns['causal_block'](self.engine,'confirmation',[{'id':'q'}],self.specs,captures=True)

if __name__=='__main__':unittest.main()
