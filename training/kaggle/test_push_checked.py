import unittest
from unittest.mock import patch
from subprocess import CompletedProcess
from push_checked import push_checked

class PushReceiptTests(unittest.TestCase):
    def test_quota_rejection_with_zero_exit_is_failure(self):
        with patch('push_checked.subprocess.run',return_value=CompletedProcess([],0,'Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.')):
            with self.assertRaises(RuntimeError):push_checked(['kaggle'],'owner/job')
    def test_wrong_job_receipt_is_failure(self):
        with patch('push_checked.subprocess.run',return_value=CompletedProcess([],0,'Kernel successfully pushed. https://www.kaggle.com/code/owner/other')):
            with self.assertRaises(RuntimeError):push_checked(['kaggle'],'owner/job')
    def test_matching_acceptance_is_success(self):
        with patch('push_checked.subprocess.run',return_value=CompletedProcess([],0,'Kernel successfully pushed. https://www.kaggle.com/code/owner/job')):
            self.assertEqual(push_checked(['kaggle'],'owner/job').returncode,0)

if __name__=='__main__':unittest.main()
