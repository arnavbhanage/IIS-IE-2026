"""Run with: python -m unittest discover -s placement-classifier/tests"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from app import app
from predict import predict_with_explanation

BASE = dict(cgpa=8.7, attendance_percent=85, leadership_roles=0, academic_gaps=0, project_complexity=3, internships=1, coding_questions_solved=180)

class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_representative_profiles(self):
        for patch, expected in [({}, '1'), ({'cgpa':8.5,'coding_questions_solved':0,'project_complexity':1}, '1'), ({'cgpa':7.7,'internships':0,'leadership_roles':1,'coding_questions_solved':100}, '2'), ({'cgpa':6.5,'internships':0,'project_complexity':2,'coding_questions_solved':35}, '3')]:
            with self.subTest(patch=patch):
                result = predict_with_explanation(BASE | patch)
                self.assertEqual(result['prediction'], expected)
                self.assertEqual(len(result['decision_path']),len(result['explanation']['steps']))
                self.assertAlmostEqual(sum(result['probabilities'].values()),1)
                self.assertAlmostEqual(sum(result['local_feature_contributions'].values()),1)
                self.assertIn(f"Category {expected}",result['explanation']['summary'])

    def test_more_experience_does_not_penalize_strong_profile(self):
        for count in [0,1,2,5,10]:
            result = predict_with_explanation(BASE | {'leadership_roles':count,'internships':max(1,count)})
            self.assertEqual(result['prediction'],'1')

    def test_invalid_requests(self):
        for payload in [None, [], {}, BASE|{'cgpa':11}, BASE|{'internships':.5}, BASE|{'leadership_roles':True}, BASE|{'cgpa':'abc'}, BASE|{'cgpa':float('nan')}, BASE|{'cgpa':float('inf')}]:
            with self.subTest(payload=payload):
                response=self.client.post('/predict',json=payload)
                self.assertEqual(response.status_code,400)
                self.assertIn('error',response.json)
        self.assertEqual(self.client.post('/predict',data='{',content_type='application/json').status_code,400)

    def test_api_and_metadata(self):
        self.assertEqual(self.client.post('/predict',json=BASE).status_code,200)
        info = self.client.get('/model-info')
        self.assertEqual(info.status_code,200)
        self.assertIn('Synthetic',info.json['data_source'])

if __name__ == '__main__':
    unittest.main()
