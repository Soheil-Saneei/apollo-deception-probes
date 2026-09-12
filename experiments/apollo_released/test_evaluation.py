from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing
import unittest

import numpy as np
from evaluate_reports import evaluation_rows, fit_layer, SEEDS


def fixture():
    rows=[]
    prefix='Reasoning: context.\nAction: send_message\n'
    report='Executed trade update: exact report text\n"""'
    for label,n,training in [('honest',666,533),('deceptive',334,267)]:
        for j in range(n):
            i=len(rows)
            rows.append(dict(raw_index=i,trajectory_id=f'trajectory-{i}',apollo_loader_id=i,
                group_id=f'group-{i}',label=label,report_label='explicit' if label=='honest' else 'concealed',
                split='train' if j<training else 'test',
                messages=[dict(role='assistant',content=prefix+report)],
                char_spans=dict(assistant=[100,100+len(prefix+report)],report=[100+len(prefix),100+len(prefix+report)]),
                token_positions=dict(report=[10,11,12])))
    return rows


class EvaluationTests(unittest.TestCase):
    def test_validation_only_from_training_and_original_test_fixed(self):
        rows=fixture()
        original=copy.deepcopy(rows)
        result=evaluation_rows(rows)
        self.assertEqual(rows,original)
        self.assertEqual(result,evaluation_rows(rows))
        self.assertEqual(Counter(r['split']+'/'+r['label'] for r in result),
            {'train/honest':433,'train/deceptive':217,'validation/honest':100,
             'validation/deceptive':50,'test/honest':133,'test/deceptive':67})
        for before,after in zip(rows,result):
            self.assertEqual(before['split']=='test',after['split']=='test')
            if after['split']=='validation':
                self.assertEqual(before['split'],'train')
            self.assertEqual(after['report_text'],'Executed trade update: exact report text\n"""')
            self.assertEqual(after['report_token_count'],3)

    def test_duplicate_trajectory_rejected(self):
        rows=fixture()
        rows[1]['raw_index']=rows[0]['raw_index']
        with self.assertRaisesRegex(ValueError,'unique trajectory'):
            evaluation_rows(rows)

    def test_parallel_fits_match_serial(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(32,4)).astype(np.float32)
        y=np.tile([0,1],16)
        masks=dict(train=np.arange(32)<16,validation=(np.arange(32)>=16)&(np.arange(32)<24),
                   test=np.arange(32)>=24)
        labels=[np.random.default_rng(s).permutation(y[:16]) for s in SEEDS]
        expected=fit_layer(0,x,masks,y,labels)
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(fit_layer,i,x,masks,y,labels) for i in range(2)]
            for future in futures:
                actual=future.result()
                np.testing.assert_array_equal(actual[2],expected[2])
                np.testing.assert_array_equal(actual[3],expected[3])


if __name__=='__main__':
    unittest.main()
