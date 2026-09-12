import unittest
from collections import Counter
from analyze_report_controls import match_lengths


class MatchingTests(unittest.TestCase):
    def test_exact_matching_preserves_splits_and_no_replacement(self):
        rows=[]
        for split in ('train','validation','test'):
            for label,lengths in [('honest',[3,3,4,9]),('deceptive',[3,4,4,12])]:
                for length in lengths:
                    rows.append(dict(raw_index=len(rows),trajectory_id=str(len(rows)),split=split,label=label,report_token_count=length))
        indices,pairs=match_lengths(rows)
        self.assertEqual(len(indices),12)
        self.assertEqual(len(set(indices)),12)
        self.assertEqual((indices,pairs),match_lengths(rows))
        for split in ('train','validation','test'):
            counts=[Counter(rows[i]['report_token_count'] for i in indices if rows[i]['split']==split and rows[i]['label']==label) for label in ('honest','deceptive')]
            self.assertEqual(counts[0],counts[1])
            self.assertEqual(counts[0],{3:1,4:1})
        for pair in pairs:
            a,b=[rows[int(pair[key])] for key in ('honest_id','deceptive_id')]
            self.assertEqual(a['split'],b['split'])

    def test_missing_overlap_rejected(self):
        with self.assertRaises(ValueError):
            match_lengths([])


if __name__=='__main__':
    unittest.main()
