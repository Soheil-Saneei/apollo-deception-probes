import hashlib
import unittest

import numpy as np

from sanity_checks import fit_text, metrics, paired_bootstrap, validate_rows


def examples():
    rows = []
    for group, split in enumerate(('train', 'train', 'validation', 'test')):
        for label in ('honest', 'deceptive'):
            text = ('truth clean' if label == 'honest' else 'lie dirty') + ' ' + split+'only'
            rows.append(dict(id=f'{group}-{label}', scenario_id=str(group), split=split,
                label=label, completion=text, completion_sha256=hashlib.sha256(text.encode()).hexdigest(),
                messages=[dict(role='system', content=f'Scenario {group}')]))
    return rows


class SanityTests(unittest.TestCase):
    def test_vocabulary_fit_on_training_text_only(self):
        rows = examples()
        masks = validate_rows(rows)
        y = np.array([r['label'] == 'deceptive' for r in rows], dtype=int)
        vectorizer, _, p, _ = fit_text(np.array([r['completion'] for r in rows]), masks, y)
        self.assertIn('trainonly', vectorizer.vocabulary_)
        self.assertNotIn('validationonly', vectorizer.vocabulary_)
        self.assertNotIn('testonly', vectorizer.vocabulary_)
        self.assertEqual(len(p), 2)

    def test_reject_split_leakage_and_changed_text(self):
        rows = examples()
        rows[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'crosses splits'):
            validate_rows(rows)
        rows = examples()
        rows[0]['completion'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            validate_rows(rows)

    def test_metric_definitions(self):
        m = metrics(np.array([0, 1, 0, 1]), np.array([.1, .8, .4, .3]))
        self.assertAlmostEqual(m['auroc'], .75)
        self.assertAlmostEqual(m['auprc'], 5/6)
        self.assertAlmostEqual(m['accuracy'], .75)

    def test_paired_bootstrap_identical_predictors(self):
        y, p = np.array([0,1,0,1]), np.array([.1,.8,.4,.3])
        result = paired_bootstrap(y, p, p, np.array(['a','a','b','b']), n=20)
        for value in result.values():
            self.assertEqual(value['activation_minus_text'], 0.)
            self.assertEqual(value['paired_scenario_bootstrap_95_ci'], [0.,0.])


if __name__ == '__main__':
    unittest.main()
