"""Check leave-one-out exclusion and empirical rank conventions."""
import unittest

try:
    import torch
    from analyze_report_outlier import compare, distances
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires analysis dependencies")
class OutlierTests(unittest.TestCase):
    def test_leave_one_out_and_tie_ranks(self):
        group = torch.tensor([[[1., 0.]], [[3., 0.]], [[5., 0.]]])
        result = compare(torch.tensor([[6., 0.]]), group)
        torch.testing.assert_close(result['l2']['control'][:, 0], torch.tensor([3., 0., 3.], dtype=torch.float64))
        self.assertEqual(result['l2']['target'].item(), 3)
        self.assertEqual(result['l2']['rank'].item(), 3)
        self.assertAlmostEqual(result['l2']['percentile'].item(), 200 / 3, places=4)
        self.assertEqual(result['cosine_distance']['percentile'].item(), 50)
        self.assertEqual(result['cosine_distance']['rank'].item(), 2.5)

    def test_farthest_candidate(self):
        group = torch.tensor([[[1., 0.]], [[3., 0.]], [[5., 0.]]])
        result = compare(torch.tensor([[10., 0.]]), group)['l2']
        self.assertEqual(result['rank'].item(), 4)
        self.assertEqual(result['percentile'].item(), 100)

    def test_matches_explicit_exclusion(self):
        torch.manual_seed(0)
        group = torch.randn(5, 3, 7, dtype=torch.float64)
        result = compare(torch.randn(3, 7), group)
        for i in range(5):
            center = torch.cat([group[:i], group[i+1:]]).mean(dim=0)
            for metric, value in distances(group[i], center).items():
                torch.testing.assert_close(result[metric]['control'][i], value)


if __name__ == '__main__':
    unittest.main()
