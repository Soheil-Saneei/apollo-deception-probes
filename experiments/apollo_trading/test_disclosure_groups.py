"""Validate exact permutation distances against explicit regrouping."""
import itertools
import unittest

try:
    import torch
    from compare_disclosure_groups import permutation_distances, holm
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'requires analysis dependencies')
class GroupTests(unittest.TestCase):
    def test_all_assignments_match_direct_means(self):
        torch.manual_seed(7)
        x = torch.randn(6, 3, 5, dtype=torch.float64)
        actual = permutation_distances(x, 2)
        expected = []
        for chosen in itertools.combinations(range(6), 2):
            rest = [i for i in range(6) if i not in chosen]
            expected.append((x[list(chosen)].mean(0) - x[rest].mean(0)).norm(dim=-1))
        torch.testing.assert_close(actual, torch.stack(expected))
        self.assertEqual(len(actual), 15)

    def test_common_offset_does_not_change_l2(self):
        torch.manual_seed(8)
        x = torch.randn(5, 2, 4, dtype=torch.float64)
        torch.testing.assert_close(permutation_distances(x, 2), permutation_distances(x + 100, 2))

    def test_holm_known_example(self):
        p = torch.tensor([.04, .01, .03], dtype=torch.float64)
        torch.testing.assert_close(holm(p), torch.tensor([.06, .03, .06], dtype=torch.float64))


if __name__ == '__main__':
    unittest.main()
