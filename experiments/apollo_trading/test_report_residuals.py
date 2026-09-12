"""Boundary alignment and vector-distance checks without a model download."""
import unittest

try:
    import torch
    from compare_report_residuals import metrics, report_boundary
except ImportError:
    torch = None


class TokenPieces:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[i] for i in ids)


@unittest.skipIf(torch is None, "requires HF analysis dependencies")
class ReportResidualTests(unittest.TestCase):
    def test_boundary_after_reasoning_and_tool_arguments(self):
        pieces = ['Reasoning: act.\n', '<parameter=text>', '\n', 'Trade', ' report']
        raw = ''.join(pieces)
        index, char_start = report_boundary(raw, list(range(len(pieces))), TokenPieces(pieces))
        self.assertEqual(index, 2)
        self.assertEqual(raw[char_start:], 'Trade report')

    def test_token_straddling_whitespace_and_text(self):
        pieces = ['<parameter=text>', '\nTrade', ' report']
        index, _ = report_boundary(''.join(pieces), [0, 1, 2], TokenPieces(pieces))
        self.assertEqual(index, 0)

    def test_ambiguous_text_parameter_rejected(self):
        with self.assertRaises(ValueError):
            report_boundary('<parameter=text>a<parameter=text>b', [], TokenPieces([]))

    def test_distances_to_group_mean(self):
        target = torch.tensor([[1., 0.], [0., 2.]])
        group = torch.tensor([[[0., 0.], [0., 1.]], [[0., 2.], [0., 3.]]])
        cosine, l2 = metrics(target, group.mean(dim=0))
        torch.testing.assert_close(cosine, torch.tensor([0., 1.]))
        torch.testing.assert_close(l2, torch.tensor([2.**.5, 0.]))


if __name__ == '__main__':
    unittest.main()
