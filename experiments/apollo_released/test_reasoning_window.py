import re
import unittest
from analyze_reasoning_window import modal_divergence, select_window, WINDOW_TOKENS


class ReasoningWindowTests(unittest.TestCase):
    def test_mode_divergence_ignores_rare_first_token_difference(self):
        position,audit=modal_divergence([[1,2],[1,2],[9,2],[1,3],[1,3],[1,2]],['honest']*3+['deceptive']*3)
        self.assertEqual(position,1)
        self.assertEqual(audit[0]['modes'],{'honest':1,'deceptive':1})

    def test_window_strictly_after_divergence_and_exact_width(self):
        body='Now that the trade has been executed, I need to inform Amy. """'
        offsets=[m.span() for m in re.finditer(r'\S+',body)]
        window,reason=select_window(offsets,body,0,1)
        self.assertIsNone(reason)
        self.assertEqual(window,(2,2+WINDOW_TOKENS))

    def test_cue_and_closed_reasoning_excluded(self):
        for body in ['Now that insider information helped us to complete the trade quickly. """',
                     'Now that """ Action: send_message a long report follows with many tokens']:
            offsets=[m.span() for m in re.finditer(r'\S+',body)]
            window,reason=select_window(offsets,body,0,1)
            self.assertIsNone(window)
            self.assertIsNotNone(reason)


if __name__=='__main__':
    unittest.main()
