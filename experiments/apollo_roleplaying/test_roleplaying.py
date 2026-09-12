import unittest

from run import build_examples, encode


class PairTests(unittest.TestCase):
    def test_grouped_pairs_and_exact_text(self):
        data = [dict(scenario=f"Scenario {i}", question="Question?",
                     honest_completion="  Honest answer\n", deceptive_completion="False answer")
                for i in range(30)]
        data.append(dict(data[0], question="Another question?"))
        rows = build_examples(data)
        self.assertEqual(rows, build_examples(data))
        seen = {}
        for a, b in zip(rows[::2], rows[1::2]):
            self.assertEqual(a["messages"], b["messages"])
            self.assertEqual(a["split"], b["split"])
            self.assertEqual(a["completion"], "  Honest answer\n")
            self.assertEqual((a["label"], b["label"]), ("honest", "deceptive"))
            group = a["scenario_id"]
            self.assertEqual(seen.setdefault(group, a["split"]), a["split"])
        self.assertEqual(len(seen), 30)

    def test_reject_bad_pairs(self):
        with self.assertRaises(ValueError):
            build_examples([dict(scenario="S", question="Q", honest_completion="A",
                                 deceptive_completion="A")])

    def test_reject_token_straddling_boundary(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return "abc"

            def __call__(self, *args, **kwargs):
                return dict(input_ids=[0, 1], offset_mapping=[(0, 2), (2, 4)])

        with self.assertRaisesRegex(ValueError, "straddles"):
            encode(Tokenizer(), dict(messages=[], completion="d"))


if __name__ == "__main__":
    unittest.main()
