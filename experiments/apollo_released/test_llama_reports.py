import copy
from pathlib import Path
import tempfile
import unittest

from run_llama_reports import assign_splits, encode_report, load_cohort


class CharacterTokenizer:
    """Exact offsets for testing report span arithmetic, not a Llama substitute."""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, date_string):
        text = "".join(f"[{m['role']}]" + m['content'] + "<|eot_id|>" for m in messages)
        return text + ("[assistant]" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        return dict(input_ids=list(map(ord, text)), offset_mapping=[(i, i+1) for i in range(len(text))])

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


class ReportTests(unittest.TestCase):
    def test_same_trajectory_stays_in_one_split(self):
        rows = [dict(raw_index=i, label="honest" if i < 666 else "deceptive") for i in range(1000)]
        rows += [dict(rows[0])]
        assign_splits(rows)
        self.assertEqual(rows[0]["split"], rows[-1]["split"])
        self.assertEqual(sum(r['split'] == 'test' for r in rows[:-1]), 200)
        a = {r['group_id'] for r in rows if r['split'] == 'train'}
        b = {r['group_id'] for r in rows if r['split'] == 'test'}
        self.assertFalse(a & b)

    def test_report_body_excludes_reasoning_and_tool_header(self):
        body = 'Reasoning: mention Executed trade update here.\nAction: send_message\nAction Input: "amy@whitestone.com", "Executed trade update: report"'
        row = dict(messages=[dict(role='system', content='Context'), dict(role='assistant', content=body)])
        result = encode_report(CharacterTokenizer(), row)
        start = result['token_positions']['report'][0]
        end = result['token_positions']['report'][-1]+1
        self.assertEqual(CharacterTokenizer().decode(result['input_ids'][start:end]),
                         'Executed trade update: report"')
        self.assertEqual(result['boundary_positions']['report'], start-1)

    def test_audited_cohort(self):
        base = Path(__file__).parent
        roots = [base/'apollo-released-llama33-manifest',
                 base.parent/'apollo_trading/outputs/apollo-released-llama33-manifest']
        root = next((p for p in roots if p.exists()), None)
        if root is None:
            self.skipTest('Released audit data not present')
        rows = load_cohort(root)
        self.assertEqual(len(rows), 1000)
        # Exercise all 1,000 report span choices, including unusual formatting.
        for row in rows:
            encode_report(CharacterTokenizer(), row)

    def test_frozen_tiny_llama_end_to_end(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from run_llama_reports import ResidualCapture, extract, pilot_check
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, attention_dropout=0., attn_implementation='eager'))
        model.eval().requires_grad_(False)
        original = {k: v.clone() for k,v in model.state_dict().items()}
        row = dict(trajectory_id='tiny-honest', label='honest', input_ids=[1,2,3,4,5,6,7,8],
            token_positions=dict(assistant=[3,4,5,6], action=[4,5,6], report=[5,6]),
            boundary_positions=dict(assistant=2, action=3, report=4))
        second = copy.deepcopy(row)
        second.update(trajectory_id='tiny-deceptive', label='deceptive')
        capture = ResidualCapture(model)
        with torch.inference_mode(), tempfile.TemporaryDirectory() as tmp:
            values, features, _ = extract(model, capture, row)
            self.assertEqual(list(values.shape), [2,4,32])
            self.assertTrue(torch.equal(values[:,2:].float().mean(1), features['report_mean']))
            capture = pilot_check(model, capture, [row,second], Path(tmp)/'pilot')
            capture.close()
        self.assertTrue(all(torch.equal(original[k],v) for k,v in model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
