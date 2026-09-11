"""Backend tests. Tiny random models test mechanics, never behavioral reproduction."""

import json
import tempfile
import unittest
from pathlib import Path

from run_interactive import parse_response
from run_transformers import ActivationRecorder, PresencePenalty, TransformersBackend, parse_qwen_output


class WireFormatTests(unittest.TestCase):
    def test_buy_and_report(self):
        raw = ('Reasoning: act.\n<tool_call>\n<function=execute_trade>\n'
               '<parameter=security_id>LING</parameter>\n<parameter=side>buy</parameter>\n'
               '<parameter=shares>500</parameter>\n</function>\n</tool_call>')
        message, error = parse_qwen_output(raw)
        self.assertIsNone(error)
        call = parse_response({"done": True, "done_reason": "stop", "message": message}, "structured")
        self.assertEqual(call["arguments"]["shares"], 500)
        self.assertEqual(message["content"], "Reasoning: act.")
        report, error = parse_qwen_output('<tool_call><function=send_message>'
            '<parameter=recipient>amy@whitestone.com</parameter>'
            '<parameter=text>Shares: 500\nRisk: high & uncertain</parameter></function></tool_call>')
        self.assertIsNone(error)
        self.assertIn("high & uncertain", report["tool_calls"][0]["function"]["arguments"]["text"])

    def test_multiple_and_malformed_calls_never_execute(self):
        call = '<tool_call><function=query_stock_analyze></function></tool_call>'
        for raw in (call + call, call + 'invented output', call[:-5],
                    '<tool_call><function=find_trades><parameter=risk_level>low</parameter>'
                    '<parameter=risk_level>high</parameter></function></tool_call>'):
            with self.subTest(raw=raw):
                message, _ = parse_qwen_output(raw)
                with self.assertRaises(ValueError):
                    parse_response({"done": True, "done_reason": "stop", "message": message}, "structured")


try:
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "install requirements-hf.txt for real PyTorch tests")
class ActivationTests(unittest.TestCase):
    def setUp(self):
        import inspect
        from unittest.mock import patch
        from transformers.models.qwen3_5 import modeling_qwen3_5

        # These tiny CPU tests exercise hooks and alignment. Installed GPU-only
        # kernels must not redirect their CPU tensors into CUDA/Triton code.
        # Real GPU kernels are validated separately with the trained-model pilot.
        for name in ("causal_conv1d_fn", "causal_conv1d_update",
                     "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            reference = inspect.unwrap(getattr(modeling_qwen3_5, name))
            replacement = patch.object(modeling_qwen3_5, name, reference)
            replacement.start()
            self.addCleanup(replacement.stop)
        torch.manual_seed(42)
        config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            layer_types=["linear_attention", "full_attention"], eos_token_id=2, pad_token_id=0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000,
                             "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0]})
        self.model = Qwen3_5ForCausalLM(config).eval()
        self.ids = torch.tensor([[3, 4, 5]])

    def test_residual_identities_and_no_effect_on_logits(self):
        with torch.inference_mode():
            expected = self.model(self.ids, use_cache=False).logits
            with ActivationRecorder(self.model) as recorder:
                actual = self.model(self.ids, use_cache=False).logits
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(recorder.positions, [2])
        self.assertEqual(set(recorder.modules.values()), {"linear_attention", "full_attention"})
        for name in recorder.modules:
            values = {suffix: recorder.values[name + '.' + suffix][0]
                      for suffix in ("resid_pre", "resid_mid", "resid_post", "mixer_out", "mlp_out")}
            torch.testing.assert_close(values["resid_mid"], values["resid_pre"] + values["mixer_out"])
            torch.testing.assert_close(values["resid_post"], values["resid_mid"] + values["mlp_out"])
        self.assertEqual(len(self.model._forward_pre_hooks), 0)

    def test_cached_generation_alignment_capture_limit_and_roundtrip(self):
        from safetensors.torch import load_file
        with torch.inference_mode():
            expected = self.model.generate(self.ids, max_new_tokens=4, do_sample=False, eos_token_id=None)
            with ActivationRecorder(self.model, max_steps=2) as recorder:
                actual = self.model.generate(self.ids, max_new_tokens=4, do_sample=False, eos_token_id=None)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(recorder.positions, [2, 3])
        self.assertEqual(recorder.step + 1, 4)
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / 'capture'
            recorder.write(stem)
            tensors = load_file(str(stem) + '-activations.safetensors')
            self.assertEqual(len(tensors), 10)
            self.assertTrue(all(t.shape == (2, 32) for t in tensors.values()))
            metadata = json.loads(Path(str(stem) + '-activations.json').read_text())
            self.assertEqual(metadata['positions_in_prompt_plus_generated'], [2, 3])

    def test_presence_penalty_is_additive_and_windowed(self):
        scores = torch.zeros(1, 8)
        result = PresencePenalty(1.5, 2)(torch.tensor([[1, 2, 2]]), scores)
        self.assertEqual(result[0, 1].item(), 0)
        self.assertEqual(result[0, 2].item(), -1.5)

    def test_conditional_model_backend_saves_tokens_prompt_and_activations(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        config = Qwen3_5Config(text_config=self.model.config.to_dict(), vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64,
            "num_heads": 2, "out_hidden_size": 32, "num_position_embeddings": 16})
        model = Qwen3_5ForConditionalGeneration(config).eval()

        class TestTokenizer:
            eos_token_id = 2
            pad_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                assert kwargs["enable_thinking"] is False
                return "test prompt"

            def __call__(self, prompt, **kwargs):
                from transformers import BatchEncoding
                return BatchEncoding({"input_ids": torch.tensor([[3, 4, 5]]),
                                      "attention_mask": torch.ones(1, 3, dtype=torch.long)})

            def decode(self, tokens, **kwargs):
                return "random output"

        with tempfile.TemporaryDirectory() as directory:
            backend = TransformersBackend(model, TestTokenizer(), Path(directory), capture_steps=2)
            result = backend.infer({"messages": [], "tools": [], "options": {
                "temperature": 1.0, "top_p": 1.0, "top_k": 0, "repeat_penalty": 1.0,
                "num_predict": 3, "num_ctx": 32, "seed": 42}}, 0, 0, 0)
            files = {f.name for f in Path(directory).iterdir()}
            stem = 'trajectory-00-turn-00-attempt-00'
            self.assertIn(stem + '-activations.safetensors', files)
            self.assertIn(stem + '-response.json', files)
            tokens = json.loads((Path(directory) / (stem + '-tokens.json')).read_text())
            self.assertEqual(tokens['prompt_length'], 3)
            self.assertEqual(result['raw_output'], 'random output')


if __name__ == '__main__':
    unittest.main()
