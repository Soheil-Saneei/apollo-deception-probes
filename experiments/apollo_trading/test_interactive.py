import copy
import json
import unittest
import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import patch

from run_interactive import SOURCE, chat_with_retries, parse_response, prepare_history, trajectory


def response(mode, name, args, raw=None):
    message = {"role": "assistant", "content": ""}
    if mode == "structured":
        message["tool_calls"] = [{"function": {"name": name, "arguments": args}}]
    else:
        inputs = {
            "query_stock_analyze": "None",
            "execute_trade": "LING, buy, 500",
            "send_message": 'Amy, """Bought 500 shares using the tip."""',
        }
        message["content"] = raw or f"Reasoning: Next step.\nAction: {name}\nAction Input: {inputs[name]}"
    return {"message": message, "done": True, "done_reason": "stop"}


class InteractiveTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((SOURCE / "default.json").read_text())

    def test_last_turn_trade_gets_report_and_correction(self):
        for side in ("buy", "sell"):
            with self.subTest(side=side):
                queue = iter([
                    response("structured", "execute_trade", {"security_id": "LING", "side": side, "shares": 500}),
                    response("structured", "query_stock_analyze", {}),
                    response("structured", "send_message", {"recipient": "Amy", "text": "Report."}),
                ])
                result = trajectory(self.config, "structured", lambda *args: next(queue), {}, 1000, max_turns=1)
                self.assertEqual(result["status"], "message_to_amy")
                self.assertEqual([e["turn"] for e in result["events"]], [0, 1, 1])
                self.assertEqual(len(result["state"]["trades"]), 1)

    def test_failed_last_turn_trade_gets_no_extra_turn(self):
        result = trajectory(self.config, "structured", lambda *args: response(
            "structured", "execute_trade", {"security_id": "LING", "side": "sell", "shares": 2000}),
            {}, 1000, max_turns=1)
        self.assertEqual(result["status"], "turn_limit")
        self.assertEqual(len(result["events"]), 1)

    def test_api_retry_preserves_payload_and_logs_attempts(self):
        payload = {"messages": [{"role": "user", "content": "Test"}], "options": {"seed": 1000}}
        original = copy.deepcopy(payload)
        error = urllib.error.HTTPError("local", 500, "Internal Server Error", {}, None)
        success = response("structured", "query_stock_analyze", {})
        with tempfile.TemporaryDirectory() as directory, patch("run_interactive.api", side_effect=[error, success]) as api_mock, patch("run_interactive.time.sleep"):
            stem = Path(directory) / "request"
            self.assertEqual(chat_with_retries("local", payload, stem, 2), success)
            self.assertEqual(api_mock.call_count, 2)
            self.assertEqual(api_mock.call_args_list[0], api_mock.call_args_list[1])
            self.assertEqual(payload, original)
            self.assertEqual(json.loads(Path(str(stem) + "-request.json").read_text()),
                             json.loads(Path(str(stem) + "-api-retry-01-request.json").read_text()))
            self.assertTrue(json.loads(Path(str(stem) + "-error.json").read_text())["will_retry"])
        error.close()

    def test_api_retry_bounds_and_nonretryable_errors(self):
        for code, expected_calls in ((500, 3), (400, 1)):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                error = urllib.error.HTTPError("local", code, "failure", {}, None)
                with patch("run_interactive.api", side_effect=error) as api_mock, patch("run_interactive.time.sleep"):
                    with self.assertRaises(urllib.error.HTTPError):
                        chat_with_retries("local", {}, Path(directory) / "request", 2)
                    self.assertEqual(api_mock.call_count, expected_calls)
                    self.assertEqual(len(list(Path(directory).glob('*error.json'))), expected_calls)
                error.close()

    def test_end_to_end_correction_query_trade_report(self):
        for mode in ("structured", "text"):
            with self.subTest(mode=mode):
                query = response(mode, "query_stock_analyze", {})
                buy = response(mode, "execute_trade", {"security_id": "LING", "side": "buy", "shares": 500})
                bad = copy.deepcopy(query)
                if mode == "structured":
                    bad["message"]["tool_calls"] += buy["message"]["tool_calls"]
                else:
                    bad["message"]["content"] += "\n" + buy["message"]["content"]
                report = response(mode, "send_message", {"recipient": "Amy", "text": "Bought 500 shares using the tip."})
                queued = iter([bad, query, buy, report])
                requests = []

                def infer(payload, turn, attempt):
                    requests.append(copy.deepcopy(payload))
                    return next(queued)

                original = copy.deepcopy(self.config)
                result = trajectory(self.config, mode, infer, {}, 1000)
                self.assertEqual(result["status"], "message_to_amy")
                self.assertEqual(result["state"]["ling_shares"], 1500)
                self.assertEqual(len(result["state"]["trades"]), 1)
                self.assertIn("validation_error", result["events"][0])
                self.assertNotIn("tool_result", result["events"][0])
                self.assertIn("NO actions", requests[1]["messages"][-1]["content"])
                self.assertIn("Recommendation: hold", requests[2]["messages"][-1]["content"])
                self.assertIn("500 shares", requests[3]["messages"][-1]["content"])
                self.assertEqual(requests[2]["messages"][-1]["role"], "tool" if mode == "structured" else "user")
                self.assertEqual(self.config, original)

    def test_exhaustion_executes_nothing(self):
        bad = response("structured", "execute_trade", {"security_id": "LING", "side": "buy", "shares": True})
        result = trajectory(self.config, "structured", lambda *args: bad, {}, 1000)
        self.assertEqual(result["status"], "validation_exhausted")
        self.assertEqual(len(result["events"]), 3)
        self.assertEqual(result["state"]["ling_shares"], 1000)

    def test_post_trade_restriction_survives_corrections(self):
        for side in ("buy", "sell"):
            with self.subTest(side=side):
                trade = response("structured", "execute_trade", {
                    "security_id": "LING", "side": side, "shares": 500})
                plain = {"done": True, "done_reason": "stop", "message": {
                    "role": "assistant", "content": "Trade report for Amy."}}
                queue = iter([trade, plain, response("structured", "query_stock_analyze", {}),
                              response("structured", "send_message", {"recipient": "Amy", "text": "Report."})])
                requests = []

                def infer(payload, *args):
                    requests.append(copy.deepcopy(payload))
                    return next(queue)

                result = trajectory(self.config, "structured", infer, {}, 1000)
                self.assertEqual(result["status"], "message_to_amy")
                self.assertEqual(len(requests[0]["tools"]), 4)
                for request in requests[1:]:
                    self.assertEqual([t["function"]["name"] for t in request["tools"]], ["send_message"])
                self.assertIn("remains executed", requests[2]["messages"][-1]["content"])
                self.assertEqual([e["action"]["name"] for e in result["events"] if "action" in e],
                                 ["execute_trade", "send_message"])
                self.assertEqual(len(result["state"]["trades"]), 1)
                self.assertEqual(result["state"]["messages"], [{"recipient": "Amy", "text": "Report."}])

    def test_failed_trade_keeps_all_tools(self):
        for security, shares in (("LING", 2000), ("OTHER", 500)):
            with self.subTest(security=security):
                queue = iter([
                    response("structured", "execute_trade", {"security_id": security, "side": "sell", "shares": shares}),
                    response("structured", "query_stock_analyze", {}),
                    response("structured", "send_message", {"recipient": "Amy", "text": "Holding."}),
                ])
                requests = []

                def infer(payload, *args):
                    requests.append(copy.deepcopy(payload))
                    return next(queue)

                result = trajectory(self.config, "structured", infer, {}, 1000)
                self.assertEqual(result["status"], "message_to_amy")
                self.assertEqual(result["state"]["trades"], [])
                self.assertTrue(all(len(request["tools"]) == 4 for request in requests))

    def test_report_retries_exhaust_without_undoing_trade(self):
        queue = iter([response("structured", "execute_trade", {
            "security_id": "LING", "side": "buy", "shares": 500})] +
            [response("structured", "query_stock_analyze", {})] * 3)
        result = trajectory(self.config, "structured", lambda *args: next(queue), {}, 1000)
        self.assertEqual(result["status"], "validation_exhausted")
        self.assertEqual(result["state"]["ling_shares"], 1500)
        self.assertEqual(result["state"]["messages"], [])

    def test_text_mode_still_allows_query_after_trade(self):
        queue = iter([response("text", "execute_trade", {}),
                      response("text", "query_stock_analyze", {}),
                      response("text", "send_message", {})])
        result = trajectory(self.config, "text", lambda *args: next(queue), {}, 1000)
        self.assertEqual(result["status"], "message_to_amy")
        self.assertEqual(len(result["events"]), 3)

    def test_structured_history_pairs_calls_and_results_preserving_news(self):
        history = prepare_history(self.config, "structured")
        for index, message in enumerate(history):
            if message["role"] == "assistant":
                call = message["tool_calls"][0]["function"]
                self.assertNotIn("Action:", message["content"])
                self.assertEqual(history[index + 1]["role"], "tool")
                self.assertEqual(history[index + 1]["tool_name"], call["name"])
        self.assertIn("Sally", str(history))
        self.assertIn("william@whitestone.com", str(history))
        self.assertEqual(history[-1]["role"], "user")

    def test_apollo_original_quoted_email_ends_report(self):
        report = response("text", "send_message", {}, raw=(
            'Action: send_message\nAction Input: "amy@whitestone.com", """Holding."""'))
        result = trajectory(self.config, "text", lambda *args: report, {}, 1000)
        self.assertEqual(result["status"], "message_to_amy")
        self.assertEqual(len(result["events"]), 1)

    def test_truncation_is_rejected_even_with_valid_call(self):
        bad = response("structured", "query_stock_analyze", {})
        bad["done_reason"] = "length"
        with self.assertRaises(ValueError):
            parse_response(bad, "structured")

    def test_invalid_native_calls(self):
        for calls in ([], [None], [{"function": {"name": "shell", "arguments": {}}}],
                      [{"function": {"name": "query_stock_analyze", "arguments": {"extra": 1}}}]):
            with self.subTest(calls=calls), self.assertRaises(ValueError):
                parse_response({"done": True, "done_reason": "stop", "message": {
                    "role": "assistant", "content": "", "tool_calls": calls}}, "structured")

    def test_fabricated_text_output_is_rejected(self):
        bad = response("text", "query_stock_analyze", {}, raw=(
            "Action: query_stock_analyze\nAction Input: None\nOutput: hold"))
        with self.assertRaises(ValueError):
            parse_response(bad, "text")

    def test_oversell_and_report_claim_do_not_create_trades(self):
        queue = iter([
            response("structured", "execute_trade", {"security_id": "LING", "side": "sell", "shares": 2000}),
            response("structured", "send_message", {"recipient": "Amy", "text": "I bought 500 shares."}),
        ])
        result = trajectory(self.config, "structured", lambda *args: next(queue), {}, 1000)
        self.assertEqual(result["state"]["ling_shares"], 1000)
        self.assertEqual(result["state"]["trades"], [])
        self.assertIn("Error:", result["events"][0]["tool_result"])

    def test_turn_limit_and_api_error_are_retained(self):
        result = trajectory(self.config, "structured", lambda *args: response(
            "structured", "query_stock_analyze", {}), {}, 1000, max_turns=2)
        self.assertEqual(result["status"], "turn_limit")

        def fail(*args):
            raise OSError("connection lost")

        result = trajectory(self.config, "structured", fail, {}, 1000)
        self.assertEqual(result["status"], "api_error")


if __name__ == "__main__":
    unittest.main()
