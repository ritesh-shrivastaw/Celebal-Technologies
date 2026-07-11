import unittest

from smart_agent import classify_intent, agent, run_quiz


class SmartAgentTests(unittest.TestCase):
    def test_classify_intent_for_power_expression(self):
        self.assertEqual(classify_intent("2 ** 8"), "calculation")

    def test_agent_handles_power_expression(self):
        response = agent("2 ** 8")
        self.assertEqual(response["type"], "calculation")
        self.assertEqual(response["result"], "256")

    def test_run_quiz_returns_score_for_custom_answers(self):
        questions = [
            ("What is 2 + 2?", "4"),
            ("What is 5 * 3?", "15"),
        ]
        result = run_quiz(questions=questions, answers=["4", "15"], interactive=False)
        self.assertEqual(result["score"], 2)
        self.assertEqual(result["max_score"], 2)


if __name__ == "__main__":
    unittest.main()
