import unittest
import tempfile
import json
from pathlib import Path
from src.dataset.schema import CanonicalPromptScenario, SourceProvenance, PromptContent, ReferenceContent, ScenarioMetadata
from src.dataset.squad_adapter import SQuADAdapter

class TestSQuADAdapter(unittest.TestCase):
    def setUp(self):
        # Create a small mock SQuAD v2 JSON file for testing
        self.mock_squad_data = {
            "version": "v2.0",
            "data": [
                {
                    "title": "Test_Article",
                    "paragraphs": [
                        {
                            "context": "The quick brown fox jumps over the lazy dog.",
                            "qas": [
                                {
                                    "id": "ans_01",
                                    "question": "What jumps over the lazy dog?",
                                    "is_impossible": False,
                                    "answers": [{"text": "quick brown fox", "answer_start": 4}]
                                },
                                {
                                    "id": "unans_01",
                                    "question": "Who is the President of Mars?",
                                    "is_impossible": True,
                                    "answers": []
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.mock_file = Path(self.tmp_dir.name) / "mock_squad.json"
        with open(self.mock_file, "w", encoding="utf-8") as f:
            json.dump(self.mock_squad_data, f)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_adapter_normalization_and_conversion(self):
        adapter = SQuADAdapter(self.mock_file, split="train")
        scenarios = adapter.normalize()
        
        self.assertEqual(len(scenarios), 2)
        
        # Test answerable scenario
        ans_scenario = scenarios[0]
        self.assertEqual(ans_scenario.sample_id, "squad_v2_ans_01")
        self.assertEqual(ans_scenario.reference.answerability, "ANSWERABLE")
        self.assertEqual(ans_scenario.metadata.expected_behavior, "ANSWER")
        self.assertEqual(ans_scenario.source.article_title, "Test_Article")
        self.assertEqual(ans_scenario.prompt.question, "What jumps over the lazy dog?")
        self.assertTrue(len(ans_scenario.reference.answers) > 0)
        
        # Test unanswerable scenario
        unans_scenario = scenarios[1]
        self.assertEqual(unans_scenario.sample_id, "squad_v2_unans_01")
        self.assertEqual(unans_scenario.reference.answerability, "UNANSWERABLE_FROM_CONTEXT")
        self.assertEqual(unans_scenario.metadata.expected_behavior, "ABSTAIN")
        self.assertEqual(unans_scenario.reference.answers, [])

    def test_unique_sample_ids(self):
        adapter = SQuADAdapter(self.mock_file)
        scenarios = adapter.normalize()
        ids = [s.sample_id for s in scenarios]
        self.assertEqual(len(ids), len(set(ids)))

    def test_forbidden_fields_rejection(self):
        scenario = CanonicalPromptScenario(
            sample_id="test_1",
            source=SourceProvenance(dataset="squad_v2", split="train", original_id="1", article_id="0", paragraph_id="0", article_title="T"),
            prompt=PromptContent(question="Q?", context="C"),
            reference=ReferenceContent(answerability="ANSWERABLE", answers=[{"text": "A"}], supporting_information=None),
            metadata=ScenarioMetadata(phenomenon="grounded_contextual_reliability", expected_behavior="ANSWER")
        )
        
        # Inject forbidden field dynamically
        setattr(scenario, "qwen_answer", "forbidden value")
        with self.assertRaises(ValueError):
            scenario.validate()

    def test_invalid_malformed_input_rejection(self):
        malformed_data = {
            "version": "v2.0",
            "data": [
                {
                    "title": "Bad",
                    "paragraphs": [
                        {
                            "context": "", # empty context should fail or be caught
                            "qas": [{"id": "bad_1", "question": "", "is_impossible": False}]
                        }
                    ]
                }
            ]
        }
        bad_file = Path(self.tmp_dir.name) / "bad.json"
        with open(bad_file, "w", encoding="utf-8") as f:
            json.dump(malformed_data, f)
            
        adapter = SQuADAdapter(bad_file)
        scenarios = adapter.normalize()
        with self.assertRaises(ValueError):
            scenarios[0].validate()

if __name__ == "__main__":
    unittest.main()
