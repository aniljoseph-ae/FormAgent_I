# tests/test_deepeval.py

import pytest
from evaluation.deepeval_evaluator import DeepEvalEvaluator
from deepeval.test_case import LLMTestCase
from config.schema import EvaluationMetricsSchema
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDeepEvalEvaluation:
    """Test cases for DeepEval evaluation functionality"""
    
    @pytest.fixture
    def deepeval_evaluator(self):
        """Initialize DeepEvalEvaluator"""
        try:
            return DeepEvalEvaluator()
        except Exception as e:
            logger.error(f"Failed to initialize DeepEvalEvaluator: {str(e)}")
            pytest.skip(f"DeepEvalEvaluator initialization failed: {str(e)}")
    
    @pytest.fixture
    def sample_test_cases(self):
        """Create sample test cases for DeepEval"""
        return [
            LLMTestCase(
                input="What is the capital of France?",
                actual_output="Paris is the capital of France.",
                expected_output="Paris",
                context=["France is a country in Europe. Its capital is Paris."],
                retrieval_context=["France is a country in Europe. Its capital is Paris."]
            ),
            LLMTestCase(
                input="Summarize the document about France.",
                actual_output="The document discusses France, a European country with Paris as its capital.",
                expected_output="France is a European country with Paris as its capital.",
                context=["France is a country in Europe. Its capital is Paris."],
                retrieval_context=["France is a country in Europe. Its capital is Paris."]
            ),
            LLMTestCase(
                input="Compare capitals across documents.",
                actual_output="Document 1 mentions Paris as the capital of France; Document 2 mentions Berlin as the capital of Germany.",
                expected_output="Paris (France) and Berlin (Germany) are mentioned as capitals.",
                context=[
                    "France is a country in Europe. Its capital is Paris.",
                    "Germany is a country in Europe. Its capital is Berlin."
                ],
                retrieval_context=[
                    "France is a country in Europe. Its capital is Paris.",
                    "Germany is a country in Europe. Its capital is Berlin."
                ]
            )
        ]

    def test_deepeval_metrics_calculation(self, deepeval_evaluator, sample_test_cases):
        """Test DeepEval metrics calculation"""
        try:
            results = deepeval_evaluator.evaluate_system(sample_test_cases)
            
            assert isinstance(results, dict)
            assert all(key in results for key in [
                "answer_relevancy",
                "faithfulness",
                "contextual_recall",
                "contextual_precision",
                "overall_score"
            ])
            assert all(0 <= results[key] <= 1 for key in [
                "answer_relevancy",
                "faithfulness",
                "contextual_recall",
                "contextual_precision",
                "overall_score"
            ])
            assert results["overall_score"] == pytest.approx(
                sum([results["answer_relevancy"], results["faithfulness"],
                     results["contextual_recall"], results["contextual_precision"]]) / 4,
                abs=1e-2
            )
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            pytest.fail(f"Metrics calculation failed: {str(e)}")

    def test_test_case_creation(self, deepeval_evaluator):
        """Test creation of LLMTestCase objects"""
        questions = ["What is the capital of France?"]
        contexts = [["France is a country in Europe. Its capital is Paris."]]
        answers = ["Paris is the capital of France."]
        ground_truths = ["Paris"]
        
        try:
            test_cases = deepeval_evaluator.create_test_cases(questions, contexts, answers, ground_truths)
            
            assert len(test_cases) == 1
            assert isinstance(test_cases[0], LLMTestCase)
            assert test_cases[0].input == questions[0]
            assert test_cases[0].actual_output == answers[0]
            assert test_cases[0].expected_output == ground_truths[0]
            assert test_cases[0].context == contexts[0]
            assert test_cases[0].retrieval_context == contexts[0]
        except Exception as e:
            logger.error(f"Test case creation failed: {str(e)}")
            pytest.fail(f"Test case creation failed: {str(e)}")

    def test_empty_input_handling(self, deepeval_evaluator):
        """Test handling of empty inputs"""
        questions = [""]
        contexts = [[]]
        answers = [""]
        ground_truths = [""]
        
        with pytest.raises(ValueError, match="Input cannot be empty"):
            deepeval_evaluator.create_test_cases(questions, contexts, answers, ground_truths)

if __name__ == "__main__":
    pytest.main([__file__])