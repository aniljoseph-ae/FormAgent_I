# tests/test_ragas.py

import pytest
from evaluation.ragas_evaluator import RagasEvaluator
from config.schema import EvaluationMetricsSchema
class TestRagasEvaluation:
    """Test RAGAS evaluation functionality"""
   
    def test_metrics_calculation(self):
        """Test that RAGAS metrics are calculated correctly"""
        evaluator = RagasEvaluator()
       
        # Sample test data
        questions = ["What is the capital of France?"]
        ground_truths = ["Paris"]
        contexts = [["France is a country in Europe. Its capital is Paris."]]
        answers = ["Paris is the capital of France."]
       
        results = evaluator.evaluate_question_answer(questions, ground_truths, contexts, answers)
       
        assert isinstance(results, EvaluationMetricsSchema)
        assert 0 <= results.faithfulness <= 1
        assert 0 <= results.answer_relevancy <= 1