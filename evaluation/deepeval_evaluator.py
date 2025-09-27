# evaluation/deepeval_evaluator.py


from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from evaluation.groq_llm import GroqDeepEvalLLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepEvalEvaluator:
    def __init__(self):
        self.llm = GroqDeepEvalLLM()
        self.metrics = [
            AnswerRelevancyMetric(threshold=0.7, model=self.llm),
            FaithfulnessMetric(threshold=0.7, model=self.llm),
            ContextualRecallMetric(threshold=0.7, model=self.llm),
            ContextualPrecisionMetric(threshold=0.7, model=self.llm)
        ]
    
    def evaluate(self, test_cases: list) -> dict:
        try:
            results = {
                "retrieval_precision": 0.0,
                "retrieval_recall": 0.0,
                "retrieval_f1": 0.0
            }
            metric_scores = {metric.__class__.__name__: [] for metric in self.metrics}
            
            for case in test_cases:
                test_case = LLMTestCase(
                    input=case["question"],
                    actual_output=case.get("generated_answer", ""),
                    expected_output=case["expected_answer"],
                    retrieval_context=case.get("retrieved_chunks", [])
                )
                for metric in self.metrics:
                    metric.measure(test_case)
                    metric_scores[metric.__class__.__name__].append(metric.score)
                
                retrieved = set(case.get("retrieved_chunks", []))
                expected = set(case.get("expected_chunks", []))
                if retrieved or expected:
                    precision = len(retrieved & expected) / len(retrieved) if retrieved else 0.0
                    recall = len(retrieved & expected) / len(expected) if expected else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    results["retrieval_precision"] += precision
                    results["retrieval_recall"] += recall
                    results["retrieval_f1"] += f1
            
            for metric_name in metric_scores:
                results[metric_name] = sum(metric_scores[metric_name]) / len(metric_scores[metric_name]) if metric_scores[metric_name] else 0.0
            for key in ["retrieval_precision", "retrieval_recall", "retrieval_f1"]:
                results[key] /= len(test_cases) if test_cases else 1.0
            
            return results
        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {str(e)}")
            return {"error": str(e)}
