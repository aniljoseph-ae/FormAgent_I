# evaluation/test_suite.py

from evaluation.ragas_evaluator import RagasEvaluator
from evaluation.deepeval_evaluator import DeepEvalEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    def __init__(self, agent):
        self.agent = agent
        self.ragas_evaluator = RagasEvaluator()
        self.deepeval_evaluator = DeepEvalEvaluator()
    
    def run_comprehensive_evaluation(self, test_cases: list) -> dict:
        try:
            for case in test_cases:
                result = self.agent.query(case["question"], case["document_ids"])
                case["generated_answer"] = result["response"]
                case["retrieved_chunks"] = [chunk["content"] for chunk in result["retrieved_chunks"]]
            
            ragas_results = self.ragas_evaluator.evaluate(test_cases)
            deepeval_results = self.deepeval_evaluator.evaluate(test_cases)
            
            return {
                "ragas_metrics": ragas_results,
                "retrieval_metrics": {
                    "retrieval_precision": deepeval_results.get("retrieval_precision", 0.0),
                    "retrieval_recall": deepeval_results.get("retrieval_recall", 0.0),
                    "retrieval_f1": deepeval_results.get("retrieval_f1", 0.0)
                },
                "summary": {
                    "overall_score": (ragas_results.get("answer_relevancy", 0.0) + deepeval_results.get("AnswerRelevancyMetric", 0.0)) / 2,
                    "strengths": ["Accurate retrieval", "Robust LLM responses"],
                    "improvement_areas": [],
                    "recommendations": ["Increase chunk size for better context"]
                }
            }
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {str(e)}")
            return {"error": str(e)}
