# evaluation/ragas_evaluator.py

from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from datasets import Dataset
from evaluation.groq_llm import GroqDeepEvalLLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagasEvaluator:
    def __init__(self):
        self.llm = GroqDeepEvalLLM()
    
    def evaluate(self, test_cases: list) -> dict:
        try:
            data = {
                "question": [],
                "answer": [],
                "contexts": [],
                "ground_truth": []
            }
            
            for case in test_cases:
                data["question"].append(case["question"])
                data["answer"].append(case.get("generated_answer", ""))
                data["contexts"].append(case.get("retrieved_chunks", []))
                data["ground_truth"].append(case["expected_answer"])
            
            dataset = Dataset.from_dict(data)
            result = evaluate(
                dataset=dataset,
                metrics=[
                    answer_relevancy,
                    faithfulness,
                    context_precision,
                    context_recall
                ],
                llm=self.llm
            )
            
            return result.to_dict()
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {str(e)}")
            return {"error": str(e)}