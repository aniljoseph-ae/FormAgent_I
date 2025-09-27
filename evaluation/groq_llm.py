# evaluation/groq_llm.py
# Create a custom GroqLLM class to make ChatGroq compatible with DeepEval.

from langchain_groq import ChatGroq
from config.config import config_manager
from deepeval.models.base_model import DeepEvalBaseLLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.config = config_manager.get_validated_config()
        try:
            self.llm = ChatGroq(
                model_name=self.config.evaluation['evaluation_model'],
                api_key=self.config.evaluation['evaluation_api_key'],
                temperature=0.1,
                max_tokens=2048
            )
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {str(e)}")
            raise
    
    def load_model(self):
        return self.llm
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.llm.invoke(prompt).content
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return f"Error: {str(e)}"
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        return self.config.evaluation['evaluation_model']