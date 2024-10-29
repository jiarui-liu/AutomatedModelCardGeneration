from typing import Any, Dict, List
from langchain.chat_models import ChatOpenAI

class ChatVLLMOpenAI(ChatOpenAI): # type: ignore
    """vLLM OpenAI-compatible API chat client"""
    
    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        openai_creds: Dict[str, Any] = {
            "api_key": self.openai_api_key,
            "api_base": self.openai_api_base,
        }

        return {
            "model": self.model_name,
            **openai_creds,
            **self._default_params,
            "logit_bias": None,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "chat-vllm-openai"
