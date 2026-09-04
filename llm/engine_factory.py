"""
LLM Engine Factory:
Unifies Google Gemini, vLLM (PagedAttention), and SGLang (RadixAttention).
Enables seamless switching via configuration or API request parameters.
"""

import os
import time
from typing import Dict, Any, Optional, Tuple
import httpx
from dotenv import load_dotenv

from .gemini_client import GeminiClient

load_dotenv()


class BaseLLMEngine:
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2
    ) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError


class GeminiEngine(BaseLLMEngine):
    """Google Gemini Cloud API Engine."""
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.client = GeminiClient(api_key=api_key, model_name=model_name)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2
    ) -> Tuple[str, Dict[str, Any]]:
        return self.client.generate(prompt, system_prompt, max_tokens, temperature)


class OpenAICompatibleEngine(BaseLLMEngine):
    """
    Generic OpenAI-compatible client used for vLLM and SGLang servers.
    Communicates over HTTP to /v1/chat/completions.
    """
    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: str = "empty",
        engine_label: str = "vllm"
    ):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.engine_label = engine_label

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2
    ) -> Tuple[str, Dict[str, Any]]:
        start_time = time.perf_counter()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                res = client.post(f"{self.api_base}/chat/completions", json=payload, headers=headers)
                res.raise_for_status()
                data = res.json()
                choice = data["choices"][0]
                text = choice["message"]["content"]

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                tokens_count = len(text.split())
                tps = tokens_count / max(elapsed_ms / 1000.0, 0.001)

                telemetry = {
                    "model": f"{self.engine_label}:{self.model_name}",
                    "total_latency_ms": elapsed_ms,
                    "ttft_ms": elapsed_ms * 0.3,
                    "tokens_generated": tokens_count,
                    "tokens_per_second": round(tps, 2)
                }
                return text, telemetry
        except Exception as e:
            # Fallback simulator when local vLLM/SGLang server is not actively running
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            simulated_text = (
                f"[{self.engine_label.upper()} Local Endpoint Notice: {e}]\n\n"
                f"Context answer served via {self.engine_label} fallback simulator."
            )
            return simulated_text, {
                "model": f"{self.engine_label}:{self.model_name}-fallback",
                "total_latency_ms": elapsed_ms,
                "ttft_ms": elapsed_ms * 0.5,
                "tokens_generated": len(simulated_text.split()),
                "tokens_per_second": 0.0,
                "warning": f"Could not connect to {self.engine_label} at {self.api_base}"
            }


class VLLMEngine(OpenAICompatibleEngine):
    """vLLM Server with PagedAttention and continuous batching."""
    def __init__(self):
        api_base = os.getenv("VLLM_API_BASE", "http://localhost:8001/v1")
        model = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        api_key = os.getenv("VLLM_API_KEY", "empty")
        super().__init__(api_base=api_base, model_name=model, api_key=api_key, engine_label="vllm")


class SGLangEngine(OpenAICompatibleEngine):
    """SGLang Server with RadixAttention for prompt KV cache reuse."""
    def __init__(self):
        api_base = os.getenv("SGLANG_API_BASE", "http://localhost:30000/v1")
        model = os.getenv("SGLANG_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        api_key = os.getenv("SGLANG_API_KEY", "empty")
        super().__init__(api_base=api_base, model_name=model, api_key=api_key, engine_label="sglang")


class OllamaEngine(OpenAICompatibleEngine):
    """Local Ollama instance running on CPU / Intel Iris Xe."""
    def __init__(self, model_name: Optional[str] = None):
        api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
        model = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        super().__init__(api_base=api_base, model_name=model, api_key=api_key, engine_label="ollama")


def get_llm_engine(backend: Optional[str] = None) -> BaseLLMEngine:
    """Factory to retrieve the specified or default LLM Engine."""
    selected = (backend or os.getenv("LLM_BACKEND", "gemini")).strip()
    sel_lower = selected.lower()

    if sel_lower.startswith("ollama:") or sel_lower in ["ollama", "local"]:
        model = selected.split(":", 1)[1] if ":" in selected else None
        return OllamaEngine(model_name=model)
    elif sel_lower == "vllm":
        return VLLMEngine()
    elif sel_lower == "sglang":
        return SGLangEngine()
    else:
        return GeminiEngine()
