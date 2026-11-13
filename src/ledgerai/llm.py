"""Lazy vLLM wrapper for Gemma 4 12B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Generation:
    text: str
    elapsed_ms: int


class GemmaEngine:
    """Loads the model once and serves deterministic chat completions."""

    def __init__(
        self,
        model_path: str | Path,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        max_tokens: int = 512,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.llm = LLM(
            model=self.model_path,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def chat(self, messages: list[dict], max_tokens: int | None = None) -> Generation:
        import time

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        params = self.sampling
        if max_tokens is not None:
            from vllm import SamplingParams

            params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        start = time.time()
        output = self.llm.generate([prompt], params)
        text = output[0].outputs[0].text
        return Generation(text=text, elapsed_ms=int((time.time() - start) * 1000))

    @staticmethod
    def available(model_path: str | Path) -> bool:
        path = Path(model_path)
        return path.exists() and (path / "config.json").exists()
