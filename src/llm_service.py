from __future__ import annotations

from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

from .config import (
    DEFAULT_DEVICE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)


class LLMService:
    """
    Minimal wrapper around a Hugging Face text-generation pipeline.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        device_map: str = DEFAULT_DEVICE,
        seed: Optional[int] = 42,
    ) -> None:
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.device_map = device_map
        if seed is not None:
            set_seed(seed)
        self.generator = self._load_generator()

    def _load_generator(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=self.device_map,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        tokens = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature
        nucleus = top_p or self.top_p
        sample_top_k = top_k or self.top_k

        outputs = self.generator(
            prompt,
            max_new_tokens=tokens,
            temperature=temp,
            top_p=nucleus,
            top_k=sample_top_k,
            do_sample=True,
        )
        generated_text = outputs[0]["generated_text"]
        if generated_text.startswith(prompt):
            return generated_text[len(prompt) :].lstrip()
        return generated_text
