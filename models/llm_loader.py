"""
LLM Loader - Loads HuggingFace transformer models for text generation.

Provides a unified interface to load different models (e.g. Mistral, Qwen, Llama)
with optional 4-bit / 8-bit quantization for lower memory usage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from transformers import Pipeline

from config.hf_config import HF_TOKEN


class LLMLoader:
    """
    Loads and caches HuggingFace causal LM models and tokenizers.
    Supports device mapping and optional 4-bit/8-bit quantization for large models.
    """

    def __init__(
        self,
        model_id: str,
        device_map: Optional[str] = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.3,
    ):
        self.model_id = model_id
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self._model = None
        self._tokenizer = None
        self._pipe: Optional[Pipeline] = None

    def load(self) -> Pipeline:
        """Load the model and tokenizer, return a text-generation pipeline."""
        if self._pipe is not None:
            return self._pipe

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }

        if self.load_in_4bit and torch.cuda.is_available():
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            model_kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
            token=HF_TOKEN,
        )

        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
        )
        return self._pipe

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from a prompt.

        Returns:
            Generated text (only the new tokens, not the prompt).
        """
        pipe = self.load()
        out = pipe(prompt, **kwargs)
        if out and len(out) > 0 and "generated_text" in out[0]:
            full = out[0]["generated_text"]
            return full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
        return ""

    def unload(self) -> None:
        """Release model and tokenizer from memory."""
        self._pipe = None
        self._model = None
        self._tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
