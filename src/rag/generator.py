"""Generator module for RAG pipeline"""

import logging
import torch
from typing import Optional, Dict, Any, Generator, List
from threading import Thread

logger = logging.getLogger(__name__)


class Generator:
    """
    LLM generator for RAG responses.
    Supports both local models and API-based models.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        device: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Args:
            model_name: Model name or path
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling
            do_sample: Whether to use sampling
            repetition_penalty: Penalty for repeating tokens
            device: "cuda", "cpu", or None (auto)
            system_prompt: System prompt override
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.default_system_prompt = system_prompt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the LLM model."""
        if self.model is not None:
            return

        logger.info(f"Loading generator model: {self.model_name} on {self.device}")

        try:
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_new_tokens + 512,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
            logger.info("Model loaded with Unsloth optimization")
        except ImportError:
            logger.warning("Unsloth not available, using standard transformers")
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            self.model.eval()
            logger.info("Model loaded with standard transformers")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from prompt.

        Args:
            prompt: User prompt (will be wrapped with system prompt)
            system_prompt: Override system prompt
            max_new_tokens: Override max tokens
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if self.model is None:
            self.load()

        system = system_prompt or self.default_system_prompt

        # Format as chat messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = f"{system}\n\nUser: {prompt}\n\nAssistant:"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                top_k=kwargs.get("top_k", self.top_k),
                do_sample=kwargs.get("do_sample", self.do_sample),
                repetition_penalty=kwargs.get("repetition_penalty", self.repetition_penalty),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def generate_from_context(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None
    ) -> str:
        """
        Generate response from question + retrieved context.

        Args:
            question: User question
            context: Retrieved context
            system_prompt: System prompt
            user_template: User prompt template with {question} and {context}

        Returns:
            Generated response
        """
        if user_template is None:
            user_template = (
                "Ngữ cảnh (Context):\n{context}\n\n"
                "Câu hỏi: {question}\n\n"
                "Trả lời:"
            )

        prompt = user_template.format(question=question, context=context)
        return self.generate(prompt, system_prompt=system_prompt)

    def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        callback=None,
        chunk_size: int = 16
    ) -> str:
        """
        Generate response with streaming output.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            callback: Function to call with each chunk
            chunk_size: Number of tokens per chunk

        Returns:
            Complete generated text
        """
        if self.model is None:
            self.load()

        system = system_prompt or self.default_system_prompt

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = f"{system}\n\nUser: {prompt}\n\nAssistant:"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        full_response = []
        generated_ids = []

        from transformers import TextIteratorStreamer
        from queue import Queue

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=self.max_new_tokens,
            streamer=streamer,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for text_chunk in streamer:
            if callback:
                callback(text_chunk)
            full_response.append(text_chunk)

        thread.join()
        return "".join(full_response)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the generator model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "loaded": self.model is not None,
        }
