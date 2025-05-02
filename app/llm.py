import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import settings
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TinyLlama:
    def __init__(self):
        """Initialize TinyLlama model with proper device handling."""
        try:
            logger.info(f"Initializing TinyLlama with model: {settings.LLM_MODEL}")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Model configuration
            kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "trust_remote_code": True
            }

            # Load model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL,
                **kwargs
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.LLM_MODEL,
                padding_side="left",
                truncation_side="left"
            )
            
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.critical(f"Model initialization failed: {str(e)}", exc_info=True)
            raise

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Faster generation parameters
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                do_sample=True,
                num_beams=1,
                early_stopping=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return "I'm having trouble generating a response."