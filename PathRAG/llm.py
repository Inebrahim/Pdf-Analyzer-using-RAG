import copy
import os
from functools import lru_cache
from typing import List, Dict, Callable, Any, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

from PathRAG.utils import (
    wrap_embedding_func_with_attrs,
    logger,
)

import sys
import ollama

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ollama_client = ollama.Client(
    host="https://ollama.com",
    headers={'Authorization': f'Bearer {os.getenv("OLLAMA_API_KEY")}'}
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def ollama_turbo_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
    model_name = kwargs.get("hashing_kv").global_config["llm_model_name"]
    
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.extend(history_messages)
    messages.append({'role': 'user', 'content': prompt})

    try:
        response = ollama_client.chat(
            model=model_name,
            messages=messages
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ollama API call failed: {e}")
        # Return a standard error message or re-raise the exception
        return "Sorry, the connection to the AI model failed."
        
@lru_cache(maxsize=1)
def initialize_hf_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Initializes and caches a Hugging Face model and tokenizer."""
    logger.info(f"Initializing Hugging Face model: {model_name}")
    # You might need to install accelerate for device_map="auto" to work well
    # pip install accelerate
    hf_tokenizer = AutoTokenizer.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    logger.info("Hugging Face model initialized successfully.")
    return hf_model, hf_tokenizer


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
async def hf_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> str:
    """
    Generates text using a Hugging Face model.
    The 'model' parameter is the Hugging Face model ID.
    """
    model_name = model
    hf_model, hf_tokenizer = initialize_hf_model(model_name)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    kwargs.pop("hashing_kv", None)

    # Apply chat template
    try:
        input_prompt = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        logger.error(f"Could not apply chat template, falling back to manual formatting. Error: {e}")
        input_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    # Tokenize and generate
    inputs = hf_tokenizer(
        input_prompt, return_tensors="pt"
    ).to(hf_model.device)

    max_new_tokens = kwargs.get("max_new_tokens", 512)
    
    output = hf_model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        num_return_sequences=1,
        pad_token_id=hf_tokenizer.eos_token_id
    )
    
    # Decode the generated text, skipping the prompt
    response_text = hf_tokenizer.decode(
        output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )

    return response_text.strip()


async def hf_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    Primary function to call for LLM completion using a local Hugging Face model.
    """
    # This function is the main entry point for using the Mistral model.
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    result = await hf_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    return result


@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=512)
async def sentence_transformer_embedding(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Generates embeddings using a SentenceTransformer model.
    Caches the model for efficiency.
    """
    @lru_cache(maxsize=1)
    def get_st_model(name):
        logger.info(f"Initializing Sentence Transformer model: {name}")
        # This ensures the model is only loaded once
        model = SentenceTransformer(name)
        logger.info("Sentence Transformer model initialized successfully.")
        return model
        
    if isinstance(texts, str):
        texts = [texts]
    st_model = get_st_model(model_name)
    return st_model.encode(texts, convert_to_numpy=True)


class Model(BaseModel):
    """
    Pydantic model to define a custom language model for use with MultiModel.
    """
    gen_func: Callable[[Any], str] = Field(
        ...,
        description="A function that generates the response from the llm.",
    )
    kwargs: Dict[str, Any] = Field(
        ...,
        description="The arguments to pass to the callable function. Eg. model name, specific configs.",
    )

    class Config:
        arbitrary_types_allowed = True


class MultiModel:
    """
    Distributes the load across multiple models or model configurations.
    """
    def __init__(self, models: List[Model]):
        self._models = models
        self._current_model = 0

    def _next_model(self):
        self._current_model = (self._current_model + 1) % len(self._models)
        return self._models[self._current_model]

    async def llm_model_func(
        self, prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        next_model = self._next_model()
        args = dict(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
            **next_model.kwargs,
        )
        return await next_model.gen_func(**args)


# Main execution block for testing
if __name__ == "__main__":
    import asyncio

    async def main():
        """
        Main function to test the local LLM and embedding models.
        """
        # --- 1. Test the Language Model (Mistral-7B) ---
        print("--- Testing LLM: TinyLlama/TinyLlama-1.1B-Chat-v1.0 ---")
        try:
            # The first run will download the model, which can be large and take time.
            llm_result = await hf_model_complete(
                prompt="What are the top 3 benefits of using Retrieval-Augmented Generation (RAG)?",
                system_prompt="You are a helpful AI assistant specializing in language models."
            )
            print("\nLLM Response:")
            print(llm_result)
        except Exception as e:
            print(f"\nAn error occurred while running the LLM: {e}")
            print("Please ensure you have 'transformers', 'torch', and 'accelerate' installed (`pip install transformers torch accelerate`).")
            print("A GPU with sufficient VRAM is required for this model.")

        print("\n" + "="*50 + "\n")

        # --- 2. Test the Embedding Model (all-MiniLM-L6-v2) ---
        print("--- Testing Embedding Model: sentence-transformers/all-MiniLM-L6-v2 ---")
        try:
            texts_to_embed = [
                "Retrieval-Augmented Generation combines the power of large language models with external knowledge bases.",
                "The capital of France is Paris.",
                "Sentence transformers are used to create dense vector representations of text."
            ]
            embeddings = await sentence_transformer_embedding(texts_to_embed)
            print(f"\nSuccessfully generated {len(embeddings)} embeddings.")
            print(f"Shape of the first embedding vector: {embeddings[0].shape}")
            print(f"First 5 dimensions of the first embedding: {embeddings[0][:5]}...")
        except Exception as e:
            print(f"\nAn error occurred while running the embedding model: {e}")
            print("Please ensure you have 'sentence-transformers' installed (`pip install sentence-transformers`).")

    asyncio.run(main())