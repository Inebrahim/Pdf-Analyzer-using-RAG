import asyncio
import html
import io
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
import xml.etree.ElementTree as ET
from typing import Any, Union, List, Dict, Optional 
import numpy as np
import tiktoken

from PathRAG.prompt import PROMPTS


class UnlimitedSemaphore:
    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


logger = logging.getLogger("PathRAG")


def set_logger(log_file: str):
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable
    concurrent_limit: int = 16

    def __post_init__(self):
        if self.concurrent_limit > 0:
            self._semaphore = asyncio.Semaphore(self.concurrent_limit)
        else:
            self._semaphore = UnlimitedSemaphore()

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        async with self._semaphore:
            result = self.func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result


def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    try:
        # This regex is good for finding a JSON object within a larger string
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            # Clean up common formatting issues from LLM outputs
            json_str = match.group(0).replace("\\n", "").replace("\n", "")
            return json_str
    except Exception:
        pass
    return None


def convert_response_to_json(response: str) -> dict:
    json_str = locate_json_string_body_from_string(response)
    if not json_str:
        raise ValueError(f"Unable to find a JSON object in the response: {response}")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def limit_async_func_call(max_size: int, waiting_time: float = 0.001):
    def final_decro(func):
        semaphore = asyncio.Semaphore(max_size)
        @wraps(func)
        async def wait_func(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)
        return wait_func
    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    def final_decro(func) -> EmbeddingFunc:
        return EmbeddingFunc(**kwargs, func=func)
    return final_decro


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


# --- CHANGE 1: Corrected tiktoken functions ---
# I've simplified these functions to always use a compatible default encoder ('cl100k_base').
# This prevents crashes when non-OpenAI model names are passed.

_TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

def encode_string_by_tiktoken(content: str, model_name: str = "cl100k_base") -> List[int]:
    """Encodes a string into tokens using a fixed, compatible tiktoken encoder."""
    # The `model_name` parameter is kept for compatibility but is no longer used to select the encoder.
    return _TIKTOKEN_ENCODER.encode(content)


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "cl100k_base") -> str:
    """Decodes tokens back into a string using a fixed, compatible tiktoken encoder."""
    return _TIKTOKEN_ENCODER.decode(tokens)


# --- CHANGE 2: Renamed function for clarity ---
# Renamed from `pack_user_ass_to_openai_messages` to be more generic.
def pack_history_to_messages(*args: str) -> List[Dict[str, str]]:
    """Packs a sequence of strings into a user/assistant message history."""
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": content} for i, content in enumerate(args)]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    if not markers:
        return [content]
    # Create a regex pattern that matches any of the markers
    pattern = "|".join(re.escape(marker) for marker in markers)
    results = re.split(pattern, content)
    # Return a list of non-empty, stripped strings
    return [r.strip() for r in results if r and r.strip()]


def clean_str(input_str: Any) -> str:
    if not isinstance(input_str, str):
        return str(input_str) # Ensure it's a string
    # Remove control characters and strip whitespace
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", input_str).strip()
    return result


def is_float_regex(value: str) -> bool:
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", str(value)))


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int) -> list:
    if max_token_size <= 0:
        return []
    total_tokens = 0
    # Find the index where the cumulative token count exceeds the max
    for i, data in enumerate(list_data):
        total_tokens += len(encode_string_by_tiktoken(key(data)))
        if total_tokens > max_token_size:
            return list_data[:i]
    return list_data


def list_of_list_to_csv(data: List[List[Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> List[List[str]]:
    if not csv_string.strip():
        return []
    output = io.StringIO(csv_string)
    reader = csv.reader(output)
    return [row for row in reader]


def process_combine_contexts(csv1: str, csv2: str) -> str:
    """Combines two CSV strings, removes duplicate rows, and preserves the header."""
    list1 = csv_string_to_list(csv1)
    list2 = csv_string_to_list(csv2)

    header = []
    if list1:
        header = list1[0]
    elif list2:
        header = list2[0]
    else:
        return "" # Both are empty

    # Use a set for efficient duplicate checking of entire rows (as tuples)
    seen = set()
    combined_rows = []

    # Process rows, skipping headers
    for row in (list1[1:] + list2[1:]):
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            combined_rows.append(row)

    # Reconstruct the final list with the header and write to CSV string
    final_list = [header] + combined_rows
    return list_of_list_to_csv(final_list)


# --- CHANGE 3: Simplified caching logic ---
# The complex embedding-based caching (`get_best_cached_response`, `quantize`, etc.)
# was tightly coupled with a system that is no longer in use.
# It has been removed to prevent errors and simplify the code.
# The remaining `handle_cache` and `save_to_cache` provide a simple, direct-hit cache.

@dataclass
class CacheData:
    args_hash: str
    content: str
    prompt: str
    mode: str = "default"

async def handle_cache(hashing_kv, args_hash: str, prompt: str, mode="default"):
    """
    A simplified cache handler that checks for a direct hit based on the hash of the query.
    """
    if hashing_kv is None:
        return None, None, None, None

    # This simple cache checks if the exact same query was run before.
    mode_cache = await hashing_kv.get_by_id(mode) or {}
    if args_hash in mode_cache:
        cached_data = mode_cache[args_hash]
        logger.info(f"Cache hit for mode '{mode}' with hash '{args_hash}'.")
        return cached_data.get("return"), None, None, None # Return only the content

    return None, None, None, None


async def save_to_cache(hashing_kv, cache_data: CacheData):
    """Saves the LLM response to a simple key-value cache."""
    if hashing_kv is None or hasattr(cache_data.content, "__aiter__"): # Don't cache streaming responses
        return

    mode_cache = await hashing_kv.get_by_id(cache_data.mode) or {}
    
    # Store only the necessary data for a direct-hit cache
    mode_cache[cache_data.args_hash] = {
        "return": cache_data.content,
        "original_prompt": cache_data.prompt,
    }

    await hashing_kv.upsert({cache_data.mode: mode_cache})


def safe_unicode_decode(content: bytes) -> str:
    """Safely decodes bytes to a string, handling potential unicode escape sequences."""
    try:
        # First, decode from utf-8
        decoded_str = content.decode("utf-8")
        # Then, replace any literal '\uXXXX' sequences
        return re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), decoded_str)
    except Exception as e:
        logger.warning(f"Unicode decode failed: {e}. Falling back to ignoring errors.")
        return content.decode("utf-8", errors="ignore")