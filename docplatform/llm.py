"""Model access: chat, JSON-validated chat, and embeddings.

Maps to Amazon Bedrock in the architecture. Works with any OpenAI-compatible
endpoint, such as a local FreeLLMAPI gateway.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Callable

import numpy as np
from openai import OpenAI

from .config import Settings

EMBED_DIM = 512


class ModelOutputError(Exception):
    """The model did not return output in the required format."""


def extract_json(text: str) -> dict:
    """Pull the JSON object out of a reply, tolerating code fences, think tags and prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ModelOutputError("The reply contained no JSON object.")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"The reply was not valid JSON ({exc.msg}).") from exc
    if not isinstance(data, dict):
        raise ModelOutputError("The JSON in the reply is not an object.")
    return data


def local_embed(text: str) -> list[float]:
    """Deterministic hashed bag-of-words embedding (unigrams + bigrams), L2-normalised."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for gram in words + [f"{a} {b}" for a, b in zip(words, words[1:])]:
        h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "little")
        vec[h % EMBED_DIM] += 1.0 if (h >> 63) & 1 else -1.0
    norm = np.linalg.norm(vec)
    return (vec / norm if norm else vec).tolist()


class LLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key or "not-set",
            timeout=settings.request_timeout,
            max_retries=2,
        )
        self.last_served_by = ""

    def chat(self, messages: list[dict], model: str | None = None, temperature: float = 0.2) -> str:
        raw = self.client.chat.completions.with_raw_response.create(
            model=model or self.settings.chat_model, messages=messages, temperature=temperature
        )
        completion = raw.parse()
        self.last_served_by = raw.headers.get("x-routed-via") or completion.model or ""
        text = completion.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def chat_json(
        self,
        messages: list[dict],
        validate: Callable[[dict], object],
        model: str | None = None,
        attempts: int = 3,
    ):
        """Ask for JSON, validate it, and feed any error back to the model before retrying."""
        messages = list(messages)
        error: ModelOutputError | None = None
        for _ in range(attempts):
            reply = self.chat(messages, model=model)
            try:
                return validate(extract_json(reply))
            except ModelOutputError as exc:
                error = exc
                messages += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": f"That reply was not usable: {exc} Reply with only the JSON object."},
                ]
        raise error  # type: ignore[misc]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.settings.embed_model:
            return [local_embed(t) for t in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            resp = self.client.embeddings.create(model=self.settings.embed_model, input=texts[start : start + 64])
            vectors += [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]
        return vectors
