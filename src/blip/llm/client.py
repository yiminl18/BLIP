from __future__ import annotations
import time
import logging
from typing import Any

from openai import AzureOpenAI
from openai import RateLimitError, APIStatusError

from blip.config import Config, DeploymentConfig, get_config
from blip.llm.usage import Usage

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BASE_DELAY = 1.0


def _make_client(dep: DeploymentConfig) -> AzureOpenAI:
    return AzureOpenAI(
        api_key=dep.api_key,
        api_version=dep.api_version,
        azure_endpoint=dep.azure_endpoint,
    )


class LLMClient:
    def __init__(self, config: Config | None = None) -> None:
        self._cfg = config or get_config()
        azure = self._cfg.azure
        self._driver_dep = azure.driver
        self._escalation_dep = azure.escalation
        self._embed_dep = azure.embed
        self._driver_client = _make_client(azure.driver)
        self._escalation_client = _make_client(azure.escalation)
        self._embed_client = _make_client(azure.embed)
        self._last_fingerprint: str | None = None

    def _chat(
        self,
        messages: list[dict],
        model: str = "driver",
        seed: int | None = None,
    ) -> tuple[str, Usage]:
        if model == "driver":
            client = self._driver_client
            dep = self._driver_dep
        elif model == "escalation":
            client = self._escalation_client
            dep = self._escalation_dep
        else:
            raise ValueError(f"Unknown model: {model}")

        if seed is None:
            seed = self._cfg.seed

        delay = _BASE_DELAY
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=dep.deployment,
                    messages=messages,
                    temperature=0,
                    top_p=1,
                    seed=seed,
                )
                fp = getattr(resp, "system_fingerprint", None)
                if fp:
                    if self._last_fingerprint and fp != self._last_fingerprint:
                        logger.warning("system_fingerprint changed: %s -> %s", self._last_fingerprint, fp)
                    self._last_fingerprint = fp

                content = resp.choices[0].message.content or ""
                u = resp.usage
                cached = 0
                if u and hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
                    cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
                usage = Usage(
                    prompt_tokens=u.prompt_tokens if u else 0,
                    cached_tokens=cached,
                    completion_tokens=u.completion_tokens if u else 0,
                    model=dep.deployment,
                )
                return content, usage
            except RateLimitError:
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(delay + delay * 0.1 * attempt)
                delay *= 2
            except APIStatusError as e:
                if "context_length_exceeded" in str(e):
                    raise
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def answer(self, text: str, question: str, model: str = "driver") -> tuple[str, Usage]:
        from blip.llm.prompts import answer_messages
        msgs = answer_messages(text, question)
        return self._chat(msgs, model=model)

    def judge(self, a: str, b: str, model: str | None = None) -> tuple[str, Usage]:
        from blip.llm.prompts import judge_messages
        m = model or self._cfg.judge_model
        msgs = judge_messages(a, b)
        return self._chat(msgs, model=m)

    def rank(self, question: str, answer: str, blocks: list[str], model: str = "driver") -> tuple[str, Usage]:
        from blip.llm.prompts import ranker_messages
        msgs = ranker_messages(question, answer, blocks)
        return self._chat(msgs, model=model)

    def provenance(self, question: str, answer: str, sentences: list[str], model: str = "driver") -> tuple[str, Usage]:
        from blip.llm.prompts import provenance_messages
        msgs = provenance_messages(question, answer, sentences)
        return self._chat(msgs, model=model)

    def embed(self, texts: list[str]) -> tuple[list[list[float]], Usage]:
        dep = self._embed_dep
        resp = self._embed_client.embeddings.create(
            model=dep.deployment,
            input=texts,
        )
        vectors = [item.embedding for item in resp.data]
        u = resp.usage
        usage = Usage(
            prompt_tokens=u.prompt_tokens if u else 0,
            cached_tokens=0,
            completion_tokens=0,
            model=dep.deployment,
        )
        return vectors, usage
