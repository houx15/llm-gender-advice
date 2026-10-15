"""Async OpenRouter client (spec §4.3), adapted from the hiring-audit project.

- async httpx; tenacity retries on 429 / 5xx / timeouts (max 5 attempts)
- NO retry on content refusals
- provider pinning (no fallbacks) and the served provider logged per call
- live price / canonical-slug verification against GET /models and /endpoints
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from ..common.config import ApiModelCfg

BASE_URL = "https://openrouter.ai/api/v1"


class RefusalError(Exception):
    """Provider-level content refusal — never retried."""


class TransportFailure(Exception):
    """Gave up after retries."""


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, RefusalError):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL, timeout: float = 180.0, max_attempts: int = 5):
        headers = {"HTTP-Referer": "https://github.com/advice-audit", "X-Title": "advice-audit",
                   "Content-Type": "application/json"}
        if api_key:  # model listings are public; chat needs a key
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout)
        self.max_attempts = max_attempts

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.client.aclose()

    # ------------------------------------------------------------ verification
    async def model_info(self) -> dict[str, dict]:
        r = await self.client.get("/models")
        r.raise_for_status()
        return {m["id"]: m for m in r.json().get("data", [])}

    async def endpoints(self, slug: str) -> list[dict]:
        """Live endpoints: [{provider_name, tag, quantization, price: (in, out) per Mtok}]."""
        base = slug.split(":")[0] if slug.endswith(":batch") else slug
        r = await self.client.get(f"/models/{slug}/endpoints")
        if r.status_code == 404 and base != slug:
            r = await self.client.get(f"/models/{base}/endpoints")
        r.raise_for_status()
        out = []
        for e in r.json().get("data", {}).get("endpoints", []):
            p = e.get("pricing", {})
            try:
                price = (float(p.get("prompt", 0)) * 1e6, float(p.get("completion", 0)) * 1e6)
            except (TypeError, ValueError):
                continue
            out.append({"provider_name": e.get("provider_name") or e.get("name"), "tag": e.get("tag") or "",
                        "quantization": e.get("quantization"), "price": price})
        return out

    @staticmethod
    def pinned_endpoint(eps: list[dict], provider: str) -> dict | None:
        """The endpoint a provider pin routes to: exact tag match (e.g. 'openai', 'deepinfra/fp8'),
        else the provider's base endpoint (tag without a '/variant' such as /fast or /flex)."""
        p = provider.lower()
        exact = [e for e in eps if e["tag"].lower() == p]
        if exact:
            return exact[0]
        same = [e for e in eps if (e["provider_name"] or "").lower() == p]
        base = [e for e in same if "/" not in e["tag"]] or [e for e in same if not e["tag"].endswith(("/fast", "/flex"))]
        return (base or same or [None])[0]

    async def verify_model(self, cfg: ApiModelCfg) -> dict:
        """Check slug exists, canonical slug unchanged, pinned provider present; return live price."""
        info = (await self.model_info()).get(cfg.slug)
        res: dict[str, Any] = {"key": cfg.key, "slug": cfg.slug, "ok": True, "problems": []}
        if info is None:
            res["ok"] = False
            res["problems"].append(f"slug {cfg.slug} not listed on OpenRouter")
            return res
        res["canonical_slug_live"] = info.get("canonical_slug")
        if cfg.canonical_slug and info.get("canonical_slug") != cfg.canonical_slug:
            res["ok"] = False
            res["problems"].append(
                f"canonical slug changed: config {cfg.canonical_slug} vs live {info.get('canonical_slug')}"
            )
        listing = info.get("pricing", {})
        res["listing_price"] = (float(listing.get("prompt", 0)) * 1e6, float(listing.get("completion", 0)) * 1e6)
        try:
            eps = await self.endpoints(cfg.slug)
        except httpx.HTTPError as e:
            eps = []
            res["problems"].append(f"could not list endpoints: {e}")
        res["endpoints"] = eps
        if not cfg.provider.order:
            res["live_price"] = res["listing_price"]
            return res
        ep = self.pinned_endpoint(eps, cfg.provider.order[0])
        if ep is None:
            res["problems"].append(f"pinned provider {cfg.provider.order[0]} not among live endpoints "
                                   f"{sorted({e['tag'] for e in eps})}")
            res["live_price"] = res["listing_price"]
        else:
            res["pinned_endpoint"] = ep["tag"]
            res["live_price"] = ep["price"]
        return res

    # ------------------------------------------------------------ chat
    async def chat(
        self,
        cfg: ApiModelCfg,
        user_prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        body: dict[str, Any] = {
            "model": cfg.slug,
            "messages": messages,
            "max_tokens": max_tokens or cfg.max_tokens,
            "provider": cfg.provider.model_dump(exclude_none=True),
            "usage": {"include": True},
        }
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        if cfg.reasoning:
            body["reasoning"] = cfg.reasoning
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            return await self._chat_once(body)
        except httpx.HTTPStatusError as e:
            # Some endpoints reject response_format; retry once without it (coder only).
            if json_mode and e.response.status_code == 400 and "response_format" in e.response.text:
                body.pop("response_format")
                return await self._chat_once(body)
            if _retryable(e):
                raise TransportFailure(str(e)) from e
            raise

    async def _chat_once(self, body: dict) -> dict:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception(_retryable),
            reraise=True,
        ):
            with attempt:
                loop = asyncio.get_running_loop()
                t0 = loop.time()
                r = await self.client.post("/chat/completions", json=body)
                latency_ms = int((loop.time() - t0) * 1000)
                if r.status_code >= 400:
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}: {r.text[:300]}", request=r.request, response=r)
                data = r.json()
                if "error" in data and not data.get("choices"):
                    code = (data["error"] or {}).get("code")
                    resp = httpx.Response(int(code) if isinstance(code, int) else 502, request=r.request, text=str(data["error"]))
                    raise httpx.HTTPStatusError(f"OpenRouter error: {data['error']}", request=r.request, response=resp)
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                finish = choice.get("finish_reason")
                if message.get("refusal") or (finish == "content_filter" and not content):
                    raise RefusalError(str(message.get("refusal") or finish))
                usage = data.get("usage") or {}
                provider = data.get("provider")
                return {
                    "content": content,
                    "finish_reason": finish,
                    "http_status": r.status_code,
                    "latency_ms": latency_ms,
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    "cost_usd": usage.get("cost"),
                    "provider_served": provider.get("name") if isinstance(provider, dict) else provider,
                    "model_served": data.get("model"),
                }
        raise TransportFailure("unreachable")  # pragma: no cover
