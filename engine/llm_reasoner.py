from __future__ import annotations

import os
import re

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
MODEL_NAME: str = os.getenv(
    "FIREWORKS_MODEL_NAME",
    "accounts/fireworks/models/qwen3-8b",
)

_FALLBACK_REASON = (
    "LLM reasoning unavailable: Fireworks API key is not configured. "
    "Running in local demo mode."
)

# ---------------------------------------------------------------------------
# Lazy client — only constructed when API key is present
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    """Return a cached OpenAI-compatible Fireworks client, or None if no key is set."""
    global _client
    # Graceful fallback! If we don't have the Fireworks key in .env, we gracefully return None instead of crashing, keeping the CPU demo alive.
    api_key = os.getenv("FIREWORKS_API_KEY", "").strip()
    if not api_key:
        return None
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.fireworks.ai/inference/v1",
        )
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain_signals_batch(signals: list[dict]) -> list[dict]:
    """Enrich each signal dict with an ``llm_reason`` field.

    For every signal in *signals* this function:

    1. Builds a concise financial-analyst prompt from the signal fields
       (symbol, action, confidence, sentiment, entry / stop-loss / target).
    2. Calls Qwen3-8B via the Fireworks AI API when ``FIREWORKS_API_KEY`` is
       present in the environment.
    3. Strips any ``<think>…</think>`` chain-of-thought blocks the model may
       emit before returning the final answer.
    4. Falls back to a deterministic local string
       (``"LLM reasoning unavailable: …"``) when the API key is absent or when
       the API call fails for any reason — the app never crashes.

    Args:
        signals: List of signal dicts produced by the signal generator.  Each
            dict must contain at least ``symbol``, ``signal``, ``confidence``,
            ``sentiment``, ``sentiment_score``, ``entry``, ``sl``, and
            ``target``.

    Returns:
        The same list of dicts, each augmented with an ``"llm_reason"`` key.
    """
    enriched = []
    client = _get_client()

    for s in signals:
        if client is None:
            reason = _FALLBACK_REASON
        else:
            prompt = (
                "You are a concise financial AI analyst. "
                "Explain in 2-3 sentences WHY this trading signal was generated.\n"
                "Signal Data:\n"
                f"- Symbol: {s['symbol']}\n"
                f"- Action: {s['signal']}\n"
                f"- Confidence: {s['confidence']:.1%}\n"
                f"- Sentiment: {s['sentiment']} (score: {s['sentiment_score']:+.4f})\n"
                f"- Entry: ${s['entry']:.2f} | Stop-Loss: ${s['sl']:.2f} | Target: ${s['target']:.2f}\n"
                "Be specific, professional, and reference the sentiment and confidence. "
                "No disclaimers."
            )
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=120,
                    temperature=0.3,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                raw = response.choices[0].message.content or ""
                raw = raw.strip()
                # Strip <think>…</think> blocks regardless of whether they appear
                reason = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                if not reason:
                    reason = raw  # keep original if stripping removed everything
            except Exception as exc:
                reason = f"LLM reasoning unavailable: {exc}"

        enriched.append({**s, "llm_reason": reason})

    return enriched