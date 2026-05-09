from __future__ import annotations
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url="https://api.fireworks.ai/inference/v1",
        )
    return _client

def explain_signals_batch(signals: list[dict]) -> list[dict]:
    enriched = []
    for s in signals:
        prompt = f"""You are a concise financial AI analyst. Explain in 2-3 sentences WHY this trading signal was generated.
Signal Data:
- Symbol: {s['symbol']}
- Action: {s['signal']}
- Confidence: {s['confidence']:.1%}
- Sentiment: {s['sentiment']} (score: {s['sentiment_score']:+.4f})
- Entry: ${s['entry']:.2f} | Stop-Loss: ${s['sl']:.2f} | Target: ${s['target']:.2f}
Be specific, professional, and reference the sentiment and confidence. No disclaimers."""
        try:
            response = _get_client().chat.completions.create(
                model="accounts/fireworks/models/qwen3-8b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.3,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = response.choices[0].message.content.strip()
            reason = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        except Exception as e:
            reason = f"LLM reasoning unavailable: {e}"
        enriched.append({**s, "llm_reason": reason})
    return enriched