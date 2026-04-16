"""Utility functions for LLM querying."""
import os
from google import genai
from openai import OpenAI
from functools import wraps

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Cost per million tokens per model (input, output)
_COST_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gpt-5-nano":           (0.05, 0.40),
}
_COST_DEFAULT = (1.0, 4.0)  # fallback for unknown models

# Clients keyed by provider to support mixing models
_clients: dict[str, genai.Client | OpenAI] = {}


def _get_client(model_name: str):
    if "gemini" in model_name:
        if "gemini" not in _clients:
            _clients["gemini"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return _clients["gemini"]
    if "gpt" in model_name:
        if "openai" not in _clients:
            _clients["openai"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return _clients["openai"]
    raise ValueError(f"Unknown model: {model_name}")


def _calc_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = _COST_PER_MTOK.get(model_name, _COST_DEFAULT)
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def query_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.0,
    model_name: str = "gemini-2.5-flash-lite", # "gpt-5-nano"
) -> tuple[str, float]:
    client = _get_client(model_name)

    if "gemini" in model_name:
        config = genai.types.GenerateContentConfig(temperature=temperature)
        if system:
            config.system_instruction = system
        response = client.models.generate_content(
            model=model_name, contents=prompt, config=config,
        )
        usage = response.usage_metadata
        cost = _calc_cost(model_name, usage.prompt_token_count, usage.candidates_token_count)
        return response.text, cost

    if "gpt" in model_name:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        # gpt-5-nano など temperature 固定モデルは省略する
        kwargs = dict(model=model_name, messages=messages)
        if model_name not in {"gpt-5-nano"}:
            kwargs["temperature"] = temperature
        response = client.chat.completions.create(**kwargs)
        usage = response.usage
        cost = _calc_cost(model_name, usage.prompt_tokens, usage.completion_tokens)
        return response.choices[0].message.content, cost


def create_call_limited_query_llm(base_query_llm, max_calls: int):
    """Wrap query_llm to raise an error after max_calls invocations."""
    state = {"calls": 0}

    @wraps(base_query_llm)
    def limited(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] > max_calls:
            raise RuntimeError(
                f"LLM call limit exceeded ({max_calls} calls allowed)."
            )
        return base_query_llm(*args, **kwargs)

    return limited
