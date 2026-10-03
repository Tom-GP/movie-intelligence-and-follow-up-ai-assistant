"""One function to call whichever LLM provider is set in .env."""

from config.settings import settings


class LLMError(Exception):
    """Raised when the LLM is not configured correctly."""


def generate(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
    """Send a prompt to the configured LLM and return its text reply."""
    provider = settings.llm_provider.lower()

    if not settings.llm_model:
        raise LLMError("LLM_MODEL is empty. Set it in your .env file.")

    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is empty. Set it in your .env file.")
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is empty. Set it in your .env file.")
        from anthropic import Anthropic

        client = Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )

    raise LLMError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use 'openai' or 'anthropic'."
    )