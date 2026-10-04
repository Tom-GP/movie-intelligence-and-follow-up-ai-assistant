"""A scripted demo of the assistant. Real emails are NEVER sent here.

Run from the project root:
    python -m scripts.demo

It needs your LLM settings in .env and the 3 invented sample movies indexed
(they are in data/subtitles by default).
"""

from agent.agent import handle_message
from mcp_tools.email_client import EmailResult, send_email_via_mcp
from rag.llm import LLMError

STEPS = [
    ("A question, answered with sources", ["Why did Tomas shut down the oxygen recycler?"]),
    (
        "An ambiguous quote: the assistant asks, you answer",
        ['Who says "I know"?', "Orbit Seven"],
    ),
    (
        "An email request (the email tool runs in dry-run mode)",
        ["Send me a summary of Orbit Seven at demo@example.com"],
    ),
    ("Something that is not in the movies", ["What is the capital of France?"]),
]


def dry_run_sender(recipient: str, subject: str, body: str) -> EmailResult:
    """Always use the email tool in dry-run mode, whatever .env says."""
    return send_email_via_mcp(recipient, subject, body, dry_run=True)


def main() -> None:
    for number, (title, messages) in enumerate(STEPS, start=1):
        print(f"\n=== Step {number}: {title} ===")
        pending = None  # a question the assistant asked and is waiting on

        for message in messages:
            print(f"\nYou: {message}")
            try:
                response = handle_message(message, pending=pending, send_email_fn=dry_run_sender)
            except LLMError as error:
                print(f"\nLLM setup problem: {error}")
                return
            print(f"\nAssistant: {response.format()}")
            pending = response.request if response.kind == "clarification" else None

    print("\n=== Demo finished ===")


if __name__ == "__main__":
    main()