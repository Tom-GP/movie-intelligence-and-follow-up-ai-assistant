"""Chat with the agent in the terminal. Type 'quit' to stop.

Run from the project root:
    python -m scripts.chat
"""

from agent.agent import handle_message
from models.schemas import AgentRequest
from rag.llm import LLMError


def main() -> None:
    pending: AgentRequest | None = None
    print("Movie assistant. Ask a question, or type 'quit' to stop.")

    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in {"quit", "exit"}:
            break

        try:
            response = handle_message(message, pending=pending)
        except LLMError as error:
            print(f"\nLLM setup problem: {error}")
            continue

        print(f"\nAssistant: {response.format()}")
        # If the agent asked a question, remember the request for the next turn.
        pending = response.request if response.kind == "clarification" else None


if __name__ == "__main__":
    main()