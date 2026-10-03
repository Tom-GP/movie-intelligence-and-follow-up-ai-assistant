"""Send one message to the agent and see how it was understood.

Examples (run from the project root):
    python -m scripts.agent_demo "Why did Tomas shut down the oxygen recycler?"
    python -m scripts.agent_demo "Send me a summary of the confrontation in Orbit Seven at john@example.com"
    python -m scripts.agent_demo "hi"
"""

import argparse

from agent.agent import handle_message
from rag.llm import LLMError


def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to the movie agent.")
    parser.add_argument("message")
    args = parser.parse_args()

    try:
        response = handle_message(args.message)
    except LLMError as error:
        print(f"LLM setup problem: {error}")
        return

    print("Understood request:")
    print(response.request.model_dump_json(indent=2))
    print(f"\nResponse type: {response.kind}\n")
    print(response.format())


if __name__ == "__main__":
    main()