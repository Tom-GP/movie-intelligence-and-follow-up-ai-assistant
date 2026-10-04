"""Test the MCP email server on its own, without the agent.

    python -m scripts.test_mcp_server                      # safe: nothing is sent
    python -m scripts.test_mcp_server --real you@gmail.com # really sends a test email
"""

import argparse

from mcp_tools.email_client import (
    EmailSendError,
    get_email_status,
    list_email_tools,
    send_email_via_mcp,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the MCP email server.")
    parser.add_argument(
        "--real",
        metavar="ADDRESS",
        help="Really send a test email to this address (needs SMTP settings in .env).",
    )
    args = parser.parse_args()

    dry_run = args.real is None
    recipient = args.real or "test@example.com"

    print("1. Starting the MCP server and listing its tools...")
    print("   Tools found:", ", ".join(list_email_tools(dry_run)))

    print("2. Asking the server for its status...")
    print("  ", get_email_status(dry_run))

    mode = "dry run" if dry_run else "REAL"
    print(f"3. Sending a test email ({mode})...")
    result = send_email_via_mcp(
        recipient, "Test from the movie assistant", "Hello! This is a test.", dry_run
    )
    print("  ", result.message)

    print("4. Sending to an invalid address (the server should reject it)...")
    try:
        send_email_via_mcp("not-an-email", "Test", "Hello", dry_run=True)
        print("   PROBLEM: the invalid address was accepted!")
    except EmailSendError as error:
        print("   Rejected as expected:", error)

    print("\nAll checks finished.")


if __name__ == "__main__":
    main()