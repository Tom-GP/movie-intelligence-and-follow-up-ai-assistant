"""Run the MCP email server by itself (for debugging).

    python -m scripts.run_mcp_server

It prints nothing and waits for a client to talk to it (stdio). Stop it with Ctrl+C.
Normally you do NOT need this, because the agent starts the server automatically.
To test the server properly, use:  python -m scripts.test_mcp_server
"""

from mcp_tools.email_server import mcp


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()