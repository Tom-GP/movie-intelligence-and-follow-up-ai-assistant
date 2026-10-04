"""The agent's connection to the MCP email server.

The client starts the server as a child process and talks to it through stdio.
The agent only ever calls send_email_via_mcp(); it never touches SMTP.
"""

import asyncio
import sys
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config.settings import PROJECT_ROOT

TIMEOUT_SECONDS = 60

T = TypeVar("T")


class EmailSendError(Exception):
    """The email could not be sent (or the tool could not be reached)."""


@dataclass(frozen=True)
class EmailResult:
    ok: bool
    message: str


def _server_parameters(dry_run: bool | None = None) -> StdioServerParameters:
    """How to start the server. dry_run=True/False overrides the .env setting."""
    env = None
    if dry_run is not None:
        env = {"EMAIL_DRY_RUN": "true" if dry_run else "false"}
    return StdioServerParameters(
        command=sys.executable,  # the same Python that runs the agent (your venv)
        args=["-m", "mcp_tools.email_server"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )


async def _with_session(
    action: Callable[[ClientSession], Awaitable[T]], dry_run: bool | None
) -> T:
    async with stdio_client(_server_parameters(dry_run)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await action(session)


def _run(action: Callable[[ClientSession], Awaitable[T]], dry_run: bool | None = None) -> T:
    """Run one action against the server and return its result."""
    try:
        return asyncio.run(
            asyncio.wait_for(_with_session(action, dry_run), timeout=TIMEOUT_SECONDS)
        )
    except Exception as error:
        detail = str(error) or type(error).__name__
        raise EmailSendError(f"Could not use the email tool: {detail}") from error


def send_email_via_mcp(
    recipient: str, subject: str, body: str, dry_run: bool | None = None
) -> EmailResult:
    """Ask the MCP server to send an email."""

    async def action(session: ClientSession) -> tuple[bool, str]:
        result = await session.call_tool(
            "send_email", {"recipient": recipient, "subject": subject, "body": body}
        )
        text = "\n".join(getattr(block, "text", "") for block in result.content)
        return bool(result.isError), text

    # We decide what to do with an error only AFTER the connection is closed.
    is_error, text = _run(action, dry_run)
    if is_error:
        raise EmailSendError(text or "The email tool reported an error.")
    return EmailResult(ok=True, message=text)


def get_email_status(dry_run: bool | None = None) -> str:
    """Ask the server whether it is ready (the UI sidebar uses this in Phase 7)."""

    async def action(session: ClientSession) -> str:
        result = await session.call_tool("email_status", {})
        return "\n".join(getattr(block, "text", "") for block in result.content)

    return _run(action, dry_run)


def list_email_tools(dry_run: bool | None = None) -> list[str]:
    """The names of the tools the server offers."""

    async def action(session: ClientSession) -> list[str]:
        listing = await session.list_tools()
        return sorted(tool.name for tool in listing.tools)

    return _run(action, dry_run)