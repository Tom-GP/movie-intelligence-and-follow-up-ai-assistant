"""The MCP email server. It offers two tools: `send_email` and `email_status`.

Only this server reads the SMTP password. The agent never sees it.
The agent's client starts this server automatically (see email_client.py).
"""

import re
import smtplib
import ssl
from email.message import EmailMessage

from mcp.server.fastmcp import FastMCP

from config.settings import settings

mcp = FastMCP("movie-email")

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
MAX_SUBJECT_LENGTH = 200
MAX_BODY_LENGTH = 20_000


def validate_email_input(recipient: str, subject: str, body: str) -> None:
    """Raise ValueError if anything about the email looks wrong."""
    if not EMAIL_PATTERN.fullmatch(recipient):
        raise ValueError(f"'{recipient}' is not a valid email address.")
    if not subject.strip():
        raise ValueError("The subject is empty.")
    if "\n" in subject or "\r" in subject:
        raise ValueError("The subject must be a single line.")
    if len(subject) > MAX_SUBJECT_LENGTH:
        raise ValueError(f"The subject is longer than {MAX_SUBJECT_LENGTH} characters.")
    if not body.strip():
        raise ValueError("The email body is empty.")
    if len(body) > MAX_BODY_LENGTH:
        raise ValueError(f"The body is longer than {MAX_BODY_LENGTH} characters.")


def build_message(sender: str, recipient: str, subject: str, body: str) -> EmailMessage:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    return message


def deliver(message: EmailMessage) -> None:
    """Hand the message to the SMTP server."""
    context = ssl.create_default_context()
    if settings.smtp_port == 465:
        with smtplib.SMTP_SSL(
            settings.smtp_host, settings.smtp_port, context=context, timeout=30
        ) as server:
            server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(message)
    else:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
            server.starttls(context=context)
            server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(message)


@mcp.tool()
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send a plain-text email.

    Args:
        recipient: The email address to send to.
        subject: A one-line subject.
        body: The plain-text body of the email.
    """
    validate_email_input(recipient, subject, body)

    if settings.email_dry_run:
        return (
            "DRY RUN: nothing was sent (EMAIL_DRY_RUN=true). "
            f"The email to {recipient} passed all checks."
        )

    if not settings.smtp_is_configured():
        raise ValueError(
            "SMTP is not configured. Fill in SMTP_HOST, SMTP_USER and SMTP_PASSWORD in .env."
        )

    sender = settings.email_from or settings.smtp_user
    message = build_message(sender, recipient, subject, body)
    try:
        deliver(message)
    except smtplib.SMTPAuthenticationError as error:
        raise ValueError(
            "The email server rejected the login. Check SMTP_USER and the App Password in .env."
        ) from error
    except (smtplib.SMTPException, OSError) as error:
        raise ValueError(f"Could not send the email: {error}") from error

    return f"Email sent to {recipient}."


@mcp.tool()
def email_status() -> str:
    """Report whether the email tool is ready. Never reveals any password."""
    if settings.email_dry_run:
        return "dry-run: emails are checked but NOT sent"
    if not settings.smtp_is_configured():
        return "not configured: fill in the SMTP settings in .env"
    sender = settings.email_from or settings.smtp_user
    return f"ready: sending as {sender} via {settings.smtp_host}"


if __name__ == "__main__":
    mcp.run(transport="stdio")