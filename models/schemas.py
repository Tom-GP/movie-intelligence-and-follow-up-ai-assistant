"""Pydantic models: the structured data passed between parts of the agent."""

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Intent = Literal["information", "email", "clarification"]

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


class AgentRequest(BaseModel):
    """What the router understood from the user's message."""

    intent: Intent
    original_message: str = ""
    movie_title: str | None = None
    query: str | None = None
    recipient_email: str | None = None
    requested_content: str | None = None
    clarification_question: str | None = None

    @field_validator("recipient_email")
    @classmethod
    def check_recipient_email(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not EMAIL_PATTERN.match(value):
            raise ValueError(f"'{value}' is not a valid email address")
        return value.lower()

    @model_validator(mode="after")
    def check_clarification_has_question(self) -> "AgentRequest":
        if self.intent == "clarification" and not self.clarification_question:
            raise ValueError("A clarification request needs a clarification_question")
        return self


class AgentResponse(BaseModel):
    """What the agent sends back to the user interface."""

    kind: Literal["answer", "clarification", "email_pending"]
    text: str
    sources: list[str] = Field(default_factory=list)
    request: AgentRequest

    def format(self) -> str:
        """The text to display, with a Sources list when there are sources."""
        if not self.sources:
            return self.text
        source_lines = "\n".join(f"- {source}" for source in self.sources)
        return f"{self.text}\n\nSources:\n{source_lines}"