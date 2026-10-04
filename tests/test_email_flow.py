from agent.email_flow import compose_body, parse_draft
from rag.qa import Citation


def test_parse_draft_reads_the_subject_line():
    reply = "Subject: Orbit Seven recap [1]\n\nHi there!\nThe core overheated [1]."
    subject, body = parse_draft(reply, "Fallback")
    assert subject == "Orbit Seven recap"
    assert body == "Hi there!\nThe core overheated [1]."


def test_parse_draft_uses_the_fallback_subject_when_missing():
    subject, body = parse_draft("Just a body.", "Fallback")
    assert subject == "Fallback"
    assert body == "Just a body."


def test_compose_body_appends_sources_from_metadata():
    citation = Citation(1, "orbit_seven_0001", "Orbit Seven", "00:02:14,000", "00:02:58,500", 2, 10)
    assert compose_body("Body", [citation]) == (
        "Body\n\nSources:\n- [1] Orbit Seven — 00:02:14–00:02:58"
    )