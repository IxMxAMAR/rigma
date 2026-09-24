"""UIUX-22: the grounded-sources half that neither end implemented.

The sidecar has always returned `citations` on every `/ask`. The only consumer was
`search_my_documents` folding them into the MODEL's plain text as a
"sources: a.md, b.md" line — so the sources reached the model and never reached
the person reading the answer. For the one feature whose entire value is "which
of MY files said this", that is exactly backwards.

F11-12 deleted the frontend half because there was no producer; this is the
producer, so the two are restored together. The contract that matters: the
citations travel on the tool CONTEXT, which the turn loop reads after each round —
never on the transcript and never into the model's view, because a display change
must not be able to change an answer.
"""
import pytest

from rigma import rag, tools


@pytest.fixture
def ctx():
    return {"_citations": [], "workspace": "."}


def test_a_citation_lands_on_the_context_with_its_source(ctx):
    tools._record_citations(ctx, [{"source": "notes/a.md", "text": "the answer"}])
    assert ctx["_citations"] == [
        {"source": "notes/a.md", "snippet": "the answer"}]


def test_a_bare_string_citation_is_kept_as_a_source(ctx):
    """The sidecar is a separate process with its own schema, and the CLI's own
    reader accepts a plain string. Dropping that shape would silently lose sources
    on a sidecar version that returns them that way."""
    tools._record_citations(ctx, ["notes/b.md"])
    assert ctx["_citations"] == [{"source": "notes/b.md", "snippet": ""}]


def test_a_page_is_carried_only_when_it_is_a_real_page(ctx):
    tools._record_citations(ctx, [
        {"source": "a.pdf", "page": 7},
        {"source": "b.pdf", "page": 0},
        {"source": "c.pdf", "page": "7"},
    ])
    assert [c.get("page") for c in ctx["_citations"]] == [7, None, None]


def test_the_same_source_is_not_listed_twice(ctx):
    """A model that searches three times in a turn gets one chip per document, not
    one per search — the reader wants the set of files, not the query log."""
    row = {"source": "a.md", "text": "same"}
    tools._record_citations(ctx, [row])
    tools._record_citations(ctx, [row])
    tools._record_citations(ctx, [{"source": "a.md", "text": "same"}])
    assert len(ctx["_citations"]) == 1


def test_a_citation_with_no_source_is_dropped(ctx):
    """"Sourced from nothing" is worse than saying nothing."""
    tools._record_citations(ctx, [{"text": "orphan"}, {"source": "  "}, None, 7])
    assert ctx["_citations"] == []


def test_a_snippet_is_clipped_because_this_is_a_display_list(ctx):
    tools._record_citations(ctx, [{"source": "a.md", "text": "x" * 5000}])
    assert len(ctx["_citations"][0]["snippet"]) == 400


def test_the_list_is_bounded(ctx):
    tools._record_citations(
        ctx, [{"source": f"f{i}.md"} for i in range(200)])
    assert len(ctx["_citations"]) == 40


def test_a_malformed_context_is_survived(ctx):
    """This runs inside a TOOL. A context without the key (an older caller, a
    nested delegate's narrowed ctx) must not raise."""
    for bad in ({}, {"_citations": None}, None, "not a dict"):
        tools._record_citations(bad, [{"source": "a.md"}])   # must not raise


def test_nothing_to_record_is_a_no_op(ctx):
    for empty in (None, [], ()):
        tools._record_citations(ctx, empty)
    assert ctx["_citations"] == []


def test_search_my_documents_hands_its_citations_to_the_context(ctx, monkeypatch):
    """End to end through the tool, which is the surface that had the data and
    threw the structure away."""
    monkeypatch.setattr(rag, "live_sidecar_port", lambda timeout=5.0: 9)
    monkeypatch.setattr(rag, "ask", lambda q, port=None: {
        "answer": "grounded", "abstained": False,
        "citations": [{"source": "paper.pdf", "page": 3, "text": "evidence"}]})
    out = tools._search_docs({"query": "what?"}, ctx)
    assert "grounded" in out
    assert ctx["_citations"] == [
        {"source": "paper.pdf", "snippet": "evidence", "page": 3}]


def test_the_tool_still_tells_the_model_the_sources(ctx, monkeypatch):
    """The model's own view must not change: it is the reader's view that was
    missing. Removing the text line would be a different regression."""
    monkeypatch.setattr(rag, "live_sidecar_port", lambda timeout=5.0: 9)
    monkeypatch.setattr(rag, "ask", lambda q, port=None: {
        "answer": "grounded", "citations": [{"source": "paper.pdf"}]})
    out = tools._search_docs({"query": "what?"}, ctx)
    assert "sources: paper.pdf" in out


def test_a_sidecar_that_returns_no_citations_records_nothing(ctx, monkeypatch):
    monkeypatch.setattr(rag, "live_sidecar_port", lambda timeout=5.0: 9)
    monkeypatch.setattr(rag, "ask", lambda q, port=None: {
        "answer": "ungrounded", "citations": []})
    tools._search_docs({"query": "what?"}, ctx)
    assert ctx["_citations"] == []
