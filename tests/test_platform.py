import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import fake_llm  # noqa: E402
from docplatform.config import Settings  # noqa: E402
from docplatform.ingest import chunk, parse_template, redact  # noqa: E402
from docplatform.llm import extract_json, local_embed  # noqa: E402
from docplatform.workflow import Workflow, WorkflowError  # noqa: E402

SAMPLES = ROOT / "samples"


@pytest.fixture(scope="module")
def server():
    srv = fake_llm.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


@pytest.fixture
def wf(server, tmp_path):
    return Workflow(Settings(base_url=server, api_key="test", data_dir=tmp_path, max_agent_steps=20))


def sample(name, kind="reference"):
    return (name, (SAMPLES / name).read_bytes(), kind)


# ------------------------------------------------------------ units
def test_redact_finds_personal_data():
    text, counts = redact("Mail a.b@example.org, call +1 415 555 0142, SSN 123-45-6789, card 4111 1111 1111 1111.")
    assert "@" not in text and "555" not in text and "6789" not in text and "1111" not in text
    assert counts == {"EMAIL": 1, "SSN": 1, "CARD": 1, "PHONE": 1}


def test_redact_keeps_ordinary_numbers():
    text, counts = redact("Bid was $1.82M over 3 years; score 78/100 in 2026.")
    assert counts == {} and "1.82M" in text


def test_chunk_respects_size():
    pieces = chunk("\n\n".join(["word " * 100] * 10), size=1200, overlap=100)
    assert len(pieces) > 1 and all(len(p) <= 1400 for p in pieces)


def test_parse_template_reads_headings_and_guidance():
    sections = parse_template((SAMPLES / "recommendation_template.md").read_text())
    assert [s.heading for s in sections] == ["Purpose", "Options considered", "Assessment",
                                             "Recommendation", "Risks and mitigations"]
    assert "decision" in sections[0].guidance


def test_extract_json_tolerates_fences():
    assert extract_json('ok\n```json\n{"a": 1}\n```') == {"a": 1}


def test_local_embedding_is_normalised():
    v = local_embed("vendor bid price")
    assert abs(sum(x * x for x in v) - 1) < 1e-5


# ------------------------------------------------------------ end to end
def test_docagent_job_end_to_end(wf):
    job_id = wf.create_job("team-a", "docagent", "Vendor recommendation", "Be concise.",
                           [sample("bid_evaluation_notes.md"), sample("project_brief.md"),
                            sample("recommendation_template.md", "template")])
    fake_llm.REQUESTS.clear()
    version = wf.start(job_id)
    job = wf.store.job(job_id)
    assert version == 1 and job["status"] == "AWAITING_REVIEW"

    # Redaction happened before anything reached the model.
    sent = json.dumps(fake_llm.REQUESTS)
    assert "priya.nair@example.org" not in sent and "555 0142" not in sent
    files = {f["name"]: f for f in wf.store.files(job_id)}
    assert files["bid_evaluation_notes.md"]["redactions"] == {"EMAIL": 1, "PHONE": 1}
    assert not Path(files["bid_evaluation_notes.md"]["path"]).exists()  # raw upload removed

    draft = wf.store.drafts(job_id)[0]["content"]
    assert [s["heading"] for s in draft["sections"]] == [s["heading"] for s in job["outline"]]
    assert all("[S1]" in s["content"] for s in draft["sections"])
    assert "S1" in draft["sources"]

    # Reject → revision v2 that applies the feedback.
    assert wf.review(job_id, approve=False, feedback="Make the purpose sharper.") == 2
    v2 = wf.store.drafts(job_id)[1]
    assert v2["feedback"] == "Make the purpose sharper."
    assert v2["content"]["sections"][0]["content"].startswith("Revised per reviewer feedback")
    assert v2["content"]["sections"][1] == draft["sections"][1]  # untouched sections kept

    exports = wf.review(job_id, approve=True)
    assert {e["format"] for e in exports} == {"md", "docx"}
    assert all(Path(e["path"]).exists() for e in exports)
    assert wf.store.job(job_id)["status"] == "EXPORTED"
    with pytest.raises(WorkflowError):
        wf.review(job_id, approve=True)


def test_aar_job_builds_deck(wf):
    job_id = wf.create_job("team-a", "aar", "RFP-2026-041 review", "", [sample("bid_evaluation_notes.md")])
    wf.start(job_id)
    job = wf.store.job(job_id)
    assert {v["name"] for v in job["vendors"]} == {"Northwind Systems", "Contoso Ltd", "Fabrikam Inc"}
    draft = wf.store.drafts(job_id)[0]["content"]
    assert len(draft["slides"]) == len(job["outline"])
    exports = {e["format"]: e["path"] for e in wf.review(job_id, approve=True)}
    from pptx import Presentation
    deck = Presentation(exports["pptx"])
    assert len(deck.slides) == len(draft["slides"]) + 2  # title + content + sources


def test_tenants_are_isolated(wf):
    a = wf.create_job("team-a", "docagent", "A", "", [sample("project_brief.md")])
    wf.ingest(a)
    vec = wf.llm.embed(["warehouse budget"])[0]
    assert wf.vectors.search("team-a", a, vec)
    assert wf.vectors.search("team-b", a, vec) == []


def test_agent_recovers_from_a_bad_reply(wf):
    job_id = wf.create_job("t", "docagent", "Recovery", "TEST_BAD_FIRST_REPLY", [sample("project_brief.md")])
    wf.start(job_id)
    events = [e["message"] for e in wf.store.events(job_id)]
    assert any("corrected the agent" in m for m in events)
    assert wf.store.job(job_id)["status"] == "AWAITING_REVIEW"


def test_api_embeddings_path(server, tmp_path):
    wf = Workflow(Settings(base_url=server, api_key="t", data_dir=tmp_path, embed_model="fake-embed"))
    job_id = wf.create_job("t", "docagent", "Emb", "", [sample("project_brief.md")])
    wf.start(job_id)
    assert wf.vectors.count("t", job_id) > 0


def test_reject_needs_feedback(wf):
    job_id = wf.create_job("t", "docagent", "F", "", [sample("project_brief.md")])
    wf.start(job_id)
    with pytest.raises(WorkflowError):
        wf.review(job_id, approve=False, feedback=" ")
    assert wf.store.job(job_id)["status"] == "AWAITING_REVIEW"


def test_job_needs_a_reference_document(wf):
    with pytest.raises(WorkflowError):
        wf.create_job("t", "docagent", "x", "", [sample("recommendation_template.md", "template")])


def test_failure_is_recorded_and_retryable(tmp_path):
    wf = Workflow(Settings(base_url="http://127.0.0.1:9/v1", api_key="t", data_dir=tmp_path, request_timeout=2))
    wf.llm.client = wf.llm.client.with_options(max_retries=0)
    job_id = wf.create_job("t", "docagent", "Down", "", [sample("project_brief.md")])
    with pytest.raises(Exception):
        wf.start(job_id)
    job = wf.store.job(job_id)
    assert job["status"] == "FAILED" and job["error"]
