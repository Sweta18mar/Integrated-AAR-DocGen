"""Streamlit front end (maps to the React UI + API Gateway in the architecture).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from docplatform.config import Settings
from docplatform.ingest import Section
from docplatform.workflow import DEFAULT_AAR_OUTLINE, WORKSPACES, Workflow, WorkflowError

SAMPLES = Path(__file__).parent / "samples"
STAGES = [("upload", "Upload"), ("redaction", "Redact"), ("extraction", "Extract"), ("embedding", "Index"),
          ("route", "Route"), ("agent", "Agent"), ("review", "Review"), ("export", "Export")]
STATUS_COLOR = {"AWAITING_REVIEW": "orange", "EXPORTED": "green", "FAILED": "red", "APPROVED": "green"}

st.set_page_config(page_title="Document Platform", page_icon="📑", layout="wide")


@st.cache_resource(show_spinner=False)
def get_workflow(base_url: str, api_key: str, chat_model: str, fast_model: str, embed_model: str) -> Workflow:
    settings = Settings()
    settings.base_url, settings.api_key = base_url, api_key
    settings.chat_model, settings.fast_model, settings.embed_model = chat_model, fast_model, embed_model
    return Workflow(settings)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Workspace")
    tenant = st.text_input("Team", os.getenv("DOCPLAT_TENANT", "my-team"),
                           help="Each team has its own document index. Jobs never search another team's documents.")
    if "page_next" in st.session_state:
        st.session_state["page"] = st.session_state.pop("page_next")
    page = st.radio("View", ["New job", "Review jobs"], key="page", horizontal=True)

    with st.expander("Model connection", expanded=not os.getenv("DOCPLAT_API_KEY")):
        base_url = st.text_input("Base URL", os.getenv("DOCPLAT_BASE_URL", "http://localhost:3001/v1"))
        api_key = st.text_input("API key", os.getenv("DOCPLAT_API_KEY", ""), type="password")
        chat_model = st.text_input("Agent model", os.getenv("DOCPLAT_CHAT_MODEL", "auto:smart"))
        fast_model = st.text_input("Fast model (vendor detection)", os.getenv("DOCPLAT_FAST_MODEL", "auto:fast"))
        embed_model = st.text_input("Embedding model", os.getenv("DOCPLAT_EMBED_MODEL", ""),
                                    help="Leave empty to use the built-in local embedding.")

wf = get_workflow(base_url, api_key, chat_model, fast_model, embed_model)


def run_with_status(label: str, fn):
    """Run a workflow step, streaming its events into a status box."""
    with st.status(label, expanded=True) as box:
        def on_event(stage: str, message: str) -> None:
            box.write(f"**{stage}** · {message}")
        try:
            result = fn(on_event)
        except WorkflowError as exc:
            box.update(label=str(exc), state="error")
            return None
        except Exception as exc:  # network, auth or model errors
            box.update(label=f"Failed: {type(exc).__name__}: {exc}", state="error")
            return None
        box.update(label=f"{label} · done", state="complete", expanded=False)
        return result


# ---------------------------------------------------------------- new job
if page == "New job":
    st.title("New job")
    workspace = st.radio("What do you want to produce?", list(WORKSPACES),
                         format_func=lambda k: WORKSPACES[k]["label"])
    use_samples = st.toggle("Use the sample documents", value=False,
                            help="A bid evaluation, a project brief and a memo template, for trying the app.")

    default_title = ("Vendor recommendation memo" if workspace == "docagent" else "RFP-2026-041 bid review") \
        if use_samples else ""
    title = st.text_input("Title", default_title, key=f"title_{workspace}_{use_samples}")
    instructions = st.text_area("Instructions for the agent (optional)",
                                placeholder="Audience, tone, length, anything the draft must cover.")

    outline: list[Section] | None = None
    template_file = None
    if workspace == "docagent":
        template_file = st.file_uploader("Template (optional)", type=["md", "txt", "docx"],
                                         help="Headings become the sections the agent must write.")
    else:
        st.caption("Sections of the review. The agent writes each one, then a slide deck is built from them.")
        table = st.data_editor(pd.DataFrame([s.__dict__ for s in DEFAULT_AAR_OUTLINE]), num_rows="dynamic",
                               hide_index=True, width="stretch",
                               column_config={"heading": "Section", "guidance": "What it should cover"})
        outline = [Section(str(r.heading).strip(), str(r.guidance or "").strip())
                   for r in table.itertuples() if pd.notna(r.heading) and str(r.heading).strip()]

    uploads = st.file_uploader("Reference documents", type=["pdf", "docx", "txt", "md", "csv"],
                               accept_multiple_files=True)

    if st.button("Start job", type="primary"):
        files = [(f.name, f.getvalue(), "reference") for f in uploads or []]
        if template_file:
            files.append((template_file.name, template_file.getvalue(), "template"))
        if use_samples:
            files += [("bid_evaluation_notes.md", (SAMPLES / "bid_evaluation_notes.md").read_bytes(), "reference")]
            if workspace == "docagent":
                files += [("project_brief.md", (SAMPLES / "project_brief.md").read_bytes(), "reference")]
                if not template_file:
                    files += [("recommendation_template.md",
                               (SAMPLES / "recommendation_template.md").read_bytes(), "template")]
        try:
            job_id = wf.create_job(tenant, workspace, title, instructions, files, outline)
        except WorkflowError as exc:
            st.error(str(exc))
        else:
            st.session_state["job_id"] = job_id
            if run_with_status("Running the job", lambda cb: wf.start(job_id, cb)) is not None:
                st.session_state["page_next"] = "Review jobs"
                st.rerun()

# ---------------------------------------------------------------- review
else:
    jobs = wf.store.jobs(tenant)
    if not jobs:
        st.title("Review jobs")
        st.info("No jobs yet for this team. Start one from **New job**; "
                "turn on *Use the sample documents* to try it without your own files.")
        st.stop()

    ids = [j["id"] for j in jobs]
    current = st.session_state.get("job_id")
    job_id = st.selectbox("Job", ids, index=ids.index(current) if current in ids else 0,
                          format_func=lambda i: next(f"{j['title']} · {j['status']} · {i}" for j in jobs if j["id"] == i))
    st.session_state["job_id"] = job_id
    job = wf.store.job(job_id)
    events = wf.store.events(job_id)
    drafts = wf.store.drafts(job_id)

    st.title(job["title"])
    st.markdown(f":{STATUS_COLOR.get(job['status'], 'blue')}-badge[{job['status'].replace('_', ' ').title()}] "
                f"&nbsp; {WORKSPACES[job['workspace']]['label']}")

    seen = {e["stage"] for e in events}
    st.markdown(" → ".join(f"**{name}**" if stage in seen else f":gray[{name}]" for stage, name in STAGES))

    if job["status"] == "FAILED":
        st.error(job["error"] or "The job failed.")
        if st.button("Retry"):
            if run_with_status("Retrying", lambda cb: wf.retry(job_id, cb)) is not None:
                st.rerun()

    left, right = st.columns([3, 2], gap="large")

    with right:
        st.subheader("Inputs")
        files = wf.store.files(job_id)
        st.dataframe(pd.DataFrame([{
            "file": f["name"], "kind": f["kind"], "characters": f["chars"],
            "redacted": ", ".join(f"{n} {k.lower()}" for k, n in f["redactions"].items()) or "none",
        } for f in files]), hide_index=True, width="stretch")
        if job["workspace"] == "aar" and job["vendors"]:
            st.subheader("Detected vendors")
            st.dataframe(pd.DataFrame(job["vendors"]), hide_index=True, width="stretch")
        with st.expander(f"Activity log ({len(events)})"):
            st.dataframe(pd.DataFrame([{"time": pd.to_datetime(e["ts"], unit="s").strftime("%H:%M:%S"),
                                        "stage": e["stage"], "event": e["message"]} for e in events]),
                         hide_index=True, width="stretch")

    with left:
        if not drafts:
            st.info("No draft yet.")
            st.stop()
        versions = [d["version"] for d in drafts]
        version = st.radio("Draft version", versions, index=len(versions) - 1, horizontal=True,
                           format_func=lambda v: f"v{v}")
        entry = drafts[versions.index(version)]
        draft = entry["content"]
        if entry["feedback"]:
            st.caption(f"Revision requested: “{entry['feedback']}”")
        meta = [f"Agent note: {draft['agent_note']}" if draft.get("agent_note") else "",
                f"Written by: {', '.join(draft['served_by'])}" if draft.get("served_by") else ""]
        st.caption(" · ".join(m for m in meta if m))

        views = ["Report", "Slides"] if draft.get("slides") else ["Report"]
        for view, tab in zip(views, st.tabs(views)):
            with tab:
                if view == "Report":
                    for s in draft["sections"]:
                        st.markdown(f"#### {s['heading']}")
                        st.markdown(s["content"])
                else:
                    for i, slide in enumerate(draft["slides"], 1):
                        with st.container(border=True):
                            st.markdown(f"**{i}. {slide['title']}**")
                            st.markdown("\n".join(f"- {b}" for b in slide["bullets"]))
                            if slide.get("notes"):
                                st.caption(slide["notes"])
        if draft.get("sources"):
            with st.expander(f"Sources ({len(draft['sources'])})"):
                for key, src in sorted(draft["sources"].items(), key=lambda kv: int(kv[0][1:])):
                    st.markdown(f"**[{key}]** {src['source']}")
                    st.caption(src["excerpt"])

        is_latest = version == versions[-1]
        if job["status"] == "AWAITING_REVIEW" and is_latest:
            st.subheader("Your review")
            feedback = st.text_area("What should change?", key=f"fb_{job_id}_{version}",
                                    placeholder="Needed only when requesting changes.")
            approve_col, change_col = st.columns(2)
            if approve_col.button("Approve and export", type="primary", width="stretch"):
                if run_with_status("Exporting", lambda cb: wf.review(job_id, True, on_event=cb)) is not None:
                    st.rerun()
            if change_col.button("Request changes", width="stretch"):
                if run_with_status("Revising the draft",
                                   lambda cb: wf.review(job_id, False, feedback, on_event=cb)) is not None:
                    st.rerun()

        exports = wf.store.exports(job_id)
        if exports:
            st.subheader("Downloads")
            mime = {"md": "text/markdown",
                    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation"}
            cols = st.columns(len(exports))
            for col, e in zip(cols, exports):
                path = Path(e["path"])
                if path.exists():
                    col.download_button(f"{e['format'].upper()} · v{e['version']}", path.read_bytes(),
                                        path.name, mime[e["format"]], key=f"dl_{path.name}", width="stretch")
