"""The job workflow. Maps to the Step Functions state machine in the architecture.

States: CREATED → INGESTING → INDEXED → GENERATING → AWAITING_REVIEW
        → (rejected: REVISING → GENERATING → AWAITING_REVIEW …)
        → (approved: APPROVED → EXPORTED)
Any step can end in FAILED. AWAITING_REVIEW is the human-in-the-loop pause,
the local equivalent of a Step Functions task token.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from . import export
from .agent import DocumentAgent, detect_vendors, generate_slides
from .config import Settings
from .ingest import Section, UnsupportedFile, chunk, extract_text, parse_template, redact
from .llm import LLM
from .store import Store, VectorStore

WORKSPACES = {
    "docagent": {
        "label": "DocAgent · draft a document from a template",
        "role": "You draft a business document section by section from a template, grounded on the reference documents.",
    },
    "aar": {
        "label": "AAR · build a review deck from documents",
        "role": "You write an after-action review of the documents (for example a bid evaluation), "
                "which will then be turned into a slide deck. Use get_vendors to learn the bidders.",
    },
}

DEFAULT_AAR_OUTLINE = [
    Section("Overview", "What was reviewed, when, and why."),
    Section("Vendors and bids", "Each bidder and what they offered, with key figures."),
    Section("Evaluation", "How bids compare on cost, schedule, compliance and risk."),
    Section("Key findings", "The most important observations, good and bad."),
    Section("Recommendations", "What to do next, and what to change for future reviews."),
]

DEFAULT_DOC_OUTLINE = [
    Section("Summary"), Section("Background"), Section("Details"), Section("Next steps"),
]

EventCallback = Callable[[str, str], None]


class WorkflowError(Exception):
    pass


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\- ]+", "_", Path(name).name).strip() or "file"


class Workflow:
    def __init__(self, settings: Settings, llm: LLM | None = None):
        self.settings = settings
        self.store = Store(settings.data_dir)
        self.vectors = VectorStore(self.store)
        self.llm = llm or LLM(settings)

    def _emitter(self, job_id: str, on_event: EventCallback | None) -> EventCallback:
        def emit(stage: str, message: str) -> None:
            self.store.log(job_id, stage, message)
            if on_event:
                on_event(stage, message)
        return emit

    # ------------------------------------------------------------ 1-2 · upload
    def create_job(self, tenant: str, workspace: str, title: str, instructions: str,
                   files: list[tuple[str, bytes, str]], outline: list[Section] | None = None) -> str:
        """files: (name, content, kind) with kind in {"reference", "template"}."""
        if workspace not in WORKSPACES:
            raise WorkflowError(f"Unknown workspace {workspace!r}.")
        if not any(kind == "reference" for _, _, kind in files):
            raise WorkflowError("Add at least one reference document.")
        outline = outline or []
        job_id = self.store.create_job(tenant, workspace, title.strip() or "Untitled", instructions,
                                       [s.__dict__ for s in outline])
        raw = self.store.job_dir(job_id) / "raw"
        raw.mkdir(exist_ok=True)
        for name, content, kind in files:
            path = raw / _safe_name(name)
            path.write_bytes(content)
            self.store.add_file(job_id, path.name, kind, path)
        self.store.log(job_id, "upload", f"job created with {len(files)} file(s)")
        return job_id

    # ------------------------------------------------------------ 3-7 · ingest
    def ingest(self, job_id: str, on_event: EventCallback | None = None) -> None:
        emit = self._emitter(job_id, on_event)
        job = self.store.job(job_id)
        self.store.update_job(job_id, status="INGESTING")
        redacted_dir = self.store.job_dir(job_id) / "redacted"
        redacted_dir.mkdir(exist_ok=True)

        template_sections: list[Section] = []
        items: list[dict] = []
        for f in self.store.files(job_id):
            try:
                text = extract_text(Path(f["path"]))
            except UnsupportedFile as exc:
                emit("warning", str(exc))
                continue
            clean, counts = redact(text)
            (redacted_dir / f"{f['name']}.txt").write_text(clean, encoding="utf-8")
            self.store.update_file(job_id, f["name"], chars=len(clean), redactions=counts)
            found = ", ".join(f"{n} {k.lower()}" for k, n in counts.items()) or "nothing"
            emit("redaction", f"{f['name']}: redacted {found}")
            if f["kind"] == "template":
                template_sections += parse_template(clean)
                emit("template", f"{f['name']}: {len(template_sections)} sections")
                continue
            pieces = chunk(clean)
            items += [{"id": f"{job_id}:{f['name']}:{i}", "source": f["name"], "text": p} for i, p in enumerate(pieces)]
            emit("extraction", f"{f['name']}: {len(clean):,} characters → {len(pieces)} chunks")

        if not items:
            raise WorkflowError("No readable text in the reference documents.")
        self.vectors.add(job["tenant"], job_id, items, self.llm.embed([it["text"] for it in items]))
        emit("embedding", f"indexed {len(items)} chunks in namespace “{job['tenant']}”")
        if not self.settings.keep_raw_files:  # raw uploads are temporary; only redacted text is kept
            for f in self.store.files(job_id):
                Path(f["path"]).unlink(missing_ok=True)

        if not job["outline"]:
            outline = template_sections or (DEFAULT_AAR_OUTLINE if job["workspace"] == "aar" else DEFAULT_DOC_OUTLINE)
            self.store.update_job(job_id, outline=[s.__dict__ for s in outline])

        if job["workspace"] == "aar":
            self._detect_vendors(job_id, emit)
        self.store.update_job(job_id, status="INDEXED")

    def _detect_vendors(self, job_id: str, emit: EventCallback) -> None:
        job = self.store.job(job_id)
        vendors = detect_vendors(self.llm, self.vectors.chunks(job["tenant"], job_id))
        self.store.update_job(job_id, vendors=vendors)
        emit("vendors", f"detected {len(vendors)} vendor(s): " + (", ".join(v["name"] for v in vendors) or "none"))

    # ------------------------------------------------------------ 8-10 · generate
    def generate(self, job_id: str, on_event: EventCallback | None = None, feedback: str = "") -> int:
        emit = self._emitter(job_id, on_event)
        job = self.store.job(job_id)
        drafts = self.store.drafts(job_id)
        previous = drafts[-1]["content"] if (feedback and drafts) else None
        self.store.update_job(job_id, status="GENERATING")
        emit("route", f"route = {'AAR (sections → slides)' if job['workspace'] == 'aar' else 'Agent (template)'}")

        agent = DocumentAgent(self.llm, self.vectors, job["tenant"], job_id, emit,
                              self.settings.max_agent_steps,
                              vendors=job["vendors"] if job["workspace"] == "aar" else None)
        sections = [Section(**s) for s in job["outline"]]
        draft = agent.run(job["title"], job["instructions"], sections, WORKSPACES[job["workspace"]]["role"],
                          previous=previous, feedback=feedback)
        if job["workspace"] == "aar":
            draft["slides"] = generate_slides(self.llm, draft)
            emit("slides", f"generated {len(draft['slides'])} slides")
        version = self.store.save_draft(job_id, draft, feedback)
        self.store.update_job(job_id, status="AWAITING_REVIEW")
        emit("review", f"draft v{version} is waiting for review")
        return version

    def start(self, job_id: str, on_event: EventCallback | None = None) -> int:
        try:
            self.ingest(job_id, on_event)
            return self.generate(job_id, on_event)
        except Exception as exc:
            self._fail(job_id, exc, on_event)
            raise

    # ------------------------------------------------------------ 11-13 · review and export
    def review(self, job_id: str, approve: bool, feedback: str = "",
               on_event: EventCallback | None = None) -> list[dict] | int:
        job = self.store.job(job_id)
        if job["status"] != "AWAITING_REVIEW":
            raise WorkflowError(f"Job is {job['status']}, not waiting for review.")
        emit = self._emitter(job_id, on_event)
        try:
            if approve:
                self.store.update_job(job_id, status="APPROVED")
                emit("review", "approved")
                return self.export(job_id, on_event)
            if not feedback.strip():
                raise WorkflowError("Say what should change when rejecting a draft.")
            self.store.update_job(job_id, status="REVISING")
            emit("review", f"rejected: {feedback.strip()[:200]}")
            return self.generate(job_id, on_event, feedback=feedback)
        except WorkflowError:
            self.store.update_job(job_id, status="AWAITING_REVIEW")
            raise
        except Exception as exc:
            self._fail(job_id, exc, on_event)
            raise

    def export(self, job_id: str, on_event: EventCallback | None = None) -> list[dict]:
        emit = self._emitter(job_id, on_event)
        job = self.store.job(job_id)
        latest = self.store.drafts(job_id)[-1]
        draft, version = latest["content"], latest["version"]
        out_dir = self.store.job_dir(job_id) / "exports"
        out_dir.mkdir(exist_ok=True)
        stem = re.sub(r"[^\w\-]+", "_", job["title"]).strip("_")[:60] or "document"
        md_path = out_dir / f"{stem}_v{version}.md"
        md_path.write_text(export.to_markdown(draft), encoding="utf-8")
        files = [("md", md_path), ("docx", export.to_docx(draft, out_dir / f"{stem}_v{version}.docx"))]
        if job["workspace"] == "aar":
            files.append(("pptx", export.to_pptx(draft, draft.get("slides", []), out_dir / f"{stem}_v{version}.pptx")))
        for fmt, path in files:
            self.store.add_export(job_id, version, fmt, path)
        self.store.update_job(job_id, status="EXPORTED")
        emit("export", "exported " + ", ".join(fmt.upper() for fmt, _ in files))
        return self.store.exports(job_id)

    def _fail(self, job_id: str, exc: Exception, on_event: EventCallback | None) -> None:
        message = f"{type(exc).__name__}: {exc}"
        self.store.update_job(job_id, status="FAILED", error=message)
        self._emitter(job_id, on_event)("error", message)

    def retry(self, job_id: str, on_event: EventCallback | None = None) -> int:
        """Resume a failed job from the right stage."""
        job = self.store.job(job_id)
        if job["status"] != "FAILED":
            raise WorkflowError("Only failed jobs can be retried.")
        self.store.update_job(job_id, error="")
        try:
            if self.vectors.count(job["tenant"], job_id) == 0:
                self.ingest(job_id, on_event)
            elif job["workspace"] == "aar" and not job["vendors"]:
                self._detect_vendors(job_id, self._emitter(job_id, on_event))
            return self.generate(job_id, on_event)
        except Exception as exc:
            self._fail(job_id, exc, on_event)
            raise
