# Document Platform (agentic)

A runnable, local version of the integrated DocAgent + AAR architecture. One app, two workspaces:

- **DocAgent** drafts a document from a template, grounded on your reference documents.
- **AAR** writes a review of documents such as a bid evaluation, detects the vendors, and builds a slide deck.

Both share the same pipeline: upload → redact → extract → chunk → index → route → agent drafts
→ human review (approve, or request changes and the agent revises) → export.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the **Model connection** panel in the sidebar and enter your endpoint and key. It defaults to a
local FreeLLMAPI gateway (`http://localhost:3001/v1`), but any OpenAI-compatible endpoint works.
You can also set them before starting (Windows `cmd` shown; use `export` on macOS/Linux):

```bash
set DOCPLAT_API_KEY=freellmapi-your-key
set DOCPLAT_BASE_URL=http://localhost:3001/v1
```

To try it, choose a workspace, turn on **Use the sample documents** and click **Start job**.

### Try it without any model

```bash
python tests/fake_llm.py 8099
```

Then set the base URL to `http://127.0.0.1:8099/v1` (any key). The fake model follows the agent
protocol with canned text, so you can click through the whole flow offline.

## How the agent works

`docplatform/agent.py` runs a tool-using loop. Each step the model replies with one JSON action:

| Tool | What it does |
|---|---|
| `search(query, k)` | Semantic search over this job's redacted documents; passages come back labelled `[S1]`, `[S2]`… |
| `list_sources()` | Names of the job's documents |
| `get_vendors()` | Vendors found by VendorDetection (AAR only) |
| `read_section(heading)` | Current text of a section |
| `write_section(heading, content)` | Write or replace one required section, with citations |
| `finish(note)` | End; refused until every section is written (and, on a revision, until something changed) |

It uses a JSON action protocol rather than native function calling, so it works with whatever model a
gateway routes to. Malformed replies are fed back to the model as errors and retried. Retrieved text is
marked as data, never instructions, and the agent stops at a step limit (`DOCPLAT_MAX_AGENT_STEPS`, default 30).

## Mapping to the AWS architecture

| Diagram component | Here |
|---|---|
| React UI, API Gateway (REST + WebSocket) | `app.py` (Streamlit; live progress replaces WebSocket) |
| Cognito | "Team" field (no sign-in locally) |
| Upload Lambda, S3 raw (temp) | `Workflow.create_job`; raw files are deleted once indexed |
| Redaction Analysis | `ingest.redact` (emails, phones, SSNs, card numbers, IPs) |
| Extraction Lambda + Textract | `ingest.extract_text` (PDF text layer, DOCX, TXT/MD/CSV; scans need OCR) |
| Chunker, Embedding Lambda | `ingest.chunk`, `LLM.embed` |
| S3 Vectors (namespace per tenant) | `store.VectorStore`, always filtered by team and job |
| VendorDetection | `agent.detect_vendors` |
| Step Functions + Choice state | `workflow.Workflow` (states listed in its docstring) |
| Orchestration / SectionGenerator + Bedrock | `agent.DocumentAgent` |
| SlideGenerator | `agent.generate_slides` |
| HITL Controller (task token) | `AWAITING_REVIEW` state + `Workflow.review` |
| Export Lambda, DownloadHandler | `export.py` (MD, DOCX, PPTX) + download buttons |
| DynamoDB job table | `store.Store` (SQLite: jobs, files, events, draft versions, exports) |

Each module is a natural Lambda boundary, so moving to AWS means replacing the local store with
DynamoDB/S3/S3 Vectors and calling the same functions from Step Functions tasks.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `DOCPLAT_BASE_URL` | `http://localhost:3001/v1` | OpenAI-compatible endpoint |
| `DOCPLAT_API_KEY` | (empty) | API key |
| `DOCPLAT_CHAT_MODEL` | `auto:smart` | Agent and slide model |
| `DOCPLAT_FAST_MODEL` | `auto:fast` | Vendor detection |
| `DOCPLAT_EMBED_MODEL` | (empty) | Empty uses the built-in local embedding |
| `DOCPLAT_DATA_DIR` | `data_store` | Where jobs and the index are stored |
| `DOCPLAT_MAX_AGENT_STEPS` | `30` | Agent step limit |
| `DOCPLAT_KEEP_RAW` | `0` | `1` keeps original uploads after indexing |

## Tests

```bash
pip install pytest
python -m pytest tests
```

14 tests, including full DocAgent and AAR runs against the fake model, a check that personal data
never reaches the model, tenant isolation, revision handling and failure recording.

## Limits

- Redaction is pattern-based. It does not catch names or addresses; for production use Amazon
  Comprehend PII detection or Bedrock Guardrails as well.
- Scanned PDFs without a text layer are skipped with a warning (Textract handles those on AWS).
- Jobs run inside the web request, so a slow model keeps the page busy until the step finishes.
- No sign-in: the Team field separates indexes but is not access control.
