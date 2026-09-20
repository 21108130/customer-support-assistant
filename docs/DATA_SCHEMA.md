# Data Schema

## Design decision: one record = one chunk

The supplied corpus (`faqs.md`, `policies.md`, `tickets.md`) is 40 entries that are
already short (100-300 words), topical, and self-contained. Splitting them further
would cut a policy's caveats away from its main statement, or split a ticket's
resolution from the problem it resolves — both hurt more than they help at this
size. So `src/corpus.py` treats **each supplied entry as exactly one chunk/record**.
This is called out explicitly in the Trade-offs section of the top-level README as
the first thing to revisit if the real corpus were large or unevenly sized.

## Record shape

Produced by `src/corpus.py`, persisted at `data/corpus.json`. This is the shape a
vector DB record or SQL row would carry in a production version (see
`docs/ARCHITECTURE.md` for how the interface stays the same if TF-IDF is swapped
for a vector index).

```json
{
  "id": "POLICY-02",
  "title": "Cancellation and Refund Policy",
  "source_type": "policy",
  "text": "You can cancel an active subscription at any time ... Effective date: January 2026.",
  "reviewed": "January 2026",
  "status": null,
  "has_deprecated_note": true,
  "tags": ["billing", "refund", "subscription"]
}
```

| Field | Type | Present on | Purpose |
|---|---|---|---|
| `id` | string (PK) | all | Stable citation key. The LLM is instructed to cite `[ID]` inline; the eval harness checks recall by ID. |
| `title` | string | all | Retrieval signal (weighted into the TF-IDF document) and display label in the UI. |
| `source_type` | enum: `policy`, `faq`, `ticket` | all | Drives the **authority ranking** (policy > FAQ > ticket) and the prompt rule that tickets are precedent, not policy. |
| `text` | text | all | The evidence sent to the LLM inside a `<context id="...">` block. Nothing outside `text` is shown to the model. |
| `reviewed` | string, nullable | policy | Freshness signal shown to the operator/user; parsed from lines like `Effective date: January 2026.` |
| `status` | string, nullable | ticket | The ticket's resolution status (`Resolved`, `Escalated`, `Pending monitoring`, ...), shown so the model doesn't imply an open case is settled. |
| `has_deprecated_note` | bool | all (mostly policy) | Set by a keyword scan (`older version`, `no longer`, `obsolete`, `should not be treated as`, ...) at ingestion time. Used two ways: (1) injected into the prompt as a warning not to restate the superseded claim, (2) read by the gate to help decide when a user-flagged conflict should escalate instead of being answered. |
| `tags` | string[] | all | Lightweight keyword tags (`refund`, `accessibility`, `billing`, ...) for human browsing / future filtered retrieval. Not used by the TF-IDF ranking itself today. |

## Where each field comes from

`src/corpus.py` parses the three markdown files with a small set of regexes:

- **Heading** — `# POLICY-02 — Cancellation and Refund Policy` → `id`, `title`.
- **Policy date fields** — `Last reviewed:` / `Effective date:` / `Updated:` / `Last updated:` / `Reviewed:` lines → `reviewed` (and removed from `text` so the date line isn't duplicated as a body sentence).
- **Ticket status** — `STATUS: Resolved.` → `status` (also stripped from `text`).
- **Deprecation scan** — a fixed phrase list (`STALE_MARKERS` in `corpus.py`) checked against the body → `has_deprecated_note`.
- **Tags** — a small keyword→tag lookup table checked against title+body → `tags`.

Rebuild the corpus at any time with:

```bash
python3 src/corpus.py
```

## Session / conversation state

Multi-turn context is a second, much simpler record, kept separate from the
knowledge-base records above:

```json
// .state/<session_id>.json
{
  "history": [
    {"role": "user", "content": "I bought a course yesterday; can I get a refund?"},
    {"role": "assistant", "content": "Yes — LearnForge's standard refund window ... [FAQ-02][POLICY-02]"},
    {"role": "user", "content": "What if I've already watched most of it?"}
  ]
}
```

- Up to **12 turn-pairs** persisted to disk per session (`MAX_HISTORY_TURNS_PERSISTED`).
- Only the **last 6 turn-pairs** are actually sent to the LLM per call (`MAX_HISTORY_TURNS_SENT_TO_MODEL`), to bound prompt size and cost.
- Keyed by an opaque `session_id` (CLI: `--session alice`; Streamlit: a random id per browser session, with a "New session" button).
- This is local-file state for the prototype; see the README trade-offs for the production replacement (a real session store, e.g. Redis/Postgres, keyed by authenticated user).

## Eval case shape

`evals/cases.json` — one record per regression test:

```json
{
  "id": "sensitive-unrecognized-charge",
  "query": "There's a charge on my card I don't recognize, I think it's fraud",
  "expected_sources_any": ["FAQ-15", "POLICY-10"],
  "expected_route": "escalate_sensitive"
}
```

`evals/run_eval.py` checks that retrieval returns at least one of
`expected_sources_any` in the top-4, and that the gate's route matches
`expected_route` exactly.

## Why not a vector DB record for this prototype

A vector DB row for this corpus would look like:

```json
{ "id": "POLICY-02", "embedding": [0.0123, -0.041, ...], "metadata": { "...": "same fields as above" } }
```

i.e. identical metadata, with `text` replaced/accompanied by an embedding vector
and the cosine-similarity search delegated to the vector index instead of
`sklearn`'s `TfidfVectorizer`. Because `Retriever.search(query, k)` is the only
interface the rest of the system depends on (see `src/retrieval.py`), swapping the
vectorizer for real embeddings + a vector store (e.g. pgvector, Qdrant) is a
localized change — the record schema above does not need to change.
