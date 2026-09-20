
from __future__ import annotations

import uuid

import streamlit as st
from dotenv import load_dotenv

from src.assistant import Session, SupportAssistant
from src.gate import Route
from src.llm_client import LLMError

load_dotenv()

st.set_page_config(page_title="LearnForge Support Assistant", page_icon="🎓", layout="wide")

ROUTE_LABELS = {
    Route.ANSWER: ("Answered from knowledge base", "green"),
    Route.ESCALATE_LOW_CONFIDENCE: ("Escalated · low confidence", "orange"),
    Route.ESCALATE_SENSITIVE: ("Escalated · sensitive case", "red"),
    Route.ESCALATE_STALE_CONFLICT: ("Escalated · stale/conflicting policy", "violet"),
}


@st.cache_resource(show_spinner=False)
def get_assistant() -> SupportAssistant:

    return SupportAssistant()


def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "session" not in st.session_state:
        st.session_state.session = Session(session_id=st.session_state.session_id)
    if "turns" not in st.session_state:
        st.session_state.turns = []  


def sidebar():
    with st.sidebar:
        st.markdown("### About")
        st.write(
            "A retrieval-augmented support assistant over LearnForge's FAQs, "
            "policies, and historical tickets. Retrieval and the escalation "
            "gate run locally (TF-IDF + rules); generation calls a free "
            "model via **OpenRouter**."
        )
        st.markdown("### Session")
        st.code(st.session_state.session_id, language=None)
        if st.button("🔄 New session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.session = Session(session_id=st.session_state.session_id)
            st.session_state.turns = []
            st.rerun()

        st.markdown("### Try asking")
        examples = [
            "I bought a course yesterday, can I get a refund?",
            "What if I've already watched most of it?",
            "My progress reset after I used my phone",
            "There's a charge on my card I don't recognize",
            "Can I download courses to watch offline?",
            "What's the weather like tomorrow?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex-{ex}", use_container_width=True):
                st.session_state.pending_query = ex

        st.markdown("---")
        st.caption(
            "OPENROUTER_API_KEY not set → retrieval/routing still works, "
            "but generation will show an escalation instead of an error."
        )


def render_sources(sources):
    if not sources:
        st.caption("No sources retrieved.")
        return

    for s in sources:
        r = s.record
        badge = {"policy": "🟦", "faq": "🟩", "ticket": "🟨"}.get(
            r.source_type, "⬜"
        )
        stale = " ⚠️ contains superseded wording" if r.has_deprecated_note else ""

        st.markdown(
            f"**{badge} {r.id} — {r.title}**  \n"
            f"Score: `{s.score:.3f}`{stale}"
        )

        st.write(r.text)

        meta = f"type: `{r.source_type}`"
        if r.reviewed:
            meta += f" · reviewed: {r.reviewed}"
        if r.status:
            meta += f" · historical status: {r.status}"

        st.caption(meta)
        st.markdown("---")


def main():
    init_state()
    st.title("🎓 LearnForge Support Assistant")
    st.caption("RAG prototype · TF-IDF retrieval + rule-based safety gate + OpenRouter generation")
    sidebar()

    assistant = get_assistant()

    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            label, color = ROUTE_LABELS.get(turn["route"], (turn["route"], "gray"))
            st.markdown(f":{color}[**{label}**]")
            st.write(turn["answer"])
            if turn.get("error"):
                st.error(f"LLM error: {turn['error']}")
            with st.expander("Retrieved sources", expanded=False):
                render_sources(turn["sources"])

    query = st.chat_input("Ask a LearnForge support question…")
    pending = st.session_state.pop("pending_query", None)
    if pending and not query:
        query = pending

    if query:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating an answer…"):
                try:
                    result = assistant.handle(query, session=st.session_state.session)
                except LLMError as exc:
                    st.error(f"LLM error: {exc}")
                    return
            label, color = ROUTE_LABELS.get(result.route, (result.route.value, "gray"))
            st.markdown(f":{color}[**{label}**]")
            st.write(result.answer)
            if result.error:
                st.error(f"LLM error: {result.error}")
            with st.expander("Retrieved sources", expanded=False):
                render_sources(result.sources)

        st.session_state.turns.append({
            "query": query,
            "answer": result.answer,
            "route": result.route,
            "sources": result.sources,
            "error": result.error,
        })


if __name__ == "__main__":
    main()
