"""
ButterCup - YouTube AI Analyst

Streamlit entrypoint. This file owns the UI only: session state, widgets,
and wiring user actions to the pure business-logic modules in this package.
No LLM prompts or graph logic live here.

Run with: streamlit run app.py
"""
import warnings

import streamlit as st

from buttercup.config import settings
from buttercup.crag import build_crag_graph
from buttercup.crag.graph import empty_state
from buttercup.memory_store import clear as clear_memory
from buttercup.memory_store import create_memory, get_history, save_turn
from buttercup.summarizer import generate_summary
from buttercup.transcript_processor import clean_transcript
from buttercup.vectorstore_builder import build_retriever
from buttercup.youtube_service import extract_video_id, fetch_transcript, get_video_meta

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ButterCup · YouTube AI Analyst",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CACHED RESOURCES ────────────────────────────────────────────────────────


@st.cache_resource
def get_model():
    from langchain_groq import ChatGroq

    return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key)


@st.cache_resource
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


model = get_model()

# ── SESSION STATE DEFAULTS ───────────────────────────────────────────────────

defaults = {
    "processed": False,
    "translated": None,
    "summary": None,
    "retriever": None,
    "crag_graph": None,
    "messages": [],  # each: {role, content, verdict?, reason?}
    "memory": None,
    "video_id": None,
    "video_title": "",
    "video_author": "",
    "lang_code": "en",
    "word_count": 0,
    "char_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎬 ButterCup")
    st.caption("🤖 YouTube AI Analyst · ⚡ Groq + LangGraph Corrective RAG")
    st.divider()

    if not st.session_state.processed:
        url_input = st.text_input(
            "🔗 YouTube URL",
            placeholder="https://youtube.com/watch?v=...",
            help="Paste any YouTube video link",
        )
        analyze_clicked = st.button("⚡ Analyze Video", type="primary", use_container_width=True)
    else:
        url_input = ""
        analyze_clicked = False
        vid = st.session_state.video_id
        if vid:
            st.image(f"https://img.youtube.com/vi/{vid}/mqdefault.jpg", use_container_width=True)
        st.markdown(f"**🎞️ {st.session_state.video_title}**")
        st.caption(f"📺 {st.session_state.video_author}")
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("📝 Words", f"{st.session_state.word_count:,}")
        c2.metric("🌐 Language", st.session_state.lang_code.upper())
        st.metric("🔤 Characters", f"{st.session_state.char_count:,}")
        st.divider()
        if st.button("🔄 Analyze New Video", use_container_width=True):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()

    st.divider()
    with st.expander("ℹ️ How to use"):
        st.markdown(
            """
1. 🔗 Paste a YouTube URL above
2. ⚡ Click **Analyze Video**
3. 📋 Read the AI summary
4. 💬 Chat with the video content — ButterCup grades its own retrieval and
   falls back to a live web search when the transcript doesn't have the
   answer (Corrective RAG)
5. ⬇️ Download transcript or summary
            """
        )

# ── PROCESS VIDEO ────────────────────────────────────────────────────────────

if analyze_clicked and url_input:
    video_id = extract_video_id(url_input)
    if not video_id:
        st.error("❌ Invalid YouTube URL. Please check and try again.")
    else:
        st.session_state.video_id = video_id
        title, author = get_video_meta(video_id)
        st.session_state.video_title = title
        st.session_state.video_author = author

        with st.status("⚙️ Processing video…", expanded=True) as status:
            st.write("📥 Fetching transcript…")
            transcript_snippets, lang_code = fetch_transcript(video_id)
            st.session_state.lang_code = lang_code

            if not transcript_snippets:
                msgs = {
                    "disabled": "🚫 Transcripts are disabled for this video.",
                    "error": "💥 An error occurred while fetching transcripts.",
                    "unknown": "🤷 No transcripts could be found.",
                }
                status.update(label="❌ Failed", state="error")
                st.error(msgs.get(lang_code, "❌ No transcripts found."))
            else:
                st.write(f"✅ Transcript fetched ({lang_code.upper()}) 🌐")

                st.write("🔄 Preprocessing & translating…")
                translate_progress = st.progress(0, text="Translating transcript…")

                def _on_translate_progress(done: int, total: int) -> None:
                    translate_progress.progress(done / total, text=f"Translating… {done}/{total}")

                translated = clean_transcript(
                    transcript_snippets, lang_code, model, on_progress=_on_translate_progress
                )
                translate_progress.empty()

                st.session_state.translated = translated
                st.session_state.word_count = len(translated.split())
                st.session_state.char_count = len(translated)

                st.write("📝 Generating summary… ✨")
                summary_progress = st.progress(0, text="Extracting facts…")

                def _on_summary_progress(done: int, total: int) -> None:
                    summary_progress.progress(done / total, text=f"Extracting facts… {done}/{total}")

                st.session_state.summary = generate_summary(
                    translated, model, on_progress=_on_summary_progress
                )
                summary_progress.empty()

                st.write("🔍 Building knowledge base… 🧠")
                retriever = build_retriever(translated, get_embeddings())
                st.session_state.retriever = retriever

                st.write("🧭 Wiring up Corrective RAG graph… 🔗")
                st.session_state.crag_graph = build_crag_graph(llm=model, retriever=retriever)

                st.session_state.memory = create_memory()
                st.session_state.messages = []
                st.session_state.processed = True
                status.update(label="✅ Ready to explore! 🚀", state="complete", expanded=False)
        st.rerun()

# ── MAIN CONTENT ─────────────────────────────────────────────────────────────

if not st.session_state.processed:
    st.title("🎬 Welcome to ButterCup")
    st.markdown("**✨ Turn any YouTube video into a conversation.** 🔗 Paste a URL in the sidebar to get started.")
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("### 🎯 Smart Summary\n✅ Get a concise, fact-based summary of any YouTube video — no fluff, just substance.")
    with c2:
        st.success(
            "### 💬 Corrective RAG Chat\n"
            "🤖 Ask questions. ButterCup grades its own retrieved context and automatically "
            "falls back to a live web search when the video doesn't cover it."
        )
    with c3:
        st.warning("### 🌐 Multi-Language\n🗺️ Automatic transcript detection and translation across 15+ languages.")

    st.divider()
    st.markdown("#### 🚀 Works great with")
    eg1, eg2, eg3, eg4 = st.columns(4)
    eg1.markdown("🎓 Lectures & Tutorials")
    eg2.markdown("📰 News & Documentaries")
    eg3.markdown("💡 Tech Talks & Podcasts")
    eg4.markdown("🏋️ Fitness & How-To Guides")

else:
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.title(f"🎬 {st.session_state.video_title}")
        st.caption(
            f"by {st.session_state.video_author}  ·  {st.session_state.word_count:,} words  ·  "
            f"{st.session_state.lang_code.upper()}"
        )
    with hcol2:
        vid = st.session_state.video_id
        st.link_button("▶️ Watch on YouTube", f"https://youtube.com/watch?v={vid}", use_container_width=True)

    st.divider()

    tab_summary, tab_chat, tab_transcript = st.tabs(["📋 Summary", "💬 Chat", "📄 Transcript"])

    # ── SUMMARY TAB ──────────────────────────────────────────────────────────

    with tab_summary:
        st.subheader("📋 AI-Generated Summary ✨")
        st.info(st.session_state.summary)

        sc1, sc2, sc3 = st.columns([1, 1, 2])
        with sc1:
            st.download_button(
                "⬇️ Download Summary",
                data=st.session_state.summary,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with sc2:
            if st.button("🔁 Regenerate ✨", use_container_width=True):
                with st.spinner("🔄 Regenerating summary…"):
                    st.session_state.summary = generate_summary(st.session_state.translated, model)
                st.rerun()

        st.divider()
        st.subheader("📊 Video Stats")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📝 Words", f"{st.session_state.word_count:,}")
        m2.metric("🔤 Characters", f"{st.session_state.char_count:,}")
        m3.metric("🌐 Language", st.session_state.lang_code.upper())
        read_time = max(1, st.session_state.word_count // 200)
        m4.metric("⏱️ Est. Read Time", f"{read_time} min")

    # ── CHAT TAB (Corrective RAG) ───────────────────────────────────────────

    with tab_chat:
        if st.session_state.crag_graph is None:
            st.warning("⚠️ Something went wrong — the CRAG graph isn't ready. Please re-analyze the video.")
            st.stop()

        VERDICT_BADGE = {
            "DIRECT": "💬 Answered directly — no retrieval needed",
            "CORRECT": "🟢 Answered from the video transcript",
            "INCORRECT": "🌐 Video didn't cover this — answered from a live web search",
            "AMBIGUOUS": "🟡 Mixed signal — combined video transcript + web search",
        }

        user_input = st.chat_input("💬 Ask anything about the video…")

        chat_container = st.container(height=500)
        with chat_container:
            if not st.session_state.messages:
                st.info("👋 Welcome! 🎬 Ask me anything about the video — I'm ready to help. 🤖✨")
            else:
                for msg in st.session_state.messages:
                    avatar = "🧑" if msg["role"] == "user" else "🤖"
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.markdown(msg["content"])
                        if msg.get("verdict"):
                            st.caption(VERDICT_BADGE.get(msg["verdict"], msg["verdict"]))

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(user_input)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("🧠 Retrieving, grading, and (if needed) searching the web…"):
                        result = st.session_state.crag_graph.invoke(
                            empty_state(
                                question=user_input,
                                history=get_history(st.session_state.memory),
                            )
                        )
                        answer = result["answer"]
                        verdict = result.get("verdict", "")
                        save_turn(st.session_state.memory, user_input, answer)
                    st.markdown(answer)
                    if verdict:
                        st.caption(VERDICT_BADGE.get(verdict, verdict))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "verdict": verdict, "reason": result.get("reason", "")}
            )

        if st.session_state.messages:
            st.divider()
            cc1, cc2 = st.columns([1, 5])
            with cc1:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.messages = []
                    clear_memory(st.session_state.memory)
                    st.rerun()
            with cc2:
                chat_export = "\n\n".join(
                    f"{'🧑 You' if m['role']=='user' else '🤖 ButterCup'}: {m['content']}"
                    for m in st.session_state.messages
                )
                st.download_button(
                    "⬇️ Export Chat 💾",
                    data=chat_export,
                    file_name="chat_export.txt",
                    mime="text/plain",
                )

    # ── TRANSCRIPT TAB ───────────────────────────────────────────────────────

    with tab_transcript:
        st.subheader("📄 Processed Transcript 🌐")
        st.caption("🧹 Cleaned and translated transcript used for analysis.")

        search_term = st.text_input("🔍 Search transcript", placeholder="🔎 Type a keyword…")
        transcript = st.session_state.translated

        if search_term:
            count = transcript.lower().count(search_term.lower())
            st.caption(f"🎯 Found **{count}** occurrence(s) of '{search_term}'")
            parts = transcript.split(". ")
            hits = [p for p in parts if search_term.lower() in p.lower()]
            if hits:
                st.success("✅ **Matching sentences:**")
                for h in hits[:20]:
                    st.markdown(f"• {h.strip()}.")
            else:
                st.warning("🤷 No matching sentences found.")
            st.divider()

        with st.expander("📖 View full transcript", expanded=False):
            st.text_area(
                label="Full transcript",
                value=transcript,
                height=400,
                label_visibility="collapsed",
            )

        st.download_button(
            "⬇️ Download Full Transcript 📄",
            data=transcript,
            file_name="transcript.txt",
            mime="text/plain",
        )

        st.divider()
        st.subheader("📦 Export Everything 🚀")
        full_export = f"""ButterCup Export
==============
Video: {st.session_state.video_title}
Channel: {st.session_state.video_author}
Language: {st.session_state.lang_code.upper()}
Word Count: {st.session_state.word_count:,}

SUMMARY
-------
{st.session_state.summary}

FULL TRANSCRIPT
---------------
{st.session_state.translated}
"""
        st.download_button(
            "📦 Download Full Report (Summary + Transcript)",
            data=full_export,
            file_name="buttercup_report.txt",
            mime="text/plain",
            use_container_width=True,
            type="primary",
        )
