"""
Chat with Transcript — Streamlit Web UI

A browser-based chat interface for asking questions about your transcripts.
Uses Ollama (local LLM) + ChromaDB (vector store) for fully offline RAG.

Launch:
    streamlit run chat.py
"""
import streamlit as st
from pathlib import Path
from rag_engine import ingest_transcript, query_transcript, list_collections, delete_collection

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Chat with Transcript",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .source-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.85rem;
    }

    .source-speaker {
        color: #667eea;
        font-weight: 600;
    }

    .source-time {
        color: #888;
        font-size: 0.8rem;
    }

    .source-text {
        color: #ccc;
        margin-top: 4px;
    }

    .status-indexed {
        color: #4ade80;
        font-weight: 600;
    }

    .status-not-indexed {
        color: #f87171;
        font-weight: 600;
    }

    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_collection" not in st.session_state:
    st.session_state.active_collection = None

if "active_file" not in st.session_state:
    st.session_state.active_file = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🎙️ Transcript Manager")
    st.markdown("---")

    # Find transcript files
    transcript_dir = Path(".")
    txt_files = sorted([
        f.name for f in transcript_dir.glob("*.txt")
        if f.name != "requirements.txt"
    ])

    if not txt_files:
        st.warning("No transcript files found in the project directory.")
        st.stop()

    # File selector
    selected_file = st.selectbox(
        "📄 Select Transcript",
        txt_files,
        help="Pick a transcript file to chat with",
    )

    # Check if this file is already indexed
    collections = list_collections()
    collection_names = {c["name"]: c for c in collections}

    # Determine the expected collection name for this file
    from rag_engine import _collection_name_from_file
    expected_col = _collection_name_from_file(selected_file)
    is_indexed = expected_col in collection_names

    if is_indexed:
        col_info = collection_names[expected_col]
        st.markdown(
            f'<span class="status-indexed">✅ Indexed</span> — {col_info["chunk_count"]} chunks',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-not-indexed">❌ Not indexed</span>',
            unsafe_allow_html=True,
        )

    # Index button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Index", use_container_width=True):
            with st.spinner(f"Indexing {selected_file}..."):
                try:
                    col_name = ingest_transcript(selected_file)
                    st.session_state.active_collection = col_name
                    st.session_state.active_file = selected_file
                    st.session_state.messages = []  # Clear chat on re-index
                    st.success(f"Indexed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if is_indexed and st.button("🗑️ Delete", use_container_width=True):
            delete_collection(expected_col)
            st.session_state.active_collection = None
            st.session_state.messages = []
            st.rerun()

    # Auto-set active collection if indexed
    if is_indexed and st.session_state.active_collection != expected_col:
        st.session_state.active_collection = expected_col
        st.session_state.active_file = selected_file

    st.markdown("---")

    # All indexed collections
    if collections:
        st.markdown("### 📚 All Indexed Transcripts")
        for col in collections:
            st.markdown(f"• **{col['name']}** ({col['source_file']}, {col['chunk_count']} chunks)")

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. Select a transcript file
    2. Click **Index** to process it
    3. Ask questions in the chat!
    
    *Powered by Ollama (local LLM) + ChromaDB*
    """)


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">Chat with Transcript</div>', unsafe_allow_html=True)

if st.session_state.active_file:
    st.markdown(
        f'<div class="subtitle">Chatting with: <strong>{st.session_state.active_file}</strong></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="subtitle">Select and index a transcript to start chatting</div>',
        unsafe_allow_html=True,
    )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📎 Sources", expanded=False):
                for src in message["sources"]:
                    st.markdown(
                        f"""<div class="source-card">
                            <span class="source-speaker">{src['speaker']}</span>
                            <span class="source-time"> [{src['start_time']:.1f}s → {src['end_time']:.1f}s]</span>
                            <div class="source-text">{src['text'][:200]}{'...' if len(src['text']) > 200 else ''}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )


# Chat input
if prompt := st.chat_input("Ask something about the transcript..."):
    if not st.session_state.active_collection:
        st.error("Please select and index a transcript first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query RAG
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = query_transcript(prompt, st.session_state.active_collection)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    # Show sources
                    with st.expander("📎 Sources", expanded=False):
                        for src in sources:
                            st.markdown(
                                f"""<div class="source-card">
                                    <span class="source-speaker">{src['speaker']}</span>
                                    <span class="source-time"> [{src['start_time']:.1f}s → {src['end_time']:.1f}s]</span>
                                    <div class="source-text">{src['text'][:200]}{'...' if len(src['text']) > 200 else ''}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                    # Save to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    error_msg = str(e)
                    if "Connection refused" in error_msg or "ConnectError" in error_msg:
                        st.error(
                            "🔌 **Ollama is not running!**\n\n"
                            "Start Ollama and pull the required models:\n"
                            "```\nollama pull llama3.2\nollama pull mxbai-embed-large\n```"
                        )
                    else:
                        st.error(f"Error: {error_msg}")
