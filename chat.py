"""
Streamlit Web UI — Chat with Transcript & Live Transcription
"""
import time
import warnings
import streamlit as st
from pathlib import Path

# Suppress annoying transformers/streamlit path warnings in the terminal
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Whisper Studio",
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
    .subtitle { color: #888; font-size: 1rem; margin-bottom: 2rem; }
    .source-card { background: #1e1e2e; border: 1px solid #333; border-radius: 8px; padding: 12px 16px; margin: 6px 0; font-size: 0.85rem; }
    .source-speaker { color: #667eea; font-weight: 600; }
    .source-time { color: #888; font-size: 0.8rem; }
    .source-text { color: #ccc; margin-top: 4px; }
    .status-indexed { color: #4ade80; font-weight: 600; }
    .status-not-indexed { color: #f87171; font-weight: 600; }
    .stChatMessage { border-radius: 12px; }
    .st-emotion-cache-1kyxreq { justify-content: center; } /* Center buttons */
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "RAG Chat"

# RAG state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "active_file" not in st.session_state:
    st.session_state.active_file = None

# Live state
if "live_transcriber" not in st.session_state:
    st.session_state.live_transcriber = None
if "is_live_recording" not in st.session_state:
    st.session_state.is_live_recording = False
if "live_history" not in st.session_state:
    st.session_state.live_history = []


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎙️ Whisper Studio")
    st.selectbox(
        "Mode",
        ["RAG Chat", "Live Transcription"],
        key="app_mode"
    )
    st.markdown("---")


# ===========================================================================
# MODE 1: RAG CHAT
# ===========================================================================
if st.session_state.app_mode == "RAG Chat":
    from rag_engine import ingest_transcript, query_transcript, list_collections, delete_collection
    
    with st.sidebar:
        st.markdown("#### Transcript Manager")
        transcript_dir = Path(".")
        txt_files = sorted([f.name for f in transcript_dir.glob("*.txt") if f.name != "requirements.txt"])

        if not txt_files:
            st.warning("No transcript files found.")
        else:
            selected_file = st.selectbox("📄 Select Transcript", txt_files)
            
            collections = list_collections()
            collection_names = {c["name"]: c for c in collections}

            from rag_engine import _collection_name_from_file
            expected_col = _collection_name_from_file(selected_file)
            is_indexed = expected_col in collection_names

            if is_indexed:
                col_info = collection_names[expected_col]
                st.markdown(f'<span class="status-indexed">✅ Indexed</span> — {col_info["chunk_count"]} chunks', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-not-indexed">❌ Not indexed</span>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Index", use_container_width=True):
                    with st.spinner(f"Indexing {selected_file}..."):
                        try:
                            col_name = ingest_transcript(selected_file)
                            st.session_state.active_collection = col_name
                            st.session_state.active_file = selected_file
                            st.session_state.messages = []
                            st.success("Indexed!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            with col2:
                if is_indexed and st.button("🗑️ Delete", use_container_width=True):
                    delete_collection(expected_col)
                    st.session_state.active_collection = None
                    st.session_state.messages = []
                    st.rerun()

            if is_indexed and st.session_state.active_collection != expected_col:
                st.session_state.active_collection = expected_col
                st.session_state.active_file = selected_file

        st.markdown("---")
        if collections:
            st.markdown("#### All Indexed Transcripts")
            for col in collections:
                st.markdown(f"• **{col['name']}** ({col['source_file']}, {col['chunk_count']} chunks)")

    st.markdown('<div class="main-header">Chat with Transcript</div>', unsafe_allow_html=True)
    if st.session_state.active_file:
        st.markdown(f'<div class="subtitle">Chatting with: <strong>{st.session_state.active_file}</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="subtitle">Select and index a transcript to start chatting</div>', unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
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

    if prompt := st.chat_input("Ask something about the transcript..."):
        if not st.session_state.active_collection:
            st.error("Please select and index a transcript first!")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = query_transcript(prompt, st.session_state.active_collection)
                        answer = result["answer"]
                        sources = result["sources"]
                        st.markdown(answer)
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
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })
                    except Exception as e:
                        if "Connection refused" in str(e) or "ConnectError" in str(e):
                            st.error("🔌 **Ollama is not running!**")
                        else:
                            st.error(f"Error: {e}")


# ===========================================================================
# MODE 2: LIVE TRANSCRIPTION
# ===========================================================================
elif st.session_state.app_mode == "Live Transcription":
    from live_transcriber import LiveTranscriber

    st.markdown('<div class="main-header">Live Transcription</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Speak into your microphone natively in real-time.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("#### Options")
        live_lang = st.text_input("Source Language (e.g. en, zh) - leave blank for auto-detect", "")
        live_lang = live_lang.strip() if live_lang.strip() else None
        
        do_translate = st.checkbox("Translate to English live")
        auto_save = st.checkbox("Auto-save session", value=True)

    def start_recording():
        task = "translate" if do_translate else "transcribe"
        transcriber = LiveTranscriber(language=live_lang, device_pref="auto", task=task)
        
        # We hook into the transcriber so it updates our session state directly
        def on_new_text(text):
            st.session_state.live_history.append(text)
            
        transcriber.on_segment_ready = on_new_text
        transcriber.start_listening()
        
        st.session_state.live_transcriber = transcriber
        st.session_state.is_live_recording = True
        st.session_state.live_history = []

    def stop_recording():
        if st.session_state.live_transcriber is not None:
            st.session_state.live_transcriber.stop_listening()
            if auto_save:
                path = st.session_state.live_transcriber.save_session()
                if path:
                    st.success(f"Session saved to {path}!")
            st.session_state.live_transcriber = None
        st.session_state.is_live_recording = False

    col1, col2 = st.columns([1, 1])
    
    with col1:
        if not st.session_state.is_live_recording:
            st.button("🔴 Start Recording", on_click=start_recording, use_container_width=True, type="primary")
        else:
            st.button("⏹ Stop Recording", on_click=stop_recording, use_container_width=True)

    # The transcript UI area
    st.markdown("### 📝 Live Transcript")
    
    transcript_container = st.empty()
    
    def render_history():
        history = st.session_state.live_transcriber.transcript_history if st.session_state.live_transcriber else []
        if not history:
            transcript_container.info("Waiting for speech... (Start Recording and speak)")
        else:
            transcript_container.text_area(
                label="Transcript", 
                value="\n".join(history), 
                height=400,
                label_visibility="collapsed"
            )

    render_history()

    # If we are recording, auto-refresh 
    if st.session_state.is_live_recording:
        time.sleep(1.0)
        st.rerun()
