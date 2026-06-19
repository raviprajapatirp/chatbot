import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langsmith import Client
import os
import time

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Swastik Chat",
    page_icon="🌟",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0d0f12;
    --surface:   #161a20;
    --border:    #2a2f3a;
    --accent:    #e87c3e;
    --accent2:   #f5a461;
    --green:     #4ade80;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --user-bg:   #1e2330;
    --bot-bg:    #13171e;
    --radius:    12px;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

[data-testid="stMainBlockContainer"] {
    max-width: 780px;
    padding: 0 1.5rem 6rem;
}

.title-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1.6rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.6rem;
}
.title-bar .icon { font-size: 2rem; line-height: 1; }
.title-bar h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text);
    margin: 0;
    letter-spacing: -0.5px;
}
.badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent);
    background: rgba(232,124,62,0.12);
    border: 1px solid rgba(232,124,62,0.3);
    border-radius: 4px;
    padding: 2px 7px;
}
.ls-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--green);
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.3);
    border-radius: 4px;
    padding: 2px 7px;
}

.msg-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 1.2rem;
}
.msg-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0 4px;
}
.msg-label.user  { color: var(--accent2); }
.msg-bubble {
    padding: 0.85rem 1.1rem;
    border-radius: var(--radius);
    font-size: 0.93rem;
    line-height: 1.65;
    border: 1px solid var(--border);
    white-space: pre-wrap;
    word-break: break-word;
}
.msg-bubble.user      { background: var(--user-bg); border-color: rgba(232,124,62,0.2); }
.msg-bubble.assistant { background: var(--bot-bg); }

.trace-box {
    background: rgba(74,222,128,0.05);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #4ade80;
    margin-top: 4px;
    word-break: break-all;
}

[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.93rem !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea:focus { border-color: var(--accent) !important; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
.stSlider > div > div > div { background: var(--accent) !important; }

.stButton button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}
.stButton button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Groq models ───────────────────────────────────────────────────
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# ── Session state ─────────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "trace_urls" not in st.session_state: st.session_state.trace_urls = []

# ── Read secrets ──────────────────────────────────────────────────
def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    # ── Groq ──────────────────────────────────────────────────────
    st.markdown("#### ⚡ Groq")
    groq_key = st.text_input(
        "Groq API Key",
        value=get_secret("GROQ_API_KEY"),
        type="password",
        placeholder="gsk_...",
        help="Free key at console.groq.com",
    )

    st.markdown("---")

    # ── LangSmith ─────────────────────────────────────────────────
    st.markdown("#### 🔍 LangSmith")
    ls_key = st.text_input(
        "LangSmith API Key",
        value=get_secret("LANGSMITH_API_KEY"),
        type="password",
        placeholder="lsv2_...",
        help="Free key at smith.langchain.com",
    )
    ls_project = st.text_input(
        "Project Name",
        value=get_secret("LANGSMITH_PROJECT", "groq-chat"),
        placeholder="groq-chat",
    )

    # ── Status ────────────────────────────────────────────────────
    groq_ok = bool(groq_key and groq_key.startswith("gsk_"))
    ls_ok   = bool(ls_key)

    st.markdown(
        f"""
        <div style="display:flex;flex-direction:column;gap:6px;margin:0.8rem 0">
            <div style="display:flex;align-items:center;gap:8px">
                <div style="width:8px;height:8px;border-radius:50%;
                    background:{'#4ade80' if groq_ok else '#f87171'}"></div>
                <span style="font-size:0.78rem;color:{'#4ade80' if groq_ok else '#f87171'}">
                    Groq {'Ready ✓' if groq_ok else 'Key required'}
                </span>
            </div>
            <div style="display:flex;align-items:center;gap:8px">
                <div style="width:8px;height:8px;border-radius:50%;
                    background:{'#4ade80' if ls_ok else '#f5a461'}"></div>
                <span style="font-size:0.78rem;color:{'#4ade80' if ls_ok else '#f5a461'}">
                    LangSmith {'Tracing ON ✓' if ls_ok else 'Key required'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    model       = st.selectbox("Model", GROQ_MODELS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens  = st.slider("Max tokens", 64, 2048, 512, 64)

    st.markdown("---")
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, concise, and friendly AI assistant.",
        height=100,
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear chat"):
            st.session_state.messages   = []
            st.session_state.trace_urls = []
            st.rerun()
    with col2:
        msg_count = len(st.session_state.messages)
        st.markdown(
            f'<div style="font-size:0.75rem;color:#6b7280;padding-top:6px">'
            f'{msg_count} message{"s" if msg_count != 1 else ""}</div>',
            unsafe_allow_html=True,
        )

# ── Enable LangSmith tracing ──────────────────────────────────────
if ls_ok:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]    = ls_key
    os.environ["LANGCHAIN_PROJECT"]    = ls_project
    os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ── Title ─────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="title-bar">
        <div class="icon">🌟</div>
        <h1>Swastik Chat</h1>
        <div style="display:flex;gap:6px;margin-left:auto;align-items:center">
            <div class="badge">Groq · {model}</div>
            {"<div class='ls-badge'>🔍 LangSmith</div>" if ls_ok else ""}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Warnings ──────────────────────────────────────────────────────
if not groq_ok:
    st.warning("⚠️ Enter your Groq API key in the sidebar. Free at [console.groq.com](https://console.groq.com)")
if not ls_ok:
    st.info("💡 Add your LangSmith API key to enable tracing. Free at [smith.langchain.com](https://smith.langchain.com)")

# ── Render chat history ───────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    role  = msg["role"]
    label = "YOU" if role == "user" else "LLAMA"
    cls   = "user" if role == "user" else "assistant"
    st.markdown(
        f"""
        <div class="msg-wrap">
            <div class="msg-label {cls}">{label}</div>
            <div class="msg-bubble {cls}">{msg["content"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Show trace link under each assistant message
    if role == "assistant":
        idx = i // 2
        if idx < len(st.session_state.trace_urls) and st.session_state.trace_urls[idx]:
            url = st.session_state.trace_urls[idx]
            st.markdown(
                f'<div class="trace-box">🔍 LangSmith Trace → '
                f'<a href="{url}" target="_blank" style="color:#4ade80">{url}</a></div>',
                unsafe_allow_html=True,
            )

# ── Chat input ────────────────────────────────────────────────────
if prompt := st.chat_input("Message LLaMA…", disabled=not groq_ok):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f"""
        <div class="msg-wrap">
            <div class="msg-label user">YOU</div>
            <div class="msg-bubble user">{prompt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Build LangChain messages
    lc_messages = [SystemMessage(content=system_prompt)]
    for m in st.session_state.messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    st.markdown('<div class="msg-wrap"><div class="msg-label">LLAMA</div>', unsafe_allow_html=True)
    response_placeholder = st.empty()
    full_response = ""

    try:
        # ── LangChain Groq — auto-traced by LangSmith ─────────────
        llm = ChatGroq(
            api_key=groq_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
        )

        run_config = {
            "run_name": f"Ravi-Chat-Turn-{len(st.session_state.messages)}",
            "tags": ["streamlit", "groq", "ravi", model],
            "metadata": {"user": "Ravi", "model": model, "project": ls_project},
        }

        # Stream tokens
        for chunk in llm.stream(lc_messages, config=run_config):
            full_response += chunk.content or ""
            response_placeholder.markdown(
                f'<div class="msg-bubble assistant">{full_response}▌</div>',
                unsafe_allow_html=True,
            )

        # Final render
        response_placeholder.markdown(
            f'<div class="msg-bubble assistant">{full_response}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Fetch LangSmith trace URL ─────────────────────────────
        trace_url = None
        if ls_ok:
            try:
                time.sleep(1)  # allow trace to be recorded
                ls_client = Client(api_key=ls_key)
                runs = list(ls_client.list_runs(
                    project_name=ls_project,
                    execution_order=1,
                    limit=1,
                ))
                if runs:
                    trace_url = f"https://smith.langchain.com/public/{runs[0].id}/r"
                else:
                    trace_url = f"https://smith.langchain.com/projects/{ls_project}"

               # st.markdown(
                   # f'<div class="trace-box">🔍 LangSmith Trace → '
                   # f'<a href="{trace_url}" target="_blank" style="color:#4ade80">{trace_url}</a></div>',
                    #unsafe_allow_html=True,
                #)
            except Exception:
                trace_url = f"https://smith.langchain.com/projects/{ls_project}"

        st.session_state.trace_urls.append(trace_url)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        response_placeholder.error(f"Error: {e}")
        st.session_state.trace_urls.append(None)
