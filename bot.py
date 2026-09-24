"""Streamlit interface for the HVAC Codes GraphRAG assistant.

Run locally with::

    .\\venv\\Scripts\\python.exe -m streamlit run bot.py
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from agent import query_agent
from validation import sanitize_query, validate_query


st.set_page_config(
    page_title="HVAC Codes Assistant",
    page_icon="🏭",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        html, body { background: #f5f8fc !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: #f5f8fc;
            color: #183247;
        }
        header[data-testid="stHeader"] { background: #f5f8fc; }
        [data-testid="stDeployButton"] { display: none; }
        [data-testid="stToolbar"] { visibility: hidden; height: 0; }

        .block-container {
            max-width: 980px;
            padding: 2rem 1.25rem 7rem;
        }

        .hero {
            padding: 1.6rem 1.8rem 1.45rem;
            margin-bottom: 1.25rem;
            border: 1px solid rgba(30, 136, 229, 0.16);
            border-radius: 22px;
            background: linear-gradient(135deg, #f4f9ff 0%, #ffffff 62%, #f5fbf8 100%);
            box-shadow: 0 12px 32px rgba(25, 55, 90, 0.08);
        }

        .hero-kicker {
            color: #1479c9;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .hero h1 {
            margin: 0.45rem 0 0.45rem;
            color: #12304a;
            font-size: clamp(1.8rem, 4vw, 2.7rem);
            line-height: 1.1;
        }

        .hero p {
            max-width: 720px;
            margin: 0;
            color: #52677a;
            font-size: 1rem;
            line-height: 1.55;
        }

        .memory-note {
            margin: 0.85rem 0 1.5rem;
            color: #718395;
            font-size: 0.82rem;
            text-align: center;
        }

        [data-testid="stChatMessage"] {
            border-radius: 18px;
            border: 1px solid #e2eaf2;
            margin: 0.7rem 0;
            padding: 0.85rem 1rem;
            background: #ffffff !important;
            box-shadow: 0 5px 16px rgba(25, 55, 90, 0.05);
            color: #183247;
        }

        [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
            margin-left: 10% !important;
            background: #eaf3ff !important;
            border-color: #cfe3f8 !important;
        }

        [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {
            margin-right: 10% !important;
        }

        [data-testid="stChatMessageContent"] {
            background: transparent !important;
            color: #183247 !important;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            color: #183247;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child {
            margin-bottom: 0;
        }

        [data-testid="stChatInput"] {
            max-width: 980px;
            margin: 0 auto;
        }

        [data-testid="stBottomBlockContainer"],
        [data-testid="stBottomBlockContainer"] > div {
            background: #f5f8fc !important;
            border-top: 0 !important;
        }

        [data-testid="stBottom"],
        [data-testid="stBottom"] > div {
            background: #f5f8fc !important;
        }

        .stChatFloatingInputContainer,
        [data-testid="stBottomBlockContainer"] section {
            background: #f5f8fc !important;
        }

        [data-testid="stChatInput"] > div {
            border: 1px solid #c9d8e7;
            border-radius: 17px;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(25, 55, 90, 0.1);
        }

        [data-testid="stChatInput"] textarea {
            background: #ffffff !important;
            color: #183247 !important;
            -webkit-text-fill-color: #183247 !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #718395 !important;
            opacity: 1 !important;
        }

        .welcome {
            margin: 2.5rem auto 1rem;
            max-width: 700px;
            color: #52677a;
            text-align: center;
        }

        .welcome h2 {
            color: #12304a;
            font-size: 1.35rem;
        }

        .examples {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 1.25rem;
        }

        .example-card {
            min-height: 74px;
            padding: 0.85rem;
            border: 1px solid #e5edf4;
            border-radius: 14px;
            background: #fff;
            color: #52677a;
            font-size: 0.86rem;
            line-height: 1.35;
        }

        @media (max-width: 640px) {
            .block-container { padding: 1rem 0.75rem 6.5rem; }
            .hero { padding: 1.25rem 1.1rem; border-radius: 17px; }
            .examples { grid-template-columns: 1fr; }
            [data-testid="stChatMessage"] { padding: 0.75rem; }
            [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) { margin-left: 0 !important; }
            [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) { margin-right: 0 !important; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Grounded HVAC code assistant</div>
        <h1>Find the rule. Understand the source.</h1>
        <p>Ask about sections, requirements, exceptions, tables, or HVAC code topics. Answers are grounded in the indexed code corpus and include source citations when code evidence is used.</p>
    </section>
    <div class="memory-note">Conversation memory is limited to this browser session. It is not saved as long-term memory.</div>
    """,
    unsafe_allow_html=True,
)


if "messages" not in st.session_state:
    st.session_state.messages = []


prompt = st.chat_input("Ask about an HVAC section, requirement, or exception…")
pending_prompt = None
prior_history = []

if prompt:
    is_valid, error_msg = validate_query(prompt)
    if not is_valid:
        st.error(f"Please revise your question: {error_msg}")
    else:
        pending_prompt = sanitize_query(prompt)
        prior_history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages[-8:]
            if message.get("role") in {"user", "assistant"}
        ]
        st.session_state.messages.append({"role": "user", "content": pending_prompt})


for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def scroll_to_latest_message() -> None:
    """Ask the browser to keep the latest chat message in view."""

    components.html(
        """
        <script>
        (() => {
          const scroll = () => {
            try {
              const doc = window.parent.document;
              const messages = doc.querySelectorAll('[data-testid="stChatMessage"]');
              const latest = messages[messages.length - 1];
              if (latest) latest.scrollIntoView({behavior: "smooth", block: "end"});
            } catch (_) { /* Streamlit may sandbox this helper; native scroll remains available. */ }
          };
          setTimeout(scroll, 0);
          setTimeout(scroll, 180);
          setTimeout(scroll, 500);
        })();
        </script>
        """,
        height=0,
    )


if not st.session_state.messages and not prompt:
    st.markdown(
        """
        <div class="welcome">
            <h2>What would you like to look up?</h2>
            <div class="examples">
                <div class="example-card">What does Section 303.3 prohibit?</div>
                <div class="example-card">Are multiple fans allowed for emergency ventilation?</div>
                <div class="example-card">Where is refrigerant piping prohibited?</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if pending_prompt:
    scroll_to_latest_message()
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        response_placeholder.markdown("Searching the HVAC code and checking the source…")
        try:
            response = query_agent(pending_prompt, conversation_history=prior_history)
        except Exception:
            response = (
                "I’m sorry, something went wrong while handling that request. "
                "Please try a specific HVAC code section or requirement."
            )
        response_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    scroll_to_latest_message()
