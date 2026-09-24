"""Streamlit interface for the HVAC Codes GraphRAG assistant.

Run locally with::

    .\\venv\\Scripts\\python.exe -m streamlit run bot.py
"""

from __future__ import annotations

import streamlit as st

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
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header[data-testid="stHeader"] { background: transparent; }
        [data-testid="stDeployButton"] { display: none; }

        .block-container {
            max-width: 980px;
            padding: 2.5rem 1.25rem 7rem;
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
            border-radius: 16px;
            border: 1px solid rgba(18, 48, 74, 0.08);
            margin-bottom: 0.75rem;
            padding: 0.9rem 1rem;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child {
            margin-bottom: 0;
        }

        [data-testid="stChatInput"] {
            max-width: 980px;
            margin: 0 auto;
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


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if not st.session_state.messages:
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


prompt = st.chat_input("Ask about an HVAC section, requirement, or exception…")
if prompt:
    is_valid, error_msg = validate_query(prompt)
    if not is_valid:
        st.error(f"Please revise your question: {error_msg}")
    else:
        sanitized_prompt = sanitize_query(prompt)
        prior_history = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages[-8:]
            if message.get("role") in {"user", "assistant"}
        ]
        st.session_state.messages.append({"role": "user", "content": sanitized_prompt})
        with st.chat_message("user"):
            st.markdown(sanitized_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the HVAC code and checking the source…"):
                try:
                    response = query_agent(sanitized_prompt, conversation_history=prior_history)
                except Exception:
                    response = (
                        "I’m sorry, something went wrong while handling that request. "
                        "Please try a specific HVAC code section or requirement."
                    )
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
