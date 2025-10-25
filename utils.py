import streamlit as st
import re

def get_session_id():
    try:
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
        return get_script_run_ctx().session_id
    except ImportError:
        # Fallback for different Streamlit versions
        try:
            from streamlit.scriptrunner.script_run_context import get_script_run_ctx
            return get_script_run_ctx().session_id
        except ImportError:
            # If all else fails, return a default session ID
            return "default_session"

def write_message(role, content, save = True):
    """
    This is a helper function that saves a message to the
     session state and then writes a message to the UI
    """
    # Append to session state
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # Write to UI
    with st.chat_message(role):
        # Check if the content contains Cypher queries and format them properly
        formatted_content = format_cypher_in_text(content)
        st.markdown(formatted_content)

def format_cypher_in_text(text):
    """
    Detect Cypher queries in text and format them as code blocks to prevent wrapping
    """
    # Pattern to match Cypher queries (starts with MATCH, CREATE, etc.)
    cypher_pattern = r'(MATCH\s+.*?RETURN\s+[^.]+(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)'
    
    def replace_cypher(match):
        cypher_query = match.group(1)
        return f"\n```cypher\n{cypher_query}\n```\n"
    
    # Replace Cypher queries with code blocks
    formatted_text = re.sub(cypher_pattern, replace_cypher, text, flags=re.IGNORECASE | re.DOTALL)
    
    return formatted_text

def get_session_id():
    try:
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
        ctx = get_script_run_ctx()
        return ctx.session_id if ctx else "default_session"
    except ImportError:
        # Fallback for different Streamlit versions
        try:
            from streamlit.scriptrunner.script_run_context import get_script_run_ctx
            ctx = get_script_run_ctx()
            return ctx.session_id if ctx else "default_session"
        except ImportError:
            # If all else fails, return a default session ID
            return "default_session"
