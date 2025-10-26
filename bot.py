"""
Streamlit Chatbot Interface for HVAC Codes Assistant
Run: streamlit run bot.py
"""
import streamlit as st
from agent import query_agent, get_statistics
from validation import validate_query, sanitize_query

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="HVAC Codes Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================

st.markdown('<h1 class="main-header">🏭 HVAC Codes Knowledge Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about HVAC codes, sections, equipment, and standards</p>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This assistant helps you navigate HVAC codes and standards using AI-powered search.
    
    **Features:**
    - Section lookups
    - Equipment requirements
    - Code relationships
    - Standards compliance
    """)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ==================== CHAT HISTORY ====================

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==================== CHAT INPUT ====================

# Handle chat input
prompt = st.chat_input("Ask about HVAC codes...")

# Process user input
if prompt:
    # Validate and sanitize input
    is_valid, error_msg = validate_query(prompt)
    
    if not is_valid:
        st.error(f"❌ Invalid query: {error_msg}")
    else:
        # Sanitize the query
        sanitized_prompt = sanitize_query(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": sanitized_prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(sanitized_prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = query_agent(sanitized_prompt)
                    st.markdown(response)
                except Exception as e:
                    response = f"Error: {str(e)}"
                    st.error(response)
        
        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

# ==================== WELCOME MESSAGE ====================

if len(st.session_state.messages) == 0:
    st.info("""
    👋 **Welcome to the HVAC Codes Assistant!**
    
    I can help you find information about:
    - **Specific sections** (e.g., "What is Section 901.1?")
    - **Equipment regulations** (e.g., "Furnace installation requirements")
    - **Code hierarchies** (e.g., "Subsections of 901")
    - **Topics** (e.g., "Clearance requirements")
    
    💬 **Try asking a question or click an example in the sidebar!**
    """)

# ==================== FOOTER ====================

st.markdown("---")
st.caption("🏭 HVAC Codes Knowledge Assistant | Powered by Neo4j GraphRAG + OpenAI")
