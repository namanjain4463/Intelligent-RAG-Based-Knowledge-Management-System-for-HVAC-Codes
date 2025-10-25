import streamlit as st
from utils import write_message
from agent import generate_response
from tools import cypher as cypher_tool

st.set_page_config("HVAC Codes Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about HVAC codes, equipment specifications, installation requirements, or regulatory standards."}]

for message in st.session_state.messages:
    write_message(message["role"], message["content"], save=False)

question = st.chat_input("Ask your HVAC codes knowledge base...")

def handle_submit(message):
    with st.spinner('Working...'):
        response = generate_response(message)
        
        # Extract just the output content from the response
        if isinstance(response, dict) and 'output' in response:
            response_content = response['output']
        else:
            # Fallback in case response format is different
            response_content = str(response)
        
        write_message("human", message)
        write_message("assistant", response_content)
        
        # Display the generated Cypher query if available
        if cypher_tool.last_generated_cypher:
            cypher_display = f"**Generated Cypher Query:**\n```cypher\n{cypher_tool.last_generated_cypher}\n```"
            write_message("assistant", cypher_display)

if question:
    handle_submit(question)