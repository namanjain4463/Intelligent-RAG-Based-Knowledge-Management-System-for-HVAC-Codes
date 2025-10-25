from llm import llm, embeddings
from graph import graph
from langchain_neo4j import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Neo4jVector is now configured for your HVAC index
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,  # (1) The embedding model
    graph=graph,  # (2) The graph object
    index_name="hvacChunkVector",  # (3) Your vector index name for HVAC
    node_label="HVACChunk",  # (4) The node label with embeddings for HVAC
    text_node_property="text",  # (5) The property with the raw text
    embedding_node_property="textEmbedding",  # (6) The property with the vector embedding
    retrieval_query="""
RETURN
node.text AS text,
score,
{
    source: [ (d)<-[:PART_OF]-(node) | d.name ][0],
    entities: [ (node)-[:MENTIONS]->(e) | labels(e)[0] + ': ' + e.name ]
} AS metadata
"""
)

retriever = neo4jvector.as_retriever()

# 2. Instructions are updated for HVAC use case
instructions = (
    "You are an assistant for an HVAC codes and regulatory knowledge base."
    "Use the given context, which contains excerpts from HVAC code documents, equipment specifications, and regulatory standards, to answer the question."
    "Focus on providing accurate information about HVAC equipment, installation requirements, code compliance, and regulatory standards."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", instructions),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)

# 3. The retrieval chain variable is renamed for clarity
knowledge_retriever = create_retrieval_chain(
    retriever,
    question_answer_chain
)

# 4. The main function is renamed to be more descriptive
def search_knowledge_base(input):
    """
    Invokes the retrieval chain to answer a question based on the HVAC knowledge base.
    """
    return knowledge_retriever.invoke({"input": input})
