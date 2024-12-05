import os
import streamlit as st
from dotenv import load_dotenv
from langchain.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import HumanMessage, AIMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Neo4jVector
from langchain.graphs import Neo4jGraph
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.output_parsers import PydanticOutputParser
from langchain_experimental.graph_transformers import LLMGraphTransformer
from pydantic import BaseModel, Field
from typing import List, Tuple

# Load environment variables
load_dotenv()

# Set up OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Neo4j setup
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j graph
graph = Neo4jGraph()

# Streamlit app
st.title("Elizabeth I Knowledge Graph QA")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_index' not in st.session_state:
    st.session_state.vector_index = None

# Sidebar for data loading and processing
with st.sidebar:
    st.header("Data Processing")
    if st.button("Load and Process Data"):
        with st.spinner("Loading and processing data..."):
            # Load and process documents
            raw_documents = WikipediaLoader(query="Elizabeth I").load()
            text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
            documents = text_splitter.split_documents(raw_documents[:3])

            # Initialize LLM and graph transformer
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
            llm_transformer = LLMGraphTransformer(llm=llm)
            graph_documents = llm_transformer.convert_to_graph_documents(documents)

            # Add documents to graph
            graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )

            # Create fulltext index
            graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

            # Initialize vector store
            st.session_state.vector_index = Neo4jVector.from_existing_graph(
                OpenAIEmbeddings(),
                search_type="hybrid",
                node_label="Document",
                text_node_properties=["text"],
                embedding_node_property="embedding"
            )

            st.session_state.data_loaded = True

        st.success("Data loaded and processed successfully!")


# Entity extraction
class Entities(BaseModel):
    names: List[str] = Field(description="All the person, organization, or business entities that appear in the text")


entities_parser = PydanticOutputParser(pydantic_object=Entities)

entity_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text."),
    ("human", "Extract entities from the following input. {format_instructions}\n\nInput: {question}"),
])

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
entity_chain = entity_prompt | llm | entities_parser


# Retriever functions
def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in input.split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({
        "question": question,
        "format_instructions": entities_parser.get_format_instructions()
    })
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result


def retriever(question: str):
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in st.session_state.vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}
    """
    return final_data


# QA Chain
def create_chain(retriever):
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
        ),
        RunnableLambda(lambda x: x["question"]),
    )

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            RunnableParallel(
                {
                    "context": _search_query | retriever,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | ChatOpenAI(temperature=0)
            | StrOutputParser()
    )

    return chain


# Main chat interface
st.header("Chat with Elizabeth I Knowledge Graph")

# Input for user question
user_question = st.text_input("Ask a question about Elizabeth I:")

if user_question and st.session_state.data_loaded:
    # Create the QA chain
    qa_chain = create_chain(retriever)

    # Generate response
    with st.spinner("Generating response..."):
        response = qa_chain.invoke({
            "question": user_question,
            "chat_history": st.session_state.chat_history
        })

    # Display response
    st.write("Answer:", response)

    # Update chat history
    st.session_state.chat_history.append((user_question, response))

# Display chat history
if st.session_state.chat_history:
    st.header("Chat History")
    for q, a in st.session_state.chat_history:
        st.write("Q:", q)
        st.write("A:", a)
        st.write("---")

elif not st.session_state.data_loaded:
    st.warning("Please load the data first using the button in the sidebar.")

# Run the Streamlit app
if __name__ == "__main__":
    st.set_page_config(page_title="Elizabeth I Knowledge Graph QA", page_icon="👑", layout="wide")