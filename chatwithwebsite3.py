import streamlit as st
from langchain.schema import AIMessage, HumanMessage
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains.combine_documents import create_stuff_documents_chain


def get_vector_store_from_url(url):
    # Load the HTML text from the document and split it into chunks
    loader = WebBaseLoader(url)
    document = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    document_chunks = text_splitter.split_documents(document)

    # Use Hugging Face Embeddings with a model name
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Extract text content from document chunks and store it in Chroma
    texts = [chunk.page_content for chunk in document_chunks]  # Extract page content text from document chunks
    vector_store = Chroma.from_texts(texts, embeddings)

    return vector_store


def get_context_retriever_chain(vector_store):
    # Set up the retriever and prompt for the retriever_chain
    retriever = vector_store.as_retriever(k=2)  # Retrieve top 2 most relevant documents

    # Using HuggingFaceHub for the model (e.g., GPT-Neo or GPT-J from Hugging Face)
    llm = HuggingFaceHub(repo_id="EleutherAI/gpt-neo-2.7B")  # Example using GPT-Neo model

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Based on the conversation above, generate a search query to look up relevant information."),
        ]
    )

    retriever_chain = create_history_aware_retriever(
        llm,
        retriever,
        prompt,
    )

    return retriever_chain


def get_conversation_rag_chain(retriever_chain):
    # Summarize the contents of the context obtained from the webpage
    llm = HuggingFaceHub(repo_id="EleutherAI/gpt-neo-2.7B")  # Example using GPT-Neo model

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user's questions based on the context below:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff_document_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_document_chain)


def get_response(user_input):
    # Invoke the chains created to generate a response to a given user query
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversation_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input,
    })

    return response['answer']


# Streamlit app config
st.set_page_config(page_title="Chat with a Website", page_icon="💻")
st.title("Chat with a Website")

# Sidebar setup
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Type the URL here")

if not website_url:
    st.info("Please enter a website URL...")

else:
    # Session State: Check for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    # Check if vector store is already populated
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store_from_url(website_url)

    # User input
    user_query = st.chat_input("Type here...")
    if user_query:
        response = get_response(user_query)

        # Append user query and AI response to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
