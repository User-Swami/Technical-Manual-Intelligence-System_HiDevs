import streamlit as st
import os

# IMPORTANT: This block forces pysqlite3-binary to be used,
# addressing the RuntimeError: Your system has an unsupported version of sqlite3.
# This must be done *before* any imports that might implicitly load sqlite3,
# such as chromadb.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    st.warning("pysqlite3-binary not found. Falling back to system sqlite3. Ensure your system sqlite3 is >= 3.35.0 if you encounter issues with ChromaDB.")


from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# --- Configuration ---
# Directly set your Groq API key here.
# IMPORTANT: Replace "YOUR_GROQ_API_KEY_HERE" with your actual Groq API key.
# For production environments, consider using environment variables for better security.
GROQ_API_KEY = "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"

# Check if the API key has been replaced. If not, stop the app and show an error.
if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE" or not GROQ_API_KEY:
    st.error("Please replace 'YOUR_GROQ_API_KEY_HERE' in app.py with your actual Groq API key.")
    st.stop()

# Define the Groq model to use for generating responses.
# You can experiment with other models like "mixtral-8x7b-32768" for different performance.
GROQ_MODEL = "llama3-8b-8192"

# Define the embedding model for converting text into numerical vectors.
# This model runs locally and is downloaded on first use.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    Iterates through each PDF, then through each page, accumulating all text.
    """
    text = ""
    for pdf in pdf_docs:  
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text: # Only append if text was successfully extracted from the page
                    text += page_text
        except Exception as e:
            st.error(f"Error reading PDF file {pdf.name}: {e}")
            return None # Indicate failure if any PDF causes an error
    return text

def get_text_chunks(text):
    """
    Splits a given text into smaller, manageable chunks for processing.
    `chunk_size`: The maximum number of characters in each text chunk.
    `chunk_overlap`: The number of characters that overlap between consecutive chunks
                     to maintain context across splits.
    """
    if not text or len(text.strip()) == 0:
        return [] # Return empty if text is empty or just whitespace

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Max characters in a chunk
        chunk_overlap=200 # Overlap between chunks to maintain context
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates a vector store (ChromaDB) from text chunks using HuggingFace embeddings.
    This store allows for efficient similarity searches to retrieve relevant document parts.
    `persist_directory`: Specifies a local directory to save the ChromaDB, so it can be reused.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_texts(text_chunks, embeddings, persist_directory="./chroma_db")
    return vector_store

def get_conversational_chain(vector_store):
    """
    Initializes and returns a conversational RAG (Retrieval Augmented Generation) chain.
    This chain combines a Language Model (LLM) with a retriever and memory for multi-turn conversations.
    """
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
    
    # Configure memory to store chat history, enabling multi-turn conversations.
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    # Define a custom system prompt to guide the LLM's behavior.
    # This prompt makes the LLM act as an expert technical support assistant,
    # ensuring detailed, accurate, and context-aware responses from the manuals.
    system_template = """
    You are an expert technical support assistant. Your primary goal is to provide accurate, detailed, and comprehensive answers based ONLY on the technical manuals provided.
    
    Instructions:
    - If the user asks a question, retrieve relevant information from the context.
    - Provide complete and step-by-step explanations when applicable.
    - If the answer is not found in the provided context, clearly state that you do not have the information.
    - Maintain a helpful and professional tone.
    - Do not invent information or use external knowledge.
    
    Context: {context}
    Chat History: {chat_history}
    """
    human_template = "{question}"

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])
    
    # Create a ConversationalRetrievalChain.
    # This chain will use the LLM, a retriever to fetch relevant document chunks,
    # and memory to maintain the conversation flow.
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt} # Apply the custom prompt
    )
    return conversation_chain

# --- Streamlit UI and Logic ---

def main():
    st.set_page_config(
        page_title="Technical Manual Intelligence System",
        page_icon="📚", # Sets a book icon for the page
        layout="wide" # Uses a wide layout for the Streamlit app
    )

    st.title("📚 Technical Manual Intelligence System")
    st.markdown(
        """
        Upload your technical manuals (PDFs) and ask questions to get precise,
        contextual answers. This system streamlines technical support by
        transforming complex manuals into an accessible knowledge base.
        """
    )

    # Initialize session state variables if they don't already exist.
    # These variables persist across user interactions in Streamlit.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None # Stores the conversational chain
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # Stores the chat history (user and AI messages)
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = False # Tracks if documents have been successfully processed

    # Sidebar for PDF uploads and processing controls.
    with st.sidebar:
        st.header("Upload Your Manuals")
        pdf_docs = st.file_uploader(
            "Upload your PDF documents here and click 'Process'",
            accept_multiple_files=True, # Allows uploading multiple PDFs
            type=["pdf"] # Only accepts PDF file types
        )
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents... This may take a moment."):
                    # Step 1: Extract text from uploaded PDFs.
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text or len(raw_text.strip()) == 0:
                        st.error("Could not extract any meaningful text from the uploaded PDF(s). "
                                 "Please ensure your PDFs contain selectable text, not just images.")
                        st.session_state.processed_docs = False
                        st.session_state.conversation = None
                        st.stop() # Stop execution here as no text was extracted

                    # Step 2: Split the raw text into smaller, manageable chunks.
                    text_chunks = get_text_chunks(raw_text)
                    
                    if not text_chunks:
                        st.error("No text chunks could be generated from the extracted text. "
                                 "The document might be too short or consist of unchunkable content.")
                        st.session_state.processed_docs = False
                        st.session_state.conversation = None
                        st.stop() # Stop execution here as no chunks were generated
                    
                    # Step 3: Create a vector store from the text chunks.
                    # This step generates embeddings and stores them for retrieval.
                    vector_store = get_vector_store(text_chunks)
                    
                    # Step 4: Initialize the conversational RAG chain with the vector store.
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.session_state.processed_docs = True # Mark documents as processed
                    st.success("Documents processed successfully! You can now ask questions.")
                    st.session_state.chat_history = [] # Clear chat history for new documents
            else:
                st.warning("Please upload at least one PDF document before processing.")

    # Main chat interface for user interaction.
    st.subheader("Chat with your Manuals")

    # Display initial guidance or allow chat input based on document processing status.
    if not st.session_state.processed_docs:
        st.info("Upload PDF documents and click 'Process Documents' in the sidebar to start chatting.")
    else:
        user_question = st.chat_input("Ask a question about your manuals:")
        if user_question:
            if st.session_state.conversation:
                with st.spinner("Getting an answer..."):
                    # Invoke the conversational chain with the user's question.
                    # The chain will retrieve relevant info, use chat history, and generate a response.
                    response = st.session_state.conversation({'question': user_question})
                    st.session_state.chat_history = response['chat_history']
                    
                    # Display the updated chat history.
                    # Messages are displayed alternately for user and assistant.
                    for i, message in enumerate(st.session_state.chat_history):
                        if i % 2 == 0: # User message (even index)
                            st.chat_message("user").write(message.content)
                        else: # AI message (odd index)
                            st.chat_message("assistant").write(message.content)
            else:
                # This error should ideally not be reached if processed_docs is True.
                st.error("The conversational chain is not initialized. Please process documents first.")

if __name__ == "__main__":
    main()
