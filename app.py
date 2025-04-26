import streamlit as st

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="Document Chatbot",
    page_icon="📚",
    layout="wide"
)

import os
from pdf_processor import PDFProcessor
from web_processor import WebProcessor
from vector_store import VectorStore
from rag_chain import RAGChain
import tempfile
from collections import defaultdict
import base64
import re
from typing import List

# Disable PostHog telemetry
os.environ["POSTHOG_DISABLED"] = "true"

# Initialize components
@st.cache_resource
def init_components():
    pdf_processor = PDFProcessor()
    web_processor = WebProcessor()
    vector_store = VectorStore()
    rag_chain = RAGChain(vector_store)
    return pdf_processor, web_processor, vector_store, rag_chain

pdf_processor, web_processor, vector_store, rag_chain = init_components()

def is_url(text: str) -> bool:
    """Check if the text is a URL."""
    url_pattern = re.compile(r'https?://\S+')
    return bool(url_pattern.match(text))

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.findall(text)

def process_urls(urls: List[str]) -> None:
    """Process a list of URLs and add them to the vector store."""
    for url in urls:
        try:
            # Process URL
            chunks = web_processor.process_url(url)
            # Add to vector store
            vector_store.add_documents(chunks)
            st.success(f"Processed {url}")
        except Exception as e:
            st.error(f"Error processing {url}: {str(e)}")

def create_download_link(file_path: str, text: str) -> str:
    """Create a download link for a file."""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(file_path)}">{text}</a>'

# Title and description
st.title("📚 RAG Document Assistant")
st.markdown("""
## How it works

This application uses Retrieval-Augmented Generation (RAG) to answer questions about your documents:

1. **Upload Documents**: Add PDF files or web links in the sidebar
2. **Document Processing**: The system extracts text and creates embeddings
3. **Question Answering**: Ask questions about your documents
4. **RAG Process**:
   - **Retrieval**: Finds relevant document sections
   - **Augmentation**: Adds context to your question
   - **Generation**: Creates accurate answers based on your documents

The answers include sources from your documents for transparency.
""")

# Sidebar for file upload and URL input
with st.sidebar:
    st.header("Upload Documents")
    
    # PDF Upload
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    
    # URL Input
    urls = st.text_area(
        "Enter URLs (one per line)",
        help="Enter web links to process their content"
    )
    
    if uploaded_files or urls:
        # Reset vector store when new content is added
        vector_store.reset()
        
        with st.spinner("Processing content..."):
            # Process PDFs
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Process PDF
                        chunks = pdf_processor.process_pdf(tmp_path, original_filename=uploaded_file.name)
                        # Add to vector store
                        vector_store.add_documents(chunks)
                        st.success(f"Processed {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_path)
            
            # Process URLs from text area
            if urls:
                url_list = [url.strip() for url in urls.split('\n') if url.strip()]
                process_urls(url_list)

# Main chat interface
st.header("Chat with your documents")

# Initialize chat history and input state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")

# Chat input with dynamic key
prompt = st.text_input(
    "Ask a question about your documents",
    key=f"chat_input_{st.session_state.input_key}"
)

# Handle Enter key press
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    st.markdown(f"**You:** {prompt}")
    
    # Check for URLs in the prompt
    urls_in_prompt = extract_urls(prompt)
    if urls_in_prompt:
        with st.spinner("Processing URLs..."):
            process_urls(urls_in_prompt)
    
    # Generate response
    with st.spinner("Thinking..."):
        try:
            # Search for relevant documents
            relevant_docs = vector_store.search(prompt)
            
            if not relevant_docs:
                response = "I couldn't find any relevant information in the uploaded documents to answer your question."
            else:
                # Generate answer
                response = rag_chain.generate_answer(prompt, relevant_docs)
                
                # Process and display sources
                source_counts = defaultdict(list)
                for doc in relevant_docs:
                    source = doc['metadata'].get('original_filename', doc['metadata']['source'])
                    source_counts[source].append({
                        'page': doc['metadata'].get('page', 'N/A'),
                        'text': doc['text']
                    })
                
                if source_counts:
                    st.markdown("**Sources:**")
                    for source, refs in source_counts.items():
                        st.markdown(f"- {source}")
                        for ref in refs:
                            st.markdown(f"  - Page {ref['page']}: {ref['text']}")
        except Exception as e:
            response = f"An error occurred while processing your question: {str(e)}"
    
    # Display response
    st.markdown(f"**Assistant:** {response}")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Increment input key to clear the input field
    st.session_state.input_key += 1
    st.rerun() 