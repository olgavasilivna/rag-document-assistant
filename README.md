# RAG Document Assistant

A powerful document processing and question-answering system that combines PDF processing, web scraping, and RAG (Retrieval-Augmented Generation) to provide intelligent responses to your questions about documents.

## 🌐 Live Demo
Visit the live demo at: [https://rag-document-assistant.streamlit.app](https://rag-document-assistant.streamlit.app)

## ⚠️ Important Note
To run this application locally, you'll need to:
1. Generate your own Groq API key from [https://console.groq.com](https://console.groq.com)
2. Set up the API key in your environment variables or Streamlit secrets

## Features

- **Document Upload**: Support for PDF files and web links
- **RAG Technology**: Uses Retrieval-Augmented Generation for accurate answers
- **Source Transparency**: Answers include references to source documents
- **User-Friendly Interface**: Simple and intuitive Streamlit UI

## How It Works

1. **Upload Documents**: Add PDF files or web links
2. **Document Processing**: The system extracts text and creates embeddings
3. **Question Answering**: Ask questions about your documents
4. **RAG Process**:
   - **Retrieval**: Finds relevant document sections
   - **Augmentation**: Adds context to your question
   - **Generation**: Creates accurate answers based on your documents

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-document-assistant.git
   cd rag-document-assistant
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Project Structure

- `app.py` - Main Streamlit application
- `pdf_processor.py` - PDF processing module
- `web_processor.py` - Web content processing module
- `vector_store.py` - Vector store implementation
- `rag_chain.py` - RAG chain implementation
- `requirements.txt` - Python dependencies
