from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

class RAGChain:
    def __init__(self, vector_store):
        self.llm = ChatGroq(
            model="llama3-8b-8192",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.vector_store = vector_store

    def _get_context(self, question: str) -> str:
        """Get relevant context from the vector store."""
        docs = self.vector_store.search(question)
        return self.format_docs(docs)

    def format_docs(self, docs):
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(doc["text"] for doc in docs)

    def generate_answer(self, question, relevant_docs=None):
        try:
            if relevant_docs is None:
                relevant_docs = self.vector_store.search(question)
            context = self.format_docs(relevant_docs)
            
            # Create a simple prompt
            prompt = f"""You are a helpful AI assistant. Use the following context to answer the question at the end.
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.

            Context: {context}

            Question: {question}

            Answer: """
            
            # Generate response
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
 