import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from modules.pdf_processor import PDFProcessor
from modules.vector_store import VectorStore
from modules.rag_engine import RAGEngine
import json
import time

# Load environment variables
load_dotenv()

# Initialize components
# pdf_processor = PDFProcessor()
pdf_processor = PDFProcessor(chunk_size=2000, chunk_overlap=250)
vector_store = VectorStore()
rag_engine = RAGEngine()

def initialize_session_state():
    """Initialize session state variables"""
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'vector_store_loaded' not in st.session_state:
        st.session_state.vector_store_loaded = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processing_error' not in st.session_state:
        st.session_state.processing_error = None
    if 'current_pdf_hash' not in st.session_state:
        st.session_state.current_pdf_hash = None
    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False

def get_file_hash(file):
    """Generate a simple hash for the file to track changes"""
    import hashlib
    file.seek(0)
    content = file.read()
    file.seek(0)  # Reset file pointer
    return hashlib.md5(content).hexdigest()

def clear_old_index():
    """Clear the existing vector store index"""
    index_path = "storage/faiss_index"
    metadata_path = "storage/metadata.json"
    
    try:
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        print("✓ Cleared old index files")
    except Exception as e:
        print(f"Error clearing old index: {e}")

def process_pdf(uploaded_file):
    """Process uploaded PDF file with proper cleanup"""
    try:
        with st.spinner("Processing PDF..."):
            # Clear any previous errors and state
            st.session_state.processing_error = None
            
            # Generate file hash to track changes
            file_hash = get_file_hash(uploaded_file)
            
            # Check if this is a different file than before
            if (st.session_state.current_pdf_hash and 
                st.session_state.current_pdf_hash == file_hash and
                st.session_state.vector_store_loaded):
                st.info("This PDF has already been processed. Using existing index.")
                return
            
            # Clear old index if it exists
            clear_old_index()
            
            # Extract text from PDF
            chunks = pdf_processor.process_pdf(uploaded_file)
            if not chunks:
                raise ValueError("No text could be extracted from the PDF")
            
            # Create new vector index
            vector_store.create_index(chunks)
            
            # Update session state
            st.session_state.processed = True
            st.session_state.vector_store_loaded = True
            st.session_state.current_pdf_hash = file_hash
            st.session_state.pdf_uploaded = True
            st.session_state.messages = []  # Clear chat history for new PDF
            
            st.success("PDF processed successfully! You can now ask questions.")
            
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        st.session_state.processing_error = error_msg
        st.error(error_msg)
        print(f"Processing error: {e}")

def load_existing_index():
    """Load existing index if available"""
    try:
        if vector_store.load_index():
            st.session_state.vector_store_loaded = True
            st.session_state.processed = True
            return True
    except Exception as e:
        print(f"Error loading existing index: {e}")
        st.session_state.vector_store_loaded = False
    return False

def main():
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Chatbot with Gemini Flash 1.5")
    st.write("Upload a PDF and chat with it using AI!")
    
    initialize_session_state()
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📄 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload any PDF document to start chatting"
        )
        
        if uploaded_file:
            st.session_state.pdf_uploaded = True
            file_hash = get_file_hash(uploaded_file)
            
            # Show processing button only for new files
            if (not st.session_state.current_pdf_hash or 
                st.session_state.current_pdf_hash != file_hash or
                not st.session_state.vector_store_loaded):
                
                if st.button("Process PDF", type="primary", key="process_btn"):
                    process_pdf(uploaded_file)
            else:
                st.success("✅ PDF already processed and ready!")
        
        # Display processing status
        if st.session_state.processed:
            st.success("✅ PDF processed successfully!")
            if st.button("🔄 Process New PDF", key="new_pdf_btn"):
                # Reset state for new PDF
                st.session_state.processed = False
                st.session_state.vector_store_loaded = False
                st.session_state.current_pdf_hash = None
                st.session_state.messages = []
                st.rerun()
        
        # Display errors if any
        if st.session_state.processing_error:
            st.error(st.session_state.processing_error)
            
        # Display chat history
        st.header("💬 Conversation History")
        for msg in st.session_state.messages[-10:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("sources") and not msg.get("used_history", False):
                    with st.expander("View Sources"):
                        for source in msg["sources"]:
                            st.write(f"Page {source['page']}: {source['text']}...")
    
    # Main chat area
    if not st.session_state.vector_store_loaded and not st.session_state.pdf_uploaded:
        # Try to load existing index on startup
        if load_existing_index():
            st.sidebar.success("Loaded existing PDF index!")
        else:
            st.info("👈 Please upload a PDF file to get started!")
    
    if st.session_state.vector_store_loaded:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("sources") and not message.get("used_history", False):
                    with st.expander("Document Sources"):
                        for source in message["sources"]:
                            st.write(f"**Page {source['page']}:** {source['text']}...")
                elif message.get("used_history", False):
                    st.caption("🤖 Answered using conversation history")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the PDF..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        # For conversation history questions, don't search the document
                        is_about_chat = any(keyword in prompt.lower() for keyword in [
                            'previous', 'before', 'earlier', 'first question', 
                            'last question', 'what did i ask', 'what was my',
                            'our conversation', 'chat history', 'did i ask'
                        ])

                        # Check if this is a data/numbers question
                        is_data_question = any(keyword in prompt.lower() for keyword in [
                            'percentage', 'percent', '%', 'chart', 'graph', 
                            'table', 'data', 'number', 'statistic', 'figure'
                        ])

                        if is_about_chat:
                            # Don't search document for chat history questions
                            retrieved_chunks = []
                        elif is_data_question:
                            # Use specialized search for data questions
                            retrieved_chunks = vector_store.search_structured_data(prompt, k=5)
                        else:
                            # Regular search for other questions
                            retrieved_chunks = vector_store.search(prompt, k=5)

                        # Generate response
                        response = rag_engine.generate_response(
                            prompt, 
                            retrieved_chunks, 
                            st.session_state.messages
                        )
                        response_text = response['answer']
                        sources = response['sources']
                        used_history = response.get('used_history', False)
                    
                    # Display response
                    st.write(response_text)
                    
                    # Display sources if available and not using history
                    if sources and not used_history:
                        with st.expander("Document Sources"):
                            for source in sources:
                                st.write(f"**Page {source['page']}:** {source['text']}...")
                    elif used_history:
                        st.caption("🤖 Answered using conversation history")
                
                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "sources": sources,
                        "used_history": used_history
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    print(f"Response generation error: {e}")
                    
                    # Add error message to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "sources": [],
                        "used_history": False
                    })
    elif st.session_state.pdf_uploaded and not st.session_state.vector_store_loaded:
        st.warning("⚠️ PDF uploaded but not processed. Click 'Process PDF' in the sidebar.")

if __name__ == "__main__":
    main()