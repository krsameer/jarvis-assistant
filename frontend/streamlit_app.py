"""
Streamlit Chat UI for Jarvis AI Assistant
Simple, clean interface for interacting with the RAG system.
"""

import streamlit as st
import requests
import json
import io
from typing import List, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

# Backend API URL (update if deployed elsewhere)
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Jarvis AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_backend_health() -> Dict:
    """Check if backend is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "unhealthy", "services": {}}
    except:
        return {"status": "offline", "services": {}}


def ingest_document(text: str, metadata: Dict = None) -> Dict:
    """Send document to backend for ingestion."""
    try:
        payload = {
            "text": text,
            "metadata": metadata or {}
        }
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def send_chat_message(query: str, conversation_history: List[Dict] = None) -> Dict:
    """Send chat message to backend."""
    try:
        payload = {
            "query": query,
            "conversation_history": conversation_history or []
        }
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "status": "error",
            "response": f"Error: {str(e)}",
            "sources": []
        }


def get_system_stats() -> Dict:
    """Get system statistics from backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return {}


def extract_text_from_file(uploaded_file) -> str:
    """Extract plain text from an uploaded file (PDF or TXT)."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="replace")

    elif filename.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            if not pages:
                raise ValueError("No extractable text found in PDF. The file may be scanned/image-based.")
            return "\n\n".join(pages)
        except ImportError:
            raise RuntimeError("pypdf is not installed. Run: pip install pypdf")

    else:
        raise ValueError(f"Unsupported file type: '{uploaded_file.name}'. Supported: PDF, TXT.")


# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_sidebar():
    """Render the sidebar with document ingestion and system info."""
    with st.sidebar:
        st.title("🤖 Jarvis")
        st.markdown("*Your Enterprise AI Assistant*")
        st.divider()
        
        # System health check
        st.subheader("🏥 System Health")
        health = check_backend_health()
        
        if health["status"] == "healthy":
            st.success("✅ All systems operational")
        elif health["status"] == "degraded":
            st.warning("⚠️ Some services unavailable")
        else:
            st.error("❌ Backend offline")
            st.info("Make sure the FastAPI backend is running:\n\n```bash\ncd backend\nuvicorn main:app --reload\n```")
        
        # Show individual service status
        if health.get("services"):
            with st.expander("Service Status"):
                for service, status in health["services"].items():
                    icon = "✅" if status else "❌"
                    st.text(f"{icon} {service.replace('_', ' ').title()}")
        
        st.divider()
        
        # Document ingestion
        st.subheader("📄 Add Knowledge")
        
        upload_tab, paste_tab = st.tabs(["📁 Upload File", "📝 Paste Text"])

        with upload_tab:
            uploaded_file = st.file_uploader(
                "Upload a PDF or TXT file",
                type=["pdf", "txt"],
                help="Drop a PDF or plain-text file here"
            )

            col1, col2 = st.columns(2)
            with col1:
                up_source = st.text_input("Source label", placeholder="e.g., policy.pdf", key="up_source")
            with col2:
                up_type = st.text_input("Type", placeholder="e.g., policy", key="up_type")

            if uploaded_file is not None:
                if st.button("📤 Ingest File", use_container_width=True, key="btn_file"):
                    with st.spinner("Extracting text and ingesting..."):
                        try:
                            doc_text = extract_text_from_file(uploaded_file)
                            metadata = {"source": up_source or uploaded_file.name, "type": up_type or "document"}
                            result = ingest_document(doc_text, metadata)
                            if result.get("status") == "success":
                                st.success(f"✅ Ingested {result.get('chunks_processed', 0)} chunks from '{uploaded_file.name}'!")
                            else:
                                st.error(f"❌ Ingestion error: {result.get('message', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"❌ {e}")

        with paste_tab:
            with st.form("ingest_form"):
                doc_text = st.text_area(
                    "Document Text",
                    height=150,
                    placeholder="Paste your document content here..."
                )

                col1, col2 = st.columns(2)
                with col1:
                    doc_source = st.text_input("Source", placeholder="e.g., policy.pdf")
                with col2:
                    doc_type = st.text_input("Type", placeholder="e.g., policy")

                submit = st.form_submit_button("📤 Ingest Text", use_container_width=True)

                if submit:
                    if not doc_text or len(doc_text.strip()) < 10:
                        st.error("Please provide valid document text (minimum 10 characters)")
                    else:
                        with st.spinner("Processing document..."):
                            metadata = {}
                            if doc_source:
                                metadata["source"] = doc_source
                            if doc_type:
                                metadata["type"] = doc_type

                            result = ingest_document(doc_text, metadata)

                            if result.get("status") == "success":
                                st.success(f"✅ Ingested {result.get('chunks_processed', 0)} chunks!")
                            else:
                                st.error(f"❌ Error: {result.get('message', 'Unknown error')}")

        
        st.divider()
        
        # System stats
        with st.expander("📊 System Stats"):
            stats = get_system_stats()
            if stats:
                st.json(stats)
            else:
                st.info("Stats unavailable")


def render_chat():
    """Render the main chat interface."""
    st.title("💬 Chat with Jarvis")
    st.markdown("Ask questions about your documents and get AI-powered answers with source citations.")
    
    # Initialize session state for conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** (Score: {source['score']:.3f})")
                        st.text(source["text"])
                        if source.get("metadata"):
                            st.caption(f"Metadata: {source['metadata']}")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Send to backend
                result = send_chat_message(
                    query=prompt,
                    conversation_history=st.session_state.conversation_history
                )
                
                if result.get("status") == "success":
                    response = result["response"]
                    sources = result.get("sources", [])
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display sources
                    if sources:
                        with st.expander(f"📚 {len(sources)} Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}** (Similarity: {source['score']:.3f})")
                                st.text(source["text"])
                                if source.get("metadata"):
                                    st.caption(f"📎 {source['metadata']}")
                                st.divider()
                    
                    # Add to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                    
                    # Update conversation history (for context)
                    st.session_state.conversation_history.append({
                        "user": prompt,
                        "assistant": response
                    })
                    
                    # Keep only last 5 turns for context
                    if len(st.session_state.conversation_history) > 5:
                        st.session_state.conversation_history = st.session_state.conversation_history[-5:]
                
                else:
                    error_msg = result.get("response", "An error occurred")
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()


def render_welcome():
    """Render welcome message when chat is empty."""
    if not st.session_state.get("messages"):
        st.markdown("""
        ### 👋 Welcome to Jarvis!
        
        I'm your AI assistant powered by Retrieval-Augmented Generation (RAG).
        
        **How to use:**
        1. **Add Knowledge**: Use the sidebar to ingest documents
        2. **Ask Questions**: Type your question in the chat below
        3. **Get Answers**: I'll search the knowledge base and provide answers with sources
        
        **Example questions:**
        - "What is our vacation policy?"
        - "Explain the employee benefits"
        - "How do I submit an expense report?"
        
        **Features:**
        - 🔍 Semantic search across your documents
        - 📚 Source citations for every answer
        - 💡 Context-aware responses
        - 🔒 Self-hosted and private
        
        Start by adding some documents in the sidebar! 👈
        """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    # Render sidebar
    render_sidebar()
    
    # Render main chat area
    render_chat()
    
    # Render welcome message if no messages
    render_welcome()
    
    # Footer
    st.divider()
    st.caption("Powered by Ollama, Pinecone, FastAPI, and Streamlit | Built with ❤️ for Enterprise")


if __name__ == "__main__":
    main()
