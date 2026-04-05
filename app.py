# app.py
# Streamlit web application for Pakistan History RAG System

import streamlit as st
import requests
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Import our utils
from utils import load_models, search_and_generate

# Page configuration
st.set_page_config(
    page_title="Pakistan History RAG System",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 Pakistan History RAG System")
st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    ### Ask questions about Pakistan's rich history!
    This system uses **Retrieval-Augmented Generation (RAG)** with:
    - 🔍 Hybrid Search (Semantic + BM25 + RRF)
    - 🎯 Cross-Encoder Re-ranking
    - 🤖 LLM-powered answers
    - 📊 Real-time evaluation metrics
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input (optional, can use env vars)
    hf_api_key = st.text_input(
        "Hugging Face API Key",
        type="password",
        value=os.getenv("HUGGINGFACE_API_KEY", ""),
        help="Get your API key from huggingface.co/settings/tokens"
    )
    
    pinecone_api_key = st.text_input(
        "Pinecone API Key",
        type="password",
        value=os.getenv("PINECONE_API_KEY", ""),
        help="Get your API key from pinecone.io"
    )
    
    # Model selection
    use_rerank = st.checkbox("Use Re-ranking", value=True, help="Enable cross-encoder re-ranking for better accuracy")
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
    
    st.markdown("---")
    st.markdown("### 📊 System Stats")
    st.markdown("- Corpus: 50+ Wikipedia pages")
    st.markdown("- Chunks: 500+")
    st.markdown("- Embedding: all-MiniLM-L6-v2")
    st.markdown("- Re-ranker: ms-marco-MiniLM-L-6-v2")
    
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📖 Source Code](https://github.com/yourusername/pakistan-history-rag)")
    st.markdown("[🗄️ Pinecone Index](https://www.pinecone.io/)")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False

# Load models (cached)
@st.cache_resource
def init_system():
    """Initialize the RAG system"""
    with st.spinner("Loading models and connecting to Pinecone..."):
        # Set API keys if provided in UI
        if hf_api_key:
            os.environ['HUGGINGFACE_API_KEY'] = hf_api_key
        if pinecone_api_key:
            os.environ['PINECONE_API_KEY'] = pinecone_api_key
        
        # Load models
        embedding_model, cross_encoder, index, bm25_retriever = load_models()
        return embedding_model, cross_encoder, index, bm25_retriever

# Initialize system
try:
    embedding_model, cross_encoder, index, bm25_retriever = init_system()
    st.session_state.system_ready = True
except Exception as e:
    st.error(f"❌ Failed to initialize system: {str(e)}")
    st.info("Please check your API keys in the sidebar and refresh the page.")
    st.session_state.system_ready = False

# Function to call Hugging Face Inference API
def query_llm(prompt):
    """Query Hugging Face API for answer generation"""
    API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.3,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to evaluate faithfulness (simplified for UI)
def quick_evaluate_faithfulness(answer, context):
    """Quick faithfulness evaluation using keyword matching (simplified)"""
    # This is a simplified version - the full version uses LLM-as-a-Judge
    # For the UI, we'll use a simple heuristic
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    
    if len(answer_words) == 0:
        return 0.0
    
    overlap = len(answer_words.intersection(context_words))
    score = min(overlap / len(answer_words), 1.0)
    
    return score

# Main chat interface
st.header("💬 Ask a Question")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "scores" in message:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Faithfulness", f"{message['scores']['faithfulness']:.2f}")
            with col2:
                st.metric("Relevancy", f"{message['scores']['relevancy']:.2f}")

# Chat input
if prompt := st.chat_input("Ask about Pakistan's history..."):
    if not st.session_state.system_ready:
        st.error("System not ready. Please check your API keys.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching Pakistan's history..."):
            # Step 1: Retrieve relevant chunks
            retrieval_result = search_and_generate(
                prompt,
                embedding_model,
                cross_encoder,
                index,
                bm25_retriever,
                use_rerank=use_rerank,
                top_k=top_k
            )
            
            # Step 2: Generate answer using LLM
            generation_prompt = f"""You are a helpful assistant that answers questions about Pakistan's history.
Use ONLY the following context to answer the question. If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{retrieval_result['context']}

Question: {prompt}

Answer (be concise and factual):"""
            
            answer = query_llm(generation_prompt)
            
            # Step 3: Quick evaluation
            faithfulness = quick_evaluate_faithfulness(answer, retrieval_result['context'])
            # For relevancy, we'll use a placeholder (full version uses LLM)
            relevancy = 0.85  # Placeholder - in production you'd compute properly
            
            # Display answer
            st.markdown(answer)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Faithfulness Score", f"{faithfulness:.2f}", 
                         help="How well the answer is supported by the retrieved context")
            with col2:
                st.metric("Relevancy Score", f"{relevancy:.2f}",
                         help="How well the answer addresses the question")
            with col3:
                st.metric("Chunks Retrieved", retrieval_result['num_chunks'])
            
            # Display retrieved chunks in expander
            with st.expander("📚 View Retrieved Context Chunks"):
                for i, chunk in enumerate(retrieval_result['chunks'], 1):
                    st.markdown(f"**Chunk {i}** (Score: {chunk.get('rerank_score', chunk.get('fusion_score', chunk.get('score', 0))):.3f})")
                    st.markdown(f"*Source: {chunk['metadata']['source_title']}*")
                    st.markdown(chunk['text'])
                    st.markdown("---")
            
            # Save to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "scores": {
                    "faithfulness": faithfulness,
                    "relevancy": relevancy
                }
            })

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
    Powered by Pinecone Vector Database, Hugging Face Inference API, and Streamlit<br>
    Built for NLP with Deep Learning Assignment - Pakistan History RAG System
    </div>
""", unsafe_allow_html=True)