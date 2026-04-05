# app.py - Works Locally AND on Hugging Face Spaces
# Pakistan History RAG System

import streamlit as st
import requests
import os
import json
import pickle
import re
import numpy as np
from collections import defaultdict

# Load .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Pakistan History RAG System",
    page_icon="📚",
    layout="wide"
)

# ============================================
# CONFIG
# ============================================

with open("pinecone_config.json", "r") as f:
    PINECONE_CONFIG = json.load(f)

EMBEDDING_MODEL_NAME = PINECONE_CONFIG["embedding_model"]
PINECONE_INDEX_NAME  = PINECONE_CONFIG["pinecone_index"]
TOTAL_VECTORS        = PINECONE_CONFIG["total_vectors"]
FIXED_CHUNKS         = PINECONE_CONFIG["fixed_chunks"]
RECURSIVE_CHUNKS     = PINECONE_CONFIG["recursive_chunks"]

# ============================================
# SECRETS
# ============================================

def get_secret(key, default=""):
    try:
        value = st.secrets.get(key, default)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(key, default)

PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
GROQ_API_KEY     = get_secret("GROQ_API_KEY")

if not PINECONE_API_KEY:
    st.error("❌ PINECONE_API_KEY not found!")
    st.info("""
    **Local:** add to `.env` file\n
    **HF Spaces:** Settings → Repository Secrets
    """)

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found!")

# ============================================
# BM25 TOKENIZER (module-level so it's always accessible)
# ============================================

def bm25_tokenize(text):
    """Simple whitespace+punctuation tokenizer. Defined at module level."""
    return re.findall(r"\w+", text.lower())


# ============================================
# TOP-LEVEL BM25 RETRIEVER CLASS
# Must be at module level — pickle requires this to deserialize.
# The search() method pre-tokenizes the query itself so BM25Okapi
# never needs to call its internal _tokenize method.
# ============================================

class BM25Retriever:
    def __init__(self, bm25_obj, chunks):
        self.bm25   = bm25_obj
        self.chunks = chunks

    def search(self, query, top_k=10):
        # Pre-tokenize query ourselves — never let BM25Okapi touch it
        tokenized_query = bm25_tokenize(query)
        scores          = self.bm25.get_scores(tokenized_query)
        top_indices     = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "id":    self.chunks[i].get("id", f"chunk_{i}"),
                "text":  self.chunks[i]["text"],
                "score": float(scores[i]),
            }
            for i in top_indices if scores[i] > 0
        ]

# ============================================
# UI
# ============================================

st.title("📚 Pakistan History RAG System")
st.markdown("""
### Ask questions about Pakistan's rich history!
This system uses **Retrieval-Augmented Generation (RAG)** with:
- 🔍 Hybrid Search (Semantic + BM25 + RRF)
- 🎯 Cross-Encoder Re-ranking
- 🤖 LLM-powered answers (Llama-3.1-8B via Groq)
- ⚖️ LLM-as-a-Judge evaluation (Faithfulness & Relevancy)
""")

with st.sidebar:
    st.header("⚙️ Configuration")
    use_rerank = st.checkbox("Use Re-ranking", value=True,
                             help="Enable cross-encoder re-ranking for better accuracy")
    top_k      = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
    run_eval   = st.checkbox("Run LLM-as-a-Judge evaluation", value=True,
                             help="Compute Faithfulness & Relevancy scores (adds ~5-10s)")

    st.markdown("---")
    st.markdown("### 📊 System Stats")
    st.markdown(f"- **Total vectors:** {TOTAL_VECTORS:,}")
    st.markdown(f"- **Fixed chunks:** {FIXED_CHUNKS:,}")
    st.markdown(f"- **Recursive chunks:** {RECURSIVE_CHUNKS:,}")
    st.markdown(f"- **Embedding:** `{EMBEDDING_MODEL_NAME.split('/')[-1]}`")
    st.markdown(f"- **Index:** `{PINECONE_INDEX_NAME}`")
    st.markdown("- **Re-ranker:** ms-marco-MiniLM-L-6-v2")
    st.markdown("- **LLM:** Llama-3.1-8B via Groq")

    st.markdown("---")
    st.markdown("### 🔑 API Status")
    st.success("✅ Pinecone: Connected") if PINECONE_API_KEY else st.error("❌ Pinecone: No API key")
    st.success("✅ Groq: Connected")     if GROQ_API_KEY     else st.error("❌ Groq: No API key")

    st.markdown("---")
    st.markdown("### 📚 Topics Covered")
    st.markdown("""
    - Indus Valley Civilization
    - Mughal Empire
    - British Raj
    - Pakistan Movement
    - Independence 1947
    - Modern Pakistan
    """)

# ============================================
# SESSION STATE
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

# ============================================
# SYSTEM LOADING
# ============================================

@st.cache_resource
def load_system():
    if not PINECONE_API_KEY or not GROQ_API_KEY:
        return None, None, None, None

    with st.spinner("Loading RAG system... This may take 1-2 minutes on first run."):
        try:
            from sentence_transformers import SentenceTransformer, CrossEncoder
            from pinecone import Pinecone
            from rank_bm25 import BM25Okapi

            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            cross_encoder   = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            pc    = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)

            # ---- BM25 ----
            # We always rebuild BM25Okapi by passing pre-tokenized lists.
            # This means BM25Okapi stores token lists internally and never
            # needs to call its own _tokenize method — fixing all version issues.
            bm25_retriever = None
            chunks         = None

            try:
                if os.path.exists("bm25_index.pkl"):
                    with open("bm25_index.pkl", "rb") as f:
                        bm25_data = pickle.load(f)
                    chunks = bm25_data["chunks"]
                    source = "bm25_index.pkl"

                elif os.path.exists("chunks_recursive.json"):
                    with open("chunks_recursive.json", "r") as f:
                        chunks = json.load(f)
                    source = "chunks_recursive.json"

                if chunks:
                    # Pre-tokenize every chunk and pass token lists directly —
                    # BM25Okapi receives lists, not strings, so _tokenize is never called
                    tokenized_corpus = [bm25_tokenize(c["text"]) for c in chunks]
                    fresh_bm25       = BM25Okapi(tokenized_corpus)
                    bm25_retriever   = BM25Retriever(fresh_bm25, chunks)
                    st.info(f"✅ BM25 ready: {len(chunks):,} chunks from {source}")
                else:
                    st.warning("⚠️ No BM25 source found. BM25 search disabled.")

            except Exception as e:
                st.warning(f"⚠️ BM25 issue: {e}")

            return embedding_model, cross_encoder, index, bm25_retriever

        except Exception as e:
            st.error(f"Failed to load system: {e}")
            return None, None, None, None

# ============================================
# RETRIEVAL
# ============================================

from utils import reciprocal_rank_fusion, rerank_results


def get_text(r):
    return r.get("text", r.get("metadata", {}).get("text", ""))


def hybrid_search(query, embedding_model, index, bm25_retriever, top_k=20):
    if not embedding_model or not index:
        return []

    query_embedding      = embedding_model.encode(query)
    semantic_results_raw = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k * 2,
        include_metadata=True,
    )
    semantic_results = [
        {
            "id":       m["id"],
            "text":     m["metadata"].get("text", ""),
            "score":    m["score"],
            "metadata": m["metadata"],
        }
        for m in semantic_results_raw["matches"]
    ]

    if bm25_retriever:
        bm25_results = bm25_retriever.search(query, top_k=top_k * 2)
        return reciprocal_rank_fusion(semantic_results, bm25_results)[:top_k]

    return semantic_results[:top_k]

# ============================================
# GROQ
# ============================================

def call_groq(prompt, max_tokens=600, temperature=0.1):
    if not GROQ_API_KEY:
        return "Error: Groq API key not configured."
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model":       "llama-3.1-8b-instant",
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return f"Error {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return f"Error: {e}"

# ============================================
# LLM-AS-A-JUDGE
# ============================================

def evaluate_faithfulness(answer, context):
    extract_prompt = f"""Extract all factual claims from the answer below as a JSON array of strings.
Return ONLY a JSON array, no explanation, no markdown fences.

Answer: {answer}

JSON array:"""

    raw = call_groq(extract_prompt, max_tokens=400, temperature=0.0)
    try:
        claims = json.loads(re.sub(r"```json|```", "", raw).strip())
        if not isinstance(claims, list):
            claims = [str(claims)]
    except Exception:
        claims = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 20]

    if not claims:
        return 1.0, [], []

    verdicts = []
    for claim in claims:
        vp = f"""Does the context below support this claim? Reply ONLY with yes or no.

Context: {context[:2000]}

Claim: {claim}

Answer:"""
        v = call_groq(vp, max_tokens=5, temperature=0.0).strip().lower()
        verdicts.append("yes" if v.startswith("yes") else "no")

    score = round(verdicts.count("yes") / len(verdicts), 3)
    return score, claims, verdicts


def evaluate_relevancy(query, answer, embedding_model):
    gen_prompt = f"""Generate exactly 3 questions that are fully answered by the text below.
Return ONLY a JSON array of 3 question strings, no explanation, no markdown fences.

Text: {answer}

JSON array:"""

    raw = call_groq(gen_prompt, max_tokens=250, temperature=0.3)
    try:
        questions = json.loads(re.sub(r"```json|```", "", raw).strip())
        if not isinstance(questions, list):
            questions = [str(questions)]
        questions = [q for q in questions if q.strip()][:3]
    except Exception:
        questions = re.findall(r'\d+[\.\)]\s*(.+)', raw)[:3]
        if not questions:
            questions = [s.strip() for s in raw.split('\n') if '?' in s][:3]

    if not questions or embedding_model is None:
        return 0.0, questions, []

    from sklearn.metrics.pairwise import cosine_similarity
    sims     = cosine_similarity(embedding_model.encode([query]),
                                 embedding_model.encode(questions))[0]
    sim_list = [round(float(s), 3) for s in sims]
    return round(float(np.mean(sims)), 3), questions, sim_list

# ============================================
# BOOT
# ============================================

if PINECONE_API_KEY and GROQ_API_KEY:
    embedding_model, cross_encoder, index, bm25_retriever = load_system()
    if embedding_model is not None:
        st.session_state.system_ready = True
        st.success("✅ System ready! Ask a question about Pakistan's history.")
    else:
        st.error("❌ System failed to load. Check the error messages above.")
else:
    st.warning("⚠️ Waiting for API keys...")
    embedding_model = cross_encoder = index = bm25_retriever = None

# ============================================
# CHAT INTERFACE
# ============================================

st.header("💬 Ask a Question")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Pakistan's history..."):
    if not st.session_state.system_ready:
        st.error("System not ready. Please check API keys and try again.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching Pakistan's history..."):
            try:
                results = hybrid_search(prompt, embedding_model, index, bm25_retriever, top_k=20)

                if use_rerank and cross_encoder:
                    results = rerank_results(prompt, results, cross_encoder)[:top_k]
                else:
                    results = results[:top_k]

                if not results:
                    answer = "I couldn't find relevant information. Please try rephrasing your question."
                    st.warning("No relevant context found.")
                else:
                    context = "\n\n---\n\n".join([get_text(r) for r in results])
                    gen_prompt = f"""You are a helpful assistant answering questions about Pakistan's history.
Use ONLY the following context. If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context[:3000]}

Question: {prompt}

Answer (concise and factual):"""
                    answer = call_groq(gen_prompt, max_tokens=500, temperature=0.3)

                st.markdown(answer)

                # Retrieved context
                if results:
                    with st.expander("📚 View Retrieved Context"):
                        for i, r in enumerate(results, 1):
                            text   = get_text(r)
                            score  = r.get("rerank_score", r.get("fusion_score", r.get("score", 0)))
                            source = r.get("metadata", {}).get("source_title", "Unknown")
                            st.markdown(f"**Chunk {i}** (Score: `{score:.3f}`) — *{source}*")
                            st.markdown(text[:500] + "..." if len(text) > 500 else text)
                            st.markdown("---")

                # LLM-as-a-Judge
                if run_eval and results and answer:
                    with st.spinner("⚖️ Running LLM-as-a-Judge evaluation..."):
                        context_for_eval = "\n\n".join([get_text(r) for r in results])
                        faith_score, claims, verdicts   = evaluate_faithfulness(answer, context_for_eval)
                        rel_score, gen_questions, sims  = evaluate_relevancy(prompt, answer, embedding_model)

                    with st.expander("⚖️ LLM-as-a-Judge Evaluation", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            color = "green" if faith_score >= 0.7 else "orange" if faith_score >= 0.4 else "red"
                            st.markdown("### 🎯 Faithfulness")
                            st.markdown(f"<h2 style='color:{color}'>{faith_score:.0%}</h2>",
                                        unsafe_allow_html=True)
                            st.caption("% of answer claims supported by retrieved context")
                        with col2:
                            color = "green" if rel_score >= 0.7 else "orange" if rel_score >= 0.4 else "red"
                            st.markdown("### 📐 Relevancy")
                            st.markdown(f"<h2 style='color:{color}'>{rel_score:.0%}</h2>",
                                        unsafe_allow_html=True)
                            st.caption("Avg cosine similarity of 3 generated questions to original query")

                        st.markdown("---")
                        st.markdown("**🔍 Faithfulness — Claim Verification**")
                        for claim, verdict in zip(claims, verdicts):
                            st.markdown(f"{'✅' if verdict == 'yes' else '❌'} {claim}")

                        st.markdown("**📝 Relevancy — Generated Questions**")
                        for q, sim in zip(gen_questions, sims):
                            st.markdown(f"- {q} *(similarity: `{sim:.3f}`)*")

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Sorry, an error occurred: {e}"}
                )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
Powered by Pinecone | Groq Llama-3.1-8B | Sentence Transformers<br>
Pakistan History RAG System - NLP with Deep Learning Assignment
</div>
""", unsafe_allow_html=True)
