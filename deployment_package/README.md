---
title: Pakistan History RAG System
emoji: 📚
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Pakistan History RAG System

A Retrieval-Augmented Generation (RAG) system for answering questions about Pakistan's history.

## Features
- Hybrid Search (Semantic + BM25 + RRF)
- Cross-Encoder Re-ranking
- Llama-3.1-8B via Groq API
- 100+ Wikipedia pages as corpus

## Setup Secrets
Add these to your Space secrets:
- `PINECONE_API_KEY`
- `GROQ_API_KEY`