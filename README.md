# 🛒 ShopEasy FAQ Bot — Agentic AI Capstone Project

**Student:** Kevin Adesara  
**Roll Number:** 23051838  
**Batch / Program:** 2027_Agentic AI (IE)  
**Course:** Agentic AI Hands-On Course | Dr. Kanthi Kiran Sirra  

---

## 📌 Project Overview

ShopEasy FAQ Bot is an intelligent 24/7 customer support chatbot built for an e-commerce platform. It answers common customer queries about returns, refunds, shipping, payments, and more — using a verified knowledge base, never hallucinating information.

---

## 🎯 Problem Statement

ShopEasy receives 500+ customer support queries daily. Most are repetitive questions about return policy, refunds, shipping, and payments. Human agents are overwhelmed. This bot handles those queries instantly, 24/7.

---

## ✅ Features

- **LangGraph StateGraph** with 8 nodes (memory, router, retrieve, skip, tool, answer, eval, save)
- **ChromaDB RAG** with 10 knowledge base documents covering all FAQ topics
- **MemorySaver + thread_id** for multi-turn conversation memory
- **Self-reflection eval node** — scores faithfulness 0.0–1.0, auto-retries if below 0.7
- **Datetime tool** for time-based queries
- **Streamlit UI** with chat bubbles, sidebar, and session management
- **Red-team tested** — handles prompt injection and out-of-scope questions correctly

---

## 🗂️ Knowledge Base Topics

| Doc ID | Topic |
|--------|-------|
| doc_001 | Return Policy |
| doc_002 | Refund Process |
| doc_003 | Shipping and Delivery |
| doc_004 | Order Cancellation |
| doc_005 | Payment Methods |
| doc_006 | Account and Login |
| doc_007 | Product Warranty |
| doc_008 | Offers, Coupons and Cashback |
| doc_009 | Customer Support |
| doc_010 | Seller and Product Authenticity |

---

## 🏗️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq — llama-3.3-70b-versatile |
| Agent Framework | LangGraph (StateGraph + MemorySaver) |
| Vector Database | ChromaDB (in-memory) |
| Embedder | SentenceTransformer — all-MiniLM-L6-v2 |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision) |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ShopEasy-FAQ-Bot.git
cd ShopEasy-FAQ-Bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API Key
Open `agent.py` and replace:
```python
GROQ_API_KEY = "your_groq_api_key_here"
```
with your actual key from [console.groq.com](https://console.groq.com)

### 4. Run the chatbot UI
```bash
streamlit run capstone_streamlit.py
```

### 5. Run the notebook
```bash
jupyter notebook day13_capstone.ipynb
```

---

## 📁 Project Structure

```
ShopEasy-FAQ-Bot/
│
├── agent.py                  # Core AI agent — all nodes, graph, knowledge base
├── capstone_streamlit.py     # Streamlit chat UI
├── day13_capstone.ipynb      # Jupyter notebook with full walkthrough
├── requirements.txt          # All dependencies
└── README.md                 # This file
```

---

## 📊 RAGAS Evaluation Scores

| Metric | Score |
|--------|-------|
| Faithfulness | 0.85 |
| Answer Relevancy | 0.88 |
| Context Precision | 0.82 |
| **Average** | **0.85** |

---

## 🏛️ Agent Architecture

```
User Question
      ↓
[memory_node]    → adds to history, extracts customer name
      ↓
[router_node]    → classifies: retrieve / tool / memory_only
      ↓
[retrieval_node / tool_node / skip_node]
      ↓
[answer_node]    → grounded LLM answer from context
      ↓
[eval_node]      → faithfulness score → retry if < 0.7
      ↓
[save_node]      → saves answer to history → END
```

---

## ⚠️ Important Note

Never commit your real Groq API key to GitHub. Always keep it only on your local machine.

---

*Agentic AI Capstone Project 2026 | Kevin Adesara*
