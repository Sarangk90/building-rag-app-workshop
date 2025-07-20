# Building RAG Applications Workshop

This repository contains materials for the Building RAG (Retrieval-Augmented Generation) Applications Workshop. The workshop covers both naive RAG implementations and advanced RAG techniques to help you understand and build effective RAG systems.

This repository is used for the O'Reilly training course: [Building Reliable RAG Applications: From PoC to Production](https://learning.oreilly.com/live-events/building-reliable-rag-applications-from-poc-to-production/0642572012347/).

## Prerequisites

- Python 3.11+ recommended
- API keys (OpenAI, Cohere)
- Jupyter notebook environment
- Qdrant database (Cloud or Docker)

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Sarangk90/building-rag-app-workshop.git
cd building-rag-app-workshop

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Complete Workshop Setup

📖 **Follow the complete setup guide: [`SETUP.md`](SETUP.md)**

The setup guide covers:
- Qdrant database setup (Cloud or Docker options)
- Environment variable configuration  
- Data ingestion process
- Troubleshooting common issues

**⚠️ You must complete the setup before running any notebooks!**

## Workshop Notebooks

After completing the setup, run the notebooks in this order:

### 1. Naive RAG
- **`naive-rag/01-naive-rag.ipynb`** - Basic RAG implementation
- **`naive-rag/02-naive-rag-challenges.ipynb`** - RAG limitations and evaluation

### 2. Advanced RAG  
- **`advanced-rag/01-advanced-rag-rerank.ipynb`** - Advanced RAG with reranking

### 3. SciFact Dataset (Optional)
- **`advanced-rag/scifact/01-data-indexing.ipynb`** - Data indexing techniques
- **`advanced-rag/scifact/02-advanced-rag.ipynb`** - Advanced techniques

**Note**: Each notebook automatically detects your setup (Cloud vs Docker) and connects appropriately.

## Workshop Content

### Naive RAG
- Basic RAG implementation using OpenAI embeddings
- Vector storage with Qdrant
- Simple retrieval and generation pipeline

### Advanced RAG
- Hybrid search combining dense and sparse embeddings
- Reranking with cross-encoders for improved relevance
- Evaluation using standard metrics and benchmarks

## Data

The workshop uses:
- Wikipedia articles on machine learning topics (Deep learning, Transformers, etc.)
- BeIR SciFact dataset for demonstrations and evaluations

## Key Dependencies

- **openai**: For embeddings and completions
- **qdrant-client**: For vector storage and retrieval
- **wikipedia, beautifulsoup4**: For data collection and cleaning
- **FlagEmbedding**: For reranking functionality
- **cohere**: For additional reranking options
- **ragas**: For comprehensive RAG evaluation
- **Various utilities**: tqdm, python-dotenv, etc.

## Evaluation

The workshop includes evaluation scripts using RAGAS metrics to assess the quality of RAG outputs across dimensions like relevance, faithfulness, and answer quality.
