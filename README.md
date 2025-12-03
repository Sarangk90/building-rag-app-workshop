# Building RAG Applications Workshop

This repository contains materials for the Building RAG (Retrieval-Augmented Generation) Applications Workshop. The workshop covers both naive RAG implementations and advanced RAG techniques to help you understand and build effective RAG systems.

This repository is used for the O'Reilly training course: [Building Reliable RAG Applications: From PoC to Production](https://learning.oreilly.com/live-events/building-reliable-rag-applications-from-poc-to-production/0642572012347/).

## Prerequisites

- API keys (OpenAI, Cohere)
- Qdrant database (Cloud or Docker)

## Quick Start

### 1. Install uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. It automatically handles Python and virtual environments.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Install

```bash
git clone https://github.com/Sarangk90/building-rag-app-workshop.git
cd building-rag-app-workshop

# Install Python 3.11 and all dependencies
uv sync
```

### 3. Complete Workshop Setup

📖 **Follow the complete setup guide: [`SETUP.md`](SETUP.md)**

The setup guide covers:
- Qdrant database setup (Cloud or Docker options)
- Environment variable configuration  
- Data ingestion process
- Troubleshooting common issues

**⚠️ You must complete the setup before running any notebooks!**

### 4. Start Jupyter

```bash
uv run jupyter lab
```

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

### Wikipedia Article Management

The repository includes pre-downloaded Wikipedia articles in `data/wiki_articles/` to avoid repetitive API calls during workshops. Use the following scripts to manage articles:

#### List Available Articles
```bash
uv run python scripts/fetch_additional_articles.py --list-available
```

#### Fetch Additional Articles
```bash
# Fetch specific articles
uv run python scripts/fetch_additional_articles.py "Machine learning" "Computer vision"

# Fetch from extended list (30+ ML/AI topics)
uv run python scripts/fetch_additional_articles.py

# View the extended article list
uv run python scripts/fetch_additional_articles.py --list-extended
```

#### Force Re-fetch Existing Articles
```bash
uv run python scripts/fetch_additional_articles.py --force "Deep learning"
```

**Available Pre-downloaded Articles:**
- Artificial neural network
- BERT (language model) 
- Deep learning
- Generative pre-trained transformer
- Overfitting
- Transformer (machine learning model)

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
