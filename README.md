# Building RAG Applications Workshop

This repository contains materials for the Building RAG (Retrieval-Augmented Generation) Applications Workshop. The workshop covers both naive RAG implementations and advanced RAG techniques to help you understand and build effective RAG systems.

## Prerequisites

- Python 3.11+ recommended
- Docker (for running Qdrant vector database)
- API keys (OpenAI, Cohere)
- Jupyter notebook environment

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/building-rag-app-workshop.git
cd building-rag-app-workshop
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Set up API keys**

Create a `.env` file in the root directory with the following content:

```
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=your_openai_api_base  # Optional: for Azure OpenAI or other endpoints
COHERE_API_KEY=your_cohere_api_key    # For reranking functionality
```

## Running the Workshop

The workshop is organized into progressive notebooks that build on each other:

### Naive RAG

1. **Basic RAG implementation**:

```bash
jupyter notebook naive-rag/naive-rag.ipynb
```

2. **Testing with larger dataset**:

```bash
jupyter notebook naive-rag/naive-rag-extended.ipynb
```

### Advanced RAG

1. **Implementation with reranking**:

```bash
jupyter notebook naive-rag/naive-rag-extended-with-rerank.ipynb
```

2. **Start the Qdrant vector database**:

```bash
cd advanced-rag
bash run-qdrant.sh
```

3. **Data indexing for advanced techniques**:

```bash
jupyter notebook advanced-rag/01-data-indexing.ipynb
```

4. **Advanced RAG techniques notebook**:

```bash
jupyter notebook advanced-rag/02-advanced-rag.ipynb
```

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
