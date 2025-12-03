# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational workshop repository for building Retrieval-Augmented Generation (RAG) applications, used for the O'Reilly training course "Building Reliable RAG Applications: From PoC to Production". The workshop follows a progressive structure from basic to advanced RAG techniques.

## Workshop Structure

1. **Naive RAG** (`naive-rag/`) - Basic RAG implementation concepts
2. **Naive RAG Challenges** - Identifying limitations and areas for improvement  
3. **Advanced RAG** (`advanced-rag/`) - Reranking and hybrid search techniques
4. **SciFact Dataset** (`advanced-rag/scifact/`) - Advanced demonstrations using scientific data

## Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.11 and all dependencies
uv sync

# Set environment variables in .env file
OPENAI_API_KEY=your_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here  # For cloud setup only
COHERE_API_KEY=your_cohere_key_here  # Optional, for advanced features
```

## Running Notebooks

Start Jupyter and run notebooks in order:

```bash
uv run jupyter lab
```

1. `naive-rag/01-naive-rag.ipynb` - Basic RAG implementation
2. `naive-rag/02-naive-rag-challenges.ipynb` - Exploring limitations
3. `advanced-rag/01-advanced-rag-rerank.ipynb` - Advanced techniques with reranking

All notebooks automatically detect Qdrant setup (cloud vs local) based on environment variables.

## Data Setup Commands

Two setup options available:

**Option A: Qdrant Cloud (Recommended)**
```bash
uv run python scripts/ingest_to_qdrant_cloud.py
```

**Option B: Local Docker Setup**
```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.13.2
uv run python scripts/ingest_to_qdrant_cloud.py
```

The ingestion script loads 61 Wikipedia articles, creates 1,210 chunks (300 chars with 50 char overlap), and uploads to Qdrant.

## Key Architecture Components

### Data Processing Pipeline
- **Raw Data**: Wikipedia articles fetched via `scripts/create_extended_dataset.py`
- **Text Processing**: Clean markup, remove citations using mwparserfromhell and BeautifulSoup
- **Chunking**: 300 character chunks with 50 character overlap (configurable)
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Vector Storage**: Qdrant with COSINE distance metric

### RAG Components
- **Retrieval**: Vector similarity search via Qdrant
- **Reranking**: Cross-encoders using FlagEmbedding and Cohere
- **Generation**: OpenAI GPT models for answer synthesis
- **Evaluation**: RAGAS metrics for comprehensive assessment

### Evaluation Framework
- **Location**: `naive-rag/rag_evaluator_v2.py`
- **Metrics**: Context recall, precision, relevance using RAGAS
- **Test Data**: `data/wiki_eval_dataset.json` with question-answer pairs

## Technical Configuration

### Vector Database Settings
- Collection Name: `workshop_wikipedia_extended`
- Vector Dimension: 1536
- Distance Metric: COSINE
- Batch Size: 50 (for API rate limiting)

### Model Parameters
- Embedding Model: `text-embedding-3-small`
- Chunk Size: 300 characters
- Chunk Overlap: 50 characters
- Temperature: 0.1 (for consistent outputs)

## Development Guidelines

### Notebook Structure
- Start with clear objectives and prerequisites
- Include comprehensive imports section
- Add configuration cells for API keys and parameters
- Structure: Setup → Data Loading → Processing → Evaluation
- End with results summary and next steps

### Error Handling
- Validate environment variables before API calls
- Handle API rate limits gracefully with retry logic
- Provide fallback options for different LLM providers
- Include clear error messages for common setup issues

### Performance Considerations
- Monitor token usage and costs
- Implement caching for embeddings when possible
- Use batch processing for large datasets
- Optimize chunk sizes based on content type

## Common Issues & Troubleshooting

**Collection Not Found**: Run data ingestion script from project root
**Docker Issues**: Check container status with `docker ps`
**API Rate Limits**: Ingestion script includes delay and batch processing
**Environment Variables**: Ensure `.env` file is in project root and restart Jupyter kernel

## File Structure

```
├── naive-rag/                    # Basic RAG implementation
├── advanced-rag/                 # Advanced techniques (reranking, hybrid)
├── scripts/                      # Data preparation and ingestion
├── data/                         # Processed datasets and evaluation data
├── pyproject.toml                # Python dependencies (uv)
├── uv.lock                       # Locked dependencies
├── SETUP.md                      # Detailed setup instructions
└── .env                          # Environment variables (not in git)
```

When working with this codebase, prioritize educational clarity and maintain the progressive learning structure from naive to advanced RAG techniques.
