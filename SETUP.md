# RAG Workshop Setup Guide

This guide walks you through setting up your environment for the RAG workshop. You'll need to complete these steps before running any of the workshop notebooks.

---

## 🚀 Quick Start with uv

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management. uv automatically handles Python version management and virtual environments.

### Step 1: Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Install Dependencies

```bash
# This installs Python 3.11 (if needed) and all dependencies
uv sync
```

That's it! uv handles everything automatically.

---

## 🔑 Configure API Keys

Create a `.env` file in the project root with your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_cluster_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Optional: For advanced RAG features
COHERE_API_KEY=your_cohere_api_key_here
```

---

## 🗄️ Choose Your Qdrant Setup

You have **two options** for the vector database. Choose the one that works best for your environment:

### Option A: Qdrant Cloud (Recommended)

1. Go to [Qdrant Cloud](https://cloud.qdrant.io/) and sign up for a free account
2. Create a new cluster (free tier is sufficient)
3. Get your cluster URL and API key from the dashboard
4. Add them to your `.env` file

### Option B: Local Docker Setup

```bash
# Run Qdrant locally
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.13.2
```

For local setup, use these `.env` values:
```bash
QDRANT_URL=http://localhost:6333
# Note: No QDRANT_API_KEY needed for local setup
```

---

## 📥 Run Data Ingestion

```bash
# From the project root directory
uv run python scripts/ingest_to_qdrant_cloud.py
```

The script automatically detects whether you're using cloud or local setup based on your `QDRANT_URL`.

### What the Ingestion Script Does

- ✅ Load the extended Wikipedia dataset (61 articles)
- 🔪 Create 1,210 chunks with 300 character chunks, 50 character overlap  
- 🤖 Generate embeddings using OpenAI text-embedding-3-small
- 📤 Upload everything to your Qdrant instance (cloud or local)
- ⏱️ Takes approximately 5-10 minutes to complete

---

## 📓 Start the Workshop

```bash
uv run jupyter lab
```

Then open the notebooks in order:
1. `naive-rag/01-naive-rag.ipynb` - Basic RAG implementation
2. `naive-rag/02-naive-rag-challenges.ipynb` - Exploring RAG limitations
3. `advanced-rag/01-advanced-rag-rerank.ipynb` - Advanced RAG with reranking
4. `advanced-rag/scifact/` - SciFact dataset examples (optional)

Each notebook automatically detects your setup and connects appropriately.

---

## 🔧 Troubleshooting

### Collection Not Found Error
```bash
# Make sure you're in the project root directory
cd path/to/building-rag-app-workshop

# Run the ingestion script
uv run python scripts/ingest_to_qdrant_cloud.py
```

### Docker Setup Issues (Option B)
```bash
# Check if Qdrant container is running
docker ps

# If not running, start it again
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.13.2

# Test connection
curl http://localhost:6333
```

### Cloud Setup Issues (Option A)
- Verify your Qdrant Cloud cluster is running in the dashboard
- Double-check your cluster URL and API key
- Make sure you're using the correct cluster region

### Environment Variables Issues
- Double-check your `.env` file is in the project root
- Restart your Jupyter kernel after creating/updating `.env`
- For local setup: `QDRANT_URL=http://localhost:6333` (no API key needed)
- For cloud setup: Both `QDRANT_URL` and `QDRANT_API_KEY` required

### OpenAI API Issues
- Make sure you have credits in your OpenAI account
- Verify your OpenAI API key is correct

### Cohere API Issues (for Advanced RAG)
- Sign up for a free Cohere account at [cohere.ai](https://cohere.ai)
- Get your API key from the dashboard
- Add it to your `.env` file as `COHERE_API_KEY=your_key_here`

### Still Having Issues?
- Check the `data/ingestion_summary.json` file (created after successful ingestion)
- Look at the terminal output from the ingestion script for error messages
- For Docker: Check Docker logs with `docker logs <container_id>`

---

## ✅ Verification

After completing the setup, you should see:
- A `.env` file in your project root with the required API keys
- Output from the ingestion script showing "🎉 INGESTION COMPLETED SUCCESSFULLY!"
- A `data/ingestion_summary.json` file with ingestion details

**You're ready to start the workshop once you see "Expected number of chunks found! Ingestion was successful." in any notebook!**
