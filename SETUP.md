# RAG Workshop Setup Guide

This guide walks you through setting up your environment for the RAG workshop. You'll need to complete these steps before running any of the workshop notebooks.

## 🚀 Choose Your Setup Option

You have **two options** to set up Qdrant and ingest the extended Wikipedia dataset. Choose the option that works best for your environment:

---

## Option A: Qdrant Cloud (Recommended)

This is the easiest option and works on all platforms without requiring Docker.

### Step 1: Create Qdrant Cloud Account
1. Go to [Qdrant Cloud](https://cloud.qdrant.io/)
2. Sign up for a free account
3. Create a new cluster (free tier is sufficient)
4. Get your cluster URL and API key from the dashboard

### Step 2: Set Environment Variables
Create a `.env` file in the project root with:
```bash
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_cluster_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Optional: For advanced RAG features
COHERE_API_KEY=your_cohere_api_key_here
```

### Step 3: Run Data Ingestion Script
```bash
# From the project root directory
python scripts/ingest_to_qdrant_cloud.py
```

---

## Option B: Local Docker Setup

If you prefer to run Qdrant locally using Docker, follow these steps:

### Step 1: Start Local Qdrant with Docker
```bash
# Run Qdrant locally
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.13.2
```

### Step 2: Set Environment Variables
Create a `.env` file in the project root with:
```bash
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=http://localhost:6333
# Note: No QDRANT_API_KEY needed for local setup

# Optional: For advanced RAG features
COHERE_API_KEY=your_cohere_api_key_here
```

### Step 3: Run Data Ingestion Script
```bash
# From the project root directory
python scripts/ingest_to_qdrant_cloud.py
```

**Note**: The same ingestion script automatically detects whether you're using cloud or local setup based on your QDRANT_URL!

---

## 📋 What the Ingestion Script Does

Regardless of which option you choose, the ingestion script will:
- ✅ Load the extended Wikipedia dataset (61 articles)
- 🔪 Create 1,210 chunks with 300 character chunks, 50 character overlap  
- 🤖 Generate embeddings using OpenAI text-embedding-3-small
- 📤 Upload everything to your Qdrant instance (cloud or local)
- ⏱️ Takes approximately 5-10 minutes to complete

---

## 🔧 Troubleshooting

### Collection Not Found Error
```bash
# Make sure you're in the project root directory
cd path/to/building-rag-app-workshop

# Run the ingestion script
python scripts/ingest_to_qdrant_cloud.py
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

---

## 📚 Workshop Notebooks

After completing this setup, you can run any of these notebooks:

1. **`naive-rag/01-naive-rag.ipynb`** - Basic RAG implementation
2. **`naive-rag/02-naive-rag-challenges.ipynb`** - Exploring RAG limitations
3. **`advanced-rag/01-advanced-rag-rerank.ipynb`** - Advanced RAG with reranking
4. **`advanced-rag/scifact/`** - SciFact dataset examples

Each notebook will automatically detect your setup and connect appropriately. 