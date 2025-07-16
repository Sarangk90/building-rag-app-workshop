# RAG Workshop Scripts

This directory contains automation scripts to streamline the workshop experience by pre-processing data and setting up cloud infrastructure.

## Overview

The scripts eliminate the need for students to repeatedly fetch, clean, and embed Wikipedia articles during the workshop, allowing them to focus on learning RAG concepts rather than data preparation.

## Scripts

### 1. `create_extended_dataset.py`

**Purpose**: Fetches and processes Wikipedia articles for the extended dataset used in naive-rag-challenges.

**Features**:
- Fetches 47 Wikipedia articles covering ML, AI, and related topics
- Cleans text by removing wiki markup and citations
- Saves articles as individual text files with metadata
- Idempotent operation (skips already downloaded articles)
- Robust error handling with retry logic

**Usage**:
```bash
python scripts/create_extended_dataset.py
```

**Output**:
- Creates `data/extended_wiki_articles/` directory
- Saves individual article files (e.g., `Deep_learning.txt`)
- Creates metadata files (e.g., `Deep_learning_metadata.json`)
- Generates `dataset_summary.json` with overview

**Requirements**:
- `wikipedia`, `mwparserfromhell`, `beautifulsoup4`, `tqdm`

### 2. `ingest_to_qdrant_cloud.py`

**Purpose**: Ingests the extended dataset into Qdrant Cloud for workshop use.

**Features**:
- Loads processed articles from extended dataset
- Chunks text using consistent parameters (1000 chars, 100 overlap)
- Creates embeddings using OpenAI text-embedding-3-small
- Uploads to Qdrant Cloud with proper collection setup
- Batch processing for efficiency and API rate limit management
- Verification and testing of ingested data

**Usage**:
```bash
python scripts/ingest_to_qdrant_cloud.py
```

**Prerequisites**:
1. Run `create_extended_dataset.py` first
2. Set environment variables:
   ```bash
   OPENAI_API_KEY=your_openai_key
   QDRANT_URL=your_qdrant_cloud_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

**Output**:
- Creates `workshop_wikipedia_extended` collection in Qdrant Cloud
- Uploads ~2000+ chunks with embeddings
- Generates `data/ingestion_summary.json` with details

**Requirements**:
- `openai`, `qdrant-client`, `python-dotenv`, `tqdm`

## Workflow

### For Workshop Instructors:

1. **Setup Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Create Extended Dataset**:
   ```bash
   python scripts/create_extended_dataset.py
   ```

3. **Ingest to Qdrant Cloud**:
   ```bash
   python scripts/ingest_to_qdrant_cloud.py
   ```

4. **Distribute to Students**:
   - Provide students with Qdrant Cloud URL and API keys
   - Students use the streamlined notebook version

### For Students:

Students only need:
- OpenAI API key for query embeddings
- Qdrant Cloud URL and API key (provided by instructor)
- Use `naive-rag-challenges-streamlined.ipynb`

## Configuration

### Environment Variables

```bash
# Required for both scripts
OPENAI_API_KEY=sk-...

# Required for ingestion script
QDRANT_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your-api-key
```

### Collection Settings

- **Collection Name**: `workshop_wikipedia_extended`
- **Vector Dimension**: 1536 (text-embedding-3-small)
- **Distance Metric**: COSINE
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 100 characters

## Benefits

### Time Savings
- **Data Fetching**: ~5-10 minutes saved per student
- **Embedding Creation**: ~10-15 minutes saved per student
- **Setup Complexity**: Reduced from complex to simple connection

### Consistency
- All students work with identical, pre-processed data
- Eliminates variations due to Wikipedia content changes
- Standardized chunking and embedding parameters

### Scalability
- Qdrant Cloud handles concurrent student access
- No individual API rate limit issues during workshops
- Reliable performance regardless of class size

### Focus
- Students concentrate on RAG concepts, not infrastructure
- More time for advanced techniques and evaluation
- Reduced troubleshooting of data preparation issues

## Troubleshooting

### Common Issues

1. **Wikipedia API Errors**:
   - Script includes retry logic and disambiguation handling
   - Check internet connection and Wikipedia availability

2. **OpenAI API Rate Limits**:
   - Ingestion script includes batch processing and delays
   - Monitor API usage and adjust batch sizes if needed

3. **Qdrant Connection Issues**:
   - Verify QDRANT_URL and QDRANT_API_KEY are correct
   - Check Qdrant Cloud cluster status

4. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Verification

After running scripts, verify:

1. **Dataset Creation**:
   ```bash
   ls data/extended_wiki_articles/
   # Should show ~47 .txt files and metadata
   ```

2. **Qdrant Ingestion**:
   ```python
   from qdrant_client import QdrantClient
   client = QdrantClient(url=..., api_key=...)
   info = client.get_collection("workshop_wikipedia_extended")
   print(f"Points: {info.points_count}")
   ```

## File Structure

```
scripts/
├── README.md                    # This file
├── create_extended_dataset.py   # Dataset creation script
└── ingest_to_qdrant_cloud.py   # Qdrant ingestion script

data/
├── extended_wiki_articles/      # Created by dataset script
│   ├── Deep_learning.txt
│   ├── Deep_learning_metadata.json
│   ├── ...
│   └── dataset_summary.json
└── ingestion_summary.json       # Created by ingestion script
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify environment variables are set correctly
3. Ensure all dependencies are installed
4. Check script output for specific error messages

The scripts are designed to be robust and provide clear feedback about their progress and any issues encountered.
