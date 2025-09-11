#!/usr/bin/env python3
"""
Qdrant Cloud Ingestion Script for RAG Workshop

This script ingests the extended Wikipedia dataset into Qdrant Cloud,
creating embeddings and uploading them for workshop use.

Prerequisites:
- Extended dataset created using create_extended_dataset.py
- QDRANT_URL and QDRANT_API_KEY environment variables set
- OPENAI_API_KEY environment variable set

Usage:
    python scripts/ingest_to_qdrant_cloud.py
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, CollectionInfo
from tqdm import tqdm
import re
from mwparserfromhell import parse
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Configuration
COLLECTION_NAME = "workshop_wikipedia_extended"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
BATCH_SIZE = 50  # Process embeddings in batches to manage API limits


def setup_clients():
    """Initialize OpenAI and Qdrant clients."""
    # Check required environment variables
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Always require OpenAI API key and Qdrant URL
    if not openai_api_key:
        raise ValueError("Missing required environment variable: OPENAI_API_KEY")
    if not qdrant_url:
        raise ValueError("Missing required environment variable: QDRANT_URL")
    
    # Determine if this is a local or cloud setup
    is_local_setup = "localhost" in qdrant_url.lower()
    
    if not is_local_setup and not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY is required for cloud setup. For local Docker setup, use QDRANT_URL=http://localhost:6333")
    
    print(f"🔧 Setting up clients for {'local Docker' if is_local_setup else 'cloud'} Qdrant...")
    
    openai_client = OpenAI()
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,  # Will be None for local setup, which is fine
        timeout=60.0  # 60 second timeout for cloud operations
    )
    
    return openai_client, qdrant_client


def clean_text(text: str) -> str:
    """
    Clean Wikipedia text by removing markup and citations.
    
    Args:
        text: Raw Wikipedia content
        
    Returns:
        Cleaned text string
    """
    # Remove wiki markup and citation numbers
    text = ''.join(parse(text).strip_code())
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    return re.sub(r'\[\d+\]', '', text).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks


def create_embeddings_batch(texts: List[str], openai_client: OpenAI) -> List[List[float]]:
    """
    Create embeddings for a batch of texts.
    
    Args:
        texts: List of texts to embed
        openai_client: OpenAI client instance
        
    Returns:
        List of embedding vectors
    """
    # Clean texts for embedding
    cleaned_texts = [text.replace("\n", " ") for text in texts]
    
    try:
        response = openai_client.embeddings.create(
            input=cleaned_texts,
            model=EMBEDDING_MODEL
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        raise


def setup_qdrant_collection(qdrant_client: QdrantClient, recreate: bool = True) -> bool:
    """
    Setup Qdrant collection for the workshop.
    
    Args:
        qdrant_client: Qdrant client instance
        recreate: Whether to recreate the collection if it exists (default: True for clean ingestion)
        
    Returns:
        True if collection is ready, False otherwise
    """
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_exists = any(col.name == COLLECTION_NAME for col in collections)
        
        if collection_exists:
            if recreate:
                print(f"🗑️  Deleting existing collection: {COLLECTION_NAME}")
                qdrant_client.delete_collection(COLLECTION_NAME)
            else:
                print(f"✅ Collection '{COLLECTION_NAME}' already exists")
                return True
        
        # Create collection
        print(f"🏗️  Creating collection: {COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE
            )
        )
        
        print(f"✅ Collection '{COLLECTION_NAME}' created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up collection: {e}")
        return False


def load_extended_dataset(data_dir: Path) -> List[Dict]:
    """
    Load the extended Wikipedia dataset.
    
    Args:
        data_dir: Directory containing the extended dataset
        
    Returns:
        List of article dictionaries
    """
    articles = []
    
    # Load dataset summary to get article list
    summary_file = data_dir / "dataset_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Dataset summary not found: {summary_file}")
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print(f"📚 Loading {summary['total_articles']} articles from extended dataset")
    
    # Load each article
    for title in tqdm(summary['articles'], desc="Loading articles"):
        # Create safe filename
        safe_title = re.sub(r'[^\w\s-]', '', title)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        
        content_file = data_dir / f"{safe_title}.txt"
        metadata_file = data_dir / f"{safe_title}_metadata.json"
        
        if content_file.exists() and metadata_file.exists():
            try:
                # Load content
                with open(content_file, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                
                # Load metadata
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                articles.append({
                    "title": metadata['title'],
                    "url": metadata['url'],
                    "content": clean_text(raw_content),
                    "metadata": metadata
                })
                
            except Exception as e:
                print(f"Warning: Could not load article '{title}': {e}")
        else:
            print(f"Warning: Files missing for article '{title}'")
    
    print(f"✅ Loaded {len(articles)} articles successfully")
    return articles


def process_articles_to_chunks(articles: List[Dict]) -> List[Dict]:
    """
    Process articles into chunks with metadata.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        List of chunk dictionaries
    """
    all_chunks = []
    
    print("🔪 Chunking articles...")
    for article in tqdm(articles, desc="Processing articles"):
        chunks = chunk_text(article['content'])
        
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk,
                "title": article['title'],
                "url": article['url'],
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            all_chunks.append(chunk_data)
    
    print(f"✅ Created {len(all_chunks)} chunks from {len(articles)} articles")
    return all_chunks


def ingest_chunks_to_qdrant(chunks: List[Dict], openai_client: OpenAI, qdrant_client: QdrantClient):
    """
    Create embeddings and ingest chunks into Qdrant.
    
    Args:
        chunks: List of chunk dictionaries
        openai_client: OpenAI client instance
        qdrant_client: Qdrant client instance
    """
    print(f"🚀 Ingesting {len(chunks)} chunks to Qdrant Cloud...")
    
    # Process in batches
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    points_to_upload = []
    
    for batch_idx in tqdm(range(0, len(chunks), BATCH_SIZE), 
                         desc="Creating embeddings", 
                         total=total_batches):
        
        batch_chunks = chunks[batch_idx:batch_idx + BATCH_SIZE]
        batch_texts = [chunk['text'] for chunk in batch_chunks]
        
        try:
            # Create embeddings for batch
            embeddings = create_embeddings_batch(batch_texts, openai_client)
            
            # Create points for Qdrant
            for i, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                point_id = batch_idx + i
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk['text'],
                        "title": chunk['title'],
                        "url": chunk['url'],
                        "chunk_index": chunk['chunk_index'],
                        "total_chunks": chunk['total_chunks']
                    }
                )
                points_to_upload.append(point)
            
            # Add small delay to respect API limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_idx // BATCH_SIZE + 1}: {e}")
            raise
    
    # Upload all points to Qdrant
    print(f"📤 Uploading {len(points_to_upload)} points to Qdrant...")
    
    try:
        # Upload in smaller batches to avoid timeouts
        upload_batch_size = 25  # Reduced batch size for better reliability
        total_upload_batches = (len(points_to_upload) + upload_batch_size - 1) // upload_batch_size
        
        for i in tqdm(range(0, len(points_to_upload), upload_batch_size), 
                     desc="Uploading to Qdrant", total=total_upload_batches):
            batch_points = points_to_upload[i:i + upload_batch_size]
            
            # Retry mechanism for upload failures
            max_retries = 3
            for retry in range(max_retries):
                try:
                    qdrant_client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=batch_points
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    if retry == max_retries - 1:
                        raise e  # Re-raise on final retry
                    print(f"⚠️  Upload batch {i//upload_batch_size + 1} failed (retry {retry + 1}/{max_retries}): {e}")
                    time.sleep(2 ** retry)  # Exponential backoff
            
            # Small delay between batches to avoid rate limits
            time.sleep(0.5)
        
        print("✅ Successfully uploaded all points to Qdrant Cloud")
        
    except Exception as e:
        print(f"❌ Error uploading to Qdrant: {e}")
        raise


def verify_ingestion(qdrant_client: QdrantClient):
    """
    Verify the ingestion was successful.
    
    Args:
        qdrant_client: Qdrant client instance
    """
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        point_count = collection_info.points_count
        
        print(f"\n🔍 Ingestion Verification:")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Total points: {point_count}")
        print(f"Vector dimension: {collection_info.config.params.vectors.size}")
        print(f"Distance metric: {collection_info.config.params.vectors.distance}")
        
        # Test search functionality
        print("\n🧪 Testing search functionality...")
        test_query = "What is deep learning?"
        
        # Create test embedding
        openai_client = OpenAI()
        response = openai_client.embeddings.create(
            input=test_query,
            model=EMBEDDING_MODEL
        )
        query_embedding = response.data[0].embedding
        
        # Perform search
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            with_payload=True,
            limit=3
        ).points
        
        print(f"✅ Search test successful! Found {len(search_results)} results")
        if search_results:
            print(f"Top result from: {search_results[0].payload.get('title', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")


def create_ingestion_summary(output_dir: Path, total_chunks: int, total_articles: int):
    """
    Create a summary of the ingestion process.
    
    Args:
        output_dir: Directory to save the summary
        total_chunks: Total number of chunks ingested
        total_articles: Total number of articles processed
    """
    summary = {
        "ingestion_completed_at": datetime.now().isoformat(),
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_articles": total_articles,
        "total_chunks": total_chunks,
        "qdrant_url": os.getenv("QDRANT_URL"),
        "usage_instructions": {
            "connection": "Use QDRANT_URL and QDRANT_API_KEY environment variables",
            "collection": COLLECTION_NAME,
            "search_example": "client.query_points(collection_name, query=embedding, limit=5)"
        }
    }
    
    summary_file = output_dir / "ingestion_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Ingestion summary saved to: {summary_file}")


def main():
    """Main function to ingest the extended dataset to Qdrant Cloud."""
    print("🚀 Ingesting Extended Dataset to Qdrant Cloud")
    print("=" * 60)
    
    try:
        # Setup clients
        print("🔧 Setting up clients...")
        openai_client, qdrant_client = setup_clients()
        print("✅ Clients initialized successfully")
        
        # Setup collection
        if not setup_qdrant_collection(qdrant_client):
            return
        
        # Load dataset
        base_dir = Path(__file__).parent.parent
        data_dir = base_dir / "data" / "extended_wiki_articles"
        
        if not data_dir.exists():
            print(f"❌ Extended dataset not found at: {data_dir}")
            print("💡 Run 'python scripts/create_extended_dataset.py' first")
            return
        
        articles = load_extended_dataset(data_dir)
        
        # Process articles to chunks
        chunks = process_articles_to_chunks(articles)
        
        # Ingest to Qdrant
        ingest_chunks_to_qdrant(chunks, openai_client, qdrant_client)
        
        # Verify ingestion
        verify_ingestion(qdrant_client)
        
        # Create summary
        create_ingestion_summary(base_dir / "data", len(chunks), len(articles))
        
        print("\n" + "=" * 60)
        print("🎉 INGESTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Ingested {len(chunks)} chunks from {len(articles)} articles")
        print(f"🏗️  Collection: {COLLECTION_NAME}")
        print(f"🌐 Qdrant URL: {os.getenv('QDRANT_URL')}")
        print("\n💡 Students can now use the collection with their API keys!")
        print("📝 Next step: Update the naive-rag-challenges notebook")
        
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        print("Please check your environment variables and try again.")


if __name__ == "__main__":
    main()
