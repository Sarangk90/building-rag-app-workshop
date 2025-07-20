#!/usr/bin/env python3
"""
Simple script to check what's actually in the Qdrant database
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# Load environment
load_dotenv()

# Initialize clients
openai_client = OpenAI()
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "workshop_wikipedia_extended"

def main():
    print("🔍 Checking Database Contents")
    print("=" * 50)
    
    # Get collection info
    info = qdrant_client.get_collection(collection_name)
    print(f"Total points: {info.points_count}")
    
    # Get all unique titles
    all_points = []
    offset = None
    
    while True:
        result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        points, next_offset = result
        all_points.extend(points)
        
        if next_offset is None:
            break
        offset = next_offset
    
    # Extract unique titles
    titles = set()
    for point in all_points:
        title = point.payload.get('title', 'Unknown')
        titles.add(title)
    
    print(f"\n📚 Found {len(titles)} unique articles:")
    for title in sorted(titles):
        print(f"  - {title}")
    
    # Check for GPT-3 specifically
    gpt3_titles = [t for t in titles if 'gpt' in t.lower() or 'language models are few-shot' in t.lower()]
    print(f"\n🎯 GPT-3 related articles:")
    for title in gpt3_titles:
        print(f"  - {title}")
    
    # Test search for GPT-3
    print(f"\n🔍 Testing search for 'GPT-3 parameters':")
    query = "GPT-3 parameters"
    response = openai_client.embeddings.create(input=query, model="text-embedding-3-small")
    query_embeddings = response.data[0].embedding
    
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embeddings,
        with_payload=True,
        limit=5,
    ).points
    
    for i, result in enumerate(search_result):
        title = result.payload.get('title', 'Unknown')
        text_preview = result.payload.get('text', '')[:150]
        print(f"  {i+1}. {title}")
        print(f"     {text_preview}...")
        print()

if __name__ == "__main__":
    main()
