#!/usr/bin/env python3
"""
Extended Dataset Creation Script for RAG Workshop

This script fetches and processes Wikipedia articles for the naive-rag-challenges notebook.
It creates a consistent, reusable dataset that students can use without repeating the
data fetching and cleaning process.

Usage:
    python scripts/create_extended_dataset.py
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import wikipedia
import re
from mwparserfromhell import parse
from bs4 import BeautifulSoup
from tqdm import tqdm

# Extended article list for demonstrating naive RAG limitations
EXTENDED_ARTICLE_TITLES = [
    # Original core articles
    "Deep learning", "Transformer (machine learning model)", 
    "Natural language processing", "Reinforcement learning",
    "Artificial neural network", "Generative pre-trained transformer",
    "BERT (language model)", "Overfitting",
    
    # Previous expansion pack (broad adjacents)
    "Statistics", "Linear algebra", "Computer science", 
    "Neuroscience", "Psychology", "Algorithm",
    "Information theory", "Probability theory",
    "Optimization (mathematics)", "Pattern recognition",
    "Signal processing", "Software engineering",
    "Human–computer interaction", "Cognitive science",
    "Data mining", "Knowledge representation and reasoning",
    "History of artificial intelligence", "Expert system",
    "Cybernetics", "TensorFlow",
    
    # New additions (to highlight semantic limitations)
    "Machine learning", "Bayesian network", "Graph theory",
    "Electrical engineering", "Quantum computing", "Robotics",
    "Game theory", "Control theory", "Big data",
    "Database management system", "Cryptography",
    "Evolutionary computation", "Fuzzy logic",
    "Decision tree learning", "Support vector machine",
    "Cluster analysis", "Dimensionality reduction",
    "Feature selection", "Ensemble learning", "Transfer learning",
    
    # CONFUSING ARTICLES - Added to demonstrate naive RAG limitations
    # Ambiguous terms (same word, different contexts) - using exact Wikipedia titles
    "Computer network", "Operating system", "Database",
    "Tree (data structure)", "Node (computer science)",
    
    # Cross-domain vocabulary overlap (finance)
    "Finance", "Risk", "Portfolio (finance)",
    "Mathematical optimization", "Quantitative analysis (finance)",
    
    # Cross-domain vocabulary overlap (biology)
    "Biology", "Evolution", "Genetics",
    "Bioinformatics", "Computational biology",
    
    # Cross-domain vocabulary overlap (physics)
    "Physics", "Mathematics", "Statistical mechanics",
    
    # Biographical articles (will match AI queries but provide different context)
    "Alan Turing", "John von Neumann", "Claude Shannon",
    "Geoffrey Hinton", "Yann LeCun",
    
    # Broader topics that will dilute specific queries
    "Artificial intelligence", "Engineering",
    "Data structure", "Software", "Hardware"
]


def setup_directories():
    """Create necessary directories for the extended dataset."""
    base_dir = Path(__file__).parent.parent
    extended_dir = base_dir / "data" / "extended_wiki_articles"
    extended_dir.mkdir(parents=True, exist_ok=True)
    return extended_dir


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


def fetch_wikipedia_article(title: str, max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch a Wikipedia article with retry logic.
    
    Args:
        title: Wikipedia article title
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with article data or None if failed
    """
    for attempt in range(max_retries):
        try:
            page = wikipedia.page(title)
            return {
                "title": title,
                "url": page.url,
                "raw_content": page.content,
                "fetched_at": datetime.now().isoformat()
            }
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"  Disambiguation for '{title}', using: {e.options[0]}")
            return fetch_wikipedia_article(e.options[0], max_retries - 1)
        except wikipedia.exceptions.PageError:
            print(f"  Page not found: {title}")
            return None
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for '{title}': {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"  Failed to fetch '{title}' after {max_retries} attempts")
                return None
    
    return None


def save_article(article_data: Dict, output_dir: Path) -> bool:
    """
    Save article data to individual files.
    
    Args:
        article_data: Dictionary containing article information
        output_dir: Directory to save the article
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Clean filename
        safe_title = re.sub(r'[^\w\s-]', '', article_data['title'])
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        
        # Save raw content
        content_file = output_dir / f"{safe_title}.txt"
        with open(content_file, 'w', encoding='utf-8') as f:
            f.write(article_data['raw_content'])
        
        # Save metadata
        metadata = {
            "title": article_data['title'],
            "url": article_data['url'],
            "fetched_at": article_data['fetched_at'],
            "content_file": f"{safe_title}.txt",
            "cleaned_length": len(clean_text(article_data['raw_content']))
        }
        
        metadata_file = output_dir / f"{safe_title}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    except Exception as e:
        print(f"  Error saving article '{article_data['title']}': {e}")
        return False


def load_existing_articles(output_dir: Path) -> set:
    """
    Load list of already processed articles.
    
    Args:
        output_dir: Directory containing existing articles
        
    Returns:
        Set of article titles already processed
    """
    existing = set()
    for metadata_file in output_dir.glob("*_metadata.json"):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                existing.add(metadata['title'])
        except Exception as e:
            print(f"Warning: Could not read {metadata_file}: {e}")
    
    return existing


def create_dataset_summary(output_dir: Path, processed_articles: List[str]):
    """
    Create a summary file for the extended dataset.
    
    Args:
        output_dir: Directory containing the articles
        processed_articles: List of successfully processed article titles
    """
    summary = {
        "dataset_name": "Extended Wikipedia Articles for RAG Workshop",
        "created_at": datetime.now().isoformat(),
        "total_articles": len(processed_articles),
        "articles": processed_articles,
        "purpose": "Demonstrating naive RAG limitations with expanded knowledge base",
        "usage": "Used in naive-rag-challenges.ipynb for workshop demonstrations"
    }
    
    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Dataset summary saved to: {summary_file}")


def main():
    """Main function to create the extended dataset."""
    print("🚀 Creating Extended Wikipedia Dataset for RAG Workshop")
    print("=" * 60)
    
    # Setup
    output_dir = setup_directories()
    print(f"📁 Output directory: {output_dir}")
    
    # Check existing articles
    existing_articles = load_existing_articles(output_dir)
    articles_to_fetch = [title for title in EXTENDED_ARTICLE_TITLES 
                        if title not in existing_articles]
    
    if existing_articles:
        print(f"✅ Found {len(existing_articles)} existing articles")
    
    if not articles_to_fetch:
        print("🎉 All articles already exist! Dataset is up to date.")
        create_dataset_summary(output_dir, list(existing_articles))
        return
    
    print(f"📥 Fetching {len(articles_to_fetch)} new articles...")
    
    # Fetch and process articles
    processed_articles = list(existing_articles)
    failed_articles = []
    
    for title in tqdm(articles_to_fetch, desc="Fetching articles"):
        print(f"\n📖 Processing: {title}")
        
        article_data = fetch_wikipedia_article(title)
        if article_data:
            if save_article(article_data, output_dir):
                processed_articles.append(title)
                print(f"  ✅ Saved successfully")
            else:
                failed_articles.append(title)
        else:
            failed_articles.append(title)
    
    # Results summary
    print("\n" + "=" * 60)
    print("📊 DATASET CREATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully processed: {len(processed_articles)} articles")
    
    if failed_articles:
        print(f"❌ Failed to process: {len(failed_articles)} articles")
        print("Failed articles:", failed_articles)
    
    # Create dataset summary
    create_dataset_summary(output_dir, processed_articles)
    
    print(f"\n🎯 Extended dataset ready for workshop!")
    print(f"📍 Location: {output_dir}")
    print(f"📚 Total articles: {len(processed_articles)}")
    print("\n💡 Next step: Run the ingestion script to upload to Qdrant Cloud")


if __name__ == "__main__":
    main()
