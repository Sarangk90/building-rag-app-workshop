#!/usr/bin/env python3
"""
Fetch Additional Wikipedia Articles

Script to fetch additional Wikipedia articles that are not already 
in the data/wiki_articles directory. Useful for expanding the dataset.
"""

import argparse
from pathlib import Path
from typing import List
from wiki_article_loader import (
    fetch_wikipedia_article, 
    save_article_to_file, 
    get_available_article_titles
)


def fetch_additional_articles(
    article_titles: List[str], 
    data_dir: str = "data/wiki_articles",
    skip_existing: bool = True
) -> None:
    """
    Fetch additional Wikipedia articles and save them.
    
    Args:
        article_titles: List of article titles to fetch
        data_dir: Directory to save articles
        skip_existing: Whether to skip articles that already exist
    """
    # Get existing articles if skip_existing is True
    existing_titles = set()
    if skip_existing:
        existing_titles = set(get_available_article_titles(data_dir))
    
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Fetching articles to {data_dir}")
    print(f"Skip existing articles: {skip_existing}")
    
    successful = 0
    skipped = 0
    failed = 0
    
    for title in article_titles:
        if skip_existing and title in existing_titles:
            print(f"Skipping existing article: {title}")
            skipped += 1
            continue
        
        print(f"Fetching: {title}")
        article = fetch_wikipedia_article(title)
        
        if article and save_article_to_file(article, data_dir):
            successful += 1
        else:
            failed += 1
    
    print(f"\nSummary:")
    print(f"  Successfully fetched: {successful}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total processed: {len(article_titles)}")


# Extended list of ML/AI related articles for workshop expansion
EXTENDED_ARTICLE_TITLES = [
    # Core ML concepts
    "Machine learning",
    "Supervised learning", 
    "Unsupervised learning",
    "Semi-supervised learning",
    
    # Neural Networks
    "Convolutional neural network",
    "Recurrent neural network",
    "Long short-term memory",
    "Attention mechanism",
    
    # NLP specific
    "Word embedding",
    "Named-entity recognition",
    "Part-of-speech tagging",
    "Sentiment analysis",
    
    # Computer Vision
    "Computer vision",
    "Image classification",
    "Object detection",
    "Image segmentation",
    
    # Advanced topics
    "Transfer learning",
    "Meta-learning",
    "Few-shot learning",
    "Zero-shot learning",
    
    # Optimization
    "Gradient descent",
    "Backpropagation", 
    "Adam optimizer",
    "Learning rate",
    
    # Evaluation
    "Cross-validation",
    "Bias-variance tradeoff",
    "ROC curve",
    "Precision and recall"
]


def main():
    parser = argparse.ArgumentParser(
        description="Fetch additional Wikipedia articles for RAG workshop"
    )
    parser.add_argument(
        "articles", 
        nargs="*",
        help="Article titles to fetch (if not provided, uses extended list)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/wiki_articles",
        help="Directory to save articles (default: data/wiki_articles)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fetch articles even if they already exist"
    )
    parser.add_argument(
        "--list-available",
        action="store_true",
        help="List currently available articles and exit"
    )
    parser.add_argument(
        "--list-extended",
        action="store_true", 
        help="List extended article titles and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_available:
        print("Currently available articles:")
        titles = get_available_article_titles(args.data_dir)
        for i, title in enumerate(titles, 1):
            print(f"{i:2d}. {title}")
        print(f"\nTotal: {len(titles)} articles")
        return
    
    if args.list_extended:
        print("Extended article list:")
        for i, title in enumerate(EXTENDED_ARTICLE_TITLES, 1):
            print(f"{i:2d}. {title}")
        print(f"\nTotal: {len(EXTENDED_ARTICLE_TITLES)} articles")
        return
    
    # Determine which articles to fetch
    if args.articles:
        titles_to_fetch = args.articles
        print(f"Fetching user-specified articles: {titles_to_fetch}")
    else:
        titles_to_fetch = EXTENDED_ARTICLE_TITLES
        print(f"Fetching extended article list ({len(titles_to_fetch)} articles)")
    
    # Fetch the articles
    fetch_additional_articles(
        titles_to_fetch,
        data_dir=args.data_dir,
        skip_existing=not args.force
    )


if __name__ == "__main__":
    main()