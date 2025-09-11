#!/usr/bin/env python3
"""
Wikipedia Article Loader Utilities

This module provides functions to load existing wiki articles and 
fetch additional ones when needed for the RAG workshop.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import wikipedia
from mwparserfromhell import parse
from bs4 import BeautifulSoup


def load_existing_wiki_articles(
    data_dir: str = "data/wiki_articles", 
    article_titles: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Load existing wiki articles from the data directory.
    
    Args:
        data_dir: Directory containing wiki article text files
        article_titles: Optional list of specific article titles to load.
                       If None, loads all available articles.
        
    Returns:
        List of dictionaries with title, content, and file_path
    """
    articles = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Warning: {data_dir} directory not found")
        return articles
    
    # Convert article titles to set for faster lookup if specified
    target_titles = set(article_titles) if article_titles else None
    found_titles = set()
    
    for file_path in data_path.glob("*.txt"):
        try:
            # Extract title from filename (remove .txt and replace underscores)
            title = file_path.stem.replace('_', ' ')
            
            # Skip if we have specific titles and this isn't one of them
            if target_titles and title not in target_titles:
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            articles.append({
                "title": title,
                "content": content,
                "file_path": str(file_path),
                "url": f"https://en.wikipedia.org/wiki/{file_path.stem}"
            })
            
            if target_titles:
                found_titles.add(title)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Report any missing articles if specific titles were requested
    if target_titles:
        missing_titles = target_titles - found_titles
        if missing_titles:
            print(f"Warning: The following articles were not found: {', '.join(missing_titles)}")
            print(f"Available articles: {', '.join(get_available_article_titles(data_dir))}")
    
    return articles


def clean_text(text: str) -> str:
    """
    Clean Wikipedia text by removing markup and citation numbers.
    
    Args:
        text: Raw Wikipedia text
        
    Returns:
        Cleaned text
    """
    # Remove wiki markup and citation numbers
    text = ''.join(parse(text).strip_code())
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    return re.sub(r'\[\d+\]', '', text).strip()


def fetch_wikipedia_article(title: str) -> Optional[Dict[str, str]]:
    """
    Fetch a single Wikipedia article and clean it.
    
    Args:
        title: Wikipedia article title
        
    Returns:
        Dictionary with article data or None if failed
    """
    try:
        page = wikipedia.page(title)
        return {
            "title": title,
            "url": page.url,
            "raw_content": page.content,
            "content": clean_text(page.content)
        }
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation for '{title}', trying first option: {e.options[0]}")
        return fetch_wikipedia_article(e.options[0])
    except wikipedia.exceptions.PageError:
        print(f"Page not found: {title}")
        return None
    except Exception as e:
        print(f"Error fetching {title}: {e}")
        return None


def save_article_to_file(article: Dict[str, str], data_dir: str = "data/wiki_articles") -> bool:
    """
    Save an article to a text file.
    
    Args:
        article: Dictionary with article data (must have 'title' and 'content')
        data_dir: Directory to save the article
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename from title
        filename = article["title"].replace(' ', '_').replace('/', '_') + '.txt'
        file_path = data_path / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(article["content"])
        
        print(f"Saved: {filename}")
        return True
    except Exception as e:
        print(f"Error saving article '{article['title']}': {e}")
        return False


def get_available_article_titles(data_dir: str = "data/wiki_articles") -> List[str]:
    """
    Get list of available article titles from the data directory.
    
    Args:
        data_dir: Directory containing wiki articles
        
    Returns:
        List of article titles
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    titles = []
    for file_path in data_path.glob("*.txt"):
        title = file_path.stem.replace('_', ' ')
        titles.append(title)
    
    return sorted(titles)


# Default article titles used in the workshop
DEFAULT_ARTICLE_TITLES = [
    "Deep learning",
    "Transformer (machine learning model)",
    "Natural language processing", 
    "Reinforcement learning",
    "Artificial neural network",
    "Generative pre-trained transformer",
    "BERT (language model)",
    "Overfitting"
]

# Predefined article combinations for different workshop scenarios
ARTICLE_COMBINATIONS = {
    "minimal": ["Deep learning", "Artificial neural network"],
    "core_ml": ["Deep learning", "Artificial neural network", "Overfitting"],
    "transformers": ["Transformer (machine learning model)", "BERT (language model)", "Generative pre-trained transformer"],
    "all_neural": ["Deep learning", "Artificial neural network", "Transformer (machine learning model)", "BERT (language model)"],
    "default": DEFAULT_ARTICLE_TITLES
}


def load_article_combination(
    combination: str, 
    data_dir: str = "data/wiki_articles"
) -> List[Dict[str, str]]:
    """
    Load a predefined combination of articles.
    
    Args:
        combination: Name of the predefined combination
        data_dir: Directory containing wiki articles
        
    Returns:
        List of article dictionaries
    """
    if combination not in ARTICLE_COMBINATIONS:
        available = ", ".join(ARTICLE_COMBINATIONS.keys())
        raise ValueError(f"Unknown combination '{combination}'. Available: {available}")
    
    article_titles = ARTICLE_COMBINATIONS[combination]
    print(f"Loading '{combination}' combination: {', '.join(article_titles)}")
    
    return load_existing_wiki_articles(data_dir, article_titles)


if __name__ == "__main__":
    # Example usage
    print("Available articles:")
    titles = get_available_article_titles()
    for title in titles:
        print(f"  - {title}")
    
    print(f"\n=== Loading ALL articles (default behavior) ===")
    articles = load_existing_wiki_articles()  # No params = load all
    print(f"Successfully loaded {len(articles)} articles")
    
    print(f"\n=== Loading SPECIFIC articles ===")
    specific_articles = load_existing_wiki_articles(
        article_titles=["Deep learning", "Artificial neural network"]
    )
    print(f"Successfully loaded {len(specific_articles)} specific articles")
    
    print(f"\n=== Loading PREDEFINED combination ===")
    combo_articles = load_article_combination("minimal")
    print(f"Successfully loaded {len(combo_articles)} articles from 'minimal' combination")
    
    print(f"\n=== Available combinations ===")
    for combo_name, combo_titles in ARTICLE_COMBINATIONS.items():
        print(f"  - {combo_name}: {combo_titles}")
    
    # Show example content
    if articles:
        print(f"\nExample content from '{articles[0]['title']}':") 
        print(f"{articles[0]['content'][:200]}...")