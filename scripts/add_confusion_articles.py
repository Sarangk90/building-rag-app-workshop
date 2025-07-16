#!/usr/bin/env python3
"""
Add long, confusion-inducing articles to the extended dataset.
Fetches from multiple sources including academic papers, technical blogs, and documentation.
"""

import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import PyPDF2
from io import BytesIO
import arxiv

# Configure paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "extended_wiki_articles"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Article sources configuration
CONFUSION_ARTICLES = {
    "lilian_weng": [
        {
            "url": "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "title": "LLM Powered Autonomous Agents",
            "confusion_terms": ["agent", "model", "architecture", "memory", "planning"]
        },
        {
            "url": "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/",
            "title": "The Transformer Family Version 2.0",
            "confusion_terms": ["transformer", "architecture", "attention", "model"]
        },
        {
            "url": "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/",
            "title": "What are Diffusion Models?",
            "confusion_terms": ["diffusion", "model", "sampling", "optimization"]
        },
        {
            "url": "https://lilianweng.github.io/posts/2018-10-13-flow-models/",
            "title": "Flow-based Deep Generative Models",
            "confusion_terms": ["flow", "model", "transformation", "optimization"]
        },
        {
            "url": "https://lilianweng.github.io/posts/2022-02-20-active-learning/",
            "title": "Learning with not Enough Data Part 3: Active Learning",
            "confusion_terms": ["learning", "sampling", "optimization", "model"]
        },
        {
            "url": "https://lilianweng.github.io/posts/2021-05-31-contrastive/",
            "title": "Contrastive Representation Learning",
            "confusion_terms": ["representation", "learning", "similarity", "optimization"]
        }
    ],
    "arxiv_papers": [
        {
            "arxiv_id": "2307.09288",  # LLaMA 2 paper
            "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
            "confusion_terms": ["model", "architecture", "optimization", "training"]
        },
        {
            "arxiv_id": "1706.03762",  # Original Transformer paper
            "title": "Attention Is All You Need",
            "confusion_terms": ["attention", "architecture", "model", "network"]
        },
        {
            "arxiv_id": "2005.14165",  # GPT-3 paper
            "title": "Language Models are Few-Shot Learners",
            "confusion_terms": ["model", "learning", "optimization", "architecture"]
        }
    ],
    "technical_blogs": [
        {
            "url": "https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback",
            "title": "Constitutional AI: Harmlessness from AI Feedback",
            "confusion_terms": ["model", "training", "optimization", "feedback"]
        },
        {
            "url": "https://openai.com/research/gpt-4",
            "title": "GPT-4 Technical Report",
            "confusion_terms": ["model", "architecture", "optimization", "evaluation"]
        }
    ],
    "confusion_topics": [
        # These are topics that naturally create confusion
        {
            "search_term": "Network Theory graph theory",
            "expected_confusion": "network (computer vs neural vs graph)"
        },
        {
            "search_term": "Kernel methods operating systems",
            "expected_confusion": "kernel (OS vs ML)"
        },
        {
            "search_term": "Agent based modeling simulation",
            "expected_confusion": "agent (software vs RL)"
        },
        {
            "search_term": "Pipeline architecture computing",
            "expected_confusion": "pipeline (data vs ML)"
        },
        {
            "search_term": "Distributed systems architecture",
            "expected_confusion": "architecture (software vs neural)"
        }
    ]
}


def fetch_lilian_weng_article(url: str, title: str) -> Optional[Dict]:
    """Fetch article from Lilian Weng's blog."""
    try:
        print(f"  Fetching: {title}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main content
        content_div = soup.find('div', class_='post-content') or soup.find('article')
        if not content_div:
            print(f"  ⚠️  Could not find content div")
            return None
            
        # Extract text content
        # Remove code blocks and math to focus on text
        for element in content_div.find_all(['pre', 'code', 'script', 'style']):
            element.decompose()
            
        text = content_div.get_text(separator='\n', strip=True)
        
        # Clean up the text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
        text = re.sub(r' {2,}', ' ', text)  # Remove excessive spaces
        
        if len(text) < 5000:  # We want long articles
            print(f"  ⚠️  Article too short: {len(text)} characters")
            return None
            
        return {
            "title": title,
            "content": text,
            "source": "Lilian Weng's Blog",
            "url": url,
            "word_count": len(text.split()),
            "character_count": len(text)
        }
        
    except Exception as e:
        print(f"  ❌ Error fetching {title}: {str(e)}")
        return None


def fetch_arxiv_paper(arxiv_id: str, title: str) -> Optional[Dict]:
    """Fetch paper from arXiv and extract text."""
    try:
        print(f"  Fetching arXiv paper: {title}")
        
        # Search for the paper
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        
        # Download PDF
        pdf_url = paper.pdf_url
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        
        # Extract text from PDF
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
            
        # Clean up the text
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove references section if it's too long
        ref_index = text.lower().rfind("references")
        if ref_index > len(text) * 0.7:  # If references are in last 30%
            text = text[:ref_index]
            
        if len(text) < 10000:  # We want long papers
            print(f"  ⚠️  Paper too short: {len(text)} characters")
            return None
            
        return {
            "title": f"{title} (arXiv:{arxiv_id})",
            "content": text,
            "source": "arXiv",
            "url": pdf_url,
            "arxiv_id": arxiv_id,
            "word_count": len(text.split()),
            "character_count": len(text)
        }
        
    except Exception as e:
        print(f"  ❌ Error fetching arXiv paper {arxiv_id}: {str(e)}")
        return None


def fetch_web_article(url: str, title: str) -> Optional[Dict]:
    """Fetch article from a web URL."""
    try:
        print(f"  Fetching: {title}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try different content selectors
        content = None
        selectors = [
            'article', 
            'main',
            'div.content',
            'div.post-content',
            'div.entry-content',
            'div.article-content'
        ]
        
        for selector in selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem
                break
                
        if not content:
            # Fallback to body
            content = soup.body
            
        if not content:
            print(f"  ⚠️  Could not find content")
            return None
            
        # Remove unwanted elements
        for element in content.find_all(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
            
        text = content.get_text(separator='\n', strip=True)
        
        # Clean up
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        if len(text) < 5000:
            print(f"  ⚠️  Article too short: {len(text)} characters")
            return None
            
        return {
            "title": title,
            "content": text,
            "source": "Web",
            "url": url,
            "word_count": len(text.split()),
            "character_count": len(text)
        }
        
    except Exception as e:
        print(f"  ❌ Error fetching {title}: {str(e)}")
        return None


def save_article(article_data: Dict, filename_base: str) -> bool:
    """Save article data to files."""
    try:
        # Save text content
        text_path = DATA_DIR / f"{filename_base}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(article_data['content'])
            
        # Save metadata
        metadata = {
            "title": article_data['title'],
            "source": article_data['source'],
            "url": article_data.get('url', ''),
            "word_count": article_data['word_count'],
            "character_count": article_data['character_count'],
            "fetched_at": datetime.now().isoformat(),
            "confusion_article": True,
            "arxiv_id": article_data.get('arxiv_id', '')
        }
        
        metadata_path = DATA_DIR / f"{filename_base}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"  ✅ Saved: {article_data['title']}")
        print(f"     Words: {article_data['word_count']:,}")
        print(f"     Characters: {article_data['character_count']:,}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error saving {filename_base}: {str(e)}")
        return False


def main():
    """Main function to fetch and save confusion articles."""
    print("🚀 Adding Confusion Articles to Extended Dataset")
    print("=" * 60)
    
    # Check for required packages
    try:
        import PyPDF2
        import arxiv
        from bs4 import BeautifulSoup
    except ImportError as e:
        print("❌ Missing required packages. Please install:")
        print("   pip install PyPDF2 arxiv beautifulsoup4")
        return
    
    successful_articles = []
    failed_articles = []
    
    # Fetch Lilian Weng articles
    print("\n📚 Fetching Lilian Weng's Blog Articles...")
    for article_info in CONFUSION_ARTICLES["lilian_weng"]:
        article_data = fetch_lilian_weng_article(
            article_info["url"], 
            article_info["title"]
        )
        if article_data:
            filename_base = f"LilianWeng_{article_info['title'].replace(' ', '_').replace(':', '')}"
            filename_base = re.sub(r'[^\w\-_]', '', filename_base)
            if save_article(article_data, filename_base):
                successful_articles.append(article_info["title"])
            else:
                failed_articles.append(article_info["title"])
        else:
            failed_articles.append(article_info["title"])
        time.sleep(2)  # Be polite
    
    # Fetch arXiv papers
    print("\n📚 Fetching arXiv Papers...")
    for paper_info in CONFUSION_ARTICLES["arxiv_papers"]:
        article_data = fetch_arxiv_paper(
            paper_info["arxiv_id"],
            paper_info["title"]
        )
        if article_data:
            filename_base = f"arXiv_{paper_info['arxiv_id']}"
            if save_article(article_data, filename_base):
                successful_articles.append(paper_info["title"])
            else:
                failed_articles.append(paper_info["title"])
        else:
            failed_articles.append(paper_info["title"])
        time.sleep(3)  # Be polite to arXiv
    
    # Fetch other technical blogs
    print("\n📚 Fetching Technical Blog Articles...")
    for article_info in CONFUSION_ARTICLES["technical_blogs"]:
        article_data = fetch_web_article(
            article_info["url"],
            article_info["title"]
        )
        if article_data:
            filename_base = f"Blog_{article_info['title'].replace(' ', '_').replace(':', '')}"
            filename_base = re.sub(r'[^\w\-_]', '', filename_base)
            if save_article(article_data, filename_base):
                successful_articles.append(article_info["title"])
            else:
                failed_articles.append(article_info["title"])
        else:
            failed_articles.append(article_info["title"])
        time.sleep(2)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CONFUSION ARTICLES SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully added: {len(successful_articles)} articles")
    for title in successful_articles:
        print(f"   - {title}")
    
    if failed_articles:
        print(f"\n❌ Failed to add: {len(failed_articles)} articles")
        for title in failed_articles:
            print(f"   - {title}")
    
    # Update dataset summary
    summary_path = DATA_DIR / "dataset_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Add confusion articles info
        summary["confusion_articles_added"] = {
            "timestamp": datetime.now().isoformat(),
            "successful": successful_articles,
            "failed": failed_articles,
            "total_added": len(successful_articles)
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(f"\n💡 Next steps:")
    print(f"   1. Run the ingestion script to add these to Qdrant")
    print(f"   2. Update the confusion tests in the notebook")
    print(f"   3. Test with queries that should trigger confusion")


if __name__ == "__main__":
    main()
