#!/usr/bin/env python3
"""
Quick test script to identify GPT-3 questions that create the most confusion
in naive RAG by pulling from multiple different article sources.
"""

import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# Load environment
load_dotenv(find_dotenv())

# Initialize clients
openai_client = OpenAI()
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "workshop_wikipedia_extended"
embedding_model = "text-embedding-3-small"

def vector_search(query, top_k=5):
    """Search the Qdrant Cloud collection for relevant chunks."""
    response = openai_client.embeddings.create(
        input=query,
        model=embedding_model
    )
    query_embeddings = response.data[0].embedding
    
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embeddings,
        with_payload=True,
        limit=top_k,
    ).points
    
    return [result.payload for result in search_result]

def analyze_confusion(question, results):
    """Analyze how much confusion a question creates based on source diversity."""
    sources = set()
    domains = set()
    
    for result in results:
        title = result.get('title', 'Unknown').lower()
        text = result.get('text', '').lower()
        
        # Categorize sources - updated to match actual titles in database
        if any(term in title for term in ['language models are few-shot learners', 'arxiv_2005.14165', 'gpt-3']):
            sources.add('GPT-3 Paper')
            domains.add('GPT-3')
        elif any(term in title for term in ['attention is all you need', 'arxiv_1706.03762']):
            sources.add('Transformer Paper')
            domains.add('Transformers')
        elif 'bert' in title.lower():
            sources.add('BERT Articles')
            domains.add('BERT/Language Models')
        elif 'generative pre-trained transformer' in title.lower():
            sources.add('GPT Articles')
            domains.add('GPT/Language Models')
        elif 'transformer' in title.lower() and ('family' in title.lower() or 'machine learning' in title.lower()):
            sources.add('Transformer Articles')
            domains.add('Transformers')
        elif 'deep learning' in title.lower():
            sources.add('Deep Learning Articles')
            domains.add('Deep Learning')
        elif 'artificial neural network' in title.lower():
            sources.add('Neural Network Articles')
            domains.add('Neural Networks')
        elif any(term in title.lower() for term in ['mathematics', 'mathematical']):
            sources.add('Mathematics Articles')
            domains.add('Mathematics')
        elif 'optimization' in title.lower():
            sources.add('Optimization Articles')
            domains.add('Optimization')
        elif 'transfer learning' in title.lower():
            sources.add('Transfer Learning Articles')
            domains.add('Transfer Learning')
        elif 'active learning' in title.lower():
            sources.add('Active Learning Articles')
            domains.add('Active Learning')
        elif any(term in title.lower() for term in ['llm powered', 'autonomous agents']):
            sources.add('LLM Agents Blog')
            domains.add('AI Agents')
        elif 'artificial intelligence' in title.lower():
            sources.add('AI Articles')
            domains.add('Artificial Intelligence')
        else:
            sources.add('Other Articles')
            domains.add('General')
    
    confusion_score = len(sources)
    domain_diversity = len(domains)
    
    return {
        'sources': sources,
        'domains': domains,
        'confusion_score': confusion_score,
        'domain_diversity': domain_diversity,
        'has_gpt3': 'GPT-3' in domains
    }

# Test questions that should create confusion between different articles
confusion_questions = [
    # Questions that should pull from multiple transformer/LLM articles
    "What is the transformer architecture?",
    "How does attention mechanism work?",
    "What are the key components of a language model?",
    "How do generative pre-trained transformers work?",
    
    # Questions about training that appear across multiple papers
    "What training datasets are used for language models?",
    "How are large language models trained?",
    "What optimization techniques are used in transformer training?",
    "What is the training objective for autoregressive models?",
    
    # Questions about capabilities that span multiple domains
    "What are few-shot learning capabilities?",
    "How do models perform on reasoning tasks?",
    "What is in-context learning?",
    "How do language models handle arithmetic?",
    
    # Questions about model comparison
    "What are the differences between BERT and GPT models?",
    "How do transformer models compare to each other?",
    "What are the scaling laws for language models?",
    
    # Questions about AI/ML concepts that appear everywhere
    "What is artificial intelligence?",
    "How does machine learning work?",
    "What are neural networks?",
    "What is deep learning?",
    
    # Questions about specific technical terms
    "What are attention heads?",
    "What is model parallelism?",
    "How does gradient descent work?",
    "What is overfitting?",
]

def main():
    print("🧪 Testing GPT-3 Questions for Confusion Potential")
    print("=" * 60)
    
    confusion_results = []
    
    for question in confusion_questions:
        print(f"\n❓ Question: {question}")
        
        # Get search results
        results = vector_search(question, top_k=5)
        
        # Analyze confusion
        analysis = analyze_confusion(question, results)
        
        print(f"   📚 Sources found: {', '.join(analysis['sources'])}")
        print(f"   🎯 Domains: {', '.join(analysis['domains'])}")
        print(f"   📊 Confusion score: {analysis['confusion_score']}/5")
        print(f"   🔍 Has GPT-3 content: {'✅' if analysis['has_gpt3'] else '❌'}")
        
        if analysis['confusion_score'] >= 3:
            print(f"   ⚠️  HIGH CONFUSION POTENTIAL")
        elif analysis['confusion_score'] >= 2:
            print(f"   ⚡ MODERATE CONFUSION")
        else:
            print(f"   ✅ Low confusion")
        
        confusion_results.append({
            'question': question,
            'analysis': analysis,
            'results': results
        })
    
    # Summary of best confusion questions
    print(f"\n📈 CONFUSION ANALYSIS SUMMARY")
    print("=" * 60)
    
    high_confusion = [r for r in confusion_results if r['analysis']['confusion_score'] >= 3]
    moderate_confusion = [r for r in confusion_results if r['analysis']['confusion_score'] == 2]
    
    print(f"🔥 HIGH CONFUSION QUESTIONS ({len(high_confusion)}):")
    for result in sorted(high_confusion, key=lambda x: x['analysis']['confusion_score'], reverse=True):
        print(f"   • {result['question']} (Score: {result['analysis']['confusion_score']})")
    
    print(f"\n⚡ MODERATE CONFUSION QUESTIONS ({len(moderate_confusion)}):")
    for result in moderate_confusion:
        print(f"   • {result['question']} (Score: {result['analysis']['confusion_score']})")
    
    # Questions with GPT-3 content but also confusion
    gpt3_with_confusion = [r for r in confusion_results 
                          if r['analysis']['has_gpt3'] and r['analysis']['confusion_score'] >= 2]
    
    print(f"\n🎯 BEST WORKSHOP EXAMPLES (GPT-3 + Confusion) ({len(gpt3_with_confusion)}):")
    for result in sorted(gpt3_with_confusion, key=lambda x: x['analysis']['confusion_score'], reverse=True):
        analysis = result['analysis']
        print(f"   • {result['question']}")
        print(f"     Sources: {', '.join(analysis['sources'])}")
        print(f"     Score: {analysis['confusion_score']}")
    
    return confusion_results

if __name__ == "__main__":
    results = main()
