"""
Simplified RAG Evaluator using modern RAGAS API
Evaluates RAG systems using context-focused metrics.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Callable

from ragas import evaluate
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference, ContextRelevance
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI


def evaluate_naive_rag_v2(
    vector_search_func: Callable,
    generate_answer_func: Callable,
) -> Dict[str, Any]:
    """
    Evaluate RAG system using RAGAS context metrics.
    
    Args:
        vector_search_func: Function to perform vector search
        generate_answer_func: Function to generate answers
        
    Returns:
        Dictionary containing evaluation results
    """
    # Load evaluation dataset
    base_dir = Path(__file__).resolve().parent.parent
    eval_dataset_path = base_dir / "data" / "wiki_eval_dataset.json"
    
    try:
        with eval_dataset_path.open("r", encoding="utf-8") as f:
            eval_data = json.load(f)
        print(f"✅ Loaded {len(eval_data)} questions from evaluation dataset")
    except FileNotFoundError:
        print(f"❌ Evaluation dataset not found at {eval_dataset_path}")
        return {"error": "Dataset not found"}
    
    # Initialize LLM for RAGAS metrics
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
    
    # Initialize metrics
    metrics = [
        LLMContextRecall(llm=llm),
    ]
    
    # Process each question
    samples = []
    print(f"\nEvaluating {len(eval_data)} questions...\n")
    
    for i, item in enumerate(eval_data):
        print(f"Question {i+1}/{len(eval_data)}: {item['question'][:50]}...")
        
        try:
            # Get RAG system outputs
            search_results = vector_search_func(item["question"])
            retrieved_contexts = [result.get("text", "") for result in search_results]
            generated_answer = generate_answer_func(item["question"])
            
            # Create RAGAS sample with reference context
            sample = SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=retrieved_contexts,
                response=generated_answer,
                reference=item["ground_truth"],
                reference_contexts=[item["reference_context"]]  # Use reference_context!
            )
            samples.append(sample)
            
        except Exception as e:
            print(f"  ⚠️ Error processing question: {e}")
    
    # Run RAGAS evaluation
    print("\n🔍 Running RAGAS evaluation...")
    dataset = EvaluationDataset(samples)
    results = evaluate(dataset, metrics=metrics)
    
    # Format and display results
    formatted_results = _format_results(results)
    _print_results(formatted_results)
    
    return formatted_results


def _format_results(results) -> Dict[str, Any]:
    """Format RAGAS results into a clean dictionary."""
    formatted = {
        "metrics": {},
        "aggregate_scores": {}
    }
    
    # Extract scores from results DataFrame
    df = results.to_pandas()
    
    # Get metric columns (they might have different names)
    metric_mapping = {
        "context_recall": ["context_recall", "llm_context_recall"],
    }
    
    for metric_name, possible_columns in metric_mapping.items():
        for col in possible_columns:
            if col in df.columns:
                score = df[col].mean()
                formatted["metrics"][metric_name] = float(score)
                formatted["aggregate_scores"][metric_name] = float(score)
                break
    
    # Calculate overall score
    if formatted["metrics"]:
        formatted["aggregate_scores"]["overall_context_score"] = float(
            sum(formatted["metrics"].values()) / len(formatted["metrics"])
        )
    
    return formatted


def _print_results(results: Dict[str, Any]):
    """Print formatted evaluation results."""
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    scores = results.get("aggregate_scores", {})
    
    print("\nCONTEXT RECALL METRIC (0.0 - 1.0 scale):")
    if "context_recall" in scores:
        score = scores["context_recall"]
        icon = _get_score_icon(score)
        print(f"  {icon} Context Recall: {score:.3f}")
    
    if "context_recall" in scores:
        recall_score = scores["context_recall"]
        print(f"\n{_get_score_interpretation(recall_score)}")
    
    print("=" * 60)


def _get_score_icon(score: float) -> str:
    """Get icon based on score."""
    if score >= 0.8:
        return "🟢"
    elif score >= 0.6:
        return "🟡"
    elif score >= 0.4:
        return "🟠"
    else:
        return "🔴"


def _get_score_interpretation(score: float) -> str:
    """Get interpretation based on score."""
    if score >= 0.8:
        return "🟢 EXCELLENT: Your context retrieval is highly effective!"
    elif score >= 0.6:
        return "🟡 GOOD: Your context retrieval is working well."
    elif score >= 0.4:
        return "🟠 NEEDS IMPROVEMENT: Consider optimizing your retrieval strategy."
    else:
        return "🔴 POOR: Significant improvements needed in context retrieval."
