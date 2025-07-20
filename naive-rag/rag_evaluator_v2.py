"""
Simplified RAG Evaluator using modern RAGAS API
Evaluates RAG systems using context-focused metrics.
"""

import json
import pandas as pd
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
    show_detailed: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate RAG system using RAGAS context metrics.
    
    Args:
        vector_search_func: Function to perform vector search
        generate_answer_func: Function to generate answers
        show_detailed: Whether to show detailed individual question results
        
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
    individual_results = []
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
            
            # Store individual question data for later display
            individual_results.append({
                "question": item["question"],
                "retrieved_contexts": retrieved_contexts,
                "generated_answer": generated_answer,
                "ground_truth": item["ground_truth"],
                "reference_context": item["reference_context"]
            })
            
        except Exception as e:
            print(f"  ⚠️ Error processing question: {e}")
    
    # Run RAGAS evaluation
    print("\n🔍 Running RAGAS evaluation...")
    dataset = EvaluationDataset(samples)
    results = evaluate(dataset, metrics=metrics)
    
    # Format and display results
    formatted_results = _format_results(results, individual_results)
    _print_results(formatted_results, show_detailed)
    
    return formatted_results


def _format_results(results, individual_results: List[Dict] = None) -> Dict[str, Any]:
    """Format RAGAS results into a clean dictionary."""
    formatted = {
        "metrics": {},
        "aggregate_scores": {},
        "individual_results": []
    }
    
    # Extract scores from results DataFrame
    df = results.to_pandas()
    
    # Get metric columns (they might have different names)
    metric_mapping = {
        "context_recall": ["context_recall", "llm_context_recall"],
    }
    
    # Find the actual column name for context recall
    actual_recall_column = None
    for metric_name, possible_columns in metric_mapping.items():
        for col in possible_columns:
            if col in df.columns:
                actual_recall_column = col
                score = df[col].mean()
                formatted["metrics"][metric_name] = float(score)
                formatted["aggregate_scores"][metric_name] = float(score)
                break
        if actual_recall_column:
            break
    
    # Calculate overall score
    if formatted["metrics"]:
        formatted["aggregate_scores"]["overall_context_score"] = float(
            sum(formatted["metrics"].values()) / len(formatted["metrics"])
        )
    
    # Add individual question results with their scores
    if individual_results and not df.empty and actual_recall_column:
        for i, individual_result in enumerate(individual_results):
            question_result = {
                "question": individual_result["question"],
                "generated_answer": individual_result["generated_answer"],
                "ground_truth": individual_result["ground_truth"],
                "scores": {}
            }
            
            # Extract individual score for this question
            if i < len(df):
                score = df.iloc[i][actual_recall_column]
                if not pd.isna(score):
                    question_result["scores"]["context_recall"] = float(score)
            
            formatted["individual_results"].append(question_result)
    
    return formatted


def _print_results(results: Dict[str, Any], show_detailed: bool = False):
    """Print formatted evaluation results."""
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Display individual question results if requested or show summary
    individual_results = results.get("individual_results", [])
    if individual_results:
        if show_detailed:
            print("\n📋 DETAILED INDIVIDUAL QUESTION RESULTS:")
            print("-" * 60)
            
            for i, result in enumerate(individual_results, 1):
                print(f"\n{'='*40} Question {i} {'='*40}")
                print(f"❓ QUESTION: {result['question']}")
                
                # Display scores
                scores = result.get("scores", {})
                if "context_recall" in scores:
                    score = scores["context_recall"]
                    icon = _get_score_icon(score)
                    print(f"\n📊 {icon} Context Recall Score: {score:.3f}")
                
                # Display generated answer
                print(f"\n🤖 GENERATED ANSWER:")
                print(f"{result.get('generated_answer', 'No answer generated')}")
                
                # Display ground truth
                print(f"\n✅ GROUND TRUTH:")
                print(f"{result.get('ground_truth', 'No ground truth available')}")
                
                print(f"\n{'-'*80}")
        else:
            print("\n📋 INDIVIDUAL QUESTION SCORES:")
            print("-" * 60)
            
            individual_scores_shown = 0
            for i, result in enumerate(individual_results, 1):
                scores = result.get("scores", {})
                if "context_recall" in scores:
                    score = scores["context_recall"]
                    icon = _get_score_icon(score)
                    question_preview = result['question'][:60] + "..." if len(result['question']) > 60 else result['question']
                    print(f"{i:2d}. {icon} {score:.3f} - {question_preview}")
                    individual_scores_shown += 1
                else:
                    question_preview = result['question'][:60] + "..." if len(result['question']) > 60 else result['question']
                    print(f"{i:2d}. ❓ N/A   - {question_preview}")
            
            if individual_scores_shown == 0:
                print("⚠️ Individual scores not available - showing aggregate only")
    
    # Display aggregate results
    scores = results.get("aggregate_scores", {})
    
    print("\n" + "=" * 60)
    print("📊 AGGREGATE RESULTS")
    print("=" * 60)
    
    print("\nCONTEXT RECALL METRIC (0.0 - 1.0 scale):")
    if "context_recall" in scores:
        score = scores["context_recall"]
        icon = _get_score_icon(score)
        print(f"  {icon} Context Recall: {score:.3f}")
    
    print("=" * 60)
    
    if not show_detailed and individual_results:
        print("\n💡 Tip: Add show_detailed=True to see full question details")


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
