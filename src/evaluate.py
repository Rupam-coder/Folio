"""Evaluation script for Document QA RAG Retriever.

Loads a test JSON dataset of questions and ground-truth chunks,
executes queries against the ChromaDB retriever, calculates
the Retrieval Hit Rate (and MRR), and outputs a summary report.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.rag_pipeline import (
    DocumentRAGPipeline,
    RetrievedChunk,
    get_default_embeddings,
    save_to_chroma,
)


def load_evaluation_dataset(dataset_path: str | Path) -> List[Dict[str, Any]]:
    """Load evaluation questions and ground truth chunks from JSON file.

    Args:
        dataset_path: Path to the JSON dataset file.

    Returns:
        List of evaluation sample dictionaries.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a JSON array of question samples.")

    return data


def seed_test_corpus_if_needed(
    dataset: List[Dict[str, Any]],
    pipeline: DocumentRAGPipeline,
) -> None:
    """Index dataset ground-truth chunks into vectorstore if database is empty."""
    if pipeline.vectorstore is not None:
        try:
            count = pipeline.vectorstore._collection.count()
            if count > 0:
                print(f"[*] Found existing vectorstore collection with {count} items.")
                return
        except Exception:
            pass

    print("[*] Vectorstore empty. Indexing ground truth documents into ChromaDB...")
    documents: List[Document] = []
    for item in dataset:
        doc = Document(
            page_content=item.get("ground_truth_text", ""),
            metadata={
                "chunk_id": item.get("ground_truth_chunk_id", ""),
                "source": item.get("ground_truth_source", "unknown.pdf"),
                "page": max(0, item.get("ground_truth_page", 1) - 1),  # 0-indexed for consistency
            },
        )
        documents.append(doc)

    if pipeline.vectorstore is not None:
        pipeline.vectorstore.add_documents(documents)
    else:
        pipeline.vectorstore = save_to_chroma(
            documents=documents,
            embeddings=pipeline.embeddings,
            persist_directory=pipeline.persist_directory,
            collection_name=pipeline.collection_name,
        )
    print(f"[+] Successfully indexed {len(documents)} chunks into {pipeline.persist_directory}.")


def is_chunk_match(
    ground_truth: Dict[str, Any],
    retrieved_chunk: RetrievedChunk,
    min_overlap_ratio: float = 0.5,
) -> bool:
    """Determine whether a retrieved chunk matches ground truth.

    Checks:
    1. Exact metadata chunk_id match, or
    2. Document source + page number match, or
    3. Ground truth text substring or significant word overlap.
    """
    gt_chunk_id = ground_truth.get("ground_truth_chunk_id")
    if gt_chunk_id and retrieved_chunk.metadata.get("chunk_id") == gt_chunk_id:
        return True

    gt_source = ground_truth.get("ground_truth_source", "")
    gt_page = ground_truth.get("ground_truth_page")
    if gt_source and gt_page is not None:
        ret_source = Path(retrieved_chunk.source).name
        if ret_source == Path(gt_source).name and retrieved_chunk.page == gt_page:
            return True

    # Text containment or lexical overlap
    gt_text = ground_truth.get("ground_truth_text", "").strip().lower()
    ret_text = retrieved_chunk.content.strip().lower()

    if gt_text and (gt_text in ret_text or ret_text in gt_text):
        return True

    # Word set overlap
    gt_words = set(gt_text.split())
    if gt_words:
        ret_words = set(ret_text.split())
        overlap = len(gt_words.intersection(ret_words))
        if overlap / len(gt_words) >= min_overlap_ratio:
            return True

    return False


def evaluate_retrieval(
    dataset: List[Dict[str, Any]],
    pipeline: DocumentRAGPipeline,
    top_k: int = 4,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run evaluation over dataset queries and compute Hit Rate and MRR.

    Args:
        dataset: List of question and ground truth items.
        pipeline: DocumentRAGPipeline instance.
        top_k: Number of chunks retrieved per query.

    Returns:
        Tuple of (metrics_dict, query_results_list).
    """
    total_queries = len(dataset)
    hits_at_k = 0
    hits_at_1 = 0
    reciprocal_ranks = []
    detailed_results = []

    for item in dataset:
        q_id = item.get("id", "N/A")
        question = item.get("question", "")

        try:
            retrieved_chunks = pipeline.retrieve(query=question, top_k=top_k)
        except Exception as e:
            print(f"[!] Error retrieving query '{q_id}': {e}")
            retrieved_chunks = []

        hit_found = False
        hit_rank = None

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if is_chunk_match(item, chunk):
                hit_found = True
                hit_rank = rank
                break

        if hit_found:
            hits_at_k += 1
            if hit_rank == 1:
                hits_at_1 += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)

        # Record detail
        top_chunk = retrieved_chunks[0] if retrieved_chunks else None
        detailed_results.append(
            {
                "id": q_id,
                "question": question,
                "hit": hit_found,
                "hit_rank": hit_rank,
                "retrieved_count": len(retrieved_chunks),
                "top_source": Path(top_chunk.source).name if top_chunk else "None",
                "top_page": top_chunk.page if top_chunk else "None",
            }
        )

    hit_rate_at_k = (hits_at_k / total_queries * 100.0) if total_queries > 0 else 0.0
    hit_rate_at_1 = (hits_at_1 / total_queries * 100.0) if total_queries > 0 else 0.0
    mrr = (sum(reciprocal_ranks) / total_queries) if total_queries > 0 else 0.0

    metrics = {
        "total_queries": total_queries,
        "hits_at_k": hits_at_k,
        "top_k": top_k,
        "hit_rate_at_k_percent": round(hit_rate_at_k, 2),
        "hit_rate_at_1_percent": round(hit_rate_at_1, 2),
        "mean_reciprocal_rank": round(mrr, 4),
    }

    return metrics, detailed_results


def print_summary_report(metrics: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    """Print clean terminal report table."""
    print("\n" + "=" * 80)
    print(">> RAG RETRIEVAL EVALUATION REPORT")
    print("=" * 80)
    print(f"{'Query ID':<10} | {'Hit?':<8} | {'Rank':<6} | {'Top Source':<28} | {'Top Page':<8}")
    print("-" * 80)

    for res in results:
        hit_str = "[HIT]" if res["hit"] else "[MISS]"
        rank_str = str(res["hit_rank"]) if res["hit_rank"] else "-"
        source_str = res["top_source"][:27]
        print(f"{res['id']:<10} | {hit_str:<8} | {rank_str:<6} | {source_str:<28} | {str(res['top_page']):<8}")

    print("=" * 80)
    print(">> OVERALL PERFORMANCE METRICS:")
    print(f"   * Total Queries Evaluated   : {metrics['total_queries']}")
    print(f"   * Successful Hits (Top-{metrics['top_k']})  : {metrics['hits_at_k']} / {metrics['total_queries']}")
    print(f"   * Retrieval Hit Rate (Hit@{metrics['top_k']}): {metrics['hit_rate_at_k_percent']:.2f}%")
    print(f"   * Top-1 Hit Rate (Hit@1)    : {metrics['hit_rate_at_1_percent']:.2f}%")
    print(f"   * Mean Reciprocal Rank (MRR): {metrics['mean_reciprocal_rank']:.4f}")
    print("=" * 80 + "\n")


def save_report(
    output_path: str | Path,
    metrics: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> None:
    """Save evaluation metrics and breakdown to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "metrics": metrics,
        "details": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[+] Saved evaluation report to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Retriever Hit Rate.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval_dataset.json",
        help="Path to the JSON evaluation dataset.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="ChromaDB persistence directory (defaults to fast in-memory).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of retrieved chunks per query (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_report.json",
        help="Path to save the JSON evaluation report.",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        default=True,
        help="Auto-index dataset ground truth chunks if vectorstore is empty.",
    )

    args = parser.parse_args()

    # 1. Load dataset
    print(f"[*] Loading evaluation dataset from {args.dataset}...")
    dataset = load_evaluation_dataset(args.dataset)
    print(f"[+] Loaded {len(dataset)} evaluation samples.")

    # 2. Initialize RAG pipeline
    pipeline = DocumentRAGPipeline(persist_directory=args.persist_dir)

    # 3. Seed vectorstore if empty
    if args.seed:
        seed_test_corpus_if_needed(dataset, pipeline)

    # 4. Evaluate
    print(f"[*] Running retrieval evaluation (top_k={args.top_k})...")
    metrics, results = evaluate_retrieval(dataset, pipeline, top_k=args.top_k)

    # 5. Output report
    print_summary_report(metrics, results)

    if args.output:
        save_report(args.output, metrics, results)


if __name__ == "__main__":
    main()
