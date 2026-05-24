import numpy as np
from typing import List, Dict


# ── Thresholds ────────────────────────────────
PASS_HIGH  = 0.60
PASS_LOW   = 0.40
WARN       = 0.20

# ── Test queries ──────────────────────────────
TEST_QUERIES = [
    "Can employees access internal systems from outside office premises?",
    "Can unused annual leave be carried forward to the next year?",
    "What is the late arrival attendance policy and its consequences?"
]



def check_relevance(retrieved_docs: List[Dict]) -> Dict:
    if not retrieved_docs:
        return {"scores": [], "avg_score": 0.0, "min_score": 0.0,
                "max_score": 0.0, "status": "FAIL",
                "note": "No documents retrieved."}

    scores = [doc["similarity_score"] for doc in retrieved_docs]
    avg    = round(float(np.mean(scores)), 4)
    
    if avg >= PASS_LOW:
        status, note = "PASS", f"Avg {avg} — relevant retrieval."
    elif avg >= WARN:
        status, note = "WARN", f"Avg {avg} — weak match, reduce top_k or rewrite query."
    else:
        status, note = "FAIL", f"Avg {avg} — poor retrieval. Re-index or fix chunk size."

    return {
        "avg_score": avg,
        "min_score": round(float(np.min(scores)), 4),
        "max_score": round(float(np.max(scores)), 4),
        "status":    status,
        "note":      note,
        "scores": [
            {
                "rank":    doc["rank"],
                "score":   round(doc["similarity_score"], 4),
                "source":  doc.get("metadata", {}).get("source_file", "unknown"),
                "preview": doc["content"][:100] + "..." if len(doc["content"]) > 100 else doc["content"],
            }
            for doc in retrieved_docs
        ],
    }


def run_relevance_tests(retriever, top_k: int = 3):
    print("\n" + "="*55)
    print("  RELEVANCE SCORE TEST")
    print("="*55)

    statuses = []

    for query in TEST_QUERIES:
        docs   = retriever.retrieve(query, top_k=top_k)
        result = check_relevance(docs)
        statuses.append(result["status"])

        icon = "✓" if result["status"] == "PASS" else "⚠" if result["status"] == "WARN" else "✗"
        print(f"\n  {icon} {result['status']} | Avg: {result['avg_score']} | Query: {query}")
        for c in result["scores"]:
            print(f"     Rank {c['rank']} | {c['score']} | {c['source']}")
            print(f"     {c['preview']}")

    # Summary
    print(f"\n{'─'*55}")
    print(f"  PASS:{statuses.count('PASS')}  WARN:{statuses.count('WARN')}  FAIL:{statuses.count('FAIL')}  TOTAL:{len(statuses)}")

    if "FAIL" in statuses:
        verdict = "FAIL — Fix retrieval before deploying"
    elif "WARN" in statuses:
        verdict = "WARN — Some queries need attention"
    else:
        verdict = "PASS — Retrieval is healthy"

    print(f"  VERDICT: {verdict}")
    print("="*55 + "\n")


if __name__ == "__main__":
    from main import load_retriever
    retriever = load_retriever()
    run_relevance_tests(retriever, top_k=3)