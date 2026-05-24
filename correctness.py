# Checks: Relevance, Correctness, Hallucination, Response Quality
import re
import json
import time
import numpy as np
from difflib import SequenceMatcher
from langchain_core.messages import HumanMessage, SystemMessage
from rag_pipeline import llm, rag_simple
from main import load_retriever
from score import check_relevance

TOP_K = 3

TEST_CASES = [
    {"query": "How many annual leave days are employees entitled to?",
     "ground_truth": "Employees are entitled to 18 paid annual leave days per calendar year."},
    {"query": "How long is the probation period for new employees?",
     "ground_truth": "New employees serve a probation period of 6 months."},
    {"query": "What is the notice period after resignation?",
     "ground_truth": "Employees must serve a 60-day notice period after resignation."},
    {"query": "How often must passwords be changed?",
     "ground_truth": "Passwords must be changed every 90 days."},
    {"query": "What is the response time for a P1 critical issue?",
     "ground_truth": "First response within 30 minutes, resolved within 4 hours."},
    {"query": "What is the annual L&D budget for a junior employee?",
     "ground_truth": "Junior employees with 0 to 2 years experience get an annual L&D budget of Rs 20,000."},
    {"query": "Can an employee on probation work from home?",
     "ground_truth": "No. Work from home requires 6 months of service, which covers the full probation period."},
]

# ── Prompts ───────────────────────────────────────────────────────────

CORRECTNESS_PROMPT = """You are evaluating a RAG answer against ground truth.
Score 1-5:
  5=All facts correct  4=Mostly correct  3=Partial  2=Mostly wrong  1=Wrong
Return ONLY valid JSON: {"score": <1-5>, "reason": "<one sentence>"}"""

HALLUCINATION_PROMPT = """You are a fact-checker for a RAG system.
Split the ANSWER into factual claims.
Check if each claim is supported by the CONTEXT.
Return ONLY valid JSON:
{
  "claims": [{"claim": "<text>", "grounded": true/false}],
  "total_claims": <int>,
  "hallucinated_claims": <int>,
  "hallucination_rate": <float 0.0-1.0>
}"""

QUALITY_PROMPT = """Score this RAG answer 1-5 per dimension:
  coherence    — logically structured and readable
  completeness — fully addresses the query
  conciseness  — no padding, not truncated
  groundedness — every claim traceable to context
Return ONLY valid JSON:
{"coherence": <1-5>, "completeness": <1-5>, "conciseness": <1-5>, "groundedness": <1-5>}"""

QUALITY_WEIGHTS = {
    "coherence": 0.25, "completeness": 0.35,
    "conciseness": 0.15, "groundedness": 0.25
}


# ── Check functions ───────────────────────────────────────────────────

def check_correctness(query, answer, ground_truth, llm):
    fuzzy = round(
        SequenceMatcher(None, answer.lower(), ground_truth.lower()).ratio() * 5, 2
    )
    llm_score, llm_reason = None, None
    try:
        r          = llm.invoke([
            SystemMessage(content=CORRECTNESS_PROMPT),
            HumanMessage(content=f"Query: {query}\nGround Truth: {ground_truth}\nAnswer: {answer}"),
        ])
        parsed     = json.loads(re.sub(r"```json|```", "", r.content.strip()))
        llm_score  = int(parsed["score"])
        llm_reason = parsed.get("reason", "")
    except Exception as e:
        llm_reason = str(e)[:80]

    final  = float(llm_score) if llm_score else fuzzy
    status = "PASS" if final >= 4.0 else "WARN" if final >= 3.0 else "FAIL"
    return {"score": final, "fuzzy": fuzzy, "llm_score": llm_score,
            "reason": llm_reason, "status": status}


def check_hallucination(query, answer, retrieved_docs, llm):
    context = "\n\n".join(
        f"[Chunk {doc['rank']}]: {doc['content']}" for doc in retrieved_docs
    )
    try:
        r      = llm.invoke([
            SystemMessage(content=HALLUCINATION_PROMPT),
            HumanMessage(content=f"QUERY:\n{query}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"),
        ])
        parsed = json.loads(re.sub(r"```json|```", "", r.content.strip()))
        rate   = parsed.get("hallucination_rate", 0.0)
        status = "PASS" if rate <= 0.05 else "WARN" if rate <= 0.15 else "FAIL"
        return {"rate": rate, "total": parsed.get("total_claims", 0),
                "hallucinated": parsed.get("hallucinated_claims", 0),
                "status": status}
    except Exception as e:
        return {"rate": 0.0, "total": 0, "hallucinated": 0,
                "status": "ERROR", "error": str(e)[:80]}


def check_quality(query, answer, retrieved_docs, llm):
    context = "\n\n".join(
        f"[Chunk {doc['rank']}]: {doc['content']}" for doc in retrieved_docs
    )
    try:
        r      = llm.invoke([
            SystemMessage(content=QUALITY_PROMPT),
            HumanMessage(content=f"QUERY:\n{query}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"),
        ])
        parsed = json.loads(re.sub(r"```json|```", "", r.content.strip()))
        dims   = {d: parsed.get(d, 3) for d in QUALITY_WEIGHTS}
        score  = round(sum(QUALITY_WEIGHTS[d] * dims[d] for d in QUALITY_WEIGHTS), 4)
        status = "PASS" if score >= 4.0 else "WARN" if score >= 3.0 else "FAIL"
        return {"score": score, "dimensions": dims, "status": status}
    except Exception as e:
        return {"score": 0.0, "dimensions": {},
                "status": "ERROR", "error": str(e)[:80]}


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = load_retriever()
    results   = []

    print(f"\n{'='*60}")
    print(f"  RAG PRE-DEPLOY EVAL -- {len(TEST_CASES)} queries")
    print(f"{'='*60}")

    for case in TEST_CASES:
        query        = case["query"]
        ground_truth = case["ground_truth"]

        # retrieve once — reuse for all checks
        retrieved_docs = retriever.retrieve(query, top_k=TOP_K)
        context        = "\n\n".join([doc["content"] for doc in retrieved_docs])

        # generate answer + track latency
        start  = time.perf_counter()
        answer = llm.invoke(
            f"Use this context to answer concisely.\nContext:\n{context}\n\nQuestion: {query}"
        ).content
        latency = round((time.perf_counter() - start) * 1000, 2)

        # run all 4 checks
        rel  = check_relevance(retrieved_docs)
        cor  = check_correctness(query, answer, ground_truth, llm)
        hal  = check_hallucination(query, answer, retrieved_docs, llm)
        qual = check_quality(query, answer, retrieved_docs, llm)

        results.append({
            "query":         query,
            "answer":        answer,
            "latency_ms":    latency,
            "relevance":     rel,
            "correctness":   cor,
            "hallucination": hal,
            "quality":       qual,
        })

        print(f"\n  Query   : {query[:65]}")
        print(f"  Answer  : {answer[:65]}")
        print(f"  Latency : {latency}ms")
        print(f"  Relevance     -- {rel['status']:<4}  Avg score : {rel['avg_score']}")
        print(f"  Correctness   -- {cor['status']:<4}  Score     : {cor['score']}/5  | {str(cor['reason'])[:50]}")
        print(f"  Hallucination -- {hal['status']:<4}  Rate      : {hal['rate']:.0%}  | {hal['hallucinated']}/{hal['total']} claims")
        print(f"  Quality       -- {qual['status']:<4}  Score     : {qual['score']}/5  | {qual['dimensions']}")

    # ── Summary ───────────────────────────────────────────────────────
    def avg_of(key, field):
        return round(float(np.mean([r[key][field] for r in results])), 4)

    def count_status(key):
        s = [r[key]["status"] for r in results]
        return s.count("PASS"), s.count("WARN"), s.count("FAIL")

    r_p, r_w, r_f = count_status("relevance")
    c_p, c_w, c_f = count_status("correctness")
    h_p, h_w, h_f = count_status("hallucination")
    q_p, q_w, q_f = count_status("quality")

    avg_latency = round(float(np.mean([r["latency_ms"] for r in results])), 2)

    print(f"\n{'─'*60}")
    print(f"  SUMMARY")
    print(f"{'─'*60}")
    print(f"  Relevance     -- PASS:{r_p}  WARN:{r_w}  FAIL:{r_f}  Avg: {avg_of('relevance', 'avg_score')}")
    print(f"  Correctness   -- PASS:{c_p}  WARN:{c_w}  FAIL:{c_f}  Avg: {avg_of('correctness', 'score')}/5")
    print(f"  Hallucination -- PASS:{h_p}  WARN:{h_w}  FAIL:{h_f}  Avg: {avg_of('hallucination', 'rate'):.0%}")
    print(f"  Quality       -- PASS:{q_p}  WARN:{q_w}  FAIL:{q_f}  Avg: {avg_of('quality', 'score')}/5")
    print(f"  Avg Latency   -- {avg_latency}ms")

    all_statuses = (
        [r["relevance"]["status"]     for r in results] +
        [r["correctness"]["status"]   for r in results] +
        [r["hallucination"]["status"] for r in results] +
        [r["quality"]["status"]       for r in results]
    )

    verdict = ("PASS -- Ready to deploy"        if "FAIL" not in all_statuses and "WARN" not in all_statuses
               else "WARN -- Review before deploying" if "FAIL" not in all_statuses
               else "FAIL -- Do not deploy")

    print(f"\n  VERDICT: {verdict}")
    print(f"{'='*60}\n")