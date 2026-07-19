"""
evaluator.py — LLM-as-a-Judge evaluation for DriveWise.

After each successful answer generation, Gemini evaluates itself on:
  • Context Relevance   — did the retriever fetch the right chunks?
  • Faithfulness        — does the answer stay within the context (no hallucination)?
  • Answer Correctness  — does the answer actually address the question?

Each metric returns {"score": float, "reason": str}.
Scores are stored in SQLite and shown in the Analytics tab.
"""

import json
import re

from langchain_groq import ChatGroq

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    CONTEXT_RELEVANCE_PROMPT,
    FAITHFULNESS_PROMPT,
    ANSWER_CORRECTNESS_PROMPT,
)


def _judge(prompt: str, api_key: str = "") -> dict:
    """
    Send an evaluation prompt to Groq and parse the JSON response.
    Returns {"score": float, "reason": str} or a safe default on failure.
    """
    try:
        llm = ChatGroq(
            model=GROQ_MODEL_NAME,
            groq_api_key=api_key or GROQ_API_KEY,
            temperature=0.0,
        )
        response = llm.invoke(prompt)
        raw = response.content.strip()
        # Extract the first JSON object from the response
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "score":  float(data.get("score", 0.0)),
                "reason": str(data.get("reason", "—")),
            }
    except Exception as exc:
        print(f"[Evaluator] judge() error: {exc}")
    return {"score": 0.0, "reason": "Evaluation failed"}


# ── Individual metrics ────────────────────────────────────────────────────────

def eval_context_relevance(query: str, context: str, api_key: str = "") -> dict:
    return _judge(CONTEXT_RELEVANCE_PROMPT.format(query=query, context=context), api_key)


def eval_faithfulness(context: str, answer: str, api_key: str = "") -> dict:
    return _judge(FAITHFULNESS_PROMPT.format(context=context, answer=answer), api_key)


def eval_answer_correctness(query: str, context: str, answer: str, api_key: str = "") -> dict:
    return _judge(
        ANSWER_CORRECTNESS_PROMPT.format(
            query=query, context=context, answer=answer
        ),
        api_key,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_all(query: str, context: str, answer: str, api_key: str = "") -> dict:
    """
    Run all three metrics sequentially and return combined results.

    Returns:
        {
            "context_relevance":   {"score": float, "reason": str},
            "faithfulness":        {"score": float, "reason": str},
            "answer_correctness":  {"score": float, "reason": str},
        }
    """
    print("[Evaluator] Running LLM-as-a-judge evaluation …")
    cr    = eval_context_relevance(query, context, api_key)
    faith = eval_faithfulness(context, answer, api_key)
    ac    = eval_answer_correctness(query, context, answer, api_key)
    print(
        f"[Evaluator] CR={cr['score']:.2f} "
        f"Faith={faith['score']:.2f} "
        f"AC={ac['score']:.2f}"
    )
    return {
        "context_relevance":  cr,
        "faithfulness":       faith,
        "answer_correctness": ac,
    }
