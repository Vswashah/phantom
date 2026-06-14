import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from litellm import completion
from dotenv import load_dotenv
from ddgs import DDGS


load_dotenv()


# ── 1. State ──────────────────────────────────────────────
# Same as before — shared memory between all agents.
# Added: pro_sources and con_sources so agents can store
# what they found before making their argument.

class DebateState(TypedDict):
    topic: str
    pro_sources: str      # NEW: what PRO agent searched and found
    con_sources: str      # NEW: what CON agent searched and found
    pro_argument: str
    con_argument: str
    verdict: str
    winner: str


# ── 2. Tool: web search ───────────────────────────────────
# This is the "tool" agents can call.
# It searches DuckDuckGo and returns the top 3 results
# as a single string the agent can read.

def search_web(query: str) -> str:
    print(f"\n  [TOOL CALL] Searching: '{query}'")
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            results.append(f"- {r['title']}: {r['body']}")
    output = "\n".join(results)
    print(f"  [TOOL RESULT] Found {len(results)} results\n")
    return output


# ── 3. Helper: call Groq via LiteLLM ─────────────────────
# Same as before.

def call_llm(prompt: str, model: str = "groq/llama-3.1-8b-instant") -> str:
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ── 4. Agent Nodes ────────────────────────────────────────
# NEW PATTERN: each agent now has TWO steps:
#   Step A — decide what to search (tool call)
#   Step B — use search results to build argument
#
# This is what makes it "agentic" — the agent decides
# what information it needs, goes and gets it, then acts.

def pro_agent(state: DebateState) -> dict:
    """PRO agent: search for evidence, then argue FOR."""

    # Step A: ask the LLM what to search for
    search_query_prompt = f"""You are preparing to argue IN FAVOR of this topic:
"{state['topic']}"

What is the SINGLE best search query to find supporting evidence?
Reply with ONLY the search query, nothing else. No quotes, no explanation."""

    search_query = call_llm(search_query_prompt, model="groq/llama-3.3-70b-versatile")

    # Step B: actually search (tool call)
    sources = search_web(search_query)

    # Step C: build argument using real search results
    argument_prompt = f"""You are a skilled debater arguing IN FAVOR of:
"{state['topic']}"

You searched the web and found this evidence:
{sources}

Now write 3 concise bullet points using this evidence to support your position.
Be specific — cite the evidence you found."""

    argument = call_llm(argument_prompt, model="groq/llama-3.3-70b-versatile")
    print(f"\n[PRO AGENT]\n{argument}\n")
    return {"pro_sources": sources, "pro_argument": argument}


def con_agent(state: DebateState) -> dict:
    """CON agent: search for counter-evidence, then argue AGAINST."""

    # Step A: decide what to search
    search_query_prompt = f"""You are preparing to argue AGAINST this topic:
"{state['topic']}"

The PRO side already argued:
{state['pro_argument']}

What is the SINGLE best search query to find counter-evidence?
Reply with ONLY the search query, nothing else. No quotes, no explanation."""

    search_query = call_llm(search_query_prompt, model="groq/llama-3.1-8b-instant")

    # Step B: search (tool call)
    sources = search_web(search_query)

    # Step C: build counter-argument using real evidence
    argument_prompt = f"""You are a skilled debater arguing AGAINST:
"{state['topic']}"

The PRO side argued:
{state['pro_argument']}

You searched the web and found this counter-evidence:
{sources}

Write 3 concise bullet points using this evidence to counter the PRO side.
Be specific — cite the evidence you found."""

    argument = call_llm(argument_prompt, model="groq/llama-3.1-8b-instant")
    print(f"\n[CON AGENT]\n{argument}\n")
    return {"con_sources": sources, "con_argument": argument}


def judge_agent(state: DebateState) -> dict:
    """Judge: evaluate both arguments AND their sources."""

    prompt = f"""You are an impartial judge evaluating a debate.
Topic: {state['topic']}

PRO argument (based on their research):
{state['pro_argument']}

CON argument (based on their research):
{state['con_argument']}

Evaluate both sides on: quality of evidence, logic, and persuasiveness.
Declare a winner and explain why in 2-3 sentences.
Format EXACTLY as:
WINNER: [pro/con/tie]
REASONING: [your explanation]"""

    verdict = call_llm(prompt, model="groq/llama-3.1-8b-instant")
    print(f"\n[JUDGE]\n{verdict}\n")

    winner = "tie"
    lower = verdict.lower()
    if "winner: pro" in lower:
        winner = "pro"
    elif "winner: con" in lower:
        winner = "con"

    return {"verdict": verdict, "winner": winner}


# ── 5. Build the Graph ────────────────────────────────────
# Same structure as before. The graph doesn't change —
# only what happens INSIDE each node changed.

def build_graph():
    graph = StateGraph(DebateState)

    graph.add_node("pro_agent", pro_agent)
    graph.add_node("con_agent", con_agent)
    graph.add_node("judge_agent", judge_agent)

    graph.set_entry_point("pro_agent")
    graph.add_edge("pro_agent", "con_agent")
    graph.add_edge("con_agent", "judge_agent")
    graph.add_edge("judge_agent", END)

    return graph.compile()


# ── 6. Run ────────────────────────────────────────────────
if __name__ == "__main__":
    debate = build_graph()

    result = debate.invoke({
        "topic": "Python is better than JavaScript for backend development",
        "pro_sources": "",
        "con_sources": "",
        "pro_argument": "",
        "con_argument": "",
        "verdict": "",
        "winner": "",
    })

    print("\n" + "=" * 50)
    print(f"FINAL WINNER: {result['winner'].upper()}")
    print("=" * 50)