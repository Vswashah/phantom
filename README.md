# Phantom 🧠⚡
 
**Multi-agent adversarial debate system** built with LangGraph, LiteLLM, and Groq.
 
Phantom pits multiple LLM agents against each other in structured debates — each agent searches the web for real evidence before arguing, and an impartial judge evaluates both sides.
 
## Architecture
 
```
User Query (debate topic)
        ↓
  [PRO AGENT]
  → decides what to search
  → calls DuckDuckGo tool
  → builds argument from real sources
        ↓
  [CON AGENT]
  → reads PRO's argument
  → searches for counter-evidence
  → builds rebuttal from real sources
        ↓
  [JUDGE AGENT]
  → evaluates both sides
  → scores on logic, evidence, persuasiveness
  → declares winner
        ↓
   Final Verdict
```
 
## Key Design Decisions
 
- **Multi-model inference** — each agent uses a different LLM (llama-3.3-70b for PRO, llama-3.1-8b for CON and Judge) via LiteLLM, demonstrating that orchestration is model-agnostic
- **Tool calling** — agents don't just generate from memory; they decide what to search, retrieve real evidence, then reason over it
- **LangGraph state machine** — shared `DebateState` TypedDict is the single source of truth passed between all nodes; no agent calls another directly
- **LiteLLM abstraction** — swap `groq/llama-3.3-70b-versatile` for `gpt-4o` or `claude-3-5-sonnet` by changing one string
## Tech Stack
 
| Layer | Tool |
|---|---|
| Agent Orchestration | LangGraph |
| LLM Interface | LiteLLM |
| LLM Provider | Groq (free tier) |
| Models | llama-3.3-70b-versatile, llama-3.1-8b-instant |
| Web Search Tool | ddgs (DuckDuckGo) |
| Package Manager | uv |
 
## Setup
 
**1. Clone and install dependencies**
```bash
git clone https://github.com/Vswashah/phantom
cd phantom
uv sync
```
 
**2. Get a free Groq API key**
 
Go to https://console.groq.com → Sign up → API Keys → Create key
 
**3. Add to `.env`**
```
GROQ_API_KEY=your_key_here
```
 
**4. Run**
```bash
uv run python graph.py
```
 
## Example Output
 
```
[TOOL CALL] Searching: 'python vs javascript backend performance benchmarks'
[TOOL RESULT] Found 3 results
 
[PRO AGENT]
- Python has a stronger ML/AI ecosystem with broader Data Science tooling...
- Python's ease of use for data-driven backend development...
- Python's performance remains competitive despite Node.js speed advantages...
 
[TOOL CALL] Searching: 'Node.js outperforms Python I/O bound tasks benchmarks'
[TOOL RESULT] Found 3 results
 
[CON AGENT]
- Node.js outperforms Python in I/O-bound tasks by 40-70% (DEV Community, 2025)...
- Python's performance is bottlenecked by GC and JIT overhead...
- JavaScript ecosystem for ML/AI is rapidly closing the gap...
 
[JUDGE]
WINNER: tie
REASONING: Both sides presented strong arguments with quality evidence...
 
==================================================
FINAL WINNER: TIE
==================================================
```
 
## Roadmap
 
- [x] Step 1 — Core LangGraph multi-agent loop
- [x] Step 2 — Tool calling with real-time web search
- [ ] Step 3 — FastAPI wrapper (`POST /debate` endpoint)
- [ ] Step 4 — Agent memory with pgvector (RAG over past debates)
- [ ] Step 5 — Conditional routing (judge sends agents back to re-argue)
## What I Learned
 
Building Phantom clarified a key distinction: **a chatbot generates text, an agent acts**. The tool-calling pattern in Step 2 — where each agent decides what information it needs, retrieves it, and then reasons over real data — is what separates agentic systems from simple LLM wrappers. The LangGraph state machine enforces clean separation: nodes never call each other directly, they only read and write shared state.
 
