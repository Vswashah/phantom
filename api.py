from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph import build_graph

app = FastAPI(
    title="Phantom API",
    description="Multi-agent adversarial debate system",
    version="1.0.0"
)

# ── 1. Request & Response Models ──────────────────────────
# Pydantic validates incoming JSON automatically.
# If the request is missing "topic", FastAPI returns 422.

class DebateRequest(BaseModel):
    topic: str

class DebateResponse(BaseModel):
    topic: str
    winner: str
    pro_argument: str
    con_argument: str
    verdict: str


# ── 2. Routes ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Phantom is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/debate", response_model=DebateResponse)
def run_debate(request: DebateRequest):
    """
    Run a multi-agent debate on any topic.

    - PRO agent searches for supporting evidence
    - CON agent searches for counter-evidence
    - JUDGE agent evaluates both and picks a winner
    """
    try:
        debate = build_graph()

        result = debate.invoke({
            "topic": request.topic,
            "pro_sources": "",
            "con_sources": "",
            "pro_argument": "",
            "con_argument": "",
            "verdict": "",
            "winner": "",
        })

        return DebateResponse(
            topic=request.topic,
            winner=result["winner"],
            pro_argument=result["pro_argument"],
            con_argument=result["con_argument"],
            verdict=result["verdict"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))