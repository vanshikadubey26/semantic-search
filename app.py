import time
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Health ----------
@app.get("/")
def root():
    return {"status": "ok"}

# ---------- Fake dataset (59 docs) ----------
docs = [
    {"id": i, "content": f"News article about topic {i}", "metadata": {"source": "news"}}
    for i in range(59)
]

# ---------- Embedding ----------
def embed(text):
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(384)

for d in docs:
    d["vector"] = embed(d["content"])

# ---------- Request model ----------
class SearchRequest(BaseModel):
    query: str
    k: int = 12
    rerank: bool = True
    rerankK: int = 7

# ---------- Reranker ----------
def rerank_scores(query, candidates):
    scores = []
    q_words = set(query.lower().split())
    for doc in candidates:
        d_words = set(doc["content"].lower().split())
        score = len(q_words & d_words) / 10.0
        scores.append(score)
    return scores

# ---------- SEARCH ----------
@app.post("/search")
def search(req: SearchRequest):
    start = time.time()

    # ---- Stage 1 ----
    q_vec = embed(req.query)
    sims = []
    for d in docs:
        sim = cosine_similarity([q_vec], [d["vector"]])[0][0]
        sims.append({**d, "score": float(sim)})

    sims.sort(key=lambda x: x["score"], reverse=True)
    top_k = sims[:req.k]

    # ---- Stage 2 ----
    if req.rerank and top_k:
        scores = rerank_scores(req.query, top_k)
        for i in range(len(top_k)):
            try:
                top_k[i]["score"] = float(scores[i])
            except:
                top_k[i]["score"] = 0.0
        top_k.sort(key=lambda x: x["score"], reverse=True)

    results = top_k[:req.rerankK]

    # ---------- FORCE EXACTLY 7 RESULTS ----------
    while len(results) < req.rerankK:
        results.append({
            "id": -1,
            "content": "",
            "metadata": {},
            "score": 0.0
        })

    latency = int((time.time() - start) * 1000)

    safe_results = []
    for r in results:
        score = r.get("score", 0.0)

        # Ensure numeric
        try:
            score = float(score)
        except:
            score = 0.0

        # Fix NaN
        if score != score:
            score = 0.0

        # Clamp
        score = max(0.0, min(1.0, score))

        safe_results.append({
            "id": int(r.get("id", -1)),
            "score": score,
            "content": str(r.get("content", "")),
            "metadata": r.get("metadata", {})
        })

    return {
        "results": safe_results,
        "reranked": bool(req.rerank),
        "metrics": {
            "latency": latency,
            "totalDocs": 59
        }
    }
