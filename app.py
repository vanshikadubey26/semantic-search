import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

docs = [
    {"id": i, "content": f"News article about topic {i}", "metadata": {"source": "news"}}
    for i in range(59)
]

def embed(text):
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(384)

for d in docs:
    d["vector"] = embed(d["content"])

class SearchRequest(BaseModel):
    query: str
    k: int = 12
    rerank: bool = True
    rerankK: int = 7

def rerank_scores(query, candidates):
    scores = []
    for doc in candidates:
        score = len(set(query.split()) & set(doc["content"].split()))
        scores.append(score / 10.0)
    return scores

@app.post("/search")
def search(req: SearchRequest):
    start = time.time()
    q_vec = embed(req.query)

    sims = []
    for d in docs:
        sim = cosine_similarity([q_vec], [d["vector"]])[0][0]
        sims.append({**d, "score": float(sim)})

    sims.sort(key=lambda x: x["score"], reverse=True)
    top_k = sims[:req.k]

    if req.rerank and top_k:
        scores = rerank_scores(req.query, top_k)
        for i in range(len(top_k)):
            top_k[i]["score"] = float(scores[i])
        top_k.sort(key=lambda x: x["score"], reverse=True)

    results = top_k[:req.rerankK]
    latency = int((time.time() - start) * 1000)

    return {
        "results": [
            {
                "id": r["id"],
                "score": r["score"],
                "content": r["content"],
                "metadata": r["metadata"]
            }
            for r in results
        ],
        "reranked": req.rerank,
        "metrics": {"latency": latency, "totalDocs": 59}
    }
