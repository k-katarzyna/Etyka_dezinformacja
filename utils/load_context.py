import json
import streamlit as st
import numpy as np

@st.cache_resource
def load_chunks():
    with open("data/chunks.jsonl", "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
         return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def filter_chunks_by_tags(chunks, required_tag=None, topic_tags=None):
    topic_tags = topic_tags or []
    filtered = []

    for chunk in chunks:
        if required_tag and required_tag not in chunk.get("tags", []):
             continue
        if topic_tags and not any(t in chunk.get("tags", []) for t in topic_tags):
             continue
        filtered.append(chunk)

    return filtered

def search_chunks(query_embedding, required_tag=None, topic_tags=None, top_k=5):
    all_chunks = load_chunks()
    if required_tag:
        chunks = filter_chunks_by_tags(all_chunks, required_tag, topic_tags)
    else:
        chunks = all_chunks

    scored = []
    for c in chunks:
        score = cosine_similarity(query_embedding, c.get("embedding", []))
        scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]