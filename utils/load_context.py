import json
import streamlit as st
import numpy as np
from openai import OpenAI


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

def translate(text, to_lang="en", api_key=None):
    client = OpenAI(api_key=api_key)
    prompt = f"Na język {to_lang}:\n\n{text}"

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Jesteś tłumaczem. Zwracasz WYŁĄCZNIE przetłumaczony tekst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def search_chunks(query_text, embed_func, api_key, required_tag=None, topic_tags=None, top_k=8):

    all_chunks = load_chunks()

    def filter_and_score(query_emb, tags):

        filtered = filter_chunks_by_tags(all_chunks, required_tag, tags)
        scored = []
        for c in filtered:
            score = cosine_similarity(query_emb, c.get("embedding", []))
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    emb_orig = embed_func(query_text)
    scored_orig = filter_and_score(emb_orig, topic_tags)

    translated_text = translate(query_text, to_lang="en", api_key=api_key)
    emb_trans = embed_func(translated_text)
    scored_trans = filter_and_score(emb_trans, topic_tags)

    combined = scored_orig + scored_trans
    seen_ids = set()
    unique_chunks = []
    for score, chunk in combined:
        id = chunk.get("id")
        if id not in seen_ids:
            seen_ids.add(id)
            unique_chunks.append(chunk)
        if len(unique_chunks) >= top_k:
            break

    return unique_chunks