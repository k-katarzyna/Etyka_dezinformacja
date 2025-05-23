import re
from copy import deepcopy
import streamlit as st
from openai import OpenAI, OpenAIError

TAGS = [
    "syntetyczne_tresci",
    "manipulacja_spoleczna",
    "detekcja_ai",
    "etyka_ai",
    "algorytmy_rekomendacji",
    "edukacja_odbiorcy",
    "przyklady_dezinformacji",
    "regulacje_prawne",
    "psychologia_manipulacji",
    "technologie_generatywne",
    "kontrola_jakosci_tresci",
    "wpływ_na_społeczeństwo",
    "zarzadzanie_ryzykiem",
    "rozpoznawanie_dezinformacji",
    "bezpieczenstwo_online",
    "media_literacy",
    "zapobieganie_dezinformacji"
]

def embed_text(text, api_key, embedding_model="text-embedding-ada-002"):
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=embedding_model,
        input=text
    )
    return response.data[0].embedding


def extract_tags(question, api_key, model="gpt-4o-mini"):
    client = OpenAI(api_key=api_key)

    function_def = [
        {
            "name": "select_tags",
            "description": "Wybiera najtrafniejsze tagi pasujące do pytania użytkownika",
            "parameters": {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": TAGS
                        },
                        "description": "Lista tagów pasujących do pytania"
                    }
                },
                "required": ["tags"]
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Jesteś asystentem pomagającym przypisać tagi do pytań użytkowników dotyczących dezinformacji AI. "
                    "Zwróc 2-3 najtrafniejsze tagi pasujące do pytania. Wybieraj jedynie z podanego zbioru."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        functions=function_def,
        function_call={"name": "select_tags"}
    )

    args = response.choices[0].message.function_call.arguments
    import json
    data = json.loads(args)
    return data["tags"]

FOLLOW_UP_PATTERNS = [
    r"^(dlaczego)\b",
    r"^(a co z)\b",
    r"^(czy możesz)\b",
    r"^(rozwiń)\b",
    r"^(wyjaśnij)\b",
    r"^(more|explain|why)"
]

def is_follow_up(question: str) -> bool:
    q = question.strip().lower()
    return any(re.match(pat, q) for pat in FOLLOW_UP_PATTERNS)

def is_question_allowed(question: str, api_key: str) -> bool:
    client = OpenAI(api_key=api_key)

    prompt = (
        f"Jesteś klasyfikatorem treści. Odpowiedz TAK, jeśli pytanie:"
        f"\n- dotyczy dozwolonych tematów: {TAGS},"
        f"\n- NIE zawiera obraźliwego języka, hejtu, gróźb ani toksycznych sformułowań."
        f"\nW przeciwnym razie odpowiedz NIE."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
    )
    return resp.choices[0].message.content.strip().upper() == "TAK"

def is_final_question_acceptable(question: str, api_key: str) -> bool:
    return is_question_allowed(question, api_key=api_key)

def requires_new_context(question: str, previous_response: str, api_key: str) -> bool:

    if is_follow_up(question):
        return False

    client = OpenAI(api_key=api_key)
    prompt = (
        "Użytkownik zadał nowe pytanie.\n"
        "Oceń, czy pytanie wymaga nowych danych (inny wątek), czy rozwija poprzednią odpowiedź.\n\n"
        "Poprzednia odpowiedź:\n"
        f"{previous_response}\n\n"
        "Nowe pytanie:\n"
        f"{question}\n\n"
        "Odpowiedz tylko: TAK (jeśli nowe dane potrzebne) lub NIE (jeśli kontynuacja)"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )

    return resp.choices[0].message.content.strip().upper() == "TAK"


def ask_openai(messages, api_key, required_tag, model="gpt-4o-mini"):
    if 'context_instruction' not in st.session_state:
        st.session_state['context_instruction'] = ''
    if 'context_tags' not in st.session_state:
        st.session_state['context_tags'] = []

    client = OpenAI(api_key=api_key)
    last_user = messages[-1]["content"]

    if not is_question_allowed(last_user, api_key):
        system_prompt = (
            "Twoim zadaniem jest grzecznie, konkretnie i empatycznie poinformować użytkownika, "
            "dlaczego nie możesz odpowiedzieć na jego pytanie. "
            "Możliwe powody to:\n"
            "- pytanie jest poza zakresem tematyki dezinformacji generowanej przez AI\n"
            "- pytanie jest nieprecyzyjne lub niezrozumiałe\n\n"
            "Zachęć użytkownika do przeformułowania pytania w zgodzie z tematyką dezinformacji, jeśli to możliwe. "
            "Nie podawaj odpowiedzi na samo pytanie."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": last_user}
            ]
        )
        return response.choices[0].message.content

    # === OCENA NOWEGO KONTEKSTU ===
    previous_response = messages[-2]["content"] if len(messages) >= 2 else ""
    new_context_needed = requires_new_context(last_user, previous_response, api_key)
    no_context_yet = not st.session_state.get("context")

    if new_context_needed or no_context_yet:
        from .load_context import search_chunks
        new_tags = extract_tags(last_user, api_key)
        emb = embed_text(last_user, api_key)
        chunks = search_chunks(emb, required_tag, new_tags)

        block = "\n\n".join(
            f"Artkuł: {c.get('source')}\nTags: {c.get('tags', '')}\nURL: {c.get('url', '')}\n{c['content']}\n\n"
            for c in chunks
        )

        context = "\n\n".join(
            f"Artkuł: {c.get('source')}\nTags: {c.get('tags', '')}\nURL: {c.get('url', '')}\n{c['content']}\n\n"
            for c in chunks
        )
        st.session_state['context'] = context
        st.session_state['context_tags'] = new_tags

    system_msg = deepcopy(st.session_state.messages[0])
    convo = (
            [{
                "role": "system",
                "content": system_msg["content"] + st.session_state['context']
            }] +
            [msg for msg in st.session_state.messages if msg["role"] != "system"]
    )
    print(convo)
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=convo
        )
        return resp.choices[0].message.content
    except OpenAIError as e:
        return f"Błąd OpenAI: {e}"
