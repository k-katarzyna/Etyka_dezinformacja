import streamlit as st
from utils.openai import ask_openai

st.set_page_config(
    page_title="Asystent ds. dezinformacji",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    html, body, [data-testid="stApp"] {
        background-color: #FFF3E6;
        color: #1B1F3B;
    }
    .stButton>button {
        background-color: #FF8C69;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5em 1em;
        transition: background-color 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #E2735F;
        color: white;
    }
    .typing {
        font-style: italic;
        color: #2B2B2B;
        animation: blink 1s steps(1) infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 0; }
        50% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([3, 1])
with col1:
    st.title("Asystent ds. dezinformacji")
with col2:
    st.image("images/img1.jpg", width=350)

st.markdown("""
    <style>
        div[data-testid="column"] > div:nth-child(2) {
            display: flex;
            align-items: center;
            height: 100%;
        }
    </style>
""", unsafe_allow_html=True)

openai_api_key = st.secrets["openai"]["api_key"]


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "Jesteś ekspertem od dezinformacji generowanej przez AI. Odpowiadasz rzeczowo, konkretnie, przyjaźnie i empatycznie."}
    ]
if "context_mode" not in st.session_state:
    st.session_state.context_mode = None
if "context_initialized" not in st.session_state:
    st.session_state.context_initialized = False
if "locked" not in st.session_state:
    st.session_state.locked = False

if st.button("🔄 Resetuj czat"):
    st.session_state.clear()
    st.experimental_set_query_params()
    st.rerun()

if st.session_state.context_mode is None:
    st.markdown("""
    ### Witaj!  
    Jestem specjalistą od dezinformacji generowanej przez AI.  
    Pomogę Ci w poszerzeniu wiedzy w tym temacie, abyś był bardziej odporny na manipulację.  
    Wybierz swoją potrzebę:
    """)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Tworzę i publikuję treści przy pomocy AI, na co powinienem uważać?"):
            st.session_state.context_mode = "creator"
            st.session_state.messages.append({
                "role": "system",
                "content": (
                    "Użytkownik to twórca treści. "
                    "Potrzebuje informacji, jak odpowiedzialnie korzystać z AI i unikać szerzenia dezinformacji."
                )
            })
            st.session_state.context_initialized = True
            st.rerun()
    with col2:
        if st.button("🔍 Chcę się dowiedzieć, jak nie paść ofiarą dezinformacji"):
            st.session_state.context_mode = "consumer"
            st.session_state.messages.append({
                "role": "system",
                "content": (
                    "Użytkownik chce nauczyć się rozpoznawać dezinformację "
                    "i nie paść jej ofiarą. Szuka sposobów na ochronę przed fałszywymi treściami."
                )
            })
            st.session_state.context_initialized = True
            st.rerun()

if st.session_state.context_mode:
    if st.session_state.context_mode == "creator":
        st.success("Świetnie! Jako twórca treści musisz znać odpowiedzialność, która wiąże się z użyciem AI. Nauczę Cię jak unikać nieświadomego szerzenia dezinformacji.")
    else:
        st.success("Doskonale! Wiedza to najlepsza broń przeciwko manipulacji. Nauczę Cię, jak rozpoznawać podejrzane informacje i nie dać się złapać na fałszywe treści.")

    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Zadaj pytanie...", disabled=st.session_state.locked)

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("""
            <style>
            .typing-dots span {
              animation: blink 1.4s infinite both;
            }
            .typing-dots span:nth-child(2) {
              animation-delay: 0.2s;
            }
            .typing-dots span:nth-child(3) {
              animation-delay: 0.4s;
            }
            @keyframes blink {
              0%   { opacity: 0.2; }
              20%  { opacity: 1; }
              100% { opacity: 0.2; }
            }
            </style>
            <div class="typing-dots">⏳ Przetwarzam<span>.</span><span>.</span><span>.</span></div>
            """, unsafe_allow_html=True)

            response = ask_openai(st.session_state.messages, openai_api_key, st.session_state.context_mode)
            placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

