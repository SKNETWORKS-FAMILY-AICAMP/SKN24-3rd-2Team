import os
import sys
from pathlib import Path

import streamlit as st
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(
    page_title="For every1",
    page_icon="🏎️",
    layout="centered",
)

LOGO_PATH = APP_DIR / "logo.png"
VECTOR_DIR = str(REPO_ROOT / "vectorstore" / "chroma_f1_e5")
COLLECTION_NAME = "f1_rules_e5"

HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

MODEL_ID = "google/gemma-3-12b-it"
ADAPTER_ID = "YHPark0208/SKN24_3rd_2Team"

SYSTEM_MSG = """# SYSTEM RULE
- 역할: F1 스포츠 경기 전문 질의응답 시스템
- 조건: 사용자가 입문자라고 가정하고, 쉽고 간단하게 경기 규칙, 용어, 현상을 설명합니다.
- 제한: context의 내용을 그대로 해석합니다.
  context에 명시된 조항 번호와 내용을 우선으로 하고,
  절대 context 밖의 지식으로 보완하거나 재해석하지 않습니다.
  정확한 근거가 없는 답변은 금지합니다.
  유사한 답변을 찾을 수 없는 경우 '현재 데이터베이스에서 찾을 수 없습니다.'라고 답합니다.
- 언어: 사용자는 한국어 사용자입니다.
  참고문서는 영어일 수 있으므로, 가능하면 한국어 질문을 영어 검색 질의로 바꿔 retrieval에 활용합니다.
  출력은 한국어 구어체 답변으로 작성합니다.
- 사용자의 한국어 구어체 맥락을 적절히 이해합니다.
- 친절한 말투로 답변합니다.
- F1 기술 용어는 원문 그대로 사용하거나 정확하게 번역합니다.
- 답변 마지막에는 반드시 출처를 명시합니다.
  형식: [출처: {파일명} {조항번호}]
  예시: [출처: section_b.md B3.2]
""".strip()

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/sunn-us/SUIT/fonts/static/woff2/SUIT.css');

html, body, [class*="css"] {
    font-family: "SUIT", "Pretendard", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
}

.stApp {
    background: #ffffff;
}

.block-container {
    max-width: 860px;
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

.logo-wrap {
    text-align: center;
    margin-bottom: 1.2rem;
}

.logo-title {
    text-align: center;
    font-size: 1.85rem;
    font-weight: 800;
    color: #111111;
    letter-spacing: -0.03em;
    margin-top: 0.95rem;
    margin-bottom: 0.1rem;
    line-height: 1.32;
}

.top-bar {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 0.9rem;
}

div.stButton > button {
    border-radius: 999px;
    border: 1px solid #e3e3e3;
    background: #ffffff;
    color: #222222;
    font-weight: 600;
    padding: 0.42rem 0.85rem;
    box-shadow: none;
}

div.stButton > button:hover {
    border-color: #cfcfcf;
    color: #111111;
    background: #fafafa;
}

div[data-testid="stChatMessage"] {
    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
}

div[data-testid="stChatMessageContent"] {
    border-radius: 22px;
    padding: 1rem 1.08rem;
    font-size: 1rem;
    line-height: 1.82;
    box-shadow: none;
}

[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) div[data-testid="stChatMessageContent"] {
    background: #ffffff;
    border: 1px solid #ebebeb;
    color: #111111;
}

[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"] {
    background: #ffffff;
    border: 1px solid #dddddd;
    color: #111111;
}

div[data-testid="stChatInput"] {
    border-radius: 999px !important;
    background: #ffffff !important;
    border: 1px solid #e5e5e5 !important;
}

div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] input {
    font-size: 1rem !important;
    color: #111111 !important;
}

.small-fallback {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 900;
    color: #222222;
    margin-bottom: 0.8rem;
}

header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0);
}

[data-testid="stBottomBlockContainer"] {
    background: #ffffff;
}
</style>
""", unsafe_allow_html=True)

def sanitize(text: str) -> str:
    if text is None:
        return ""
    return str(text).encode("utf-8", errors="ignore").decode("utf-8")


def get_article(metadata: dict) -> str:
    return metadata.get("article") or metadata.get("Clause") or metadata.get("clause") or ""


def format_docs_with_source(docs):
    chunks = []
    for doc in docs:
        metadata = doc.metadata or {}
        source = metadata.get("source", "")
        article = get_article(metadata)
        label = f"{source} {article}".strip()
        chunks.append(f"[{label}]\n{doc.page_content}")
    return "\n\n".join(chunks)


def build_prompt(question: str, context: str, tokenizer) -> str:
    messages = [
        {
            "role": "user",
            "content": f"{SYSTEM_MSG}\n\n질문: {question}\n\ncontext:\n{context}\n\n답변:"
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "prompt": "passage: ",
        },
    )


@st.cache_resource(show_spinner=False)
def load_retriever():
    embedding_model = load_embedding_model()
    vector_store = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


@st.cache_resource(show_spinner=False)
def load_llm_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        max_length=None,
        temperature=0.1,
        do_sample=True,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe), tokenizer


@st.cache_resource(show_spinner=False)
def load_translator():
    if not OPENAI_API_KEY:
        return None
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


@st.cache_resource(show_spinner=False)
def load_reranker():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=device)


def translate_query(query: str) -> str:
    translator = load_translator()
    if translator is None:
        return query

    translated = translator.invoke(
        f"""Translate the following question to English as a natural search query for FIA F1 regulation documents.
Keep technical terms in English.

Question: {query}"""
    ).content.strip()
    return translated if translated else query


def rag_answer(user_prompt: str) -> str:
    if not os.path.exists(VECTOR_DIR):
        return (
            "벡터DB 폴더를 찾지 못했어요.\n\n"
            f"- 확인 경로: {VECTOR_DIR}\n"
            "- 먼저 build_db.py를 실행해서 벡터DB를 만든 뒤 다시 실행해주세요."
        )

    if not HF_TOKEN:
        return "HF_TOKEN 환경변수가 설정되지 않았어요. Hugging Face 토큰을 먼저 설정해주세요."

    try:
        retriever = load_retriever()
        llm, tokenizer = load_llm_and_tokenizer()
        reranker = load_reranker()

        translated = translate_query(user_prompt)
        retrieved = retriever.invoke("query: " + translated)

        if not retrieved:
            return "현재 데이터베이스에서 찾을 수 없습니다."

        pairs = [(translated, doc.page_content) for doc in retrieved]
        scores = reranker.predict(pairs)
        reranked = [doc for _, doc in sorted(zip(scores, retrieved), reverse=True)]
        reranked = reranked[:3]

        context = format_docs_with_source(reranked)
        prompt = build_prompt(user_prompt, context, tokenizer)
        answer = llm.invoke(prompt)
        answer = sanitize(answer).strip()

        if not answer:
            return "현재 데이터베이스에서 찾을 수 없습니다."

        return answer

    except Exception as e:
        return f"응답 생성 중 오류가 발생했어요: {sanitize(str(e))}"


def handle_prompt(prompt: str):
    prompt = sanitize(prompt).strip()
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("답변 생성 중..."):
        answer = rag_answer(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "안녕하세요! For every1입니다. 궁금한 F1 규정, 경기, 용어를 편하게 물어보세요."
        }
    ]

st.markdown('<div class="logo-wrap">', unsafe_allow_html=True)

if LOGO_PATH.exists():
    c1, c2, c3 = st.columns([1, 3.2, 1])
    with c2:
        st.image(str(LOGO_PATH), use_container_width=True)
else:
    st.markdown('<div class="small-fallback">For every1</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="logo-title">F1, 당신도 이해할 수 있습니다!</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="top-bar">', unsafe_allow_html=True)
if st.button("대화 초기화"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "대화가 초기화되었습니다. 다시 질문해주세요!"
        }
    ]
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "🙂"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("예: 트랙 리밋은 언제 페널티가 돼?"):
    handle_prompt(prompt)
    st.rerun()
