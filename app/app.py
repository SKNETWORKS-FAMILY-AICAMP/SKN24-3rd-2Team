import os
import sys
from pathlib import Path

import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="For every1",
    page_icon="🏎️",
    layout="centered",
)

LOGO_PATH = APP_DIR / "logo.png"

# =========================
# 스타일
# =========================
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


# =========================
# 유틸
# =========================
def sanitize(text: str) -> str:
    if text is None:
        return ""
    return str(text).encode("utf-8", errors="ignore").decode("utf-8")


# =========================
# API 호출
# =========================
def agent_answer(user_prompt: str) -> str:
    try:
        res = requests.post(
            "http://localhost:8000/ask",
            json={"question": user_prompt},
            timeout=120,
        )
        return sanitize(res.json()["answer"])
    except Exception as e:
        return f"응답 생성 중 오류가 발생했어요: {sanitize(str(e))}"


def handle_prompt(prompt: str):
    prompt = sanitize(prompt).strip()
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("답변 생성 중..."):
        answer = agent_answer(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})


# =========================
# 세션 상태
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "안녕하세요! For every1입니다. 궁금한 F1 규정, 경기, 용어를 편하게 물어보세요."
        }
    ]


# =========================
# 상단
# =========================
st.markdown('<div class="logo-wrap">', unsafe_allow_html=True)

if LOGO_PATH.exists():
    c1, c2, c3 = st.columns([1, 3.2, 1])
    with c2:
        st.image(str(LOGO_PATH), use_container_width=True)
else:
    st.markdown('<div class="small-fallback">For every1</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="logo-title">F1을 쉽게 이해하게 하는 대화형 챗봇 </div>',
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

# =========================
# 채팅
# =========================
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "🙂"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# =========================
# 입력
# =========================
if prompt := st.chat_input("예: 트랙 리밋은 언제 페널티가 돼?"):
    handle_prompt(prompt)
    st.rerun()