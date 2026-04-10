"""
db 구축용 파일 1개
rag_invoke 1개
로 분리하기 
"""

import os
import json
import shutil
import glob
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from collections import defaultdict

hf_token = os.environ.get("HF_TOKEN")
openai_key = os.environ.get("OPENAI_API_KEY")

docs = []

# glossary json
with open("/workspace/data/f1_glossary_all.json", "r", encoding="utf-8") as f:
    glossary_data = json.load(f)

for i, item in enumerate(glossary_data):
    term = item.get("term", "").strip()
    desc = item.get("description", "").strip()
    if term and desc:
        docs.append(Document(
            page_content=f"Term: {term}\nDescription: {desc}",
            metadata={"source": "f1_glossary_all.json", "doc_type": "glossary", "row": i, "term": term},
        ))

print("glossary 추가 후 문서 수:", len(docs))

# history wiki json
with open("/workspace/data/f1_history_wiki.json", "r", encoding="utf-8") as f:
    wiki_data = json.load(f)

wiki_text = wiki_data if isinstance(wiki_data, str) else json.dumps(wiki_data, ensure_ascii=False)
docs.append(Document(
    page_content=wiki_text,
    metadata={"source": "f1_history_ko_wiki.json", "doc_type": "wiki"},
))

print("wiki 추가 후 문서 수:", len(docs))

# tire txt
with open("/workspace/data/pirelli_f1_tires.txt", "r", encoding="utf-8") as f:
    tire_text = f.read()

docs.append(Document(
    page_content=tire_text,
    metadata={"source": "pirelli_f1_tires.txt", "doc_type": "tires"},
))

print("최종 원본 문서 수:", len(docs))

md_files = glob.glob("/workspace/data/*.md")

for md_path in md_files:
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    filename = os.path.basename(md_path)
    docs.append(Document(
        page_content=md_text,
        metadata={
            "source": filename,
            "doc_type": "regulation",
        },
    ))

print(f"마크다운 {len(md_files)}개 추가 후 문서 수:", len(docs))
print("최종 원본 문서 수:", len(docs))

from langchain_text_splitters import MarkdownHeaderTextSplitter

glossary_docs = [doc for doc in docs if doc.metadata.get("doc_type") == "glossary"]
long_docs = [doc for doc in docs if doc.metadata.get("doc_type") != "glossary"]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ".", " ", ""]
)

# 마크다운은 헤더 기준으로 먼저 분리
headers_to_split_on = [
    ("##", "article")
]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

regulation_docs = [doc for doc in long_docs if doc.metadata.get("doc_type") == "regulation"]
other_long_docs = [doc for doc in long_docs if doc.metadata.get("doc_type") != "regulation"]

# 규정 마크다운 헤더 기준 청킹
md_chunks = []
for doc in regulation_docs:
    splits = md_splitter.split_text(doc.page_content)
    for split in splits:
        split.metadata.update({
            "source": doc.metadata["source"],
            "doc_type": "regulation",
        })
        md_chunks.append(split)

# 긴 조항은 추가로 자르기
md_chunks = splitter.split_documents(md_chunks)

def get_article_group(article_str: str) -> str:
    clean = re.sub(r'\*+', '', article_str).strip()
    match = re.match(r'(B\d+\.\d+)', clean)
    if match:
        return match.group(1)
    return clean

grouped = defaultdict(list)
for chunk in md_chunks:
    article = chunk.metadata.get("article", "")
    group_key = (chunk.metadata["source"], get_article_group(article))
    grouped[group_key].append(chunk.page_content)

merged_chunks = []
for (source, group), contents in grouped.items():
    merged_text = "\n\n".join(contents)
    merged_chunks.append(Document(
        page_content=merged_text,
        metadata={"source": source, "doc_type": "regulation", "article": group}
    ))

final_md_chunks = splitter.split_documents(merged_chunks)
print("병합 전 청크 수:", len(md_chunks))
print("병합 후 청크 수:", len(final_md_chunks))

# wiki, tire 등 나머지는 기존 방식
split_long_docs = splitter.split_documents(other_long_docs)

docs = glossary_docs + split_long_docs + final_md_chunks

print("glossary 문서 수:", len(glossary_docs))
print("일반 긴 문서 chunk 수:", len(split_long_docs))
print("규정 마크다운 chunk 수:", len(md_chunks))
print("최종 chunk 수:", len(docs))

from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda"},
    encode_kwargs={
        "normalize_embeddings": True,
        "prompt": "passage: "
    }
)
print("임베딩 모델 로드 완료")

persist_dir = "/workspace/chroma_f1_e5"

if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir, ignore_errors=True)

os.makedirs(persist_dir, exist_ok=True)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_dir,
    collection_name="f1_rules_e5"
)

print("벡터 저장 완료:", vector_store._collection.count(), "개 청크")

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_huggingface import HuggingFacePipeline

MODEL_ID = "google/gemma-3-12b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=True,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)
print("LLM 로드 완료")

from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

translator = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_MSG = """# SYSTEM RULE
-역할: F1 스포츠 경기 전문 질의응답 시스템
-조건: 사용자가 입문자라고 가정, 쉽고 간단하게 경기 규칙, 용어, 현상을 설명
-제한: context의 내용을 그대로 해석할 것. 
 context에 명시된 조항 번호와 내용을 우선으로 하고,
 절대 context 밖의 지식으로 보완하거나 재해석하지 말 것.(정확한 근거 없는 답변 금지, 유사한 답변을 찾을 수 없는 경우, '현재 데이터베이스에서 찾을 수 없습니다.' 답변)
-언어: 사용자는 한국어 사용자 → 참고문서는 영어이므로 입력된 한국어 질문을 영어로 번역한 후 문서에서 검색, 출력은 한국어 구어체 답변
-사용자의 한국어 구어체의 맥락을 적절히 이해 필요
-친절한 말투로 답변
-F1 기술 용어는 원문 그대로 사용하거나 혹은 정확하게 번역"""

def build_gemma_prompt(question: str, context: str) -> str:
    messages = [
        {"role": "user", "content": f"{SYSTEM_MSG}\n\n질문: {question}\n\ncontext:\n{context}\n\n답변:"}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

retriever = vector_store.as_retriever(
    search_type="similarity",  # mmr → similarity
    search_kwargs={"k": 10}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

def rag_invoke(query: str):
    translated = translator.invoke(
        f"Translate to English for F1 regulation search, keep technical terms: {query}"
    ).content
    
    retrieved = retriever.invoke("query: " + translated)
    
    # reranking — 질문과 각 청크를 정밀 비교
    pairs = [(translated, doc.page_content) for doc in retrieved]
    scores = reranker.predict(pairs)
    reranked = [doc for _, doc in sorted(zip(scores, retrieved), reverse=True)]
    
    context = format_docs(reranked)
    prompt = build_gemma_prompt(query, context)
    answer = llm.invoke(prompt)
    return {"answer": answer, "context": reranked}

print("체인 준비 완료")