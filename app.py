import hashlib
import os
import re
import tempfile
import uuid
from typing import Iterable

import numpy as np
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_TITLE = "🎉 El7amalawy 🎉"
APP_VERSION = "V1.2"
DEFAULT_SYSTEM_PROMPT = (
    "You are El7amalawy, a helpful RAG assistant. Answer only from the retrieved PDF "
    "context when possible. If the answer is not in the retrieved context, say that "
    "you do not have enough information from the uploaded PDFs. Keep answers clear and concise."
)
FREE_GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]
RETRIEVAL_K = 3


class LocalHashEmbeddings(Embeddings):
    """A lightweight local embedding fallback so Groq can be used only for generation."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=float)
        for token in self._tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
            vector[index] += sign

        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("doc_stats", None)
    st.session_state.setdefault("processed_files", [])
    st.session_state.setdefault("active_collection", None)


def reset_chat() -> None:
    st.session_state.messages = []


def reset_knowledge_base() -> None:
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.doc_stats = None
    st.session_state.processed_files = []
    st.session_state.active_collection = None


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def build_prompt(question: str, docs_with_scores: list[tuple[Document, float]]) -> str:
    context_blocks = []
    for index, (doc, score) in enumerate(docs_with_scores, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page_label", doc.metadata.get("page", "N/A"))
        similarity = clamp_score(score) * 100
        context_blocks.append(
            f"[Source {index}] File: {source} | Page: {page} | Similarity: {similarity:.2f}%\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_blocks)
    return (
        "Answer the user's question using the retrieved PDF context below.\n"
        "If the answer is not supported by the context, say clearly that you do not have "
        "enough information from the uploaded PDFs.\n"
        "When you answer, mention the source numbers you relied on when helpful.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}"
    )


def process_uploaded_files(uploaded_files: Iterable) -> tuple[Chroma, dict]:
    uploaded_files = list(uploaded_files)
    all_pages: list[Document] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = uploaded_file.name
                page.metadata["page_label"] = page.metadata.get("page", 0) + 1
            all_pages.extend(pages)

    if not all_pages:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_pages)

    collection_name = f"el7amalawy_{uuid.uuid4().hex}"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=LocalHashEmbeddings(),
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )

    stats = {
        "files": len(uploaded_files),
        "pages": len(all_pages),
        "chunks": len(chunks),
        "collection_name": collection_name,
    }
    return vectorstore, stats


def render_sources(sources: list[dict]) -> None:
    with st.expander("Top 3 Retrieved Chunks"):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**{index}. {source['source']}** | Page {source['page']} | "
                f"Similarity: `{source['similarity']:.2f}%`"
            )
            st.code(source["excerpt"], language="text")


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} {APP_VERSION}",
        page_icon="🎉",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    st.title(APP_TITLE)
    st.caption(f"{APP_VERSION} | Streamlit + Groq + Day4-style RAG over multiple PDFs")

    with st.sidebar:
        st.header("Settings")
        default_key = os.environ.get("GROQ_API_KEY", "")
        api_key = st.text_input(
            "Groq API Key",
            value=default_key,
            type="password",
            help="Paste your Groq API key here.",
        )
        st.markdown("[Get a Groq API key](https://console.groq.com/keys)")

        model_name = st.selectbox(
            "Free Groq Model",
            FREE_GROQ_MODELS,
            index=0,
            help="These are current Groq model IDs commonly available for free-tier experimentation.",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        system_prompt = st.text_area(
            "System Prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            height=120,
        )
        st.caption(f"Retrieval is fixed to K = {RETRIEVAL_K}.")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDFs, then click Process PDFs.",
        )

        if st.button("Process PDFs", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload at least one PDF first.")
            else:
                with st.spinner("Processing PDFs and building the knowledge base..."):
                    try:
                        vectorstore, stats = process_uploaded_files(uploaded_files)
                    except Exception as exc:
                        st.error(f"Processing failed: {exc}")
                    else:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.doc_stats = stats
                        st.session_state.processed_files = [file.name for file in uploaded_files]
                        st.session_state.active_collection = stats["collection_name"]
                        st.success(
                            f"Ready: {stats['files']} file(s), {stats['pages']} page(s), "
                            f"{stats['chunks']} chunk(s)."
                        )

        if st.button("Clear Chat", use_container_width=True):
            reset_chat()
            st.rerun()

        if st.button("Clear PDFs + Chat", use_container_width=True):
            reset_knowledge_base()
            st.rerun()

        if st.session_state.doc_stats:
            stats = st.session_state.doc_stats
            st.divider()
            st.markdown("**Knowledge Base**")
            st.write(f"Files: {stats['files']}")
            st.write(f"Pages: {stats['pages']}")
            st.write(f"Chunks: {stats['chunks']}")

    if st.session_state.processed_files:
        st.info("Loaded PDFs: " + ", ".join(st.session_state.processed_files))
    else:
        st.info("Upload PDFs from the sidebar, then click Process PDFs to start.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    prompt = st.chat_input("Ask something about your uploaded PDFs...")
    if not prompt:
        return

    if not api_key:
        st.warning("Enter your Groq API key in the sidebar before chatting.")
        return

    if st.session_state.vectorstore is None:
        st.warning("Process your PDFs first so I can search them.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    docs_with_scores = st.session_state.vectorstore.similarity_search_with_relevance_scores(
        prompt,
        k=RETRIEVAL_K,
    )
    prompt_with_context = build_prompt(prompt, docs_with_scores)

    sources = []
    for doc, score in docs_with_scores:
        page = doc.metadata.get("page_label", doc.metadata.get("page", "N/A"))
        sources.append(
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": page,
                "similarity": clamp_score(score) * 100,
                "excerpt": doc.page_content[:700].strip(),
            }
        )

    with st.chat_message("assistant"):
        try:
            client = Groq(api_key=api_key)
            messages = [{"role": "system", "content": system_prompt}]
            for message in st.session_state.messages[:-1]:
                messages.append({"role": message["role"], "content": message["content"]})
            messages.append({"role": "user", "content": prompt_with_context})

            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            answer = st.write_stream(
                chunk.choices[0].delta.content or "" for chunk in stream
            )
            render_sources(sources)
        except Exception as exc:
            answer = f"Error while calling Groq: {exc}"
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


if __name__ == "__main__":
    main()
