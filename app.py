import time
import tempfile

import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq

from rank_bm25 import BM25Okapi


# -------------------------
# LOAD ENV
# -------------------------

load_dotenv()

# -------------------------
# GROQ LLM
# -------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# -------------------------
# STREAMLIT PAGE
# -------------------------

st.set_page_config(
    page_title="Advanced AI Assistant",
    layout="wide"
)

st.title("📄 Advanced Vector-less AI Assistant")

st.write("Multi-Document AI Assistant using BM25 + Groq")

# -------------------------
# SESSION STATE
# -------------------------

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

if "cache" not in st.session_state:
    st.session_state.cache = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# FILE UPLOAD
# -------------------------

with st.sidebar:

    st.header("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    st.markdown("---")

    st.subheader("📄 Uploaded Files")

    uploaded_sources = {
        chunk.metadata.get("source")
        for chunk in st.session_state.all_chunks
    }

    if uploaded_sources:

        for file in uploaded_sources:
            st.success(file)

    else:
        st.info("No files uploaded yet")

# -------------------------
# PROCESS DOCUMENTS
# -------------------------

if uploaded_files:

    existing_sources = {
        chunk.metadata.get("source")
        for chunk in st.session_state.all_chunks
    }

    for uploaded_file in uploaded_files:

        # avoid duplicate upload
        if uploaded_file.name in existing_sources:
            continue

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())

            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        for chunk in chunks:

            chunk.metadata["source"] = uploaded_file.name

            st.session_state.all_chunks.append(chunk)

    st.success("Documents processed successfully!")

# -------------------------
# BM25 INDEX
# -------------------------

bm25 = None

if st.session_state.all_chunks:

    chunk_texts = [
        chunk.page_content
        for chunk in st.session_state.all_chunks
    ]

    tokenized_chunks = [
        text.lower().split()
        for text in chunk_texts
    ]

    bm25 = BM25Okapi(tokenized_chunks)

# -------------------------
# DISPLAY OLD CHAT
# -------------------------

for chat in st.session_state.chat_history:

    with st.chat_message(chat["role"]):

        st.write(chat["message"])

        # assistant metadata
        if chat["role"] == "assistant":

            st.markdown("---")

            st.markdown(
                f"⏱️ **Response Time:** `{chat['response_time']} sec`"
            )

            if (
                "information is not available"
                not in chat["message"].lower()
                and "answer not found"
                not in chat["message"].lower()
                and chat["citations"]
            ):

                sources_text = " | ".join(
                    chat["citations"]
                )

                st.markdown(
                    f"📚 **Sources:** {sources_text}"
                )

# -------------------------
# USER QUESTION
# -------------------------

query = st.chat_input("Ask a question...")

if query and st.session_state.all_chunks:

    # -------------------------
    # SAVE USER MESSAGE
    # -------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "message": query
        }
    )

    st.chat_message("user").write(query)

    normalized_query = query.lower().strip()

    # -------------------------
    # CACHE CHECK
    # -------------------------

    if normalized_query in st.session_state.cache:

        # -------------------------
        # CACHE TIMER
        # -------------------------

        cache_start = time.time()

        cached_data = st.session_state.cache[
            normalized_query
        ]

        answer = cached_data["answer"]

        citations = cached_data["citations"]

        # ultra fast cache response
        response_time = round(
            time.time() - cache_start,
            4
        )

        with st.chat_message("assistant"):

            st.write(answer)

            st.markdown("---")

            st.markdown(
                f"⏱️ **Response Time:** `{response_time} sec`"
            )

            # -------------------------
            # SHOW SOURCES ONLY
            # IF ANSWER EXISTS
            # -------------------------

            if (
                "information is not available"
                not in answer.lower()
                and "answer not found"
                not in answer.lower()
                and citations
            ):

                sources_text = " | ".join(citations)

                st.markdown(
                    f"📚 **Sources:** {sources_text}"
                )

            st.success("⚡ Response fetched from cache")

        # -------------------------
        # SAVE CHAT
        # -------------------------

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "message": answer,
                "response_time": response_time,
                "citations": citations,
                "cached": True
            }
        )

    else:

        start_time = time.time()

        # -------------------------
        # BM25 RETRIEVAL
        # -------------------------

        tokenized_query = normalized_query.split()

        scores = bm25.get_scores(
            tokenized_query
        )

        scored_chunks = list(
            zip(
                st.session_state.all_chunks,
                scores
            )
        )

        scored_chunks = sorted(
            scored_chunks,
            key=lambda x: x[1],
            reverse=True
        )

        # -------------------------
        # FILTER RELEVANT CHUNKS
        # -------------------------

        filtered_chunks = [
            (chunk, score)
            for chunk, score in scored_chunks
            if score > 0
        ]

        top_chunks = filtered_chunks[:5]

        # -------------------------
        # IF NOTHING RELEVANT
        # -------------------------

        if not top_chunks:

            answer = (
                "The information is not available "
                "in the uploaded documents."
            )

            response_time = round(
                time.time() - start_time,
                2
            )

            citations = []

            with st.chat_message("assistant"):

                st.write(answer)

                st.markdown("---")

                st.markdown(
                    f"⏱️ **Response Time:** `{response_time} sec`"
                )

            # -------------------------
            # SAVE CHAT
            # -------------------------

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "message": answer,
                    "response_time": response_time,
                    "citations": [],
                    "cached": False
                }
            )

        else:

            # -------------------------
            # CONTEXT
            # -------------------------

            context = "\n\n".join(
                [
                    chunk.page_content
                    for chunk, score in top_chunks
                ]
            )

            # -------------------------
            # PROMPT
            # -------------------------

            prompt = f"""
            You are a strict document question-answering assistant.

            Answer ONLY using the provided context.

            Do NOT use your own knowledge.

            If the answer is not explicitly present
            in the context, reply exactly with:

            'The information is not available in the uploaded documents.'

            Context:
            {context}

            Question:
            {query}
            """

            # -------------------------
            # LLM RESPONSE
            # -------------------------

            response = llm.invoke(prompt)

            answer = response.content

            end_time = time.time()

            response_time = round(
                end_time - start_time,
                2
            )

            # -------------------------
            # SOURCES
            # -------------------------

            shown = set()

            citations = []

            for chunk, score in top_chunks:

                source = chunk.metadata.get(
                    "source"
                )

                page = chunk.metadata.get(
                    "page"
                )

                citation = (
                    f"{source} - Page {page}"
                )

                if citation not in shown:

                    shown.add(citation)

                    citations.append(citation)

            # -------------------------
            # SHOW RESPONSE
            # -------------------------

            with st.chat_message("assistant"):

                st.write(answer)

                st.markdown("---")

                st.markdown(
                    f"⏱️ **Response Time:** `{response_time} sec`"
                )

                if (
                    "information is not available"
                    not in answer.lower()
                    and "answer not found"
                    not in answer.lower()
                    and citations
                ):

                    sources_text = " | ".join(citations)

                    st.markdown(
                        f"📚 **Sources:** {sources_text}"
                    )

            # -------------------------
            # SAVE CHAT
            # -------------------------

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "message": answer,
                    "response_time": response_time,
                    "citations": citations,
                    "cached": False
                }
            )

            # -------------------------
            # SAVE CACHE
            # -------------------------

            st.session_state.cache[
                normalized_query
            ] = {
                "answer": answer,
                "citations": citations
            }