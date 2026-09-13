"""
app.py

Basic Streamlit UI for the YouTube RAG chatbot.
Run locally with: streamlit run app.py
"""

import streamlit as st
from rag_project import (
    extract_video_id,
    get_transcript,
    split_transcript,
    build_vector_store,
    get_retriever,
    get_llm,
    get_answer,
)

st.set_page_config(page_title="Vid-Query", page_icon="🎬")
st.title("🎬 YouTube Video Chatbot")
st.caption("Ask questions about any YouTube video using its transcript (RAG + LangChain + Groq)")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Load a video")
    url_or_id = st.text_input("YouTube URL or Video ID")
    load_btn = st.button("Load video", use_container_width=True)

    if load_btn and url_or_id:
        try:
            video_id = extract_video_id(url_or_id)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Fetching transcript..."):
            try:
                transcript = get_transcript(video_id)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

        if not transcript:
            st.error("No transcript found for this video.")
            st.stop()

        with st.spinner("Splitting and embedding transcript..."):
            chunks = split_transcript(transcript)
            vector_store = build_vector_store(chunks)
            retriever = get_retriever(vector_store)
            llm = get_llm()

        st.session_state.retriever = retriever
        st.session_state.llm = llm
        st.session_state.video_loaded = video_id
        st.session_state.messages = []
        st.success(f"Video loaded: {video_id} ({len(chunks)} chunks)")

    if st.session_state.video_loaded:
        st.info(f"Currently loaded: {st.session_state.video_loaded}")

if not st.session_state.retriever or not st.session_state.llm:
    st.info("Load a video from the sidebar to start chatting.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something about the video...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(
                    st.session_state.retriever,
                    st.session_state.llm,
                    question,
                )
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
