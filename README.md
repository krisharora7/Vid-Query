# Vid-Query🎬

Chat with long YouTube videos using their English subtitles — powered by a Retrieval-Augmented Generation (RAG) pipeline built with LangChain.

Instead of watching a 40-minute video to find one answer, paste the URL and just ask.

## How it works

1. **Transcript extraction** — pulls the English captions of a YouTube video using `youtube-transcript-api`.
2. **Chunking** — splits the transcript into overlapping chunks (`RecursiveCharacterTextSplitter`) so context isn't lost at boundaries.
3. **Embedding + Vector Store** — each chunk is embedded using a local HuggingFace model (`all-MiniLM-L6-v2`, free, no API key needed) and stored in a FAISS vector index.
4. **Retrieval** — on a question, the most relevant chunks are pulled from FAISS using similarity search.
5. **Augmentation + Generation** — the retrieved chunks are inserted into a strict prompt template and sent to a Groq-hosted LLM, which answers **only** from the transcript context (no hallucination, no outside knowledge).

Both a manual step-by-step pipeline and an LCEL (`RunnableParallel | prompt | llm | parser`) chain version are implemented.

## Tech stack

- **LangChain** — orchestration (prompts, retrievers, LCEL chains)
- **Groq** — LLM inference (`openai/gpt-oss-20b`)
- **HuggingFace Embeddings** — local, free embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **FAISS** — vector similarity search
- **youtube-transcript-api** — transcript extraction
- **Streamlit** — web UI

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

**Command line:**
```bash
python rag_project.py
```
Enter a YouTube URL (or video ID) and your question when prompted.

**Web UI:**
```bash
streamlit run app.py
```
Paste a video URL in the sidebar, click **Load video**, then ask questions in the chat box.

## Notes

- Only works on videos with English captions/subtitles available.
- Answers are strictly grounded in the video transcript — if the answer isn't in the video, the assistant says so instead of guessing.

