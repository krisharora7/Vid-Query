from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re
load_dotenv()
# step 1 - Indexing

## Document Ingestion
def extract_video_id(url_or_id: str) -> str:
    patterns = [
        r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id.strip()

def get_transcript(video_id: str) -> str:
    try:
        # transcript_list has text, start duration, end duration and we only need text so we will flatten it, it returns an object
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        # print(transcript_list)
        # print(type(transcript_list))

        #flatten to plain list
        transcript = " ".join(chunk.text for chunk in transcript_list)
        #print(transcript)
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(" No caption available for this video",e)
    except Exception as e:
        print("Error -> ",e)
    


## text splitting
def split_transcript(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([transcript])

## embedding and storing in vector store
def build_vector_store(chunks: list):
    embedd_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.from_documents(documents=chunks, embedding=embedd_model)

'''testing 'VECTOR STORE' by fetching ids of each chunk
transcript = get_transript(video_id)
chunks = split_transcript(transcript)
vector_store = build_vector_store(chunks)
print(vector_store.index_to_docstore_id)'''


# step-2 retrieval
def get_retriever(vector_store, k :int=4):
    return vector_store.as_retriever(search_type='similarity', search_kwargs={'k':k})

""" testing 'RETRIEVER'
transcript = get_transript(video_id)
chunks = split_transcript(transcript)
vector_store = build_vector_store(chunks)
retriever=get_retriever(vector_store)
print(retriever.invoke("Who is mark zuckerberg"))"""

# step-3 and 4 -> Augmentation and Generation
template = PromptTemplate.from_template(
        '''You are a helpful assistant helping a user understand a YouTube video transcript.

    Use only the transcript context below to answer the user's question.
    If the required answer is not in the transcript, say:
    "I don't know based on the provided Video."

    Rules:
    - Do not hallucinate
    - Do not use external knowledge
    - Keep the answer grounded in the transcript
    - Prefer short, direct answers
    - For summaries, keep it concise

    Transcript:
    {context}

    Question:
    {question}

    Answer:
    '''
)

def get_llm():
    return ChatGroq(model='openai/gpt-oss-20b')
def format_doc(retrieved_doc):
    return "\n\n".join(doc.page_content for doc in retrieved_doc)

''' Manual Pipeline ->'''
def get_answer_manually(retriever, question, llm):
    retrieved_docs = retriever.invoke(question)
    context = format_doc(retrieved_docs)
    prompt = template.invoke({'context':context, 'question':question})
    answer = llm.invoke(prompt)
    return answer.content

'''Using Chains'''
def get_answer(retriever, llm, question):
    parallel_chain = RunnableParallel({
        'question':RunnablePassthrough(),
        'context': retriever | RunnableLambda(format_doc)
    })
    sequential_chain = template | llm | StrOutputParser()
    main_chain = parallel_chain | sequential_chain | StrOutputParser()
    return main_chain.invoke(question)


if __name__ == "__main__":
    video_url_or_id = input("Enter a video url or id -> ").strip()
    question = input("Enter your query -> ")
    id = extract_video_id(video_url_or_id)
    transcript = get_transcript(id)
    if not transcript:
        print("no transcript found")
        exit()
    chunks = split_transcript(transcript)
    vector_store = build_vector_store(chunks)
    retriever = get_retriever(vector_store)
    llm = get_llm()
    ans = get_answer(retriever, llm, question)
    print(ans)
