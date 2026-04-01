import os
import streamlit as st
import time

from langchain_openai import OpenAI
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

# ======================
# CHECK API KEY (FROM .env)
# ======================
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()

# ======================
# UI
# ======================
st.title("IngestIQ 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
faiss_store_path = "faiss_store_openai"

main_placeholder = st.empty()

# ======================
# LLM (UNCHANGED)
# ======================
llm = OpenAI(temperature=0.9, max_tokens=500)

# ======================
# PROCESS URLS
# ======================
if process_url_clicked:

    urls = [u for u in urls if u.strip()]
    if not urls:
        st.error("Please enter at least one valid URL")
        st.stop()

    # FIX: add headers to avoid blocking
    loader = UnstructuredURLLoader(
        urls=urls,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    main_placeholder.text("Data Loading...Started...")

    data = loader.load()

    # ======================
    # VALIDATION
    # ======================
    if not data:
        st.error("No data loaded. URLs might be blocked on Streamlit.")
        st.stop()

    # ======================
    # SPLITTING
    # ======================
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    main_placeholder.text("Text Splitter...Started...")

    docs = text_splitter.split_documents(data)

    # remove empty docs
    docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]

    if not docs:
        st.error("No valid content after splitting.")
        st.stop()

    st.write(f"Documents loaded: {len(docs)}")

    # ======================
    # EMBEDDINGS + FAISS
    # ======================
    embeddings = OpenAIEmbeddings()

    main_placeholder.text("Embedding Vector Started Building...")

    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    vectorstore_openai.save_local(faiss_store_path)

    time.sleep(1)

    st.success("Processing complete!")

# ======================
# QUERY
# ======================
query = st.text_input("Question:")

if query:

    if not os.path.exists(faiss_store_path):
        st.error("Please process URLs first")
        st.stop()

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        faiss_store_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    with st.spinner("Generating answer..."):
        result = chain({"question": query}, return_only_outputs=True)

    st.header("Answer")
    st.write(result.get("answer", "No answer found"))

    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        for source in sources.split("\n"):
            st.write(source)
