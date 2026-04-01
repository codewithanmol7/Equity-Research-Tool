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
# CHECK API KEY
# ======================
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()

# ======================
# UI
# ======================
st.title("IngestIQ 📈")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

faiss_store_path = "faiss_store_openai"
main_placeholder = st.empty()

# ======================
# LLM
# ======================
llm = OpenAI(temperature=0.9, max_tokens=500)

# ======================
# PROCESS URLS
# ======================
if process_url_clicked:

    urls = [u.strip() for u in urls if u.strip()]
    if not urls:
        st.error("Please enter at least one valid URL")
        st.stop()

    main_placeholder.text("Data Loading...Started...")

    try:
        loader = UnstructuredURLLoader(
            urls=urls,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading URLs: {str(e)}")
        st.stop()

    # ======================
    # VALIDATION
    # ======================
    if not data:
        st.error("No data loaded. URLs might be blocked on Streamlit.")
        st.stop()

    # ======================
    # SPLITTING
    # ======================
    main_placeholder.text("Text Splitter...Started...")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(data)

    # remove empty docs
    docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]

    if len(docs) == 0:
        st.error("No valid content after splitting.")
        st.stop()

    st.write(f"Documents created: {len(docs)}")

    # ======================
    # EMBEDDINGS + FAISS
    # ======================
    main_placeholder.text("Embedding Vector Building...")

    try:
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Embedding/FAISS error: {str(e)}")
        st.stop()

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

    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            faiss_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        st.stop()

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    with st.spinner("Generating answer..."):
        try:
            result = chain({"question": query}, return_only_outputs=True)
        except Exception as e:
            st.error(f"Model error: {str(e)}")
            st.stop()

    st.header("Answer")
    st.write(result.get("answer", "No answer found"))

    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        for source in sources.split("\n"):
            st.write(source)
