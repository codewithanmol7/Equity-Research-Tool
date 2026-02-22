import os
import streamlit as st
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# UI
# =========================
st.set_page_config(page_title="EquityBot", layout="wide")

st.title("EquityBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    urls.append(st.sidebar.text_input(f"URL {i+1}"))

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# =========================
# CHECK API KEY
# =========================
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

# =========================
# LLM
# =========================
llm = ChatOpenAI(
    temperature=0.7,
    max_tokens=500
)

# =========================
# PROCESS URLS
# =========================
if process_url_clicked:

    valid_urls = [u for u in urls if u.strip()]

    if not valid_urls:
        st.error("Enter at least one URL")
    else:
        loader = WebBaseLoader(valid_urls)

        main_placeholder.text("Loading data...")
        data = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.split_documents(data)

        # ensure metadata exists
        for doc in docs:
            if "source" not in doc.metadata:
                doc.metadata["source"] = "web"

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index")

        st.success("Processing complete!")

# =========================
# ASK QUESTION
# =========================
query = st.text_input("Question:")

if query:

    if not os.path.exists("faiss_index"):
        st.error("Process URLs first")
    else:
        embeddings = OpenAIEmbeddings()

        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        result = chain.invoke({"question": query})

        st.header("Answer")
        st.write(result.get("answer", "No answer found"))

        sources = result.get("sources")
        if sources:
            st.subheader("Sources")
            for s in sources.split("\n"):
                st.write(s)
