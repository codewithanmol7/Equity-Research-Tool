import os
import streamlit as st
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

st.title("EquityBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    urls.append(st.sidebar.text_input(f"URL {i+1}"))

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

llm = ChatOpenAI(temperature=0.7, max_tokens=500)

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

        # IMPORTANT FIX
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
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("Sources")
            for s in result["sources"].split("\n"):
                st.write(s)
