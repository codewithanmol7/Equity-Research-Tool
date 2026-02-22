import os
import streamlit as st
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS


# =========================
# UI
# =========================
st.title("EquityBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()


# =========================
# LLM
# =========================
llm = ChatOpenAI(temperature=0.9, max_tokens=500)


# =========================
# PROCESS URLS
# =========================
if process_url_clicked:

    valid_urls = [u for u in urls if u.strip() != ""]

    if len(valid_urls) == 0:
        st.error("Please enter at least one valid URL")

    else:
        loader = WebBaseLoader(valid_urls)

        main_placeholder.text("Data Loading...Started...✅")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        main_placeholder.text("Text Splitting...Started...✅")
        docs = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        main_placeholder.text("Saving FAISS index...✅")
        vectorstore.save_local("faiss_index")

        time.sleep(1)
        st.success("Processing complete!")


# =========================
# QUESTION
# =========================
query = st.text_input("Question:")

if query:

    if not os.path.exists("faiss_index"):
        st.error("Please process URLs first")

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

        # ----- Answer -----
        st.header("Answer")
        st.write(result.get("answer"))

        # ----- Sources -----
        sources = result.get("sources")
        if sources:
            st.subheader("Sources")
            for s in sources.split("\n"):
                if s.strip():
                    st.write(s)
