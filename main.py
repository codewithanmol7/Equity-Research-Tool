import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS


# ==============================
# LOAD ENVIRONMENT VARIABLES
# ==============================
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()


# ==============================
# UI
# ==============================
st.set_page_config(page_title="EquityBot", layout="wide")

st.title("EquityBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    urls.append(st.sidebar.text_input(f"URL {i+1}"))

process_clicked = st.sidebar.button("Process URLs")

VECTOR_DB_PATH = "faiss_index"


# ==============================
# LLM
# ==============================
llm = ChatOpenAI(
    temperature=0.7,
    max_tokens=500,
    model="gpt-4o-mini"  # stable + cheap
)


# ==============================
# PROCESS URLS
# ==============================
if process_clicked:

    valid_urls = [u for u in urls if u.strip()]

    if not valid_urls:
        st.error("Please enter at least one URL")
        st.stop()

    try:
        with st.spinner("Loading articles..."):
            loader = WebBaseLoader(valid_urls)
            data = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.split_documents(data)

        # ensure source metadata exists
        for doc in docs:
            doc.metadata["source"] = doc.metadata.get("source", "web")

        with st.spinner("Creating embeddings..."):
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(VECTOR_DB_PATH)

        st.success("Processing complete!")

    except Exception as e:
        st.error(f"Processing failed: {e}")
        st.stop()


# ==============================
# ASK QUESTION
# ==============================
query = st.text_input("Question:")

if query:

    if not os.path.exists(VECTOR_DB_PATH):
        st.error("Please process URLs first")
        st.stop()

    try:
        embeddings = OpenAIEmbeddings()

        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
        )

        with st.spinner("Generating answer..."):
            result = chain.invoke({"question": query})

        st.header("Answer")
        st.write(result.get("answer", "No answer generated"))

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources")
            for s in sources.split("\n"):
                if s.strip():
                    st.write(s)

    except Exception as e:
        st.error(f"Query failed: {e}")
