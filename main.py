import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

# =========================
# ENV
# =========================
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Add it in Streamlit secrets.")
    st.stop()

# =========================
# UI
# =========================
st.title("EquityBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

# =========================
# LLM
# =========================
llm = ChatOpenAI(
    temperature=0.5,
    max_tokens=500
)

# =========================
# PROCESS URLS
# =========================
if process_url_clicked:

    valid_urls = [u for u in urls if u.strip()]
    if not valid_urls:
        st.error("Enter at least one URL")
        st.stop()

    with st.spinner("Loading data..."):
        loader = WebBaseLoader(valid_urls)
        data = loader.load()

    if not data:
        st.error("No content loaded from URLs")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(data)

    # FIX: ensure metadata
    for doc in docs:
        doc.metadata["source"] = doc.metadata.get("source", "web")

    with st.spinner("Creating embeddings..."):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index")

    st.success("Processing complete!")

# =========================
# QUESTION
# =========================
query = st.text_input("Question:")

if query:

    if not os.path.exists("faiss_index"):
        st.error("Process URLs first")
        st.stop()

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    with st.spinner("Generating answer..."):
        try:
            result = chain.invoke({"question": query})
        except Exception as e:
            st.error(f"Model error: {str(e)}")
            st.stop()

    st.header("Answer")
    st.write(result.get("answer", "No answer found"))

    if result.get("sources"):
        st.subheader("Sources")
        for s in result["sources"].split("\n"):
            st.write(s)
