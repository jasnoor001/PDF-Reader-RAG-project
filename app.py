import os
import tempfile

import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PERSIST_DIRECTORY = "chroma-db"

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")

st.write("Upload a PDF and ask questions about it.")

# -----------------------------
# Upload PDF
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)

# -----------------------------
# Create Database
# -----------------------------

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("Reading PDF..."):

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        embeddings = MistralAIEmbeddings()

        # Remove old DB (optional)
        if os.path.exists(PERSIST_DIRECTORY):
            import shutil
            shutil.rmtree(PERSIST_DIRECTORY)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )

    st.success("PDF processed successfully!")

# -----------------------------
# Ask Question
# -----------------------------

if os.path.exists(PERSIST_DIRECTORY):

    question = st.text_input("Ask a question")

    if st.button("Submit"):

        embeddings = MistralAIEmbeddings()

        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":4,
                "fetch_k":10,
                "lambda_mult":0.5
            }
        )

        llm = ChatMistralAI(
            model="mistral-small-2603"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful AI assistant.

                    Use ONLY the provided context.

                    If the answer is not present in the context,
                    say:
                    I could not find the answer in the document.
                    """
                ),
                (
                    "human",
                    """
                    Context:
                    {context}

                    Question:
                    {question}
                    """
                )
            ]
        )

        with st.spinner("Searching..."):

            docs = retriever.invoke(question)

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

            response = llm.invoke(final_prompt)

        st.subheader("Answer")

        st.write(response.content)

        with st.expander("Retrieved Chunks"):

            for i, doc in enumerate(docs, start=1):
                st.markdown(f"### Chunk {i}")
                st.write(doc.page_content)