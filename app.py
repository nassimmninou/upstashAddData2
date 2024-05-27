import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from typing import List
from openai import OpenAI
from upstash_vector import Index
import httpx
import time

load_dotenv()

class UpstashVectorStore:
    def __init__(self, url: str, token: str):
        self.client = OpenAI()
        self.index = Index(url=url, token=token)

    def get_embeddings(self, documents: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """
        Given a list of documents, generates and returns a list of embeddings
        """
        documents = [document.replace("\n", " ") for document in documents]
        embeddings = self.client.embeddings.create(
            input=documents,
            model=model
        )
        return [data.embedding for data in embeddings.data]

    def add(self, ids: List[str], documents: List[str]) -> None:
        """
        Adds a list of documents to the Upstash Vector Store
        """
        embeddings = self.get_embeddings(documents)
        self.index.upsert(
            vectors=[
                (
                    id,
                    embedding,
                    {
                        "text": document,
                        "author": "Ramon RIOS",
                    }
                )
                for id, embedding, document
                in zip(ids, embeddings, documents)
            ]
        )

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def main():
    load_dotenv()
    st.set_page_config(page_title="Add data to OrgocatAI", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Add data to RiosAI")
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            
            # get Upstash credentials from environment variables
            upstash_url = os.environ.get("UPSTASH_VECTOR_REST_URL")
            upstash_token = os.environ.get("UPSTASH_VECTOR_REST_TOKEN")
            print(upstash_url)

            # create vector store
            vector_store = UpstashVectorStore(url=upstash_url, token=upstash_token)
            ids = [str(i) for i in range(len(text_chunks))]
            
            # Process in smaller batches to avoid timeouts
            batch_size = 10
            for i in range(0, len(text_chunks), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_chunks = text_chunks[i:i + batch_size]
                try:
                    vector_store.add(batch_ids, batch_chunks)
                except httpx.ReadTimeout:
                    st.error("Read timeout occurred. Retrying...")
                    time.sleep(5)
                    vector_store.add(batch_ids, batch_chunks)

if __name__ == '__main__':
    main()
