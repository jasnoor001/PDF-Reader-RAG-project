# load pdf
# create chunks
# create embeddings
# store into vector db(chroma)



from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()


loader=PyPDFLoader("document loaders/deeplearning.pdf")
docs=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks=splitter.split_documents(docs)

model=MistralAIEmbeddings()

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=model,
    persist_directory="chroma-db"

)