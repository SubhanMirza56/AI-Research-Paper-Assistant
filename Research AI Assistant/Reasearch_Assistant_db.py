from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

file_path = "Adversarial-Attacks-and-Defenses-in-Deep-Learning_2020_Engineering.pdf"

file_load = PyPDFLoader(file_path=file_path, mode="page").load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
)

chunks = splitter.split_documents(file_load)

embedding_model = MistralAIEmbeddings()

# 4. Store chunks directly in Chroma
vectorstores = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="vector_db",
)

retriver = vectorstores.as_retriever()
docs = retriver.invoke("what is Definition and Notation ",k=2)
for d in docs:
    print(d)
