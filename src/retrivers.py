from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
df =  pd.read_excel(DOCS_DIR / "Tickets.xlsx")
from langchain_core.documents import Document

BASE_DIR = Path().resolve().parent
DOCS_DIR = BASE_DIR / "docs"

import os
os.environ["TRANSFORMERS_NO_LAZY_IMPORT"] = "1"
import transformers.models.bert

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_pdf_vectorstore(filepath: str, save_path: str):
    loader = PyPDFLoader(DOCS_DIR / filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    documents = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents, embedding)    
    vectorstore.save_local(f'vectostores/{save_path}')
    retriver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 7}) 
    return retriver


def load_excel_vectorstore(filepath: str, save_path: str):
    df =  pd.read_excel(DOCS_DIR / filepath)
    documents = []
    for idx, row in df.iterrows():
        text  = ' '.join([str(cell) for cell in row if pd.notna(cell)])
        documents.append(Document(page_content=text, metadata={'row': idx}))
      
    vectorstore_tickets = FAISS.from_documents(documents, embedding)    
    vectorstore_tickets.save_local('vectostores/vectorstore_tickets')
    retriver_tickets = vectorstore_tickets.as_retriever(search_type="similarity", search_kwargs={"k": 7})
    
    return retriver_tickets

retriver_perguntas_frequentes = load_pdf_vectorstore("Perguntas Frequentes.pdf", "vectorstore_perguntas_frequentes")
retriver_manual_tecnico = load_pdf_vectorstore("Manual Tecnico de Produtos.pdf", "vectorstore_manual_tecnico_produtos")
retriver_politicas_procedimentos = load_pdf_vectorstore("Politicas e Procedimentos.pdf", "vectorstore_politica_procedimentos")   

retriver_tickets = load_excel_vectorstore("Tickets.xlsx", "vectorstore_tickets")

