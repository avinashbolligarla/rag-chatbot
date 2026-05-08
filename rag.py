from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def build_rag(pdf_path):
    print("Loading document...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print("Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("Setting up chain...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
Context: {context}
Question: {question}
""")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Ready! Ask your questions. Type 'exit' to quit.\n")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = chain.invoke(question)
        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    build_rag("sample.pdf")
