from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
import chromadb

app = FastAPI()

openai_lc_client = None

class APIKey(BaseModel):
    api_key: str

class Question(BaseModel):
    question: str

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_vectorstore(file_path):
    global openai_lc_client
    if file_path is not None:
        # Read the PDF file
        pdf_loader = PyPDFLoader(file_path)
        pages = pdf_loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings()
        new_client = chromadb.EphemeralClient()
        openai_lc_client = Chroma.from_documents(
            splits, embeddings, client=new_client, collection_name="openai_collection"
        )

        return "Data Successfully ingested"
    return "No file provided"

def doc_qa(question):
    retriever = openai_lc_client.as_retriever()
    prompt_rag = (
        PromptTemplate.from_template("Generate answers for given question: {question} based on the context {context}. In generated response provide reference to metadata from which context used in generating response")
    )
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_rag
        | llm
    )

    res = rag_chain.invoke(question)
    return str(res.content)



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    temp_file_path = tempfile.mkstemp(suffix='.pdf')[1]
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    try:
        result = create_vectorstore(temp_file_path)
        return JSONResponse(content={"message": result})
    finally:
        print(1)
        #os.remove(temp_file_path)

@app.post("/ask")
async def ask_question(question: Question):
    if not question.question:
        raise HTTPException(status_code=400, detail="No question provided")

    answer = doc_qa(question.question)
    return JSONResponse(content={"answer": answer})

@app.post("/set_api_key")
async def set_api_key(api_key: APIKey):
    if not api_key.api_key:
        raise HTTPException(status_code=400, detail="No API key provided")

    os.environ["OPENAI_API_KEY"] = api_key.api_key
    return JSONResponse(content={"message": "API key set successfully"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
