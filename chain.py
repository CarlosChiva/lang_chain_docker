# Load
import os
import time
from urllib.error import HTTPError
import arxiv
from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

dirpath= "arxiv_papers"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)
search = arxiv.Search(
    query="LLM",
    max_results= 10,
    sort_by=arxiv.SortCriterion.LastUpdatedDate,
    sort_order=arxiv.SortOrder.Descending
)
for result in search.results():
    while True:
        try:
            result.download_pdf(dirpath=dirpath)
            print(f"-> Paper id {result.get_short_id()} with title '{result.title}'")    
            break
        except FileNotFoundError:
            print("File not found")
            break
        except HTTPError:
            print("Forbidden")
            break
        except ConnectionResetError:
            print("Connection reset by peer")
            time.sleep(5)    
papers=[]
loader= DirectoryLoader(dirpath,glob="./*.pdf",loader_cls=PyPDFLoader)
papers= loader.load()
print("Total number of pages loaded", len(papers))
full_text=''
for paper in papers:
    full_text=full_text+paper.page_content
full_text=" ".join(l for l in full_text.splitlines() if l)
    
# Split

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
paper_chunks= text_splitter.create_documents([full_text])
qdrant = Qdrant.from_documents(
    documents=paper_chunks,
    embedding=GPT4AllEmbeddings(),
    path="./tmp/local_qdrant",
    collection_name="arxiv_papers",
)
retriever= qdrant.as_retriever()


# Prompt
# Optionally, pull from the Hub
# from langchain import hub
# prompt = hub.pull("rlm/rag-prompt")
# Or, define your own:
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
# Select the LLM that you downloaded
ollama_llm = "llama2:7b-chat"
model = ChatOllama(model=ollama_llm)

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
