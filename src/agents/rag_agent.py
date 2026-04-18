# type: ignore

import os
import chromadb
from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from dotenv import load_dotenv

load_dotenv()

llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
hf_token = os.getenv("HF_TOKEN")

# 1. Parse the pdf documents
parser = LlamaParse(
   api_key= llama_api_key,  
   result_type="markdown",
   verbose=True
)

sustainability_report = "microsoft_sustainability_report_2025.pdf"
extra_info = {"file_name": sustainability_report}

#Parse documents
parsed_documents = parser.load_data(sustainability_report, extra_info=extra_info)


# 2. Indexing and storing

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(parsed_documents)

db = chromadb.PersistentClient(path="./esg_audit_db")
chroma_collection = db.get_or_create_collection("ms_esg")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")


index = VectorStoreIndex.from_documents(
    parsed_documents, 
    storage_context=storage_context,
    embed_model = embed_model
)
print(index)
