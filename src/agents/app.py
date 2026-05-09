import os
from llm import load_gemini_model
from ingestion.ingest_data import ingest_markdown

#load llm 
llm = load_gemini_model()

#Ingest data
documents = ingest_markdown()
