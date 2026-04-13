# type: ignore

import os
from llama_parse import LlamaParse
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

# Saving the parsed documents
with open("output.md", "w", encoding="utf-8") as f:
   for doc in parsed_documents:
       f.write(doc.text)