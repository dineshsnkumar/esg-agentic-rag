# type: ignore
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_parse import LlamaParse

def ingest_pdf(file_name: str) -> list[Document]:
    llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

    parser = LlamaParse(
        api_key= llama_api_key,  
        result_type="markdown",
        verbose=True
    )
    sustainability_report = "microsoft_sustainability_report_2025.pdf"
    extra_info = {"file_name": sustainability_report}

    #Parse documents
    parsed_documents = parser.load_data(sustainability_report, extra_info=extra_info)
    return parsed_documents


def ingest_markdown() -> list[Document]:
    """
    Convert the markdown file to Documents
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "ms_sustainability_report_2025.md")
    reader = SimpleDirectoryReader(input_files=[file_path])
    return reader.load_data()