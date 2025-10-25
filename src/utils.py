from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
import traceback
import re

def filter_metadata(docs: List[Document])-> List[Document]:
    filtered_docs = []
    for doc in docs:
        source = doc.metadata["source"]
        page_label=doc.metadata["page_label"]
        filtered_docs.append(Document(page_content=doc.page_content,
                                      metadata={"source": source, "page_label": page_label}))
    return filtered_docs


def load_and_filter_pdfs(folder_name):
    try:
        loader = PyPDFDirectoryLoader(f"../{folder_name}/")
        docs = loader.load()
        if docs==[]:
            print("No documents found")
        return filter_metadata(docs)
    except Exception as e:
        print("An error occurred while loading the data!")
        traceback.print_exc()


def clean_text(text):
  text = text.replace('\n', ' ')
  text = re.sub(r'\s+', ' ', text)
  text = text.strip()
  return text

def initiate_embedding(texts="",model_name="all-MiniLM-L6-v2"):
  try:
    model = SentenceTransformer(model_name)
    print("Initiating embedding...")
    embeddings = model.encode(texts,show_progress_bar = True)
    print("Embedding completed successfully.")
    return embeddings
  except Exception as e:
    print("An error occurred:")
    traceback.print_exc()


