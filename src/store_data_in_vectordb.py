from utils import *
from components.datachunker import StatisticalChunkerManager
from components.vectormanager import VectorDBManager


# load the pdfs and filter the metadata
docs = load_and_filter_pdfs("data")

# import StatisticalChunkerManager class and chunk the documents
chunker = StatisticalChunkerManager()
chunks = chunker.initiate_chunker(docs)

# make every chunk a full paragraph instead of separate sentences
chunks_text = [' '.join(chunk.splits) for chunk in chunks]
embedded_chunks = initiate_embedding(texts=chunks_text)

# import the VectorDBManager class to create and add the documents to the vectordb
vectordbmanager = VectorDBManager()
vectordbmanager.add_documents_to_collection(chunks,embedded_chunks)