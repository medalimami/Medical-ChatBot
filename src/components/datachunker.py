from semantic_chunkers import StatisticalChunker
from semantic_router.encoders import HuggingFaceEncoder
import traceback
from src.utils import clean_text

class StatisticalChunkerManager:
  def __init__(self):
    self.encoder = HuggingFaceEncoder()
    self.chunker = StatisticalChunker(encoder=self.encoder)


  def initiate_chunker(self,docs):
    try:
      print("Initiating chunking...")
      full_text = [clean_text(doc.page_content) for doc in docs]
      raw_chunks = self.chunker(docs=full_text)
      cleaned_chunks=[]
      for doc,chunks in zip(docs,raw_chunks):
        for chunk in chunks:
          chunk.metadata = {**doc.metadata}
          cleaned_chunks.append(chunk)
      print("Chunking completed successfully.")
      return cleaned_chunks
    except Exception as e:
      print("An error occurred while chunking the documents:")
      traceback.print_exc()

