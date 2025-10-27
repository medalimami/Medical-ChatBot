import os
from chromadb import PersistentClient
from uuid import uuid4
import traceback

class VectorDBManager:
  def __init__(self,vectordb_name:str="Medical-ChatBot",collection_name:str="medibot-collection"):
    self.vectordb_name = vectordb_name
    self.collection_name = collection_name
    self.vectordb_path=os.path.join("..","chromadb",self.vectordb_name)
    self.client=None
    self.collection=None
    self._initiate_vector_db()

  def _initiate_vector_db(self):
    try:
      print("Initiating vector database...")
      os.makedirs(self.vectordb_path,exist_ok=True)
      self.client = PersistentClient(self.vectordb_path)
      self.collection=self.client.get_or_create_collection(
          name=self.collection_name,
          metadata={"description": "medical-chatbot collection"}
      )
      print(f"Vector database initiated successfully.\ndatabase path:{self.vectordb_path}\ncollection name: {self.collection_name}")
    except Exception as e:
      traceback.print_exc()


  def add_documents_to_collection(self,chunks,embedded_chunks):
    try:
      print("Adding documents to vector database...")

      ids=[]
      documents = []
      embeddings = []
      metadatas=[]

      for i,(chunk,emb_chunk) in enumerate(zip(chunks,embedded_chunks)):
        ids.append(f"doc_{uuid4().hex[:8]}_{i}")
        documents.append(' '.join(chunk.splits))
        embeddings.append(emb_chunk)
        metadatas.append(chunk.metadata)

      self.collection.add(
          ids=ids,
          documents=documents,
          embeddings=embeddings,
          metadatas=metadatas
      )
      print("Documents added to vector database successfully.")
    except Exception as E:
      print(f"\nError occured while adding documents to vector database: \n{E}")
      traceback.print_exc()
