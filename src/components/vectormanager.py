import os
from chromadb import PersistentClient
from uuid import uuid4
import traceback

class VectorDBManager:
    def __init__(self, vectordb_name: str = "Medical-ChatBot", collection_name: str = "medibot-collection"):
        self.vectordb_name = vectordb_name
        self.collection_name = collection_name

        # Absolute path to project root's chromadb folder
        self.vectordb_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "chromadb", self.vectordb_name)
        )
        self.client = None
        self.collection = None
        self._initiate_vector_db()


    def _initiate_vector_db(self):
        try:
            print("Initiating vector database...")
            os.makedirs(self.vectordb_path, exist_ok=True)
            self.client = PersistentClient(self.vectordb_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "medical-chatbot collection"}
            )
            print(f"Vector database initiated successfully.\ndatabase path: {self.vectordb_path}\ncollection name: {self.collection_name}")
        except Exception as e:
            traceback.print_exc()
