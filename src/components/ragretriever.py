from vectormanager import VectorDBManager
from src.utils import initiate_embedding
import traceback



class RAGretriever:
  def __init__(self,vector_store:VectorDBManager):
    self.vector_store = vector_store



  def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
    print(f"Retrieving documents for query: '{query}'")
    print(f"Top K: {top_k}, Score threshold: {score_threshold}")

    # Generate query embedding
    query_embeddings = initiate_embedding([query])[0]

    # Search in vector store
    try:
        results = self.vector_store.collection.query(
            query_embeddings=[query_embeddings],
            n_results=top_k
        )

        # Process results
        retrieved_docs = []

        if not results['documents']:
          print("No documents found")
        else:
            #print(results) [{'ids':[[id1,id2,idn]], 'documents':[[d1,d2..]],.}]
            print(results.keys())

            ids = results['ids'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                # Convert distance to similarity score (ChromaDB uses cosine distance, smaller=closer , similarity becomes bigger=closer)
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'distance': distance,
                        'similarity_rank': i + 1
                    })

            print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")

        return retrieved_docs

    except Exception as e:
      print(f"\nError occured while adding documents to vector database: \n{e}")
      traceback.print_exc()
      #raise