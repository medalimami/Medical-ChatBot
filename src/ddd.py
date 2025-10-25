from src.components.vectormanager import VectorDBManager
from src.components.ragretriever import RAGretriever
from src.components.llmgenerator import LLMGenerator

# Initialize your vector database
vectordb = VectorDBManager()
retriever = RAGretriever(vectordb)
llm_generator = LLMGenerator()

# Test query
query = "what is acne?"
print(f"Testing query: {query}")

results = llm_generator.llm_generate(query, retriever, top_k=5, threshold=0.0)

print("\nResults:")
print("Answer:", results['answer'])
print("Sources:", results['sources'])
print("Confidence:", results['confidence'])
print("Context:", results['context'][:500], "...")  # print only first 500 chars of context
print("Documents in collection:", len(vectordb.collection.get()))


