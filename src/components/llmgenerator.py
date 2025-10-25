from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
from src.utils import initiate_embedding


class LLMGenerator:
    def __init__(self,model_name="llama-3.1-8b-instant",temperature=0.2,max_tokens=1024):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None

    def llm_generate(self, query, retriever, top_k=5, threshold=0.2):
        load_dotenv()
        groq_api_key = os.environ.get("GROQ_API_KEY")
        self.llm = ChatGroq(
            model=self.model_name,
            api_key=groq_api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        results = retriever.retrieve(query, top_k=top_k, score_threshold=threshold)

        if not results:  # fallback to normal LLM response
            prompt = f"{query}"
            message = [
                SystemMessage(content="You are a helpful medical assistant. Answer the question concisely."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(message)
            output= {
                'answer': response.content,
                'sources': [],
                'confidence': 0.0,
                'context': ''
            }
            return output

        # RAG logic if documents are found
        context = "\n\n".join([result['content'] for result in results])
        sources = [{
            'source': result['metadata']['source'],
            'page_number': result['metadata']['page_number']
        } for result in results]
        confidence = max([result['similarity_score'] for result in results])

        prompt = f"""Use the following context to answer the question concisely. Only include the direct answer — do not add explanations or extra commentary.
    Context: {context}
    Question: {query}
    Answer: """

        message = [
            SystemMessage(
                content="You are a medical assistant. Be concise and only answer the question based on the given context."),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(message)

        output= {
            'answer': response.content,
            'sources': sources,
            'confidence': confidence,
            'context': context
        }
        return output
