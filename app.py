from flask import Flask, render_template, jsonify, request
from src.utils import initiate_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from src.components.ragretriever import RAGretriever
from src.components.vectormanager import VectorDBManager
from src.components.llmgenerator import LLMGenerator

vectorstore = VectorDBManager()
retriever = RAGretriever(vectorstore)
llm_generator = LLMGenerator(model_name="llama-3.1-8b-instant",temperature=0.2,max_tokens=1024)


app = Flask(__name__)




from flask import jsonify

@app.route('/get', methods=['POST'])
def get():
    msg = request.form.get("msg", "")
    results = llm_generator.llm_generate(msg, retriever, top_k=3, threshold=0.0)
    return jsonify(results)









@app.route('/')
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)