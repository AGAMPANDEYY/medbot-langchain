import os
import time
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from tqdm import tqdm
from operator import itemgetter





'''
Draft 1 of the modular codes to be run in this .py file not for deployment--- refer rag_pipelin.py for that

'''

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "llama3.2:1b"
INDEX_NAME = "medbot-index"
index_name="medbot-index"
DIMENSION = 2048
METRIC = "cosine"

# Initialize Pinecone
def initialize_pinecone(api_key, index_name=INDEX_NAME):
    pc = Pinecone(api_key=api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    return pc.Index(index_name)

# Initialize model and embeddings
def initialize_model_and_embeddings(model_name=MODEL_NAME):
    model = Ollama(model=model_name)
    embeddings = OllamaEmbeddings(model=model_name)
    parser = StrOutputParser()
    return model, embeddings, parser

# Load and process documents
def document_chunking(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)

# Populate Pinecone with embeddings
def populate_pinecone(index, documents, embeddings, batch_size=1):
    ids = [f"page_{i}" for i in range(len(documents))]
    for i in tqdm(range(0, len(documents), batch_size), desc="Adding documents to Pinecone"):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        index.add_documents(documents=batch_docs, ids=batch_ids)

# Main query function
def query_medbot(question,chain ,model, parser, prompt, retriever):
    context = retriever.invoke(question)
    response = chain.invoke({"context": context, "question": question})
    return  response

# Setup and run the RAG system
def run_medbot(pdf_path, question):

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Pinecone API key is missing!")
    
    # Initialize Pinecone, model, embeddings, and vector store
    index = initialize_pinecone(pinecone_api_key)
    model, embeddings, parser = initialize_model_and_embeddings()
    
    # Load and split documents, then populate Pinecone if it's empty
    documents = document_chunking(pdf_path)

    pc_vectordb = PineconeVectorStore(index=index, embedding=embeddings)
    '''
    if not pc_vectordb.count_documents():
        populate_pinecone(index, documents, embeddings) # Assuming the pinecone VectorDB is populated with chunks
    '''

    # Set up the prompt and retriever
    prompt_template='''
        You are a highly qualified doctor specializing in gynecology. I will provide you with an excerpt from a gynecology resource and a question related to it.
        Your task is to provide an accurate, concise answer based on the context provided and your own medical knowledge. 

        - If the provided context is clear and relevant, prioritize it in your response.
        - If the context is incomplete or unclear, supplement your answer with reliable information from your medical knowledge.
        - If you don’t know the answer, simply respond with "I am not certain" and offer guidance for the next steps the user might take.

        Context:
        {context}

        Question:
        {question}

        Answer:  
        '''
    prompt = PromptTemplate.from_template(prompt_template)
    retriever = pc_vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.5})
    # Run query

    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | parser
    )
    return query_medbot(question,chain,model, parser, prompt, retriever)

# Example usage
if __name__ == "__main__":
    pdf_path = r"C:\Agam\Work\medbot\dataset\Gynaecology-DC-Dutta.pdf"
    question = "I have a pain in my abdomen. I had periods 1 week back."
    answer = run_medbot(pdf_path, question)
    print(f"Question: {question}\nAnswer: {answer}")
