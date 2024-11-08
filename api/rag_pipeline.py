# rag_pipeline.py

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

# Load environment variables
load_dotenv(dotenv_path="./api/.env")
load_dotenv()

index_name="medbot-index"
DIMENSION = 2048
METRIC = "cosine"

class PineconeManager:
    @staticmethod
    def initialize_pinecone(api_key, index_name=index_name):
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

class RAGPipeline:


    @staticmethod
    def initialize_pinecone(api_key, index_name=index_name):
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

    def __init__(self):

        model_name="llama3.2:1b"
        model = Ollama(model=model_name)
        embeddings = OllamaEmbeddings(model=model_name)
        parser = StrOutputParser()
        #pdf_path = "/app/dataset/Gynaecology-DC-Dutta.pdf" # USE INCASE OF DOCKER 
        pdf_path=r"C:\Agam\Work\medbot-langchain\dataset\Gynaecology-DC-Dutta.pdf" #USE INCASE OF STREAMLIT LOCAL HOST

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("Pinecone API key is missing!")

        self.index = PineconeManager.initialize_pinecone(pinecone_api_key, index_name=index_name)
        self.embeddings=embeddings
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
        self.model=model
        self.parser=parser
        self.pdf_path=pdf_path


        # Load and process documents
    def document_chunking(self,pdf_path):
        ''''
        chunking strategies can be used different as mentioned in the rag-medllama.ipynb notebook.

        '''
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        return text_splitter.split_documents(documents)

    # Populate Pinecone with embeddings
    def populate_pinecone(self,index, documents, embeddings, batch_size=1):
        ids = [f"page_{i}" for i in range(len(documents))]
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding documents to Pinecone"):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            self.index.add_documents(documents=batch_docs, ids=batch_ids)

    def generate_prompt(self, context, question):
            
        """Generate prompt for the language model."""
        prompt_template = '''
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
        return PromptTemplate.from_template(prompt_template)
    def query_medbot(self, context,question, chain):

        """Retrieve context and generate answer for the question."""
             
        response = chain.invoke({"context": context, "question": question})
        return self.parser.parse(response)
    
    def run_medbot(self, question):

        """Main function to load documents, populate Pinecone, and answer the question."""
        documents = self.document_chunking(self.pdf_path)
        '''
        if not self.vector_store.count_documents():
            self.populate_pinecone(documents)
        '''

        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"k": 5, "score_threshold": 0.5}
        )
        context = retriever.invoke(question)  
        prompt=self.generate_prompt(context, question)

        chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | self.model
        | self.parser
        )

        return self.query_medbot(context,question,chain)
    