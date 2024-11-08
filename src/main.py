# main.py

import os
from api.rag_pipeline import RAGPipeline

def main():
    # Load environment variables
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Pinecone API key is missing. Please set it in the environment.")

    # Initialize RAGPipeline
    pipeline = RAGPipeline()

    # Define the path to your PDF and the question
    pdf_path = "C:\Agam\Work\medbot\dataset\Gynaecology-DC-Dutta.pdf"  # Replace with the actual path to your PDF
    question = "I have pain in my abdomen even after 3 weeks of giving birth why and what to do?"

    # Run the pipeline
    answer = pipeline.run_medbot(pdf_path, question)

    # Display the answer
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
