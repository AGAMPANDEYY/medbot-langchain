
# RAG Model for Gynecological Education Chatbot

![image](https://github.com/user-attachments/assets/b78951f4-889f-4957-beb1-bf893b21b209)


This project is a Retrieval-Augmented Generation (RAG) model specifically designed to educate users on gynecological topics through a conversational chatbot. This chatbot leverages a Local Large Language Model (LLM) and is optimized to run on CPU, making it suitable for devices without a GPU. The pipeline integrates various tools and techniques to ensure accuracy, contextual memory, and efficiency in resource usage.

## Table of Contents
- [Project Overview](#project-overview)
- [Pipeline Overview](#pipeline-overview)
- [Model Selection](#model-selection)
- [Dataset Preparation](#dataset-preparation)
- [Text Chunking Strategy](#text-chunking-strategy)
- [Vector Database Choice](#vector-database-choice)
- [Retriever and Conversational Chain](#retriever-and-conversational-chain)
- [Evaluation](#evaluation)
- [Future Enhancements](#future-enhancements)

## Project Overview

This project focuses on developing an AI chatbot for gynecological education, aiming to answer common questions related to gynecology. The system is designed to be modular and scalable, with a retrieval-augmented generation approach that leverages LangChain for chaining responses and Pinecone as the vector database for efficient information retrieval.

## Pipeline Overview

The development pipeline consists of the following steps:

1. **Model Selection:** Choosing a suitable medical LLM for CPU-only environments.
2. **Data Preparation:** Gathering and processing relevant gynecological information from various sources.
3. **Text Chunking:** Segmenting the data into manageable chunks for better model comprehension.
4. **Vector Database Selection:** Storing embeddings and efficiently retrieving relevant chunks.
5. **Retriever and Conversational Chain Setup:** Implementing contextual memory for seamless user interactions.
6. **Evaluation:** Ensuring model accuracy using a mix of ground truth checks, LLM scoring, and retriever accuracy tests.

---

## Model Selection

![image](https://github.com/user-attachments/assets/c410af00-8840-4433-b1b1-83765b2643b3)


### Approach
Since the system runs entirely locally on a CPU, model selection was a critical step. Initially, I explored different medical LLMs by examining the medical leaderboard on [Hugging Face](https://huggingface.co/blog/leaderboard-medicalllm). Options included:
- **Open-source medical LLMs** with fine-tuning on specialized datasets.
- **Generic open-source LLMs** capable of handling medical knowledge.
- **Commercial API models** (not suitable for our local setup).

After comparing performance, I chose **MedLLaMA**, ranked 8th among open-source medical LLMs, due to its fine-tuning on medical data. However, as MedLLaMA requires a GPU, I adapted my setup by using **Ollama** to run a locally available LLaMA v3.2 (7B parameters) model, balancing performance with hardware limitations.

### Reasoning
Choosing MedLLaMA initially was due to its strong medical dataset fine-tuning, crucial for providing accurate gynecological information. Ultimately, LLaMA v3.2 was used in development to align with hardware constraints and avoid overburdening the system.

---

## Dataset Preparation

### Data Sources
For version 1 of the chatbot, I prioritized data that would be comprehensive and authoritative:
1. **Gynecology-focused PDFs** – These included textbooks and medical case studies.
2. **QA Dataset** – Aggregated from various medical Q&A websites.
3. **Web-Scraped Articles** – Supplementary information from reputable sources.

![image](https://github.com/user-attachments/assets/57390ef2-dad4-42d9-955e-67834b5f31a3)


### Reasoning
Starting with the PDF-based dataset ensured a solid foundation of structured, authoritative knowledge. Later versions can incorporate more diverse datasets to improve answer specificity.

---

## Text Chunking Strategy

### Chunking Options
For efficient processing, I tested three chunking methods:
1. **Page-Based Split** – Dividing content per page.
2. **Recursive Text Splitter** – Ideal for books, splitting text based on natural language boundaries.
3. **Semantic Splitter** – Used to break content into coherent, meaning-based sections.

### Choice and Reasoning
After testing, the recursive text splitter proved optimal for PDF-based content, achieving a good balance between speed and relevance. The semantic splitter, although conceptually ideal, took over 30 minutes to process even 100 pages of an 800-page document, making it infeasible for this setup.

---

## Vector Database Choice

### Options Considered
1. **ChromaDB** – An open-source vector database with limitations in monitoring, storage, and high-dimensionality handling.
2. **Pinecone** – A cloud-based solution offering efficient indexing, monitoring, and scalability.

### Choice and Reasoning
I selected **Pinecone** for its robustness in handling large datasets and its monitoring capabilities, which streamline development and enhance performance. Pinecone also supports high-dimensionality vectors, which are essential for accurate retrieval in RAG.

---

## Retriever and Conversational Chain

### Setup
The system incorporates a **Retriever Chain** using LangChain, which enables conversation flow and contextual memory management. This setup allows the chatbot to recall previous interactions, enhancing the overall user experience.

### Reasoning
Using LangChain's retriever chain allows seamless integration of tools and makes it easier to maintain a modular approach. The contextual memory feature is crucial for maintaining coherence across interactions, as users may ask follow-up questions.

---

## Evaluation

### Evaluation Methods
Evaluation is essential to gauge the chatbot's accuracy and reliability. I used three primary methods:
1. **QA Ground Truth Comparison** – Comparing model responses with established QA pairs.
2. **LLM-Based Scoring** – Using LLMs to rate response relevance and coherence.
3. **Retriever Accuracy Check** – Ensuring the retriever fetches relevant information.

### Resources
The evaluation framework was developed based on guidelines from [Hugging Face’s Cookbook on RAG Evaluation](https://huggingface.co/learn/cookbook/en/rag_evaluation).

### Reasoning
These evaluation methods ensure the chatbot's responses are accurate, relevant, and contextually appropriate. Ground truth QA checks establish a baseline, while LLM scoring and retriever accuracy enhance precision.

---

## Future Enhancements

This project is currently in its initial version, focusing on foundational setup and data accuracy. Planned improvements include:
- **Expanding Dataset** – Incorporating web-scraped articles and QA datasets for broader coverage.
- **Real-Time Feedback Integration** – Allowing users to rate responses to improve future interactions.
- **Enhanced Chunking and Retrieval** – Exploring faster and more efficient chunking methods for larger datasets.

---

## Conclusion

This RAG-based Gynecological Education Chatbot was developed with a focus on adaptability and efficiency, considering CPU limitations and resource constraints. The project aims to provide accessible, accurate, and relevant medical information through a local setup, making it suitable for educational purposes in environments with limited GPU access.
