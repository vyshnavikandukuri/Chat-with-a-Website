# Chat-with-a-Website

This project implements a chatbot using Langchain and Streamlit, which allows users to interact with the content of a webpage.
The chatbot retrieves information from a given URL, processes it, and provides relevant answers to user queries.



**Features**

**Web Scraping**: Automatically scrapes content from a given website URL.
**Text Chunking**: Splits the content into manageable chunks to improve context retrieval.
**Embedding**: Uses Hugging Face embeddings to convert text chunks into numerical vectors.
**Retrieval-Augmented Generation (RAG)**: Combines retrieved context and user input to generate accurate responses using GPT models (e.g., GPT-Neo).
**Streamlit Interface**: Provides a user-friendly web interface for interacting with the chatbot.


**Requirements**

To run this project, make sure you have the following Python libraries installed:

**streamlit
langchain
huggingface_hub
sentence-transformers
chromadb**
