# Insight PDF Chatbot

## Overview

The **Insight PDF Chatbot** is an interactive web application  that allows users to upload PDF documents and ask questions related to the content of the documents. 
The chatbot utilizes advanced language models and embeddings to process and retrieve relevant information from the uploaded PDFs. 

Here is an example on how the chatbot works for an SBI FAQ document:

https://github.com/user-attachments/assets/3d5baadb-b1a1-4d10-b703-c49ca0e816b2
## Features

- **PDF Upload**: Users can upload PDF documents, which are then processed for text extraction and embeddings creation.
- **Question-Answering**: The chatbot answers user queries based on the content of the uploaded PDF. The chatbot excels in answering combination of two or more questions related to the content of the PDF.
- **Get Specific Information**: With this chatbot, you can get specific informations present in the PDF, like writer of the document,date of publishing etc.
  
## Getting Started

### Prerequisites

- Python 3.7 or above
- An active Hugging Face API key
- An active Google Generative AI API key

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/insight-pdf-chatbot.git
    cd insight-pdf-chatbot
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment variables:
    - Create a `.env` file in the root directory with the following content:
      ```plaintext
      HUGGINGFACE_ACCESS_TOKEN=your_huggingface_token
      GOOGLE_API_KEY=your_google_api_key
      ```

4. Run the application:
    ```bash
    streamlit run chatbot.py
    ```

## Usage

1. **Upload a PDF**: Click on the "Upload your PDF here" button and select a PDF file.
2. **View the PDF**: Use the "View uploaded File" toggle to display the uploaded PDF within the app.
3. **Ask a Question**: Type your question in the input box and press Enter. The chatbot will process your question and provide a relevant answer.
4. **Clear Chat and Files**: Click on "Clear Chat and Remove Uploaded Files" to reset the session and remove all uploaded files from the local storage.

## Project Structure

The chatbot uses Retrieval Augumented Generation (RAG) method to give content based answers.

Here is a brief description of workflow:

- The document uploaded is split into chunks of sentences.

- These chunks are then vectorized by word embeddings, a pretained model which converts sentences/words into vectors of specific dimensions (384). These are stored in a vector database.

- When the user enters a query, a retriever runs a cosine similarity search to identify the chunk relevent to user's query and feeds into google flash.

- Then the google flash LLM, organizes the information and outpus in a presentable/readable way,according to user query.

## Technologies Used

- **Streamlit**: A framework for creating interactive web applications in Python.
- **LangChain**: For document processing, text splitting, and vector storage.
- **Hugging Face Inference API**: Used for generating embeddings from the document text.
- **Google Generative AI**: For generating answers to user queries based on the document context.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.


## Application

1.  The chatbot would be a very useful tool for researchers in doing literature review and saves a lot of time.
2.  The chatbot has the potential to be a support bot for a product, when trained efficiently with issues faced by product users before.
3.  The bot enhances the efficiency in searching for a particular content in the PDF. 

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any improvements or bugs.


## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Hugging Face](https://huggingface.co/)
- [Google Generative AI](https://ai.google/tools/)
- [FAISS](https://github.com/facebookresearch/faiss)
