import streamlit as st
import base64
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain.prompts import PromptTemplate
import time


def clear_cache():
    st.session_state.clear()  
    uploaded_files_path = "./docs"
    if os.path.exists(uploaded_files_path):
        for filename in os.listdir(uploaded_files_path):
            file_path = os.path.join(uploaded_files_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)  
    st.rerun()  

def display_conversation(history):
    for i in range(len(history["generated"])):
        with st.chat_message("user"):
            st.write(history["past"][i])
        with st.chat_message("assistant"):
            st.write(history["generated"][i])



def process_answer(database,question):

    retrieved_docs = database.similarity_search(query=question, k=5)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Using the information contained in the context, 
        give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        If the answer cannot be deduced from the context, do not give an answer.
        Present the answer in a readable format.

        Context:
        {context}
        ---
        Now here is the question you need to answer:

        {question}
        """
    )
    
    formatted_prompt = prompt_template.format(question=question,context=retrieved_docs)

    answer=model.generate_content(formatted_prompt)
    print(answer.text)
    return answer.text





def data_ingestion(hf_token):
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):

                loader = PyPDFLoader(os.path.join(root, file))
                data=loader.load()
                MARKDOWN_SEPARATORS = ["```\n","\n\\*\\*\\*+\n","\n---+\n","\n___+\n","\n\n","\n"," ","","."]

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=200,  
                    chunk_overlap=100,  
                    add_start_index=True, 
                    strip_whitespace=True, 
                    separators=MARKDOWN_SEPARATORS,
                    )
                docs_processed = []
                for doc in data:
                    docs_processed += text_splitter.split_documents([doc])
                embedding_model = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L12-v2")
                KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE)
    KNOWLEDGE_VECTOR_DATABASE.save_local('./vector_db')
    return KNOWLEDGE_VECTOR_DATABASE


def get_base64_of_file(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode()   


def displayPDF(file):
    
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)



def main():
    load_dotenv()
    hf_token = os.getenv('HUGGINGFACE_ACCESS_TOKEN')
    st.title("Insight PDF (RAG Chatbot)")
    st.markdown("<h2 style='text-align: center; color: #25b5f7;'>A Custom chatbot that answers questions from your documents</h2>", unsafe_allow_html=True)
    

    st.markdown("<h3 style='text-align: center; color:#ad9d0c;'>Upload your PDF here👇</h3>", unsafe_allow_html=True)

    


    uploaded_file = st.file_uploader("", type=["pdf"])

    
    
    if uploaded_file is not None:
        filepath = "docs/"+uploaded_file.name
        
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())   

        if st.button("Clear Chat and Remove Uploaded Files"):
            clear_cache()

        
        with st.spinner('Embeddings are in process...'):
            ingested_data = data_ingestion(hf_token)
        st.success('Embeddings are created successfully!')
        st.markdown("<h4 style color:black;'>Ask your Questions</h4>", unsafe_allow_html=True)
    
        
        if "past" not in st.session_state:
            st.session_state["past"] =[]
        if "generated" not in st.session_state:
            st.session_state["generated"] = []
        with st.chat_message("ai"):
            st.write("Hey there, How can I help with your document?")
        user_input = st.chat_input("Type your message ...")
        if user_input:
            with st.spinner("Bot is typing..."):
                time.sleep(1)  
                answer = process_answer(database=ingested_data, question=user_input)
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(answer)

        if st.session_state["generated"]:
            display_conversation(st.session_state)
if __name__ == "__main__":
    main()