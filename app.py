import streamlit as st
# from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile




# def initialize_session_state():
#     if 'history' not in st.session_state:
#         st.session_state['history'] = []

#     if 'generated' not in st.session_state:
#         st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

#     if 'past' not in st.session_state:
#         st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

# def display_chat_history(chain):
#     reply_container = st.container()
#     container = st.container()

#     with container:
#         with st.form(key='my_form', clear_on_submit=True):
#             user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
#             submit_button = st.form_submit_button(label='Send')

#         if submit_button and user_input:
#             with st.spinner('Generating response...'):
#                 output = conversation_chat(user_input, chain, st.session_state['history'])

#             st.session_state['past'].append(user_input)
#             st.session_state['generated'].append(output)

#     if st.session_state['generated']:
#         with reply_container:
#             for i in range(len(st.session_state['generated'])):
#                 message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
#                 message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# def create_conversational_chain(vector_store,temperature):
#     # Create llm
#     llm = LlamaCpp(
#     streaming = True,
#     model_path="/home/monish/Desktop/PGAGI /MultiPDFchatMistral-7B/mistral-7b-instruct-v0.1.Q5_0.gguf",
#     temperature=0.01,
#     # temperature=temperature,
#     top_p=1, 
#     verbose=True,
#     n_ctx=4096, 
# )
    
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#     chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
#                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
#                                                  memory=memory)
#     return chain

# def main():
#     # Initialize session state
#     initialize_session_state()
#     st.title("Multi-PDF ChatBot using Mistral-7B-Instruct :books:")
#     # Initialize Streamlit
#     st.sidebar.title("Document Processing")
#     uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)


#     if uploaded_files:
#         text = []
#         for file in uploaded_files:
#             file_extension = os.path.splitext(file.name)[1]
#             with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#                 temp_file.write(file.read())
#                 temp_file_path = temp_file.name

#             loader = None
#             if file_extension == ".pdf":
#                 loader = PyPDFLoader(temp_file_path)

#             if loader:
#                 text.extend(loader.load())
#                 os.remove(temp_file_path)
        
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
#         text_chunks = text_splitter.split_documents(text)

#         # Create embeddings
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cpu'})

#         # Create vector store
#         vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

#         # Create the chain object
#         chain = create_conversational_chain(vector_store)

#         print('hi')
#         display_chat_history(chain)

# if __name__ == "__main__":
#     main()


import gradio as gr
import random
from pymongo import MongoClient
import gridfs

def connect():
    try:
        con = MongoClient(host='127.0.0.1', port=27017)
        print("ok")
        return con.get_database('studentDB')
    except Exception as e:
        print("No")
def uploadPdf(db, data):
    name = "Book.pdf"
    fs = gridfs.GridFS(db)
    fs.put(data, filename=name)
    print("done")

def create_vector():
    # Set the path to the folder containing PDFs
    pdf_folder_path = '/home/monish/Desktop/PGAGI /MultiPDFchatMistral-7B/docs'
    texts=[]
    # Initialize the text splitter, and embeddings
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cpu'})  

    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the folder.")
    else:
        # Continue with the loop and vector database creation
        for pdf_file in pdf_files:
            pdf_file_path = os.path.join(pdf_folder_path, pdf_file)
            print(pdf_file_path)
            loader = PyPDFLoader(pdf_file_path)
            
            # Load data from the PDF
            data = loader.load()

            # Split the document into chunks
            text = text_splitter.split_documents(data)
            texts.extend(text)

    # Create the vector database
    vectordb = FAISS.from_documents(documents=texts, embedding=embed)

    # Save the vector database locally
    vectordb.save_local("vdb")
    print('vectorDC created')


# create_vector()
sub_list = ["science", "social science", "english"]


with gr.Blocks() as demo:
    
    with gr.Tab("Teacher"):
        def zip_files(files,Subject):
            print(files,sub_list[Subject])
            with open(files[0],'rb') as f:
              data = f.read()

            db = connect()
            uploadPdf(db, data)
            with open('nkn.pdf', 'wb') as f:
              f.write(data)
            f.close()
            
            return "nkn.pdf"

        T_interface = gr.Interface(
            zip_files,
            inputs=[gr.File(file_count="multiple", file_types=["text", ".pdf"]),
                    gr.Dropdown(sub_list, type="index")],
            outputs = "file",
            )

    with gr.Tab("student"):
        temp = gr.Slider(0, 1, label="Temperature", info="Choose between 0 and 1 (value closer to 1 the more is the halucination)")
        def get_ans(message,history):
            # Create llm
            llm = LlamaCpp(
                streaming = True,
                model_path="your model",
                temperature=temp,
                top_p=1,
                verbose=True,
                n_ctx=4096,
            )
            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'gpu'})
            vector_store = FAISS.load_local("/home/monish/Desktop/PGAGI /MultiPDFchatMistral-7B/vdb", embed)
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
                                                        memory=memory)
            return conversation_chat(query=message,chain=chain,history=history)
        
        stu_inter = gr.ChatInterface(fn=get_ans, title="Chat with your PDFs")

        clear = gr.ClearButton([stu_inter])
        print(stu_inter)

        # msg.submit(respond, [msg, stu_inter], [msg, stu_inter])

demo.launch(debug=True)