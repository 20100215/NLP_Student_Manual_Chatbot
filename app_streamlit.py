import streamlit as st
import os
import requests

# initialize new sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Langchain and HuggingFace
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load embeddings, model, and vector store
@st.cache_resource # Singleton, prevent multiple initializations
def init_chain():
    model_kwargs = {'trust_remote_code': True}
    embedding = HuggingFaceEmbeddings(model_name='nomic-ai/nomic-embed-text-v1.5', model_kwargs=model_kwargs)
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.2)
    vectordb = Chroma(persist_directory='db_1800_200-with-ug-programs', embedding_function=embedding)

    # Create chain
    chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=vectordb.as_retriever(k=6),
                                  return_source_documents=True)

    return chain



# App title
st.set_page_config(page_title="Carolinian Chatbot")

# App sidebar - Refactored from https://github.com/a16z-infra/llama2-chatbot
with st.sidebar:
    st.title('📖💬 Carolinian Chatbot')
    st.subheader('Ask anything USC Related here!')
    st.markdown('''
                - About USC, History, Core Values
                - Admission, Enrollment, Graduation
                - Tutorial, Petition, Overload
                - Simultaneous Enrollment, Override
                - Academic and Grade Policies
                - Code of Conduct and Offenses
                - Motor Vehicle Pass / Car Stickers
                - Carolinian Honors List / Latin Honors
                - Directory of Student Support Services
                - Directory of Academic Departments
                - Undergraduate Academic Programs
                ''')
    
    st.markdown('''
                Access the resources here:

                - [USC Student Manual 2023](https://drive.google.com/file/d/1rFThhqMrVqMF0k0wMFMOIZuraF4AywYN/view?usp=drive_link)
                - [USC Enrollment Guide](https://enrollmentguide.usc.edu.ph)
                - [USC Undergraduate Programs](https://www.usc.edu.ph/academics/undergraduate-programs)
                ''')
    st.markdown('2024.05.12 - Developed by: [Wayne Dayata](https://github.com/20100215)')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    with st.spinner("Initializing, please wait..."):
        st.session_state.chain = init_chain()
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you today, Carolinian?"}]


def ask_question(str):
    st.session_state.messages.append({"role": "user", "content": str})

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Display sample questions
if len(st.session_state.messages) <= 1:
    container = st.container(border=True)
    container.subheader('Sample questions:')
    container.button('How do I enroll?', on_click=ask_question, args=['How do I enroll?'])
    container.button('How do I apply for a car sticker?', on_click=ask_question, args=['How do I apply for a car sticker?'])
    container.button('Who to contact about student organizations?', on_click=ask_question, args=['Who to contact about student organizations?'])
    container.button('What is the difference between BS CS and BS IT?', on_click=ask_question, args=['What is the difference between BS CS and BS IT?'])
    container.button('What is the difference between overload, tutorial, and override?', on_click=ask_question, args=['What is the difference between overload, tutorial, and override?'])
    container.button('Can you explain more about family previleges?', on_click=ask_question, args=['Can you explain more about family previleges?'])
    container.button('Can you show me the guidelines on civilian clothing?', on_click=ask_question, args=['Can you show me the guidelines on civilian clothing?'])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you today, Carolinian?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating response
def generate_response(prompt_input):
    # Initialize result
    result = ''
    # Invoke chain
    res = st.session_state.chain.invoke(prompt_input)
    # Process response
    if('According to the provided context, ' in res['result']):
        res['result'] = res['result'][35:]
        res['result'] = res['result'][0].upper() + res['result'][1:]
    elif('Based on the provided context, ' in res['result']):
        res['result'] = res['result'][31:]
        res['result'] = res['result'][0].upper() + res['result'][1:]    
    result += res['result']
    # Process sources
    result += '\n\nSources: '
    sources = [] 
    for source in res["source_documents"]:
        sources.append(source.metadata['source'][4:-4]) # Remove AXX- and .txt
    top_source = sources[0]
    sources = set(sources) # Remove duplicate sources (multiple chunks)
    result += ", ".join(sources)

    return result, res['result'], top_source

# User-provided prompt
if prompt := st.chat_input(placeholder="Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            placeholder = st.empty()
            response, answer, top_source = generate_response(prompt)
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}

    # Post question and answer to Google Sheets via Apps Script
    url = os.environ['SCRIPT_URL']
    requests.post(url, data = {"question": prompt, "answer": answer, "top_source": top_source})
    st.session_state.messages.append(message)
