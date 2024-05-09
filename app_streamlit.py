import streamlit as st
import os

# initialize new sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Langchain and HuggingFace
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory


# Load embeddings, model, and vector store
os.environ['HF_TOKEN'] = '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'
groq_api_key = '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'
model_kwargs = {'trust_remote_code': True}

def init_chain():
    embedding = HuggingFaceEmbeddings(model_name='nomic-ai/nomic-embed-text-v1.5', model_kwargs=model_kwargs)
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=0.2)
    vectordb = Chroma(persist_directory='db', embedding_function=embedding)

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    return chain



# App title
st.set_page_config(page_title="Student Manual Chatbot")

# Replicate Credentials
with st.sidebar:
    st.title('📖💬 Student Manual Chatbot')
    st.info('Chat about anything USC related here!')

    # Refactored from https://github.com/a16z-infra/llama2-chatbot
    st.subheader('Topics Covered:')
    st.markdown('''
                - About USC, History, Core Values
                - Admission, Enrollment, Graduation
                - Tutorial, Petition, Overload
                - Simultaneous Enrollment, Override
                - Academic and Grade Policies
                - Code of Conduct and Offenses
                - Carolinian Honors List / Latin Honors
                - Directory of Student Support Services
                - Directory of Academic Departments
                ''')
    
    st.markdown('''
                Access the resources here:

                - [USC Student Manual 2023](https://drive.google.com/file/d/1rFThhqMrVqMF0k0wMFMOIZuraF4AywYN/view?usp=drive_link)
                - [USC Enrollment Guide](https://enrollmentguide.usc.edu.ph)
                ''')
    st.markdown('Developed by: [Wayne Dayata](https://github.com/20100215)')



# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.chain = init_chain()
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you today, Carolinian?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you today, Carolinian?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)



# Function for generating response
def generate_response(prompt_input):
    # Initialize result
    result = ''
    # Invoke chain
    res = chain.invoke(prompt_input)
    # Process response
    if('According to the provided context, ' in res['answer']):
        res['answer'] = res['answer'][35:]
        res['answer'] = res['answer'][0].upper() + res['answer'][1:]
    result += res['answer']
    # Process sources
    result += '\n\nSources:\n'
    sources = [] 
    for source in res["source_documents"]:
        sources.append(source.metadata['source'][3:-4]) # Remove XX- and .txt
    sources = set(sources) # Remove duplicate sources (multiple chunks)
    for source in sources:
        result += source
        result += '\n'

    return result

# User-provided prompt
if prompt := st.chat_input(placeholder="Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)