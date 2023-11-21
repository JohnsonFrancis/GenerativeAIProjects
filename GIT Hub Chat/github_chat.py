import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import configparser
import os

@st.cache_resource  # This will run only once
def get_llm_qa():
    workingFolder='C:\\Users\\jfrancis\\AI Journey\\Gen AI\\'
    # Read the configuration file
    config = configparser.ConfigParser()
    config.read(workingFolder+'\\config.ini')
    OPENAI_API_KEY=config.get('General','OPENAI_API_KEY')
    ACTIVELOOP_TOKEN=config.get('General','ACTIVELOOP_TOKEN')
    ACTIVELOOP_ORG_ID=config.get('General','ACTIVELOOP_ORG_ID')
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    os.environ["ACTIVELOOP_TOKEN"] = ACTIVELOOP_TOKEN
    my_activeloop_org_id = ACTIVELOOP_ORG_ID

    # Read from activeloop vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    my_activeloop_org_id = ACTIVELOOP_ORG_ID
    my_activeloop_dataset_name = "langchain_course_chat_with_gh"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, read_only=True, embedding=embeddings)

    # Retrieval queue from activeloop
    retriever = db.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10
    model = ChatOpenAI()
    qa = RetrievalQA.from_llm(model, retriever=retriever)
    st.success("Loaded RetrievalQA")  # 👈 Show a success message
    return qa

qa = get_llm_qa()

# Design the front end chat app
st.title(f"Chat with GitHub Repository")
if "generated" not in st.session_state:
    st.session_state["generated"] = ["i am ready to help you ser"]
if "past" not in st.session_state:
    st.session_state["past"] = ["hello"]
input_text = st.text_input("", key="input")
if input_text:
    output = qa.run(input_text)
    st.session_state.past.append(input_text)
    st.session_state.generated.append(output)
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))
