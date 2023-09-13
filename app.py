# Import Classes from Libraries
import logging
import sys
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
from streamlit_chat import message
from llama_index import TrafilaturaWebReader


# Load environment values
load_dotenv()

# Initialize OpenAI LLM
llm = OpenAI(model_name="text-davinci-003")  # we can also use 'text-davinci-002'

# Read Data From WebPage
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = TrafilaturaWebReader().load_data(["https://edoctsuj.github.io/demo_data/"])
document = documents[0]
text = document.text

# Initialize Text Splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)

# Split text into chunks
finalData = text_splitter.split_text(text)

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Create FAISS vector store from text data
documentSearch = FAISS.from_texts(finalData, embeddings)

# Load QA chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Setting page title and header
st.set_page_config(page_title="PigeonCloud bot", page_icon=":robot_face:")

# Read data from web
st.sidebar.markdown(
    "<h1 style='text-align: center;'>PigeonCloud Support</h1>", unsafe_allow_html=True
)
st.sidebar.info(
    "Chatbot, that reads data from a web page and answers questions based on the information it retrieves."
)

# Read data from web
st.sidebar.title("Read Data From Web😎")
st.sidebar.text("https://edoctsuj.github.io/demo_data/")

# Sample questions to ask
st.sidebar.title("Sample Questions😎")
st.sidebar.text("Did Mr Jones work in George's shop?")  # No, he didn't
st.sidebar.text("Who worked in Mr Jones's shop?")  # George did.
st.sidebar.text("How many awards did Huhu won?")
st.sidebar.text("Did the shop sell tables, or food?")  # It sold food
st.sidebar.text("Were Gladys's father and mother rich?")  # No, they were not
st.sidebar.text("Did Gladys want to work in her small, quiet town?")  # No, she did not
st.sidebar.text("How many awards did Huhu won?")
st.sidebar.text("Who did Gladys want to marry?")  # A rich man
st.sidebar.text("Was Alan's shop open on Monday?")  # No, it wasn't
st.sidebar.text("Was Alan's shop shut on Saturday?")  # No, it wasn't
st.sidebar.text("How many awards did Huhu won?")
st.sidebar.text("Was Alan's shop open or shut on Sunday morning?")  # It was open
st.sidebar.text("Did Mr and Mrs Brown have any children?")  # Yes, they had four.
st.sidebar.text("Did their children marry?")  # Yes, all of them did
st.sidebar.text("How many awards did Huhu won?")
st.sidebar.text("Did they have any grandsons?")  # Two
st.sidebar.text("What was Joe's father?")  # He was a farmer.
st.sidebar.text("Was his father rich?")  # No, he was not
st.sidebar.text("How many awards did Huhu won?")
st.sidebar.text(
    "Why did Joe leave his father's farm?"
)  # Because he wanted a farm in a better place.


# Set empty array
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Get Response
def getresponse(question):
    docs = documentSearch.similarity_search(question)
    answer = chain.run(input_documents=docs, question=question)

    return answer


# Conversation
response_container = st.container()
container = st.container()

with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area("Your question goes here:", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")

        if submit_button:
            st.session_state["messages"].append(user_input)

            model_response = getresponse(user_input)
            st.session_state["messages"].append(model_response)

            with response_container:
                for i in range(len(st.session_state["messages"])):
                    if (i % 2) == 0:
                        message(
                            st.session_state["messages"][i],
                            is_user=True,
                            key=str(i) + "_user",
                        )
                    else:
                        message(st.session_state["messages"][i], key=str(i) + "_AI")
