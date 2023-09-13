
# Import Classes from Libraries
import logging
import sys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
from llama_index import TrafilaturaWebReader


# Load environment values
load_dotenv()

# Initialize OpenAI LLM
llm = OpenAI(model_name="text-davinci-003") #we can also use 'text-davinci-002'

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





# Q&A

# Unit_1
# Q1:Perform document search and question answering
our_query = "Did Mr Jones work in George's shop?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A1:Print answer
print(our_query)
print(answer) #No, he didn't

# Q2:Perform document search and question answering
our_query = "Who worked in Mr Jones's shop?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A2:Print answer
print(our_query)
print(answer) # George did. 

# Unknow Question For AI
# Q3:Perform document search and question answering
our_query = "How many awards did Huhu won?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A3:Print answer
print(our_query)
print(answer)

# Q4:Perform document search and question answering
our_query = "Did the shop sell tables, or food?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A4:Print answer
print(our_query)
print(answer) # It sold food





# Unit_2
# Q1:Perform document search and question answering
our_query = "Were Gladys's father and mother rich?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A1:Print answer
print(our_query)
print(answer) #No, they were not

# Q2:Perform document search and question answering
our_query = "Did Gladys want to work in her small, quiet town?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A2:Print answer
print(our_query)
print(answer) # No, she did not

# Unknow Question For AI
# Q3:Perform document search and question answering
our_query = "How many awards did Huhu won?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A3:Print answer
print(our_query)
print(answer)

# Q4:Perform document search and question answering
our_query = "Who did Gladys want to marry?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A4:Print answer
print(our_query)
print(answer) #A rich man






# Unit_3
# Q1:Perform document search and question answering
our_query = "Was Alan's shop open on Monday?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A1:Print answer
print(our_query)
print(answer) # No, it wasn't

# Q2:Perform document search and question answering
our_query = "Was Alan's shop shut on Saturday?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A2:Print answer
print(our_query)
print(answer) #No, it wasn't

# Unknow Question For AI
# Q3:Perform document search and question answering
our_query = "How many awards did Huhu won?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A3:Print answer
print(our_query)
print(answer)

# Q4:Perform document search and question answering
our_query = "Was Alan's shop open or shut on Sunday morning?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A4:Print answer
print(our_query)
print(answer) # It was open






# Unit_4
# Q1:Perform document search and question answering
our_query = "Did Mr and Mrs Brown have any children?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A1:Print answer
print(our_query)
print(answer) # Yes, they had four.

# Q2:Perform document search and question answering
our_query = " Did their children marry?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A2:Print answer
print(our_query)
print(answer) #Yes, all of them did

# Unknow Question For AI
# Q3:Perform document search and question answering
our_query = "How many awards did Huhu won?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A3:Print answer
print(our_query)
print(answer)

# Q4:Perform document search and question answering
our_query = "Did they have any grandsons?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A4:Print answer
print(our_query)
print(answer) #Two






# Unit_5
# Q1:Perform document search and question answering
our_query = "What was Joe's father?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A1:Print answer
print(our_query)
print(answer) #He was a farmer.

# Q2:Perform document search and question answering
our_query = "Was his father rich?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A2:Print answer
print(our_query)
print(answer) #No, he was not

# Unknow Question For AI
# Q3:Perform document search and question answering
our_query = "How many awards did Huhu won?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A3:Print answer
print(our_query)
print(answer)

# Q4:Perform document search and question answering
our_query = "Why did Joe leave his father's farm?"
docs = documentSearch.similarity_search(our_query)
answer = chain.run(input_documents=docs, question=our_query)
# A4:Print answer
print(our_query)
print(answer) #Because he wanted a farm in a better place.
