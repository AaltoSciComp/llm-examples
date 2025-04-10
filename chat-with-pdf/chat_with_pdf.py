from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from langgraph.graph import StateGraph, START
from typing import TypedDict, List

from langchain_core.documents import Document
from typing_extensions import List, TypedDict


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

print('Processing document...')
pdf_input = input("Enter local PDF path: ")

loader = PyPDFLoader(pdf_input)

data = loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
text_chunks=text_splitter.split_documents(data)
print('num of text chunks', len(text_chunks))

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
vector_store=FAISS.from_documents(text_chunks, embeddings)

model_name = "meta-llama/Llama-2-7b-chat-hf"  # or any other Hugging Face model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.7,
    device_map="auto"
)

# Initialize LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

template="""Use the following pieces of information to answer the user's question.
If you dont know the answer just say you know, don't try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answern
"""
qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = qa_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}

# Create the graph and add sequence
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")

# Compile the graph
graph = graph_builder.compile()

while True:
    query = input(f"prompt:")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
        
    # Execute the graph
    result = graph.invoke({"question": query})
    print(f"Answer: {result['answer']}")