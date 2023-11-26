from langchain.document_loaders import OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.llms import LlamaCpp
import os
import sys
loader = OnlinePDFLoader('https://acp.copernicus.org/articles/12/8911/2012/acp-12-8911-2012.pdf')
# loader = PyPDFLoader('yourdocument.pdf')
data = loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
text_chunks=text_splitter.split_documents(data)
print('num of text chunks', len(text_chunks))

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
vector_store=FAISS.from_documents(text_chunks, embeddings)

model_path = os.environ.get('MODEL_WEIGHTS')
llm = LlamaCpp(model_path=model_path, verbose=False)

template="""Use the following pieces of information to answer the user's question.
If you dont know the answer just say you know, don't try to make up an answer.

Context:{context}
Question:{query}

Only return the helpful answer below and nothing else
Helpful answer
"""
qa_prompt=PromptTemplate(template=template, input_variables=['context', 'query'])
print(type(llm))
chain = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type='stuff',
                                   retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
                                   return_source_documents=True,
                                   chain_type_kwargs={'prompt': qa_prompt})
while True:
    query=input(f"prompt:")
    if query=='exit':
        print('Exiting')
        sys.exit()
    if query=='':
        continue
    result=chain({'query':query})
    print(f"Answer:{result['result']}")

