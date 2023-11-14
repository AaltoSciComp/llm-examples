import os

os.environ["OPENAI_API_KEY"] = "none"
model_name = os.environ["MODEL_NAME"]

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = OpenAI(
    openai_api_base="http://localhost:4891/v1",
    model_name=model_name,
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

print(llm_chain.run(question))
