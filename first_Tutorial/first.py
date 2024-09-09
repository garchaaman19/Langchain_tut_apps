import os
from constants import GROQ_API_KEY,LANGCHAIN_API_KEY,LANGCHAIN_TRACING_V2,LANGCHAIN_PROJECT
from langchain_core.output_parsers import StrOutputParser
os.environ["GROQ_API_KEY"]=GROQ_API_KEY
os.environ["LANGCHAIN_TRACING_V2"]='true'
os.environ["LANGCHAIN_API_KEY"]=LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"]=LANGCHAIN_PROJECT
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

model = ChatGroq(model="llama3-8b-8192")

parser = StrOutputParser()
chain = model | parser
result=chain.invoke(messages)  
print(parser.invoke(result))