import os

from langchain_groq import ChatGroq
from constants import GROQ_API_KEY
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader


os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(model="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

loader = CSVLoader(file_path='../conversation.csv',csv_args={
   'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['message_id','sender','receiver','timestamp','message']
})
data = loader.load()



# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #to chunk the documents 

# splits = text_splitter.split_documents(docs)
messages=[doc.page_content for doc in data]
# embedded_messages = [embeddings.embed_query(message) for message in messages]

#vectorstore = Chroma.from_texts(texts=messages, embedding=embeddings)
vectorstore = Chroma.from_documents(data, embedding=embeddings)


# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

template = """Use the following pieces of context to answer the question at the end.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("Aman's favourite dish?"))