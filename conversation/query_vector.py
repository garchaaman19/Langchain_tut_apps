from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

loader = CSVLoader(file_path='../conversation.csv',csv_args={
   'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['message_id','sender','receiver','timestamp','message']
})
data = loader.load()
messages=[doc.page_content for doc in data]
#metadata = [{"message_id": doc.page_content['message_id']} for doc in data]
# embedded_messages = [embeddings.embed_query(message) for message in messages]
# vector_store = Chroma(embedding_function=embeddings,persist_directory="./chroma_langchain_db",collection_name="example_collection",)
db = Chroma.from_texts(texts=messages,embedding=embeddings)
query = "Shubham's favorite cuisine"
docs = db.similarity_search(query,k=10)
print(docs)

# print("vector store",vector_store)
# vector_store.add_texts(texts=messages, embeddings=embedded_messages)
# # Example query to retrieve documents
# query = "What's Shubham's favorite cuisine?"
# result = vector_store.similarity_search(query)

# # Print the result
# print(result)