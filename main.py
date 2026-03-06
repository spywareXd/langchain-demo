import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


#initialize embeddings and pinecone object
embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embeddings-001", output_dimensionality=1536)
vector_store=PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)


#retriever is a vector store with searching capabilities implemented by the vendors
retriever=vector_store.as_retriever(search_kwargs={"k":3})   #top k chunks where k=3

prompt_template=ChatPromptTemplate(
    """
    Answer the question based only on the following context provided:
    {context}
    
    Question: {question}
    
    Provide a detailed Answer:"""
)

def format_document(docs):
    """Format documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)



