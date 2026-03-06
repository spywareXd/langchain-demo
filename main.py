import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


#initialize embeddings and pinecone object
embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=1536)
vector_store=PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)


#retriever is a vector store with searching capabilities implemented by the vendors
retriever=vector_store.as_retriever(search_kwargs={"k":3})   #top k chunks where k=3

prompt_template=ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context provided:
    {context}
    
    Question: {question}
    
    Provide a detailed Answer:"""
)

def format_document(docs):
    """Format documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def rag_without_lcel(query: str):
    """ Simple retrieval wihout LCEL"""

    llm=ChatGoogleGenerativeAI(model="gemini-flash-latest")

    #retrieve top k chunks of documents
    #retriever is a runnable object
    docs=retriever.invoke(query)

    #format top k into string context:
    context=format_document(docs)

    #format the prompt
    messages=prompt_template.format_messages(context=context, question=query)

    #run the llm
    response=llm.invoke(messages)
    return response.content



if __name__=="__main__":
    query="How do vector databases handle massive load?"

    print("=" * 40)
    print("RAG without LCEL")
    print("=" * 40)
    print("\nAnswer: ")
    print(rag_without_lcel(query)[0].get('text'))


