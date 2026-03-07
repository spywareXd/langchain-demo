import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

#lcel:
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter


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



#better approach with LCEL:
def rag_with_lcel():

    llm=ChatGoogleGenerativeAI(model="gemini-flash-latest")
    retrieval_chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | retriever | format_document
            )
            | prompt_template
            | llm
            | StrOutputParser()
    )
    return retrieval_chain






if __name__=="__main__":
    query="How is venus fly trap prey processed in the plant?"
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
    #NO RAG
    print("=" * 40)
    print("No RAG")
    print("\nAnswer: ")
    result_raw = llm.invoke([HumanMessage(content=query)])
    print(result_raw.content[0].get("text"))
    print("=" * 40)


    #NO LCEL
    print("=" * 40)
    print("RAG without LCEL")
    print("\nAnswer: ")
    print(rag_without_lcel(query)[0].get('text'))
    print("=" * 40)


    #WITH LCEL
    print("=" * 40)
    print("RAG with LCEL")
    print("\nAnswer: ")
    runnable_chain=rag_with_lcel()
    result_lcel=runnable_chain.invoke({"question": query})
    print(result_lcel)
    print("=" * 40)


