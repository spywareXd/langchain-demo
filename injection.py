import os
from dotenv import load_dotenv
from langchain_core.prompts import format_document

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore




if __name__ == "__main__":
    print("Hello from langchain-course!")
    loader = TextLoader("blog.txt", encoding="utf-8")
    document = loader.load()  # this creates a document object (content, metadata)

    print("splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)  #define function attributes
    texts= text_splitter.split_documents(document)
    print(f"No. of chunks: {len(texts)}")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", output_dimensionality=1536)
    print("Ingesting....")
    PineconeVectorStore.from_documents(texts,embeddings, index_name=os.environ.get("INDEX_NAME"))
    print("Done!")


