from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq  # ✅ Added Groq import

# 🔹 Load environment variables
load_dotenv()

# 🔹 Get API keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # ✅ Use Groq API key
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 🔹 Step 1: Load and process PDFs
extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# 🔹 Step 2: Create embeddings (using Hugging Face)
embeddings = download_hugging_face_embeddings()

# 🔹 Step 3: Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot 1"

# Create index if not exists
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# 🔹 Step 4: Store document embeddings in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

# 🔹 Step 5: Initialize Groq LLM (for answering queries)
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",  # ✅ Recommended model
)

print("✅ Setup successful: Pinecone + HuggingFace + Groq are ready!")
