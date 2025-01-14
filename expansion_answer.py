from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv

from pypdf import PdfReader
import umap


# Load Environment Variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
# Initialize OpenAI API client
reader = PdfReader("./data/microsoft-annual-report.pdf")
# Extract text from PDF
pdf_texts = [p.extract_text().strip() for p in reader.pages]
#Filter out empty strings
pdf_texts = [text for text in pdf_texts if text]

#Split text into chunks
chunks = [] #List to store chunks

from langchain.text_splitter import (RecursiveCharacterTextSplitter,SentenceTransformersTokenTextSplitter)
# Initialize the text splitter

charcter_spliter= RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0)

# Split text into chunks
charcter_split_texts = charcter_spliter.split_text("\n\n".join(pdf_texts))


# print(f"Number of chunks: {len(charcter_split_texts)}")

token_spliter= SentenceTransformersTokenTextSplitter(chunk_overlap=0,tokens_per_chunk=256)
token_split_texts = []
for text in charcter_split_texts:
    token_split_texts += token_spliter.split_text(text)

# print(token_split_texts[0])
# print(f"Number of chunks: {len(token_split_texts)}")

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# Initialize the embedding function
embedding_function = SentenceTransformerEmbeddingFunction()
#testing the embedding function
# print(embedding_function(token_split_texts[0]))
# Create a ChromaDB client
client1 = chromadb.Client()

# Create a collection
collection = client1.create_collection("microsoft_collection", embedding_function=embedding_function)

# ids:
ids = [str(i) for i in range(len(token_split_texts))]
# Add documents to the collection
collection.add(ids=ids, documents=token_split_texts)
collection.count()
# print(collection.count())

#example of query
# query = "What was the total revenue for the year "

# Query the collection
# results = collection.query(query_texts=[query],n_results=5)
# retrieve_documents = results["documents"][0]
# print(len(retrieve_documents))

#wrap the documents
# for doc in retrieve_documents:
#     print(word_wrap(doc))
#     print("\n")

def augment_query_generated(query, model="gpt-3.5-turbo"):
    prompt = """You are a helpful expert financial research assistant. 
   Provide an example answer to the given question, that might be found in a document like an annual report."""
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

original_query = input("Enter your query: ")
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"

# print(word_wrap(joint_query))
# Query the collection with the joint query
results = collection.query(query_texts=[joint_query], n_results=5,include=["documents","embeddings"])

retrieved_documents = results["documents"][0]

# for doc in retrieved_documents:
#     print(word_wrap(doc))
#     print("")


embeddings = collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

retrieved_embeddings = results["embeddings"][0]

original_query_embedding = embedding_function([original_query])
augment_query_embedding = embedding_function([joint_query])

projected_originalQuery_embeddings = project_embeddings(original_query_embedding, umap_transform)

projected_AugmentQuery_embeddings = project_embeddings(augment_query_embedding, umap_transform)

projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)

import matplotlib.pyplot as plt

# Plot the projected query and retrieved documents in the embedding space
plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_originalQuery_embeddings[:, 0],
    projected_originalQuery_embeddings[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_AugmentQuery_embeddings[:, 0],
    projected_AugmentQuery_embeddings[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot