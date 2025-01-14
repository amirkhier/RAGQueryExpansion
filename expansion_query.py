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


# Read the PDF file
reader = PdfReader("./data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]
# Filter out empty strings
pdf_texts = [text for text in pdf_texts if text]

#importing langchain 
from langchain.text_splitter import (RecursiveCharacterTextSplitter,SentenceTransformersTokenTextSplitter)

character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
# Tokenize the text
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# Importing chromadb
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
chroma_client = chromadb.Client()
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_collection = chroma_client.create_collection("microsoft_collection", embedding_function=embedding_function)

ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

# query = "What is the revenue of Microsoft?"
# results = chroma_collection.query(query_texts=[query], n_results=5)

# retrived_documents = results["documents"][0]

# for docs in retrived_documents:
#     print(word_wrap(docs))

#generate multi query function 
def generate_multi_query(query, model="gpt-3.5-turbo"):

    prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
                """

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
    content = content.split("\n")
    return content


original_query = (
    input("Enter the original query: ")
)
multi_query = generate_multi_query(original_query)

#1.View the generated multi query
# for query in multi_query:
#     print("\n")
#     print(query)

#2. Join the original query with the generated multi query
joint_query = [original_query] + multi_query

#3. query the joint query to the chroma collection
results = chroma_collection.query(query_texts=joint_query, n_results=5,include=["documents","embeddings"])
#4. Retrieve the documents
retrieved_documents = results["documents"]
#5. view the retrieved documents
# for doc in retrieved_documents:
#     print(doc)
#     print("\n")
#6. we have conclusion that the documents are duplicated, we can remove the duplicates
#let's remove the duplicates
unique_documents = set()
for docs in retrieved_documents:
    for doc in docs:
        unique_documents.add(doc)
#7. convert the set to list 
unique_documents = list(unique_documents)
#8. view the unique documents
# for doc in unique_documents:
#     print(doc)
#     print("\n")
#9. now let's project the embeddings
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# 4. We can also visualize the results in the embedding space
original_query_embedding = embedding_function([original_query])
augmented_query_embeddings = embedding_function(joint_query)


project_original_query = project_embeddings(original_query_embedding, umap_transform)
project_augmented_queries = project_embeddings(
    augmented_query_embeddings, umap_transform
)

retrieved_embeddings = results["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]

projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

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
    project_augmented_queries[:, 0],
    project_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
)
plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    project_original_query[:, 0],
    project_original_query[:, 1],
    s=150,
    marker="X",
    color="r",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot