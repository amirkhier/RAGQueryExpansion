# Advanced RAG with Expansion Answers and Queries

This project provides tools to query and analyze financial reports using OpenAI's GPT-3.5 and ChromaDB for document embedding and retrieval. The project processes PDF documents, splits them into manageable chunks, generates embeddings, and allows querying to retrieve relevant information.
## Advanced RAG - Expansion Answer Phase :
![RAGExpansionAnswer](https://github.com/user-attachments/assets/a97754f4-fdaa-49e3-a1b9-33cf0f2baa02) 
### Example of RAG Expansion Answer Embedding Status Visualization
<img width="956" alt="ResultGraphExample1" src="https://github.com/user-attachments/assets/38c744a8-42dd-409a-a4f0-dda90815cb79" />
<img width="957" alt="ResultGraphExample2" src="https://github.com/user-attachments/assets/6e0f358f-5f0a-4cb4-a07c-ac27c32b9e98" />
### RAG Expansion Answer Embedding Use Cases : 
1.Information Retrieval
2.Question Answering Systems
3.E-commerce Search
4.Academic Research


## Advanced RAG - Expansion Mulitpile Queries Phase :
![RAGExpansionQueries](https://github.com/user-attachments/assets/f924f182-24c5-406d-bcb3-ae3ab12f79b5)
### Example of RAG Expansion Mulitpile Queries Embedding Status Visualization
<img width="953" alt="ResultGraphExampleMultiQueries" src="https://github.com/user-attachments/assets/6f13c385-b5f5-4836-bc6b-996fc33f9d1c" />

### RAG Expansion Mulitpile Queries Embedding Use Cases :
1.Exploring Data Analysis
2.Academic Research
3.Customer Support
4.Healthcare Information Systems


## Project Structure
- `expansion_query.py`: Main script to query the financial report and visualize the results.
- `expansion_answer.py`: Script to generate hypothetical answers and query the financial report.
- `helper_utils.py`: Utility functions for embedding projection, text extraction, and ChromaDB operations.
- `data/`: Directory containing the PDF file of the financial report.
- `.env`: Environment file containing the OpenAI API key.
- `.gitignore`: Git ignore file to exclude unnecessary files from version control.


## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/amirkhier/RAGQueryExpansion.git
    cd RAGQueryExpansion
    ```

2. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**

    Create a [.env](http://_vscodecontentref_/3) file in the root directory and add your OpenAI API key:

    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

4. **Download the financial report:**

    Place the PDF file of the financial report in the [data](http://_vscodecontentref_/4) directory.

## Usage

### Querying the Financial Report

1. **Run the [expansion_query.py](http://_vscodecontentref_/5) script:**

    ```sh
    python expansion_query.py
    ```

2. **Enter your query:**

    When prompted, enter your query related to the financial report.

3. **View the results:**

    The script will process the query, retrieve relevant documents, and visualize the results in the embedding space.

### Generating Hypothetical Answers

1. **Run the [expansion_answer.py](http://_vscodecontentref_/6) script:**

    ```sh
    python expansion_answer.py
    ```

2. **Enter your query:**

    When prompted, enter your query related to the financial report.

3. **View the results:**

    The script will generate a hypothetical answer, query the financial report, and visualize the results in the embedding space.

## Functions

### [helper_utils.py](http://_vscodecontentref_/7)

- [project_embeddings](http://_vscodecontentref_/8): Projects the given embeddings using the provided UMAP transformer.
- [word_wrap](http://_vscodecontentref_/9): Wraps the given text to the specified width.
- [extract_text_from_pdf](http://_vscodecontentref_/10): Extracts text from a PDF file.
- [load_chroma](http://_vscodecontentref_/11): Loads a document from a PDF, extracts text, generates embeddings, and stores it in a Chroma collection.

## License

This project is licensed under the MIT License.
