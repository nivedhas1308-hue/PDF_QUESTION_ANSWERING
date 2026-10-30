import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma

# Load questions
queries_df = pd.read_csv("dataset/queries.csv")

# Load embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Connect to ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="pdf_collection"
)

# Load LLM
llm = ChatOllama(model="llama3.2", temperature=0)

answers = []

for query in queries_df["question"]:

    results = vectorstore.similarity_search(query, k=5)

    context = "\n".join(
        [doc.page_content for doc in results]
    )

    prompt = f"""
    Answer only using the following PDF content.

    Context:
    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)

    answers.append({
        "question": query,
        "answer": response.content
    })

pd.DataFrame(answers).to_csv(
    "dataset/source_answers.csv",
    index=False
)

print("PDF Question Answering completed!")