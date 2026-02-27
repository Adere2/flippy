from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

CHROMA_DB_DIR = str(Path(__file__).resolve().parents[2] / "data/chroma_db/fuzzfiles")


@tool
def search_fuzzfile_examples(query: str) -> str:
    """
    Search for Fuzzfile (.fz) examples and templates based on a user's request.
    Use this tool when you need to see how to write a Fuzzfile for a specific job
    type (e.g., MPI, GPU, simple dependency).
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    try:
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR, embedding_function=embeddings
        )

        # We only return the top 3 so we don't blow up the LLM's context window
        results = vector_store.similarity_search(query, k=3)

        if not results:
            return "No relevant Fuzzfile examples found."

        formatted_results = []
        for doc in results:
            # Extract the filename from the metadata to give the LLM context
            source = doc.metadata.get("source", "Unknown File")
            filename = Path(source).name
            formatted_results.append(
                f"--- Example: {filename} ---\n{doc.page_content}\n"
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error querying Fuzzfile vector store: {str(e)}"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    result = search_fuzzfile_examples.invoke({"query": "GPU job example"})
    print(result)
