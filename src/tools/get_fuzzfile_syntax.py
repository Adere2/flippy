import os
from pathlib import Path

import frontmatter
from langchain_core.tools import tool


@tool
def get_fuzzfile_syntax(query: str = "") -> str:
    """
    Returns the complete official Fuzzfile syntax, schema, and specification.
    ALWAYS call this tool before you generate or modify a Fuzzfile to ensure you
    are using the correct keys, sections (like jobs, volumes, image), and formatting.
    """
    # Get the base docs directory from the environment
    docs_dir = os.getenv("FUZZDOCS_DIR")
    if not docs_dir:
        return "Error: FUZZDOCS_DIR environment variable is not set."

    # UPDATE THIS to the actual file name/path of the syntax page in your Hugo docs
    SYNTAX_FILE_NAME = "appendices/workflow-syntax.md"

    syntax_file_path = Path(docs_dir) / SYNTAX_FILE_NAME

    # Fallback: Search the docs dir if the exact path isn't right
    if not syntax_file_path.exists():
        possible_files = list(Path(docs_dir).rglob("workflow-syntax.md"))
        if possible_files:
            syntax_file_path = possible_files[0]
        else:
            return "Error: Could not find the Fuzzfile syntax file."

    try:
        # 🎯 USE FRONTMATTER: This safely parses the file and separates the
        # Hugo YAML metadata from the actual Markdown content.
        post = frontmatter.load(syntax_file_path)
        clean_content = post.content

        return f"--- OFFICIAL FUZZFILE SYNTAX SPECIFICATION ---\n\n{clean_content}"

    except Exception as e:
        return f"Error reading syntax file: {str(e)}"


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    result = get_fuzzfile_syntax.invoke({"query": ""})
    print(result)
