from pathlib import Path

from langchain_core.documents import Document


def parse_workflow_app(app_dir: Path) -> Document | None:
    """
    Reads a Fuzzball workflow application directory, concatenates its
    relevant configuration and documentation files, and returns a
    single LangChain Document representing the entire application package.
    """
    if not app_dir.is_dir():
        return None

    # We only care about config files, Fuzzfiles, and docs
    relevant_extensions = (".fz", ".yaml", ".yml", ".md", ".sh")
    relevant_files = [
        f
        for f in app_dir.iterdir()
        if f.is_file() and f.name.endswith(relevant_extensions)
    ]

    # If this folder doesn't have relevant app files, skip it
    if not relevant_files:
        return None

    app_name = app_dir.name
    combined_content = f"Application Name: {app_name}\n\n"

    for file_path in relevant_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()

            combined_content += f"==============================\n"
            combined_content += f"FILE: {file_path.name}\n"
            combined_content += f"==============================\n"
            combined_content += f"{file_content}\n\n"
        except Exception as e:
            print(f"⚠️ Could not read {file_path}: {e}")

    metadata = {
        "source": str(app_dir),
        "app_name": app_name,
    }

    return Document(page_content=combined_content, metadata=metadata)


# Quick test block
if __name__ == "__main__":
    test_dir = Path("/Users/adere/code/ciq-fuzzball-catalog/applications").resolve()
    # Grab the first subfolder to test
    if test_dir.exists():
        first_app = next(
            d for d in test_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        doc = parse_workflow_app(first_app)
        if doc:
            print("--- PARSED METADATA ---")
            print(doc.metadata)
            print("\n--- COMPOSITE CONTENT ---")
            print(doc.page_content)
