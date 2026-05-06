from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader

def load_documents(data_path: str):
    loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {data_path}")
    return documents