# read pdf from data source folder and save as vector database using chromadb in vector_store folder
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def ingest_pdfs_to_vector_store(input_folder, output_folder):
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name='nomic-ai/nomic-embed-text-v1',
        model_kwargs={'trust_remote_code': True}
    )
    if not os.listdir(input_folder):
       raise ValueError("Input folder is empty. Please provide a valid input folder.")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print('Using HuggingFace \
          Embeddings with model nomic-ai/nomic-embed-text-v1')
    print("only PDF files will be ingested from the data_source folder.")

    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            file_path = os.path.join(input_folder, filename)
            # use recursive character text splitter
            
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            
            vector_store = Chroma.from_documents(
                texts,
                embeddings,
                persist_directory=os.path.join(output_folder, filename.replace('.pdf', '_chroma'))
            )
            # vector_store.persist()

 
            print(f'Ingested and saved vector store for {filename}')


if __name__ == '__main__':
    # Get the script directory and construct absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_data_folder = os.path.join(project_root, 'data_source')
    output_vector_store_folder = os.path.join(project_root, 'vector_store')
    
    print(f'Input Data Folder: {input_data_folder}')
    print(f'Output Vector Store Folder: {output_vector_store_folder}')

    ingest_pdfs_to_vector_store(input_data_folder, output_vector_store_folder)