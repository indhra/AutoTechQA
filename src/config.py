"""
Configuration file for AutoTechQA application.
Stores all configurable parameters that can be used across different modules.
"""

# LLM Configuration
llm_model_name = "gpt-4"

# Embedding Configuration
embedding_model_name = "nomic-ai/nomic-embed-text-v1"

# Text Splitting Configuration
chunk_size = 1000
chunk_overlap = 100

# Paths Configuration
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_SOURCE_FOLDER = os.path.join(PROJECT_ROOT, 'data_source')
VECTOR_STORE_FOLDER = os.path.join(PROJECT_ROOT, 'vector_store')

# Retriever Configuration
TOP_K_RETRIEVER = 5