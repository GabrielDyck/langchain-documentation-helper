import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
# Import dotenv to load environment variables from a .env file
from dotenv import load_dotenv
# Import text splitter for breaking documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import Chroma vector store for storing embeddings
from langchain_chroma import Chroma
# Import Document class for handling document objects
from langchain_core.documents import Document
# Import OpenAIEmbeddings for generating vector embeddings
from langchain_openai import OpenAIEmbeddings
# Import PineconeVectorStore for alternative vector storage
from langchain_pinecone import PineconeVectorStore
# Import Tavily tools for web crawling and extraction
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

# Import custom logger utilities and color codes
from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

# Load environment variables from .env file
load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())  # Create SSL context with certifi CA bundle
os.environ["SSL_CERT_FILE"] = certifi.where()  # Set SSL cert file environment variable
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()  # Set requests CA bundle environment variable


# Initialize OpenAI embeddings with specific model and parameters
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Model name for embeddings
    show_progress_bar=False,          # Disable progress bar
    chunk_size=50,                    # Number of items per batch
    retry_min_seconds=10,             # Minimum seconds to wait before retry
)
# Initialize Chroma vector store for persistent storage of embeddings
# Chroma is an open-source vector database designed for storing and querying vector embeddings efficiently.
# It supports similarity search, filtering, and metadata management, making it suitable for semantic search and retrieval tasks.
# Here, we configure Chroma to persist its data in the 'chroma_db' directory, ensuring that indexed embeddings are saved across runs.
# The embedding_function argument links Chroma to the OpenAIEmbeddings instance, so all documents added are automatically converted to vector representations using the specified model.
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# Alternative: Initialize Pinecone vector store (commented out)
# vectorstore = PineconeVectorStore(
#     index_name="langchain-docs-2025", embedding=embeddings
# )
# Initialize Tavily extraction tool for extracting content from web pages
tavily_extract = TavilyExtract()
# Initialize Tavily map tool for mapping web content with depth and breadth limits
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
# Initialize Tavily crawl tool for crawling web pages
tavily_crawl = TavilyCrawl()

# Define asynchronous function to index documents in batches
async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")  # Log the start of the vector storage phase
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        # This inner async function adds a batch of documents to the vector store.
        # It uses the vectorstore.aadd_documents method, which is asynchronous and optimized for batch operations.
        # If the operation succeeds, it logs a success message with batch details.
        # If an exception occurs, it logs the error and returns False, allowing for robust error handling and reporting.
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Prepare a list of async tasks, one for each batch, to be processed concurrently.
    # This leverages asyncio.gather to run all batch additions in parallel, maximizing throughput and minimizing total processing time.
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count how many batches were successfully processed by checking the results list.
    # This provides a summary of the operation's reliability and helps identify any failures for further investigation.
    successful = sum(1 for result in results if result is True)

    # Log the outcome: if all batches succeeded, log a success; otherwise, log a warning with the number of successful batches.
    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""  # Function docstring describing its purpose
    # This function coordinates the entire ingestion pipeline, from crawling documentation to chunking and indexing.
    # It uses async/await to ensure non-blocking execution, allowing for efficient handling of I/O-bound tasks like web crawling and vector storage.
    log_header("DOCUMENTATION INGESTION PIPELINE")  # Log the start of the pipeline

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",  # Log crawling start
        Colors.PURPLE,  # Use purple color for log
    )
    # Crawl the documentation site using TavilyCrawl
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",  # URL to crawl
        "max_depth": 2,  # Maximum depth for crawling
        "extract_depth": "advanced"  # Extraction depth setting
    })
    all_docs = res["results"]  # Extract results from crawl response

    # Split documents into chunks for processing
    log_header("DOCUMENT CHUNKING PHASE")  # Log the start of chunking phase
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",  # Log chunking details
        Colors.YELLOW,  # Use yellow color for log
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)  # Initialize text splitter
    splitted_docs = text_splitter.split_documents(all_docs)  # Split documents into chunks
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"  # Log chunking success
    )

    # Process documents asynchronously by indexing them in vector store
    await index_documents_async(splitted_docs, batch_size=500)  # Index chunks in batches

    log_header("PIPELINE COMPLETE")  # Log pipeline completion
    log_success("🎉 Documentation ingestion pipeline finished successfully!")  # Log success message
    log_info("📊 Summary:", Colors.BOLD)  # Log summary header
    log_info(f"   • Documents extracted: {len(all_docs)}")  # Log number of documents extracted
    log_info(f"   • Chunks created: {len(splitted_docs)}")  # Log number of chunks created


if __name__ == "__main__":
    asyncio.run(main())
