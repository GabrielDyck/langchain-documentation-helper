from dotenv import load_dotenv  # Import the function to load environment variables from a .env file



# Import type hints for better code clarity and type checking
from typing import Any, Dict, List

# Import hub to access pre-defined prompts from LangChain's hub
from langchain import hub
# Import function to create a chain that combines document results
from langchain.chains.combine_documents import create_stuff_documents_chain
# Import function to create a retriever that uses chat history
from langchain.chains.history_aware_retriever import create_history_aware_retriever
# Import function to create a retrieval QA chain
from langchain.chains.retrieval import create_retrieval_chain
# Import OpenAI chat model and embeddings for LLM and vector search
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# Import Pinecone vector store for document retrieval
from langchain_pinecone import PineconeVectorStore

# Import the index name constant for Pinecone usage
from consts import INDEX_NAME

# The run_llm function orchestrates the retrieval-augmented generation pipeline for answering user queries.
# It leverages embeddings, a vector store, prompt engineering, and chain composition to provide context-aware answers.
def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    # Step 1: Create an embeddings object using OpenAI's small embedding model for vector search.
    # This model converts text into high-dimensional vectors, enabling semantic search and retrieval
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Step 2: Initialize Pinecone vector store with the index name and embeddings for document retrieval.
    # Pinecone stores the document vectors and allows fast similarity search for relevant context
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Step 3: Create a ChatOpenAI object for LLM responses, with verbose logging and deterministic output.
    # This is the language model that generates answers based on retrieved context and user queries
    chat = ChatOpenAI(verbose=True, temperature=0)

    # Step 4: Pull a prompt from LangChain hub to rephrase user queries using chat history.
    # This prompt helps the model understand the user's intent by considering previous conversation turns
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    # Step 5: Pull a prompt for retrieval QA chat from LangChain hub.
    # This prompt guides the LLM to answer questions using retrieved documents as context
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Step 6: Create a chain to combine retrieved documents and generate answers.
    # The stuff_documents_chain merges the retrieved context and passes it to the LLM for answer generation
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    # Step 7: Create a retriever that uses chat history to improve retrieval relevance.
    # The history-aware retriever reformulates the query based on chat history, improving context matching
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    # Step 8: Create a retrieval QA chain that combines the retriever and document chain.
    # This chain retrieves relevant documents and generates an answer using the LLM and context
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    # Step 9: Run the QA chain with the user's query and chat history, getting the result.
    # The chain returns an answer that is informed by both the retrieved documents and the conversation history
    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    # Step 10: Return the result to the caller.
    # The final output is the LLM's answer to the user's query, grounded in retrieved context
    return result
