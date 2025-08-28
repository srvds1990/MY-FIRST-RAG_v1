import streamlit as st
import os
from dotenv import load_dotenv

# We need these from LangChain to build the RAG pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Setup and Initialization ---

load_dotenv()
# Get the OpenAI API key from the environment
openai_api_key = os.getenv("OPEN_AI_API_KEY")
# Get the LangSmith tracing configuration
langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT")

# Initialize LLM and Embeddings, storing them in session_state to prevent re-initialization
if "llm" not in st.session_state:
    if not openai_api_key:
        st.error("OpenAI API key is not found. Please add it to a `.env` file.")
    else:
        st.session_state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
        # Initialize the Hugging Face Embeddings model and store in session state
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --- Document Processing and Vector Store Creation ---

@st.cache_resource
def get_vectorstore(pdf_files, embeddings_model):
    """
    Loads multiple PDFs, splits them into chunks, creates embeddings, and builds a vector store.
    This function is cached to avoid re-processing the documents on every run.
    """
    all_docs = []
    try:
        # Create a directory for temporary files if it doesn't exist
        os.makedirs("./data", exist_ok=True)
        
        for uploaded_file in pdf_files:
            # Save the uploaded file temporarily
            file_path = os.path.join("./data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the document
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)

        # Split the combined document into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_docs)

        # Create a vector store from the document chunks
        vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings_model)
        return vectorstore
    except Exception as e:
        st.error(f"Error processing PDFs: {e}")
        return None

# --- RAG Chain Definition ---

def create_rag_chain(vectorstore, llm_model):
    """
    Creates the complete RAG chain for question answering.
    """
    if vectorstore is None:
        return None

    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context.
        If the answer is not in the context, just say that you don't have enough information.

        <context>
        {context}
        </context>
        
        Question: {input}
        """
    )

    # Define the retriever to fetch relevant documents
    retriever = vectorstore.as_retriever()
    
    # Create the document chain to combine documents with the prompt
    document_chain = create_stuff_documents_chain(llm_model, prompt_template)
    
    # Create the retrieval chain that combines the retriever and the document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- Streamlit UI ---

def main():
    st.title("Document-Based Q&A App")
    st.subheader("Ask questions about your PDF files!")

    # Check for LangSmith environment variables and provide guidance
    if not all([langchain_tracing_v2, langchain_api_key, langchain_project]):
        st.info("LangSmith tracing is not fully configured. To enable it, please set the following environment variables in your `.env` file:")
        st.markdown(
            """
            `LANGCHAIN_TRACING_V2=true`
            `LANGCHAIN_API_KEY=your_langsmith_api_key`
            `LANGCHAIN_PROJECT=your_project_name`
            """
        )

    # File uploader widget for multiple files
    uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)
    
    # Check if files are uploaded
    if uploaded_files:
        # Check if the vectorstore is not in session state or if new files are uploaded
        if "vectorstore" not in st.session_state or st.button("Process Documents"):
            with st.spinner("Processing documents... This may take a moment."):
                vectorstore = get_vectorstore(uploaded_files, st.session_state.embeddings)
                st.session_state["vectorstore"] = vectorstore
                if vectorstore:
                    st.success("Documents processed and ready for questions!")
    else:
        # Display a message to the user when no files are uploaded
        st.info("Please upload one or more PDF files to get started.")

    # Check if the vectorstore is in session state
    if "vectorstore" in st.session_state and st.session_state["vectorstore"] is not None:
        # Create the RAG chain
        rag_chain = create_rag_chain(st.session_state["vectorstore"], st.session_state.llm)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Accept user input
        if prompt := st.chat_input("Ask a question about the documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Invoke the RAG chain to get the response
                    response = rag_chain.invoke({"input": prompt})
                    st.markdown(response["answer"])
                    
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    main()
