import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import SeleniumURLLoader  # Import SeleniumURLLoader
from dotenv import load_dotenv

load_dotenv()  # Take environment variables from .env

st.title("News Research Tool")
st.sidebar.title("News article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLS")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

# Assuming you set the OPENAI_API_KEY environment variable (remove unnecessary quotes)
openai_api_key = os.environ["OPENAI_API_KEY"]

if process_url_clicked:
    loader = SeleniumURLLoader(urls=urls)  # Specify the path to your chromedriver

    main_placeholder.text("Data Loading Started.....")
    data = loader.load()  # Call load without extra argument

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000
    )

    main_placeholder.text("Text Splitter Started.....")
    docs = text_splitter.split_documents(data)

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("Embedding Vector Started Building.....")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

question = main_placeholder.text_input("Question:")
if question:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

            # Replace davinci-003 with gpt-3.5-turbo-instruct (consider potential limitations)
            llm = OpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-instruct")  # Remove unnecessary quotes
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            # **Corrected:** Remove unnecessary openai_api argument
            result = chain({"question": question}, return_only_outputs=True)
            st.header("Answer:")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
