import os
import streamlit as st
import requests
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Hardcoded SerpAPI key (safe for dev/testing)
SERP_API_KEY = "202f70d3fa3ddfd0b88cf84725f31fe8605264cb618a862838cef1f852de088b"

st.set_page_config(page_title="Resume RAG App", layout="centered")
st.title("📄→💼 Resume Job Role Finder (with Apply Links)")

# Ask for Gemini API key
gemini_key = st.text_input("🔐 Enter your Gemini API Key:", type="password")
resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

if resume_file and gemini_key:
    try:
        os.environ["GOOGLE_API_KEY"] = gemini_key
        genai.configure(api_key=gemini_key)

        # Check if flash model is available
        supported_models = {
            m.name: m.supported_generation_methods for m in genai.list_models()
        }
        flash_available = any(
            "generateContent" in methods and "gemini-1.5-flash" in name
            for name, methods in supported_models.items()
        )

        if not flash_available:
            st.error("❌ Your API key does not have access to 'gemini-1.5-flash'. Please use a valid key from https://console.cloud.google.com/")
            st.stop()

        with open("resume.pdf", "wb") as f:
            f.write(resume_file.read())

        loader = PyPDFLoader("resume.pdf")
        pages = loader.load_and_split()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        # ✅ Use Gemini Flash model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        query = "Based on this resume, suggest 3-4 suitable job roles with relevant skills."
        st.subheader("🔍 Suggested Roles:")
        with st.spinner("Analyzing your resume..."):
            result = qa.run(query)

        st.success("✅ Roles Identified!")
        st.markdown(result)

        # Extract job titles
        roles = []
        for line in result.split("\n"):
            if "-" in line or "•" in line:
                title = line.strip("•- ").split("–")[0].strip()
                if title:
                    roles.append(title)

        st.subheader("🔗 Apply Links (from Google Jobs)")
        for role in roles:
            query = f"{role} jobs site:linkedin.com"
            url = f"https://serpapi.com/search.json?q={query}&api_key={SERP_API_KEY}&engine=google"
            try:
                resp = requests.get(url)
                data = resp.json().get("organic_results", [])
                if data:
                    st.markdown(f"**{role}**")
                    for job in data[:2]:
                        st.markdown(f"- [{job['title']}]({job['link']})")
                else:
                    st.write(f"No jobs found for: {role}")
            except Exception as e:
                st.write("🔌 Failed to fetch jobs:", str(e))

    except Exception as e:
        st.error(f"❌ Something went wrong:\n\n{e}")

elif not gemini_key:
    st.info("🔑 Please enter your Gemini API key to begin.")
