# ============================================================
# ADAPTIVE MULTILINGUAL RAG TUTOR
# ============================================================

import streamlit as st
import google.generativeai as genai
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Adaptive Academic RAG Tutor",
    layout="wide"
)

st.title("Adaptive Multilingual Academic RAG Tutor")

st.markdown("""
This system provides personalized academic explanations
using Retrieval-Augmented Generation (RAG) and adaptive
learning logic.
""")

# ============================================================
# GEMINI API KEY
# ============================================================

api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password"
)

if not api_key:
    st.warning("Please enter Gemini API Key")
    st.stop()

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash")

# ============================================================
# LEARNER PROFILE
# ============================================================

st.sidebar.header("Learner Profile")

language = st.sidebar.selectbox(
    "Preferred Language",
    ["English", "Malayalam", "Hindi"]
)

learning_style = st.sidebar.selectbox(
    "Learning Style",
    [
        "Step-by-Step",
        "Visual",
        "Technical",
        "Example-Based"
    ]
)

level = st.sidebar.selectbox(
    "Current Learning Level",
    [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]
)

cluster = st.sidebar.selectbox(
    "Learner Cluster",
    [
        "Cluster 0 - Moderate Learner",
        "Cluster 1 - Digitally Advanced",
        "Cluster 2 - Linguistically Constrained"
    ]
)

# ============================================================
# LOAD PREDEFINED PDF
# ============================================================

pdf_path = "finance_notes.pdf"

@st.cache_resource
def load_vector_db():

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)

    return db

db = load_vector_db()

st.success("Knowledge Base Loaded Successfully")

# ============================================================
# USER QUERY
# ============================================================

query = st.text_input(
    "Ask your academic question"
)

# ============================================================
# GENERATE RESPONSE
# ============================================================

if st.button("Generate Response"):

    if not query:
        st.warning("Please enter a question")
        st.stop()

    with st.spinner("Generating personalized explanation..."):

        results = db.similarity_search(query, k=3)

        context = "\n".join(
            [r.page_content for r in results]
        )

        # ====================================================
        # CLUSTER PERSONALIZATION
        # ====================================================

        cluster_instruction = ""

        if "Cluster 1" in cluster:

            cluster_instruction = """
            User is digitally advanced.
            Provide technically detailed explanation.
            Use conceptual depth.
            """

        elif "Cluster 2" in cluster:

            cluster_instruction = """
            User faces linguistic and learning challenges.
            Use very simple language.
            Avoid jargon.
            Explain step-by-step.
            Use multiple examples.
            """

        else:

            cluster_instruction = """
            User has moderate capability.
            Balance technical depth and simplicity.
            """

        # ====================================================
        # LANGUAGE PERSONALIZATION
        # ====================================================

        if language == "Malayalam":

            language_instruction = """
            Explain in English and Malayalam.
            """

        elif language == "Hindi":

            language_instruction = """
            Explain in English and Hindi.
            """

        else:

            language_instruction = """
            Explain only in English.
            """

        # ====================================================
        # LEARNING STYLE PERSONALIZATION
        # ====================================================

        if learning_style == "Visual":

            style_instruction = """
            Use tables and structured comparison.
            """

        elif learning_style == "Example-Based":

            style_instruction = """
            Use practical real-life examples.
            """

        elif learning_style == "Technical":

            style_instruction = """
            Include technical concepts and detailed reasoning.
            """

        else:

            style_instruction = """
            Explain step-by-step sequentially.
            """

        # ====================================================
        # LEVEL PERSONALIZATION
        # ====================================================

        if level == "Beginner":

            level_instruction = """
            Explain fundamentals first.
            Keep explanation simple.
            """

        elif level == "Advanced":

            level_instruction = """
            Provide advanced conceptual explanation.
            """

        else:

            level_instruction = """
            Use moderate explanation complexity.
            """

        # ====================================================
        # FINAL PROMPT
        # ====================================================

        prompt = f"""
        Use the following academic context:

        {context}

        Student Question:
        {query}

        Instructions:

        {cluster_instruction}

        {language_instruction}

        {style_instruction}

        {level_instruction}

        Also:
        - include concise summary
        - avoid unnecessary complexity
        """

        response = model.generate_content(prompt)

        # ====================================================
        # DISPLAY RESPONSE
        # ====================================================

        st.subheader("AI Tutor Response")

        st.write(response.text)

        # ====================================================
        # MINI ASSESSMENT
        # ====================================================

        st.subheader("Quick Understanding Check")

        quiz_prompt = f"""
        Generate ONE conceptual quiz question from:
        {query}

        Keep question concise.
        """

        quiz = model.generate_content(quiz_prompt)

        st.write(quiz.text)

        understanding = st.radio(
            "Did you understand the concept?",
            ["Yes", "No"]
        )

        # ====================================================
        # ADAPTIVE RETEACHING
        # ====================================================

        if understanding == "No":

            st.warning(
                "Re-teaching concept using simpler explanation..."
            )

            retry_prompt = f"""
            Re-teach the concept:
            {query}

            Instructions:
            - Explain fundamentals first
            - Use simple language
            - Include multiple examples
            - Avoid jargon
            """

            retry = model.generate_content(
                retry_prompt
            )

            st.subheader("Simplified Re-Explanation")

            st.write(retry.text)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Adaptive Multilingual Academic RAG Tutor | MBA BRP Prototype"
)
