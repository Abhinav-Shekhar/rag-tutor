# ============================================================
# ADAPTIVE MULTILINGUAL RAG TUTOR
# FINAL WORKING STREAMLIT VERSION
# ============================================================

import streamlit as st
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# GEMINI API CONFIG
# ============================================================

api_key = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=api_key)

# IMPORTANT FIX
model = genai.GenerativeModel(
    "gemini-2.0-flash"
)

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
# LOAD PDF + VECTOR DATABASE
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

    db = FAISS.from_documents(
        docs,
        embeddings
    )

    return db

db = load_vector_db()

st.success("Knowledge Base Loaded Successfully")

# ============================================================
# USER QUESTION
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

        # ====================================================
        # RETRIEVE RELEVANT CHUNKS
        # ====================================================

        results = db.similarity_search(
            query,
            k=3
        )

        context = "\n".join(
            [r.page_content for r in results]
        )

        # ====================================================
        # CLUSTER PERSONALIZATION
        # ====================================================

        if "Cluster 1" in cluster:

            cluster_instruction = """
            User is digitally advanced.
            Provide detailed technical explanation.
            Include conceptual depth and terminology.
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
            Balance simplicity and technical depth.
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
            Use practical examples and analogies.
            """

        elif learning_style == "Technical":

            style_instruction = """
            Include detailed technical reasoning.
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
            Keep explanation beginner-friendly.
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
        You are an adaptive academic tutor.

        Use ONLY the provided academic context.

        ===================================================

        CONTEXT:
        {context}

        ===================================================

        STUDENT QUESTION:
        {query}

        ===================================================

        INSTRUCTIONS:

        {cluster_instruction}

        {language_instruction}

        {style_instruction}

        {level_instruction}

        Also:
        - Include concise summary
        - Avoid unnecessary complexity
        - Ensure educational clarity
        """

        # ====================================================
        # GENERATE RESPONSE
        # ====================================================

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
        Generate ONE short conceptual quiz question
        based on:
        {query}

        Keep it concise and beginner-friendly.
        """

        quiz = model.generate_content(
            quiz_prompt
        )

        st.write(quiz.text)

        understanding = st.radio(
            "Did you understand the concept?",
            ["Yes", "No"]
        )

        # ====================================================
        # ADAPTIVE RE-TEACHING
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
            - Include examples
            - Avoid jargon
            """

            retry = model.generate_content(
                retry_prompt
            )

            st.subheader(
                "Simplified Re-Explanation"
            )

            st.write(retry.text)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Adaptive Multilingual Academic RAG Tutor | MBA BRP Prototype"
)
