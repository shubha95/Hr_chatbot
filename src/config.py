
import os
from pathlib import Path
from dotenv import load_dotenv

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "hr_policy"
# Load local .env
load_dotenv(BASE_DIR / ".env")

# API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Default providers: Gemini
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()
EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER", "google"
).lower()

# Model configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "google": "models/gemini-embedding-001",
}

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    EMBEDDING_MODELS.get(EMBEDDING_PROVIDER)
)

# Paths
PDF_PATH = BASE_DIR / "data" / "hr_policy.pdf"
CHROMA_PATH = BASE_DIR / "chroma_db"


def get_llm():

    if LLM_PROVIDER == "google":

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing")

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.2
        )

    elif LLM_PROVIDER == "openai":

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY
        )

    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


def get_embedding_model(for_query=False):

    if EMBEDDING_PROVIDER == "google":

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing")

        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY
        )

    elif EMBEDDING_PROVIDER == "openai":

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing")

        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )

    raise ValueError(
        f"Unsupported embedding provider: {EMBEDDING_PROVIDER}"
    )
