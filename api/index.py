from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session, url_for
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .embedder import RawFixedEmbedder
except ImportError:
    from api.embedder import RawFixedEmbedder



BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.secret_key = os.getenv("SECRET_KEY") or os.urandom(32)

# Each subject has its own PDF and its own FAISS index. Never merge these stores.
SUBJECTS = {
    "physics": {"name": "Physics", "pdf": os.getenv("PHYSICS_PDF", "subjects/physics.pdf")},
    "computer_science": {"name": "Computer Science", "pdf": os.getenv("COMPUTER_SCIENCE_PDF", "computerscience.pdf")},
    "chemistry": {"name": "Chemistry", "pdf": os.getenv("CHEMISTRY_PDF", "subjects/chemistry.pdf")},
    "mathematics": {"name": "Mathematics", "pdf": os.getenv("MATHEMATICS_PDF", "subjects/mathematics.pdf")},
}

class CustomEmbeddings(Embeddings):
    def __init__(self, embedder: Any):
        self.embedder = embedder

    def _embed(self, text: str) -> list[float]:
        if callable(self.embedder):
            values = self.embedder(text)
        elif hasattr(self.embedder, "embed_query"):
            values = self.embedder.embed_query(text)
        else:
            raise TypeError("RawFixedEmbedder must be callable or expose embed_query().")
        return [float(value) for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def resolve_pdf_path(configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else BASE_DIR / path


def build_subject_store(subject_key: str, configured_path: str, embedding: CustomEmbeddings) -> FAISS:
    pdf_path = resolve_pdf_path(configured_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF for {SUBJECTS[subject_key]['name']} not found: {pdf_path}")
    pages = PyPDFLoader(str(pdf_path)).load()
    if not pages:
        raise ValueError(f"No readable pages found in {pdf_path.name}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError(f"No text extracted from {pdf_path.name}; scanned PDFs may need OCR.")
    # Stamp metadata so retrieved chunks can be audited to the selected source.
    for chunk in chunks:
        chunk.metadata["subject"] = subject_key
        chunk.metadata["source_pdf"] = pdf_path.name
    logger.info("Indexing %s: %d chunks from %s", subject_key, len(chunks), pdf_path.name)
    return FAISS.from_documents(chunks, embedding)


vector_stores: dict[str, FAISS] = {}
subject_errors: dict[str, str] = {}
try:
    shared_embedder = CustomEmbeddings(RawFixedEmbedder(dimensions=384))
    for key, config in SUBJECTS.items():
        try:
            vector_stores[key] = build_subject_store(key, config["pdf"], shared_embedder)
        except Exception as exc:
            subject_errors[key] = str(exc)
            logger.exception("Could not initialize %s RAG index", key)
except Exception as exc:
    logger.exception("Embedding initialization failed")
    for key in SUBJECTS:
        subject_errors.setdefault(key, f"Embedding initialization failed: {exc}")

api_key = os.getenv("GROQ_API_KEY")
llm = None
if api_key:
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0.2,
        max_tokens=450,
        groq_api_key=api_key,
    )
else:
    logger.error("GROQ_API_KEY missing. Add it to .env.")

prompt = PromptTemplate.from_template("""You are DocuMind, a concise textbook tutor.
Answer ONLY from the selected subject's retrieved textbook context below.
Do not use knowledge from another subject or invent facts.
If the answer is not supported by this context, say: "I don't know based on this subject's textbook."
Keep the normal answer under 150 words. Use short bullet points when useful.
Only provide a longer explanation if the student explicitly asks for detail.

SELECTED SUBJECT: {subject_name}
TEXTBOOK CONTEXT:
{context}

STUDENT QUESTION: {question}

ANSWER:""")


def get_payload_value(name: str, default: str = "") -> str:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        value = payload.get(name, default)
    else:
        value = request.form.get(name, default)
    return value.strip() if isinstance(value, str) else default


def wants_json_response() -> bool:
    return (request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json
            or "application/json" in request.headers.get("Accept", ""))


from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    if request.method == "GET":
        return render_template("index.html", username=session.get("username"))

    query = get_payload_value("query")
    subject_key = get_payload_value("subject", "physics").lower()
    if not query:
        return (jsonify({"error": "Please enter a question."}), 400) if wants_json_response() else render_template("index.html")
    if subject_key not in SUBJECTS:
        return (jsonify({"error": "Unknown subject selected."}), 400) if wants_json_response() else render_template("index.html")
    if subject_key not in vector_stores:
        message = f"{SUBJECTS[subject_key]['name']} textbook is not configured or could not be indexed. Check its PDF path."
        if wants_json_response():
            return jsonify({"error": message, "details": subject_errors.get(subject_key)}), 503
        return render_template("index.html", response=message), 503
    if llm is None:
        message = "Groq API is not configured. Add GROQ_API_KEY to .env and restart Flask."
        return (jsonify({"error": message}), 503) if wants_json_response() else render_template("index.html", response=message), 503

    try:
        store = vector_stores[subject_key]  # Strictly this subject's FAISS store only.
        docs = store.similarity_search(query=query, k=4)
        # Defence in depth: only pass chunks tagged for the active subject.
        docs = [doc for doc in docs if doc.metadata.get("subject") == subject_key]
        context = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
        if not context.strip():
            answer = "I don't know based on this subject's textbook. No relevant text was retrieved."
        else:
            result = llm.invoke(prompt.format(subject_name=SUBJECTS[subject_key]["name"], context=context, question=query))
            answer = result.content if hasattr(result, "content") else str(result)
            if not isinstance(answer, str):
                answer = str(answer)
        if wants_json_response():
            return jsonify({"response": answer, "subject": subject_key})
        return render_template("index.html", username=session.get("username"), response=answer, responce=answer)
    except Exception:
        logger.exception("DocuMind chat failed for subject %s", subject_key)
        message = "The assistant could not process this question. Check the Flask terminal."
        return (jsonify({"error": message}), 500) if wants_json_response() else render_template("index.html", response=message), 500



@app.route("/subject-pdf/<subject_key>")
def subject_pdf(subject_key: str):
    """Serve only the configured PDF belonging to the requested subject."""
    if subject_key not in SUBJECTS:
        abort(404)
    pdf_path = resolve_pdf_path(SUBJECTS[subject_key]["pdf"])
    if not pdf_path.is_file():
        logger.warning("PDF not found for %s: %s", subject_key, pdf_path)
        abort(404)
    return send_file(pdf_path, mimetype="application/pdf", as_attachment=False, download_name=pdf_path.name, conditional=True)

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "llm_ready": llm is not None,
        "groq_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "subjects": {key: {"name": cfg["name"], "rag_ready": key in vector_stores,
                           "pdf": cfg["pdf"], "error": subject_errors.get(key)}
                     for key, cfg in SUBJECTS.items()},
    })

# Existing demo login/register kept for compatibility. Replace with a database + password hashing before production.
user_data = {"name": [], "username": [], "email": [], "password": []}

@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        try:
            idx = user_data["username"].index(username)
        except ValueError:
            idx = -1

        if idx >= 0 and user_data["password"][idx] == password:
            session["username"] = username
            return redirect(url_for("home"))

        return render_template(
            "login.html",
            error="Invalid username or password"
        )

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if "username" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""

        if not all([name, username, email, password]):
            return render_template(
                "register.html",
                error="Please fill in all fields."
            )

        if username in user_data["username"] or email in user_data["email"]:
            return render_template(
                "register.html",
                error="Username or email already exists."
            )

        user_data["name"].append(name)
        user_data["username"].append(username)
        user_data["email"].append(email)
        user_data["password"].append(password)

        session["username"] = username

        return redirect(url_for("home"))

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
