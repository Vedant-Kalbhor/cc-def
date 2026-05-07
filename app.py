import os
import logging
from dotenv import load_dotenv

from flask import Flask, flash, jsonify, render_template, request, session
import google.generativeai as genai
from groq import Groq
import PyPDF2

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "dev-secret-key"
)

app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PDF_CONTEXT_STORE = {}

# ─────────────────────────────────────────
# Models
# ─────────────────────────────────────────
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

SKIP_SIGNALS = (
    "429",
    "quota",
    "resource_exhausted",
    "rate_limit",
    "too many requests",
    "404",
    "not found",
    "not supported",
    "503",
)

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def should_skip(error_text):

    err = error_text.lower()

    return any(sig in err for sig in SKIP_SIGNALS)


def call_llm(prompt, system=""):

    errors = []

    # ─────────────────────────
    # Gemini
    # ─────────────────────────
    if GEMINI_API_KEY:

        for model_id in GEMINI_MODELS:

            try:

                log.info(f"[Gemini] Trying {model_id}")

                model = genai.GenerativeModel(model_id)

                full_prompt = (
                    f"{system}\n\n{prompt}"
                    if system else prompt
                )

                response = model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 8192,
                    }
                )

                if hasattr(response, "text") and response.text:

                    log.info(f"[Gemini] Success {model_id}")

                    return (
                        response.text,
                        f"Gemini / {model_id}"
                    )

                raise Exception("Empty response")

            except Exception as e:

                err = str(e)

                if should_skip(err):

                    log.warning(
                        f"[Gemini] Skip {model_id}: {err}"
                    )

                    errors.append(
                        f"[Gemini/{model_id}] skipped"
                    )

                    continue

                errors.append(
                    f"[Gemini/{model_id}] {err}"
                )

    # ─────────────────────────
    # Groq
    # ─────────────────────────
    if GROQ_API_KEY:

        client = Groq(api_key=GROQ_API_KEY)

        messages = []

        if system:

            messages.append({
                "role": "system",
                "content": system
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        for model_id in GROQ_MODELS:

            try:

                log.info(f"[Groq] Trying {model_id}")

                chat = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=8192,
                )

                answer = (
                    chat.choices[0]
                    .message
                    .content
                )

                log.info(f"[Groq] Success {model_id}")

                return (
                    answer,
                    f"Groq / {model_id}"
                )

            except Exception as e:

                err = str(e)

                if should_skip(err):

                    errors.append(
                        f"[Groq/{model_id}] skipped"
                    )

                    continue

                errors.append(
                    f"[Groq/{model_id}] {err}"
                )

    raise RuntimeError(
        "All models exhausted.\n"
        + "\n".join(errors)
    )

# ─────────────────────────────────────────
# PDF Helpers
# ─────────────────────────────────────────
def save_uploaded_pdfs(files):

    saved = []

    for file in files:

        if not file or not file.filename:
            continue

        filename = file.filename

        save_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        base, ext = os.path.splitext(filename)

        counter = 1

        while os.path.exists(save_path):

            save_path = os.path.join(
                UPLOAD_FOLDER,
                f"{base}_{counter}{ext}"
            )

            counter += 1

        file.save(save_path)

        saved.append(save_path)

    return saved


def extract_text_from_pdfs(file_paths):

    combined = ""

    for path in file_paths:

        try:

            with open(path, "rb") as f:

                reader = PyPDF2.PdfReader(f)

                text = ""

                for page in reader.pages:

                    content = page.extract_text()

                    if content:
                        text += content

            combined += (
                f"\n--- DOCUMENT: "
                f"{os.path.basename(path)} ---\n"
                f"{text}\n"
            )

        except Exception as e:

            flash(
                f"Error reading "
                f"{os.path.basename(path)}: {e}",
                "error"
            )

    return combined

# ─────────────────────────────────────────
# AI System Prompts
# ─────────────────────────────────────────
PAPER_SYSTEM = """
You are a world-class IEEE research paper writer.

Generate a COMPLETE detailed IEEE-style research paper.

IMPORTANT:
- Output MUST be clean HTML content
- DO NOT output markdown
- DO NOT output LaTeX

STRICT REQUIREMENTS:

- Generate 5000–7000 words
- Paper should become 7–8 pages
- Expand concepts deeply
- Add detailed technical content
- Add comparison tables
- Add methodology
- Add algorithms
- Add architecture explanations
- Add future scope
- Add references

MANDATORY SECTIONS:

1. Title
2. Abstract
3. Keywords
4. Introduction
5. Problem Statement
6. Literature Review
7. Existing System
8. Proposed Methodology
9. Architecture
10. Algorithm
11. Mathematical Model
12. Experimental Results
13. Comparative Analysis
14. Advantages
15. Limitations
16. Future Scope
17. Conclusion
18. References

Output ONLY HTML body content.
"""

CHAT_SYSTEM = """
You are an expert research assistant.

Answer ONLY from provided PDF context.
"""

# ─────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():

    paper_output = ""

    title = "Analysis of Uploaded Works"

    authors = "Vedant Kalbhor"

    saved_files = []

    if request.method == "POST":

        title = request.form.get(
            "title",
            title
        ).strip()

        authors = request.form.get(
            "authors",
            authors
        ).strip()

        uploaded = request.files.getlist("pdfs")

        if not uploaded or not uploaded[0].filename:

            flash(
                "⚠️ Upload at least one PDF",
                "warning"
            )

        else:

            saved_files = save_uploaded_pdfs(
                uploaded
            )

            flash(
                f"📄 {len(saved_files)} PDF(s) saved",
                "info"
            )

            raw_text = extract_text_from_pdfs(
                saved_files
            )

            flash(
                "⚙️ Generating 7–8 page paper...",
                "info"
            )

            try:

                prompt = f"""
Generate a COMPLETE IEEE-style research paper.

TITLE:
{title}

AUTHORS:
{authors}

SOURCE CONTENT:
{raw_text[:100000]}

IMPORTANT:
Generate a FULL academic paper
of approximately 7–8 pages.

Output ONLY HTML.
"""

                paper_output, used_model = call_llm(
                    prompt,
                    PAPER_SYSTEM
                )

                flash(
                    f"✅ Generated via {used_model}",
                    "success"
                )

            except RuntimeError as e:

                flash(
                    f"❌ {e}",
                    "error"
                )

    return render_template(
        "index.html",
        paper_output=paper_output,
        title=title,
        authors=authors,
        file_count=len(saved_files),
    )

# ─────────────────────────────────────────
# Chatbot Page
# ─────────────────────────────────────────
@app.route("/chatbot")
def chatbot():

    return render_template("chatbot.html")

# ─────────────────────────────────────────
# Upload PDFs
# ─────────────────────────────────────────
@app.route("/chatbot/upload", methods=["POST"])
def chatbot_upload():

    files = request.files.getlist("pdfs")

    if not files or not files[0].filename:

        return jsonify({
            "ok": False,
            "error": "No PDFs uploaded"
        }), 400

    saved = save_uploaded_pdfs(files)

    context = extract_text_from_pdfs(saved)

    if not context.strip():

        return jsonify({
            "ok": False,
            "error": "Could not extract text"
        }), 400

    session_id = os.urandom(16).hex()

    PDF_CONTEXT_STORE[session_id] = {
        "context": context,
        "history": []
    }

    session["pdf_session_id"] = session_id

    return jsonify({
        "ok": True,
        "files": [
            os.path.basename(p)
            for p in saved
        ],
        "chars": len(context)
    })

# ─────────────────────────────────────────
# Ask Questions
# ─────────────────────────────────────────
@app.route("/chatbot/ask", methods=["POST"])
def chatbot_ask():

    data = request.get_json(force=True)

    question = (
        data.get("question") or ""
    ).strip()

    if not question:

        return jsonify({
            "ok": False,
            "error": "Empty question"
        }), 400

    session_id = session.get(
        "pdf_session_id"
    )

    if (
        not session_id
        or session_id not in PDF_CONTEXT_STORE
    ):

        return jsonify({
            "ok": False,
            "error": "No PDF loaded"
        }), 400

    pdf_data = PDF_CONTEXT_STORE[
        session_id
    ]

    context = pdf_data["context"]

    history = pdf_data["history"]

    history_block = ""

    for turn in history[-6:]:

        history_block += (
            f"User: {turn['user']}\n"
            f"Assistant: "
            f"{turn['assistant']}\n\n"
        )

    prompt = f"""
PDF CONTEXT:
{context[:50000]}

CHAT HISTORY:
{history_block}

QUESTION:
{question}

ANSWER:
"""

    try:

        answer, used_model = call_llm(
            prompt,
            CHAT_SYSTEM
        )

    except RuntimeError as e:

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

    history.append({
        "user": question,
        "assistant": answer
    })

    pdf_data["history"] = history[-20:]

    return jsonify({
        "ok": True,
        "answer": answer,
        "model": used_model
    })

# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────
if __name__ == "__main__":

    app.run(debug=True)

