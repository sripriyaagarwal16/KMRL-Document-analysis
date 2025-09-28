import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

# === 1. CONFIGURATION ===
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

if not API_KEY:
    raise ValueError("ERROR: GOOGLE_API_KEY is not set in .env")

# Configure Gemini SDK
genai.configure(api_key=API_KEY)

# Flask app settings
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === 2. HELPER FUNCTIONS ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_document(file_path: str):
    """Extracts text from TXT or PDF files."""
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith('.pdf'):
        try:
            doc = fitz.open(file_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            return f"Error processing PDF: {e}"
    return None

def process_image(file_path: str):
    """Handles both text extraction and diagram description using Gemini Vision."""
    image = Image.open(file_path).convert("RGB")
    prompt = """
    Analyze this image carefully:
    - If it contains text, extract it (preserve formatting).
    - If it is a technical/engineering/diagrammatic image, describe it in detail.
    - Always produce an English description if text is in another language.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content([prompt, image])
    return response.text.strip()

def analyze_with_gemini(text: str):
    """Analyzes text using Gemini SDK and returns JSON string."""
    json_schema = {
        "type": "object",
        "properties": {
            "is_relevant": {"type": "boolean"},
            "summary": {"type": "string"},
            "actions_required": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                        "deadline": {"type": "string"},
                        "notes": {"type": "string"}
                    },
                    "required": ["action", "priority"]
                }
            },
            "departments_to_notify": {"type": "array", "items": {"type": "string"}},
            "cross_document_flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "related_document_type": {"type": "string"},
                        "related_issue": {"type": "string"}
                    },
                    "required": ["related_document_type", "related_issue"]
                }
            }
        },
        "required": ["is_relevant", "summary"]
    }

    prompt = f"""
    You are an AI assistant for KMRL.

    Task:
    - First, detect if the given text is in Malayalam. 
    - If Malayalam → translate it to professional English.
    - Then, summarize it clearly.
    - Finally, generate JSON strictly following the schema.

    Text to analyze:
    ---
    {text}
    ---
    """

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=json_schema
        )
    )
    return response.text

# === 3. FLASK ROUTES ===
@app.route('/')
def index():
    return jsonify({"status": "KMRL Document Analysis API is running"}), 200

@app.route('/analyze', methods=['POST'])
def analyze_document_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "No file selected or file type not allowed."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            extracted_text = process_image(file_path)
        else:
            extracted_text = extract_text_from_document(file_path)

        if not extracted_text or "Error" in extracted_text:
            return jsonify({"error": f"Could not extract text/description. Reason: {extracted_text}"}), 500

        try:
            analysis_json_string = analyze_with_gemini(extracted_text)
            analysis_result = json.loads(analysis_json_string)

            if not analysis_result.get("is_relevant"):
                final_output = {
                    "status": "Not Applicable",
                    "reason": "The document was determined to be not relevant to KMRL.",
                    "summary": analysis_result.get("summary")
                }
            else:
                final_output = analysis_result

            return jsonify(final_output)
        except Exception as e:
            return jsonify({"error": f"Failed to analyze document: {str(e)}"}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
