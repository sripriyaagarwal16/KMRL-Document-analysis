import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# === 1. CONFIGURATION ===
load_dotenv()  # Loads variables from a .env file
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Flask App Settings ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)  # Initialize CORS for the entire app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Configure Gemini API ---
if not GEMINI_API_KEY:
    print("🔴 ERROR: Gemini API key is not set in the .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully.")


# === 2. GEMINI-POWERED HELPER FUNCTIONS ===

def classify_image_with_gemini(image: Image.Image):
    """Uses Gemini to classify an image as a 'document' or 'diagram'."""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = "Is this image primarily a text document or an engineering/technical diagram? Answer with only the single word: 'document' or 'diagram'."
    response = model.generate_content([prompt, image])
    classification = response.text.strip().lower()
    print(f"✅ Image classified as: {classification}")
    return "diagram" if "diagram" in classification else "document"


def process_image_with_gemini(image_path: str):
    """
    Orchestrates image processing using Gemini: classifies, then either OCRs or summarizes.
    """
    print(f"--- Processing Image: {image_path} ---")
    try:
        image = Image.open(image_path)
        image_type = classify_image_with_gemini(image)

        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        if image_type == "diagram":
            print("-> Diagram identified. Generating summary...")
            prompt = "You are an engineering assistant for Kochi Metro Rail Limited (KMRL). Describe the contents of this technical diagram or engineering drawing in a concise summary. Identify key components and their apparent purpose."
            response = model.generate_content([prompt, image])
            return response.text.strip()
        else:  # Document
            print("-> Document identified. Performing OCR...")
            prompt = "Extract all text from this image. Preserve the formatting and line breaks as accurately as possible."
            response = model.generate_content([prompt, image])
            return response.text.strip()

    except Exception as e:
        print(f"❌ Error processing image with Gemini: {e}")
        return f"Error during image processing: {e}"


def extract_text_from_file_with_gemini(file_path: str):
    """Extracts text from PDF or TXT files using Gemini."""
    print(f"--- Processing File: {file_path} ---")
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        # For PDFs, upload the file and ask Gemini to read it
        uploaded_file = genai.upload_file(path=file_path)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = "Extract all text content from this document. Maintain the original structure and paragraphs."
        response = model.generate_content([prompt, uploaded_file])

        # Clean up the uploaded file after use
        genai.delete_file(uploaded_file.name)

        return response.text.strip()
    except Exception as e:
        print(f"❌ Error extracting text with Gemini: {e}")
        return f"Error during text extraction: {e}"


def analyze_and_translate_with_gemini(text_to_analyze: str):
    """
    Uses Gemini to perform three tasks in one call:
    1. Detect if the text is Malayalam and translate to English if needed.
    2. Generate a structured JSON analysis of the English text.
    3. Check if the content is relevant to KMRL.
    """
    print("--- Analyzing text with Gemini ---")
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # Define the desired JSON output structure
    json_schema = {
        "type": "object",
        "properties": {
            "is_relevant": {
                "type": "boolean",
                "description": "True if the document summary is related to transportation, infrastructure, railways, or metro systems. Otherwise, false."
            },
            "summary": {
                "type": "string",
                "description": "A concise summary of the document's content."
            },
            "actions_required": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                        "deadline": {"type": "string", "description": "e.g., 'YYYY-MM-DD' or 'N/A'"},
                        "notes": {"type": "string"}
                    }, "required": ["action", "priority"]
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
                    }, "required": ["related_document_type", "related_issue"]
                }
            }
        }, "required": ["is_relevant", "summary"]
    }

    prompt = f"""
    You are an AI assistant for KMRL.
    Step 1: Language Check. If the following text is primarily in Malayalam, translate it to professional English. If it's already in English, use the original text.
    Step 2: Analysis. Using the English text, generate a JSON object that strictly adheres to the provided schema.
    Text:
    ---
    {text_to_analyze}
    ---
    """

    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=json_schema
    )

    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        return json.loads(response.text)
    except Exception as e:
        print(f"❌ Gemini JSON generation failed: {e}")
        fallback_prompt = f"Summarize the key points and required actions from the following text: {text_to_analyze}"
        fallback_response = model.generate_content(fallback_prompt)
        return {"error": "Failed to generate structured JSON.", "summary": fallback_response.text}


# === 3. FLASK ROUTES ===

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Health check endpoint to confirm the API is running."""
    return jsonify({"status": "KMRL Document Analysis API is running"}), 200


@app.route('/analyze', methods=['POST'])
def analyze_document():
    """Handles the file upload and the entire analysis pipeline."""
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API key is not configured on the server."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request. Please upload a file with the key 'file'."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg']:
                original_text = process_image_with_gemini(file_path)
            else:  # .pdf, .txt
                original_text = extract_text_from_file_with_gemini(file_path)

            if not original_text or "Error" in original_text:
                return jsonify({"error": f"Could not extract text from the document. Reason: {original_text}"}), 500

            analysis_result = analyze_and_translate_with_gemini(original_text)

            if analysis_result.get("is_relevant") is False:
                final_output = {
                    "status": "Not Applicable",
                    "reason": "The document was determined to be not relevant to KMRL.",
                    "summary": analysis_result.get("summary", "No summary available.")
                }
            else:
                final_output = analysis_result

            return jsonify(final_output)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500
        finally:
            # Clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

    return jsonify({"error": "File type not allowed."}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5001)