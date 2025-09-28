import os
import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests  # Using requests library for all API calls
from PIL import Image
from dotenv import load_dotenv
import fitz  # PyMuPDF for reliable PDF text extraction

# === 1. CONFIGURATION ===
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"  # Using the stable model name
# The direct REST API endpoint URL
# Change it to this (Correct)
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_NAME}:generateContent"
# --- Flask App Settings ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not GEMINI_API_KEY:
    print("🔴 ERROR: Gemini API key is not set in the .env file.")
else:
    print("✅ Gemini API key loaded successfully.")

# === 2. HELPER FUNCTIONS for REST API ===

def image_to_base64(image: Image.Image):
    """Converts a PIL Image to a base64 encoded string for the API."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_gemini_api(payload):
    """Central function to make calls to the Gemini REST API with robust error handling."""
    headers = {'Content-Type': 'application/json'}
    params = {'key': GEMINI_API_KEY}
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        
        response_json = response.json()
        if 'candidates' not in response_json or not response_json['candidates']:
            # Handle cases where the API returns a success status but no content (e.g., safety blocks)
            prompt_feedback = response_json.get('promptFeedback', {})
            return {"error": f"API returned no content. Reason: {prompt_feedback}"}

        return response_json['candidates'][0]['content']['parts'][0]

    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP Error: {http_err}")
        # Extract detailed error message from Google's response
        error_details = http_err.response.json() if http_err.response and http_err.response.content else str(http_err)
        return {"error": f"API call failed with status {http_err.response.status_code}: {error_details}"}
    except Exception as e:
        print(f"❌ An unexpected error occurred during API call: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


def process_image_with_gemini(image_path: str):
    """Orchestrates image processing (classification, then OCR/summary) via REST API."""
    print(f"--- Processing Image: {image_path} ---")
    image = Image.open(image_path).convert("RGB")
    base64_image = image_to_base64(image)
    
    # Step 1: Classify the image
    classify_prompt = "Is this image primarily a text document or an engineering/technical diagram? Answer with only the single word: 'document' or 'diagram'."
    classify_payload = {"contents": [{"parts": [{"text": classify_prompt}, {"inline_data": {"mime_type": "image/png", "data": base64_image}}]}]}
    
    classification_result = call_gemini_api(classify_payload)
    if "error" in classification_result:
        return f"Error during classification: {classification_result['error']}"
    
    image_type = "diagram" if "diagram" in classification_result.get('text', '').lower() else "document"
    print(f"✅ Image classified as: {image_type}")

    # Step 2: Generate summary or OCR text based on classification
    if image_type == "diagram":
        prompt = "You are an engineering assistant for Kochi Metro Rail Limited (KMRL). Describe this technical diagram in a concise summary. Identify key components and their apparent purpose."
    else:
        prompt = "Extract all text from this image. Preserve formatting and line breaks as accurately as possible."
        
    process_payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": base64_image}}
            ]
        }]
    }    
    result = call_gemini_api(process_payload)
    
    return result.get('text', f"Error processing image: {result.get('error', 'Unknown error')}")


def extract_text_from_document(file_path: str):
    """Extracts text from local PDF or TXT files. Does not use the API."""
    print(f"--- Processing Document: {file_path} ---")
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
            print(f"❌ Error reading PDF: {e}")
            return f"Error processing PDF: {e}"
    return "Unsupported document type."


def analyze_and_translate_with_gemini(text_to_analyze: str):
    """Performs the final analysis and translation using the REST API with JSON mode."""
    print("--- Analyzing text with Gemini REST API ---")
    json_schema = {"type": "object", "properties": {"is_relevant": {"type": "boolean"}, "summary": {"type": "string"}, "actions_required": {"type": "array", "items": {"type": "object", "properties": {"action": {"type": "string"}, "priority": {"type": "string", "enum": ["High", "Medium", "Low"]}, "deadline": {"type": "string"}, "notes": {"type": "string"}}, "required": ["action", "priority"]}}, "departments_to_notify": {"type": "array", "items": {"type": "string"}}, "cross_document_flags": {"type": "array", "items": {"type": "object", "properties": {"related_document_type": {"type": "string"}, "related_issue": {"type": "string"}}, "required": ["related_document_type", "related_issue"]}}}, "required": ["is_relevant", "summary"]}
    
    prompt = f"You are an AI assistant for KMRL. First, if the following text is primarily in Malayalam, translate it to professional English. Then, using the English text, generate a JSON object that strictly adheres to the provided schema. Text to analyze: --- {text_to_analyze} ---"
    
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": json_schema
        }
    }
    
    result = call_gemini_api(payload)
    # The API returns the JSON as a string inside the 'text' field when a schema is used
    return result.get('text') if 'text' in result else json.dumps(result)


# === 3. FLASK ROUTES ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return jsonify({"status": "KMRL Document Analysis API is running"}), 200

@app.route('/analyze', methods=['POST'])
def analyze_document_route():
    if 'file' not in request.files: return jsonify({"error": "No file part in the request."}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename): return jsonify({"error": "No file selected or file type not allowed."}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            original_text = process_image_with_gemini(file_path)
        else: # .pdf, .txt
            original_text = extract_text_from_document(file_path)

        if not original_text or "Error" in original_text:
            return jsonify({"error": f"Could not extract text from document. Reason: {original_text}"}), 500
        
        analysis_json_string = analyze_and_translate_with_gemini(original_text)
        if not analysis_json_string or "error" in analysis_json_string:
            return jsonify({"error": f"Failed to get analysis from Gemini. Reason: {analysis_json_string}"}), 500
            
        analysis_result = json.loads(analysis_json_string)

        if analysis_result.get("is_relevant") is False:
             final_output = {"status": "Not Applicable", "reason": "The document was determined to be not relevant to KMRL.", "summary": analysis_result.get("summary")}
        else:
            final_output = analysis_result

        return jsonify(final_output)

    except Exception as e:
        print(f"An unexpected error occurred in the main route: {e}")
        return jsonify({"error": f"An unexpected server error occurred: {str(e)}"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
