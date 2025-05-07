from flask import Flask, request, jsonify
from flask_cors import CORS
print("🔧 Bắt đầu app.py...")

from model_utils import transcribe
from pronunciation_evaluator import evaluate_pronunciation

print("✅ Đã import xong model_utils và pronunciation_evaluator")

import os
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add root route to help with debugging
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Speech to Phoneme API is running. Use POST /upload to transcribe audio.",
        "features": ["Raw phoneme output", "IPA phoneme conversion", "Pronunciation evaluation"]
    })

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files["audio"]
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Secure the filename to prevent directory traversal attacks
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        
        print(f"🎤 Saving audio file to {path}")
        file.save(path)

        # Check if IPA output is requested (default is True)
        output_ipa = request.form.get('output_ipa', 'true').lower() == 'true'

        # Process the audio file
        print(f"🔍 Transcribing audio file...")
        result = transcribe(path, output_ipa=output_ipa)
        
        response = {
            "status": "success",
            "filename": filename
        }
        
        # Nếu kết quả là dictionary (chứa cả raw và ipa)
        if isinstance(result, dict):
            response["raw_phonemes"] = result["raw"]
            response["ipa_phonemes"] = result["ipa"]
        else:
            response["phonemes"] = result
            
        return jsonify(response)
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Endpoint để đánh giá phát âm của người dùng
    """
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files["audio"]
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
            
        # Kiểm tra văn bản tham chiếu
        reference_text = request.form.get('reference_text', '')
        if not reference_text:
            return jsonify({"error": "Reference text is required"}), 400

        # Lưu file âm thanh
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        
        print(f"🎤 Saving audio file to {path}")
        file.save(path)

        # Đánh giá phát âm
        print(f"🔍 Evaluating pronunciation...")
        evaluation = evaluate_pronunciation(path, reference_text)
        
        return jsonify({
            "status": "success",
            "filename": filename,
            "evaluation": evaluation
        })
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    print("✅ Server đang khởi động...")
    app.run(debug=True, host="0.0.0.0", port=5000)
