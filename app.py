# app server start!

import os
import io
import csv
import uuid
import zipfile
import tempfile
import shutil
import base64
import warnings
import logging
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

# quiet mode
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def load_env():
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'").strip('"')

load_env()

# setup flask app
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
CORS(app)

limit_mb = int(os.getenv('MAX_UPLOAD_MB', 100))
app.config['MAX_CONTENT_LENGTH'] = limit_mb * 1024 * 1024 # limit upload size

grading_sessions = {} # store grading sessions

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.heic'} # supported images

from pillow_heif import register_heif_opener # fix heif images
register_heif_opener()

_models_loaded = False # flag for model status

def ensure_models_loaded(): # load brains if needed
    global _models_loaded
    if not _models_loaded:
        import grader # get grader mod
        from ocr import load_ocr_models
        load_ocr_models()
        _models_loaded = True

@app.route('/')
def index(): # show the web page
    return render_template('index.html')


@app.route('/api/grade', methods=['POST'])
def grade(): # the main grading engine
    model_answer = request.form.get('model_answer', '').strip()
    if not model_answer:
        return jsonify({"error": "Model answer is required"}), 400

    try:
        max_marks = int(request.form.get('max_marks', 0))
        if max_marks <= 0:
            raise ValueError
    except (ValueError, TypeError):
        return jsonify({"error": "Max marks must be a positive integer"}), 400

    if 'answer_sheets' not in request.files:
        return jsonify({"error": "No answer sheets ZIP file uploaded"}), 400

    file = request.files['answer_sheets']
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    temp_dir = tempfile.mkdtemp(prefix='grader_') # make a cozy temp folder

    try:
        try:
            zip_buffer = io.BytesIO(file.read())
            with zipfile.ZipFile(zip_buffer, 'r') as z:
                z.extractall(temp_dir) # unpack the files
        except zipfile.BadZipFile:
            return jsonify({"error": "Invalid ZIP file"}), 400

        image_paths = []
        for root, dirs, files in os.walk(temp_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__MACOSX'] # ignore junk folders
            for fname in sorted(files):
                if fname.startswith('.'):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(root, fname))

        if not image_paths:
            return jsonify({"error": "No image files found in ZIP"}), 400

        ensure_models_loaded() # wake up the models
        from ocr import process_answer_sheets
        from grader import grade_entire_class

        print(f"\n[app] Processing {len(image_paths)} answer sheets...")
        from ocr import load_ocr_models
        ocr_models = load_ocr_models()

        debug_mode = request.form.get('debug') == 'true'
        extracted_data = process_answer_sheets(image_paths, ocr_models, return_debug=debug_mode) # read the text

        ocr_text_map = {k: v['text'] for k, v in extracted_data.items()}

        print(f"[app] Grading {len(ocr_text_map)} answers...")
        results = grade_entire_class(model_answer, ocr_text_map, max_marks) # score them!

        session_id = str(uuid.uuid4()) # unique id for this run
        grading_sessions[session_id] = {
            "results": results,
            "model_answer": model_answer,
            "max_marks": max_marks
        }

        response_results = []
        for r in results:
            student_id = r["student"]
            extra = extracted_data.get(student_id, {})
            
            response_results.append({
                "student": student_id,
                "suggested_marks": r["suggested_marks"],
                "max_marks": r["max_marks"],
                "similarity": r["similarity"],
                "nli_label": r["nli_label"],
                "confidence": r["confidence"],
                "extracted_text": r.get("answer_given", ""),
                "debug_ocr": extra.get("debug", []),
                "image_b64": extra.get("scanned_b64", ""),
                "nli_detail": {
                    "contradiction_count": r["nli_detail"]["contradiction_count"],
                    "entailment_count": r["nli_detail"]["entailment_count"],
                    "total_sentences": r["nli_detail"]["total_sentences"]
                }
            })

        return jsonify({
            "success": True,
            "session_id": session_id,
            "total_sheets": len(image_paths),
            "results": response_results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True) # clean up the mess


@app.route('/api/grade-manual', methods=['POST'])
def grade_manual(): # skip ocr and grade directly
    data = request.json
    model_answer = data.get('model_answer', '').strip()
    max_marks = int(data.get('max_marks', 5))
    student_answers = data.get('student_answers', {})

    if not model_answer or not student_answers:
        return jsonify({"error": "Missing model answer or student answers"}), 400

    ensure_models_loaded()
    from grader import grade_entire_class

    print(f"[app] Manual Grading {len(student_answers)} answers...")
    results = grade_entire_class(model_answer, student_answers, max_marks) # math time!

    session_id = str(uuid.uuid4())
    grading_sessions[session_id] = {
        "results": results,
        "model_answer": model_answer,
        "max_marks": max_marks
    }

    response_results = []
    for r in results:
        response_results.append({
            "student": r["student"],
            "suggested_marks": r["suggested_marks"],
            "max_marks": r["max_marks"],
            "similarity": r["similarity"],
            "nli_label": r["nli_label"],
            "confidence": r["confidence"],
            "extracted_text": r.get("answer_given", ""),
            "nli_detail": r["nli_detail"]
        })

    return jsonify({
        "success": True,
        "session_id": session_id,
        "results": response_results
    })


@app.route('/api/download/<session_id>')
def download_csv(session_id): # save as spreadsheet
    session = grading_sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found or expired"}), 404

    results = session["results"]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Student",
        "Suggested Marks",
        "Max Marks",
        "Similarity",
        "NLI Verdict",
        "Confidence",
        "Contradictions",
        "Entailments",
        "Total Sentences",
        "Extracted Text"
    ])

    for r in results:
        writer.writerow([
            r["student"],
            r["suggested_marks"],
            r["max_marks"],
            r["similarity"],
            r["nli_label"],
            r["confidence"],
            r["nli_detail"]["contradiction_count"],
            r["nli_detail"]["entailment_count"],
            r["nli_detail"]["total_sentences"],
            r.get("answer_given", "")
        ])

    csv_bytes = output.getvalue().encode('utf-8')
    buffer = io.BytesIO(csv_bytes)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name='grading_results.csv' # name the file
    )


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Answer Sheet Grader — Starting Server")
    print("=" * 50)
    print("\nModels will load on first grading request.")
    
    host = os.getenv('FLASK_HOST', '0.0.0.0') # where to run
    port = int(os.getenv('PORT', 5000)) # which port
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true' # debug mode?

    print(f"Open http://localhost:{port} in your browser.\n")
    app.run(host=host, port=port, debug=debug) # go go go!
