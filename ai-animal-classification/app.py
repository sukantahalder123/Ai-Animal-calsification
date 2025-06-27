from flask import Flask, request, jsonify, send_from_directory
from predict import predict_image
import os
import psycopg2
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT")
    )

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """Health check endpoint"""
    return "AI Animal Classifier API is running!"

@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload, prediction, and store results in the database"""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No filename provided"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        label = predict_image(filepath)

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO predictions (filename, label) VALUES (%s, %s)",
                    (filename, label)
                )
            conn.commit()

        return jsonify({"filename": filename, "prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    """Retrieve all prediction history records"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
            rows = cursor.fetchall()

    return jsonify([
        {
            "id": r[0],
            "filename": r[1],
            "label": r[2],
            "created_at": str(r[3])
        } for r in rows
    ])

@app.route('/search', methods=['GET'])
def search():
    """Search predictions by animal label"""
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Missing query. Use ?query=animal_name"}), 400

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM predictions WHERE LOWER(label) = %s ORDER BY created_at DESC",
                (query.lower(),)
            )
            rows = cursor.fetchall()

    return jsonify([
        {
            "id": r[0],
            "filename": r[1],
            "label": r[2],
            "image_url": f"http://localhost:5000/uploads/{r[1]}",
            "created_at": str(r[3])
        } for r in rows
    ])

if __name__ == '__main__':
    app.run(debug=True)
