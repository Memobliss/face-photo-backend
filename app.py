from flask import Flask, request, jsonify
import face_recognition
import requests
import tempfile
from PIL import Image
from googleapiclient.discovery import build
import os

app = Flask(__name__)

FOLDER_ID = os.getenv('FOLDER_ID')
API_KEY = os.getenv('API_KEY')

def get_drive_images():
    service = build('drive', 'v3', developerKey=API_KEY)
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and mimeType contains 'image/'",
        fields="files(id, name)"
    ).execute()
    return results.get('files', [])

@app.route('/search', methods=['POST'])
def search_faces():
    uploaded = request.files['image']
    query_image = face_recognition.load_image_file(uploaded)
    query_encodings = face_recognition.face_encodings(query_image)
    if not query_encodings:
        return jsonify({'error': 'No face detected'}), 400

    matched_images = []
    for file in get_drive_images():
        file_id = file['id']
        image_url = f"https://drive.google.com/uc?id={file_id}"
        try:
            response = requests.get(image_url, stream=True)
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
            ref_image = face_recognition.load_image_file(tmp_file.name)
            ref_encodings = face_recognition.face_encodings(ref_image)
            if ref_encodings and face_recognition.compare_faces([ref_encodings[0]], query_encodings[0])[0]:
                matched_images.append(image_url)
        except Exception as e:
            continue

    return jsonify({'matches': matched_images})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
