from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Cargar el modelo LBPH desde el archivo XML
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model-wi.xml")

@app.route("/", methods=["GET"])
def index():
    return "Servidor funcionando. Sube una imagen para procesar."

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No se proporcionó ninguna imagen."}), 400
    
    file = request.files["image"]
    file_path = os.path.join("uploads", file.filename)
    file.save(Documents)
    
    # Leer la imagen como escala de grises
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Validar tamaño de imagen
    try:
        label, confidence = recognizer.predict(img)
        result = {
            "label": label,
            "confidence": confidence
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": "Error al procesar la imagen: " + str(e)}), 500

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

