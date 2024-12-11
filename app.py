from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Permite solicitudes de otros dominios

# Ruta para servir el HTML
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Configurar el modelo para leer desde un archivo XML
recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read("model-wi.xml")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# Ruta para manejar la subida de imágenes
@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No se proporcionó ninguna imagen."}), 400
        
        file = request.files["image"]
        file_path = os.path.join("uploads", file.filename)  # Define la ruta donde se guardará el archivo
        os.makedirs("uploads", exist_ok=True)  # Crea el directorio si no existe
        file.save(file_path)  # Guarda la imagen en la ruta especificada
        
        # Leer la imagen como escala de grises
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({"error": "No se pudo leer la imagen. Asegúrate de que sea válida."}), 400

        # Validar tamaño de imagen
        label, confidence = recognizer.predict(img)
        result = {
            "label": label,
            "confidence": confidence
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
