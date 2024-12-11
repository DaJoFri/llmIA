from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import os

# Configuración de la aplicación Flask
app = Flask(__name__)
CORS(app)  # Permite solicitudes de otros dominios

# Ruta para servir el HTML
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Cargar el modelo entrenado
modelPath = 'modeloLBPHFace.xml'
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(modelPath)

# Configuración de la cámara
cap = cv2.VideoCapture(0)  # Cambia a un video si no tienes cámara
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

categories = ['Criminal', 'No Criminal']  # Etiquetas según el entrenamiento

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
        label, confidence = face_recognizer.predict(img)
        result = {
            "label": label,
            "confidence": confidence
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500

# Reconocimiento facial en tiempo real
def real_time_recognition():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            label = categories[result[0]]
            confidence = result[1]

            if confidence < 70:  # Ajusta el umbral si es necesario
                color = (0, 0, 255) if label == 'Criminal' else (0, 255, 0)
                cv2.putText(frame, f'{label} ({confidence:.2f})', (x, y - 10), 1, 1.3, color, 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            else:
                cv2.putText(frame, 'Desconocido', (x, y - 10), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        cv2.imshow('Reconocimiento', frame)
        if cv2.waitKey(1) == 27:  # Presiona ESC para salir
            break

# Si se ejecuta directamente el script, inicia la aplicación y el reconocimiento en tiempo real
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
